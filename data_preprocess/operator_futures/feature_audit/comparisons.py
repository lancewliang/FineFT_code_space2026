"""Matched sample comparison engine across horizons and data frequencies."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import polars as pl
import yaml

from .metrics import compute_spearman_correlation


@dataclass
class ComparisonConfig:
    schema_version: str
    protocol_version: str
    comparison_type: str  # "matched_frequency" or "matched_horizon"
    run_a: str
    run_b: str
    output_dir: str

    @classmethod
    def load(cls, path_str: str | Path) -> "ComparisonConfig":
        p = Path(path_str)
        if not p.exists():
            raise FileNotFoundError(f"Comparison config not found: {p}")
        text = p.read_text(encoding="utf-8")
        if p.suffix in [".yaml", ".yml"]:
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls(
            schema_version=data.get("schema_version", "multi_horizon_audit.comparison.v1"),
            protocol_version=data.get("protocol_version", "multi_horizon_audit.research.v1"),
            comparison_type=data.get("comparison_type", "matched_frequency"),
            run_a=data["run_a"],
            run_b=data["run_b"],
            output_dir=data["output_dir"],
        )


class ComparisonRunner:
    """Executes paired comparison between two audit runs."""

    def __init__(self, config: ComparisonConfig):
        self.config = config

    def run(self) -> Dict[str, Any]:
        dir_a = Path(self.config.run_a)
        dir_b = Path(self.config.run_b)
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Verify manifests
        man_a_file = dir_a / "audit_run_manifest.json"
        man_b_file = dir_b / "audit_run_manifest.json"
        if not man_a_file.exists() or not man_b_file.exists():
            raise FileNotFoundError("One or both audit run manifests are missing")

        with open(man_a_file) as f:
            man_a = json.load(f)
        with open(man_b_file) as f:
            man_b = json.load(f)

        if man_a.get("protocol_version") != man_b.get("protocol_version"):
            raise ValueError(
                f"Incompatible protocol versions between runs: '{man_a.get('protocol_version')}' vs '{man_b.get('protocol_version')}'"
            )

        if man_a.get("symbol") != man_b.get("symbol"):
            raise ValueError(f"Different symbols cannot be compared directly: '{man_a.get('symbol')}' vs '{man_b.get('symbol')}'")

        # 2. Load row labels
        labels_a = pl.read_parquet(dir_a / "row_labels.parquet").filter(pl.col("valid_mask"))
        labels_b = pl.read_parquet(dir_b / "row_labels.parquet").filter(pl.col("valid_mask"))

        # 3. Match keys
        join_keys = ["contract", "timestamp", "trading_day", "horizon"]
        matched = labels_a.join(labels_b, on=join_keys, suffix="_b", how="inner")
        matched_count = len(matched)

        status = "completed" if matched_count > 0 else "NO_COMMON_SAMPLES"

        # 4. Compute matched metrics
        comparison_records: List[Dict[str, Any]] = []
        if matched_count > 0:
            horizons = matched["horizon"].unique().to_list()
            for h in horizons:
                h_sub = matched.filter(pl.col("horizon") == h)
                ret_a = h_sub["target_return"].to_numpy()
                ret_b = h_sub["target_return_b"].to_numpy()

                corr_a, _ = compute_spearman_correlation(ret_a, ret_b)
                diff = float(np.mean(ret_a - ret_b)) if len(ret_a) > 0 else 0.0

                comparison_records.append({
                    "horizon": h,
                    "target_kind": "return",
                    "matched_samples": len(h_sub),
                    "mean_paired_difference": diff,
                    "target_correlation": corr_a,
                })

        cmp_df = pl.DataFrame(comparison_records)
        cmp_df.write_csv(out_dir / "comparison_metrics.csv")

        # 5. Emit comparison manifest
        manifest = {
            "schema_version": self.config.schema_version,
            "protocol_version": self.config.protocol_version,
            "comparison_type": self.config.comparison_type,
            "status": status,
            "run_a": str(dir_a),
            "run_b": str(dir_b),
            "matched_samples_count": matched_count,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(out_dir / "audit_comparison_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # 6. Render report
        report_lines = [
            f"# Multi-Horizon Feature Audit Comparison Report",
            "",
            f"- **Comparison Type**: `{self.config.comparison_type}`",
            f"- **Status**: `{status}`",
            f"- **Run A**: `{dir_a}`",
            f"- **Run B**: `{dir_b}`",
            f"- **Matched Samples**: `{matched_count}`",
            "",
            "## Important Interpretation Note",
            "> Matched comparisons evaluate exactly common observation points (matching contract, timestamp, and horizon).",
            "> However, note that if native data frequencies differ (e.g. 10min vs 30min), nominal lookback windows cover different physical historical spans.",
            "",
            "## Matched Comparison Metrics",
            "",
            "| Horizon | Matched Samples | Mean Paired Return Difference | Target Correlation |",
            "|---|---|---|---|",
        ]
        for row in comparison_records:
            report_lines.append(
                f"| `{row['horizon']}` | {row['matched_samples']} | {row['mean_paired_difference']:.6f} | {row['target_correlation']} |"
            )

        report_lines.append("")
        (out_dir / "comparison_report.md").write_text("\n".join(report_lines), encoding="utf-8")

        return manifest
