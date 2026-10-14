"""Automated reporting and matrix generation for multi-horizon feature audit."""

from pathlib import Path
from typing import Any, Dict, Optional
import polars as pl


def generate_matrices(metrics_df: pl.DataFrame, result_dir: Path) -> None:
    """Generates feature x horizon correlation and coverage matrix tables."""
    return_metrics = metrics_df.filter(pl.col("target_kind") == "return")

    # Pivot spearman_corr
    corr_matrix = return_metrics.pivot(
        values="spearman_corr",
        index="feature",
        on="horizon"
    )
    corr_matrix.write_csv(result_dir / "feature_horizon_corr_matrix.csv")

    # Pivot coverage_ratio
    coverage_matrix = return_metrics.pivot(
        values="coverage_ratio",
        index="feature",
        on="horizon"
    )
    coverage_matrix.write_csv(result_dir / "feature_horizon_coverage_matrix.csv")


def generate_audit_report(
    manifest: Dict[str, Any],
    metrics_df: pl.DataFrame,
    invalid_summary_df: Optional[pl.DataFrame],
    output_path: Path
) -> None:
    """Renders the publication-ready Markdown audit report."""
    symbol = manifest.get("symbol", "")
    freq = manifest.get("frequency", "")
    schema_ver = manifest.get("schema_version", "")
    protocol_ver = manifest.get("protocol_version", "")
    completed_at = manifest.get("completed_at", "")

    lines = [
        f"# Multi-Horizon Feature Audit Report: {symbol.upper()} ({freq})",
        "",
        f"- **Schema Version**: `{schema_ver}`",
        f"- **Protocol Version**: `{protocol_ver}`",
        f"- **Evaluated Symbol**: `{symbol}`",
        f"- **Data Frequency**: `{freq}`",
        f"- **Completed At**: `{completed_at}`",
        "",
        "## Important Methodological & Exploratory Disclaimers",
        "",
        "> **Exploratory Research Notice**:",
        "> - Pointwise confidence intervals are unadjusted for multiple comparisons; they reflect sampling uncertainty and must be treated as exploratory.",
        "> - Audit results provide retrospective characterization and do NOT constitute an automatic feature selection/promotion/deletion gate.",
        "> - Cross-horizon composite total scores are omitted by architectural design (ADR-0054).",
        "",
        "## Candidate Features Performance & Uncertainty (Return Horizon)",
        "",
    ]

    # Add summary table for return target kind
    return_df = metrics_df.filter(pl.col("target_kind") == "return")
    lines.append("| Feature | Stream | Horizon | Spearman Corr | 95% CI (Base) | 95% CI (2x) | Sign Consistency | Coverage |")
    lines.append("|---|---|---|---|---|---|---|---|")

    for row in return_df.iter_rows(named=True):
        feat = row.get("feature", "")
        stream = row.get("legacy_stream_membership", "unassigned")
        h = row.get("horizon", "")
        corr = f"{row['spearman_corr']:.4f}" if row.get("spearman_corr") is not None else "null"
        
        ci_l = f"{row['ci_lower']:.4f}" if row.get("ci_lower") is not None else "N/A"
        ci_u = f"{row['ci_upper']:.4f}" if row.get("ci_upper") is not None else "N/A"
        ci_str = f"[{ci_l}, {ci_u}]" if ci_l != "N/A" else "INSUFFICIENT_DATA"

        ci_l2 = f"{row['ci_lower_2x']:.4f}" if row.get("ci_lower_2x") is not None else "N/A"
        ci_u2 = f"{row['ci_upper_2x']:.4f}" if row.get("ci_upper_2x") is not None else "N/A"
        ci_str2 = f"[{ci_l2}, {ci_u2}]" if ci_l2 != "N/A" else "INSUFFICIENT_DATA"

        sc = f"{row['sign_consistency']:.2%}" if row.get("sign_consistency") is not None else "N/A"
        cov = f"{row['coverage_ratio']:.2%}" if row.get("coverage_ratio") is not None else "0.0%"

        lines.append(f"| `{feat}` | `{stream}` | `{h}` | {corr} | {ci_str} | {ci_str2} | {sc} | {cov} |")

    lines.append("")
    if invalid_summary_df is not None and len(invalid_summary_df) > 0:
        lines.append("## Invalid Label Reason Distribution")
        lines.append("")
        lines.append("| Horizon | Invalid Reason | Count |")
        lines.append("|---|---|---|")
        for row in invalid_summary_df.iter_rows(named=True):
            lines.append(f"| `{row['horizon']}` | `{row['invalid_reason']}` | {row['count']} |")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
