#!/usr/bin/env python3
"""
Commodity Futures Feature Selection Analyzer

Parses feature selection output directories containing:
- feature_selection_manifest.json
- aggregate_metrics.csv
- state_features.npy
- per_contract/*.csv

Provides statistical summaries, filter breakdowns, and quality diagnostics.
"""

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def load_feature_selection_data(dir_path: pathlib.Path) -> Dict[str, Any]:
    """Loads manifest, aggregate metrics, and selected features from target directory."""
    if not dir_path.exists():
        raise FileNotFoundError(f"Target directory does not exist: {dir_path}")

    manifest_path = dir_path / "feature_selection_manifest.json"
    metrics_path = dir_path / "aggregate_metrics.csv"
    state_path = dir_path / "state_features.npy"

    data = {
        "dir_path": str(dir_path),
        "manifest": None,
        "metrics_df": None,
        "state_features": None,
        "per_contract_df": None,
    }

    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            data["manifest"] = json.load(f)

    if metrics_path.exists():
        data["metrics_df"] = pd.read_csv(metrics_path)

    if state_path.exists():
        data["state_features"] = np.load(state_path, allow_pickle=True).tolist()

    per_contract_dir = dir_path / "per_contract"
    if per_contract_dir.exists():
        contract_files = list(per_contract_dir.glob("*_metrics.csv"))
        contract_data = {}
        for cf in contract_files:
            contract_name = cf.name.replace("_metrics.csv", "")
            try:
                contract_data[contract_name] = pd.read_csv(cf)
            except Exception:
                pass
        data["per_contract"] = contract_data

    return data


def summarize_selection(data: Dict[str, Any]) -> Dict[str, Any]:
    """Generates summary statistics and filter breakdown."""
    manifest = data.get("manifest") or {}
    metrics_df: Optional[pd.DataFrame] = data.get("metrics_df")
    state_features: List[str] = data.get("state_features") or manifest.get("selected_features", [])

    total_candidate_features = len(metrics_df) if metrics_df is not None else 0
    selected_count = len(state_features)

    summary = {
        "symbol": manifest.get("symbol", "N/A"),
        "target_freq": manifest.get("target_freq", "N/A"),
        "stage": manifest.get("stage", "N/A"),
        "candidate_feature_count": total_candidate_features,
        "selected_feature_count": selected_count,
        "selection_ratio": round(selected_count / max(1, total_candidate_features), 4),
        "mandatory_features": manifest.get("mandatory_state_features", []),
        "filter_stage_drops": {},
    }

    filter_results = manifest.get("filter_results", {})
    for stage_name, dropped in filter_results.items():
        summary["filter_stage_drops"][stage_name] = len(dropped)

    if metrics_df is not None and not metrics_df.empty:
        # Evaluate selected vs non-selected feature metrics
        if state_features:
            selected_df = metrics_df[metrics_df["feature"].isin(state_features)]
        else:
            selected_df = pd.DataFrame()

        summary["metrics_overview"] = {
            "all_features_ic_mean_avg": float(metrics_df["IC_Mean"].abs().mean()) if "IC_Mean" in metrics_df else None,
            "selected_ic_mean_avg": float(selected_df["IC_Mean"].abs().mean()) if not selected_df.empty and "IC_Mean" in selected_df else None,
            "all_features_perm_imp_avg": float(metrics_df["Permutation Importance_Mean"].mean()) if "Permutation Importance_Mean" in metrics_df else None,
            "selected_perm_imp_avg": float(selected_df["Permutation Importance_Mean"].mean()) if not selected_df.empty and "Permutation Importance_Mean" in selected_df else None,
        }

        # Top 10 features by IC and Permutation Importance
        if "IC_Mean" in metrics_df:
            top_ic = metrics_df.assign(abs_ic=metrics_df["IC_Mean"].abs()).sort_values(by="abs_ic", ascending=False).head(10)
            summary["top_10_by_ic"] = top_ic[["feature", "IC_Mean", "RankIC_Mean", "Permutation Importance_Mean"]].to_dict(orient="records")

        if "Permutation Importance_Mean" in metrics_df:
            top_perm = metrics_df.sort_values(by="Permutation Importance_Mean", ascending=False).head(10)
            summary["top_10_by_perm_imp"] = top_perm[["feature", "Permutation Importance_Mean", "IC_Mean", "Sharpe_Mean"]].to_dict(orient="records")

    return summary


def format_markdown_report(summary: Dict[str, Any]) -> str:
    """Formats summary data into a clean Markdown report."""
    lines = []
    lines.append(f"# Feature Selection Analysis Report ({summary['symbol'].upper()} - {summary['target_freq']} - {summary['stage']})")
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append(f"- **Symbol**: `{summary['symbol']}`")
    lines.append(f"- **Target Frequency**: `{summary['target_freq']}`")
    lines.append(f"- **Pipeline Stage**: `{summary['stage']}`")
    lines.append(f"- **Total Candidate Features**: {summary['candidate_feature_count']}")
    lines.append(f"- **Selected Features Count**: {summary['selected_feature_count']}")
    lines.append(f"- **Selection Ratio**: {summary['selection_ratio'] * 100:.2f}%")
    lines.append("")

    lines.append("## 2. Selection Pipeline Drop Stages")
    lines.append("| Pipeline Stage | Dropped Feature Count |")
    lines.append("|---|---|")
    for stage_name, count in summary.get("filter_stage_drops", {}).items():
        lines.append(f"| {stage_name} | {count} |")
    lines.append("")

    if "metrics_overview" in summary:
        ov = summary["metrics_overview"]
        lines.append("## 3. Metrics Comparison (Candidate vs Selected)")
        lines.append("| Metric | All Candidates | Selected Features |")
        lines.append("|---|---|---|")
        lines.append(f"| Mean |IC| | {ov.get('all_features_ic_mean_avg', 0):.4f} | {ov.get('selected_ic_mean_avg', 0):.4f} |")
        lines.append(f"| Mean Permutation Importance | {ov.get('all_features_perm_imp_avg', 0):.4f} | {ov.get('selected_perm_imp_avg', 0):.4f} |")
        lines.append("")

    if "top_10_by_ic" in summary:
        lines.append("## 4. Top 10 Features by Absolute IC")
        lines.append("| Feature | IC Mean | Rank IC Mean | Permutation Importance |")
        lines.append("|---|---|---|---|")
        for item in summary["top_10_by_ic"]:
            lines.append(f"| `{item['feature']}` | {item['IC_Mean']:.4f} | {item['RankIC_Mean']:.4f} | {item['Permutation Importance_Mean']:.4f} |")
        lines.append("")

    if "top_10_by_perm_imp" in summary:
        lines.append("## 5. Top 10 Features by Permutation Importance")
        lines.append("| Feature | Permutation Importance | IC Mean | Sharpe Mean |")
        lines.append("|---|---|---|---|")
        for item in summary["top_10_by_perm_imp"]:
            lines.append(f"| `{item['feature']}` | {item['Permutation Importance_Mean']:.4f} | {item['IC_Mean']:.4f} | {item['Sharpe_Mean']:.4f} |")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze Feature Selection Run Output")
    parser.add_argument("--dir", type=str, default="PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/30min/fu/train",
                        help="Path to feature selection directory")
    parser.add_argument("--format", type=str, choices=["markdown", "json", "both"], default="markdown",
                        help="Output report format")
    parser.add_argument("--output", type=str, default=None, help="Save report to file")
    args = parser.parse_args()

    dir_path = pathlib.Path(args.dir)
    data = load_feature_selection_data(dir_path)
    summary = summarize_selection(data)

    if args.format in ["markdown", "both"]:
        md_text = format_markdown_report(summary)
        print(md_text)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(md_text)

    if args.format in ["json", "both"]:
        json_text = json.dumps(summary, indent=2, ensure_ascii=False)
        if args.format == "json":
            print(json_text)
        if args.output and args.format == "json":
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(json_text)


if __name__ == "__main__":
    main()
