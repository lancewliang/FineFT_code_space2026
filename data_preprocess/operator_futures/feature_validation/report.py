from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from operator_futures.feature_validation.models import StageResult, ValidationConfig, ValidationReport


def build_report(config: ValidationConfig, stages: list[StageResult]) -> ValidationReport:
    return ValidationReport(config=config, stages=stages)


def render_json_report(report: ValidationReport) -> dict[str, Any]:
    status_counts = {status: 0 for status in ["pass", "fail", "partial", "error"]}
    for stage in report.stages:
        status_counts[stage.status] += 1
    return {
        "config": {
            "root_path": str(report.config.root_path),
            "report_dir": str(report.config.report_dir),
            "symbol": report.config.symbol,
            "target_freq": report.config.target_freq,
            "start_date": report.config.start_date,
            "end_date": report.config.end_date,
            "tolerance": report.config.tolerance,
            "sample_size": report.config.sample_size,
            "orderbook_depth": report.config.orderbook_depth,
        },
        "summary": {
            "total_stages": len(report.stages),
            "failed_stage_count": status_counts["fail"] + status_counts["error"],
            "partial_stage_count": status_counts["partial"],
            "status_counts": status_counts,
        },
        "stages": [asdict(stage) for stage in report.stages],
    }


def render_markdown_report(report: ValidationReport) -> str:
    lines = [
        "# Feature Validation Report",
        "",
        f"- symbol: `{report.config.symbol}`",
        f"- target_freq: `{report.config.target_freq}`",
        f"- date_range: `{report.config.start_date}` to `{report.config.end_date}`",
        f"- tolerance: `{report.config.tolerance}`",
        f"- sample_size: `{report.config.sample_size}`",
        "",
        "| stage | status | checked | missing | extra | unverified | mismatched | max_abs_diff |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in report.stages:
        lines.append(
            f"| {stage.stage} | {stage.status} | {stage.checked_columns} | "
            f"{len(stage.missing_columns)} | {len(stage.extra_columns)} | "
            f"{len(stage.unverified_columns)} | {len(stage.mismatched_columns)} | "
            f"{stage.max_abs_diff} |"
        )
    for stage in report.stages:
        lines.extend(["", f"## {stage.stage}", "", f"- status: `{stage.status}`"])
        if stage.message:
            lines.append(f"- message: {stage.message}")
        for label, values in [
            ("missing_columns", stage.missing_columns),
            ("extra_columns", stage.extra_columns),
            ("unverified_columns", stage.unverified_columns),
            ("mismatched_columns", stage.mismatched_columns),
        ]:
            if values:
                lines.append(f"- {label}: `{', '.join(values[:50])}`")
        if stage.sample_failures:
            lines.extend(["", "| column | timestamp | actual | expected | abs_diff |", "|---|---|---:|---:|---:|"])
            for failure in stage.sample_failures:
                lines.append(
                    f"| {failure.column} | {failure.timestamp} | {failure.actual} | "
                    f"{failure.expected} | {failure.abs_diff} |"
                )
    return "\n".join(lines) + "\n"


def write_reports(report: ValidationReport, report_dir: Path) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{report.config.symbol}_{report.config.target_freq}_"
        f"{report.config.start_date}_{report.config.end_date}"
    )
    markdown_path = report_dir / f"{stem}.md"
    json_path = report_dir / f"{stem}.json"
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
    json_path.write_text(
        json.dumps(render_json_report(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return markdown_path, json_path
