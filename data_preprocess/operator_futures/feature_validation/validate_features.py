from __future__ import annotations

import argparse
from pathlib import Path

from operator_futures.feature_validation.models import ValidationConfig, ValidationReport
from operator_futures.feature_validation.report import build_report, write_reports
from operator_futures.feature_validation.validators import validate_all_stages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate commodity feature artifacts")
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--target_freq", required=True)
    parser.add_argument("--start_date", required=True)
    parser.add_argument("--end_date", required=True)
    parser.add_argument("--report_dir")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--sample_size", type=int, default=200)
    parser.add_argument("--orderbook_depth", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root_path = Path(args.root_path).resolve()
    report_dir = (
        Path(args.report_dir).resolve()
        if args.report_dir
        else root_path / "log_futures" / "feature_validation"
    )
    config = ValidationConfig(
        root_path=root_path,
        symbol=args.symbol,
        target_freq=args.target_freq,
        start_date=args.start_date,
        end_date=args.end_date,
        report_dir=report_dir,
        tolerance=args.tolerance,
        sample_size=args.sample_size,
        orderbook_depth=args.orderbook_depth,
    )
    stages = validate_all_stages(config)
    report = build_report(config, stages)
    markdown_path, json_path = write_reports(report, report_dir)
    print(
        "feature validation configured: "
        f"symbol={args.symbol} target_freq={args.target_freq} "
        f"start_date={args.start_date} end_date={args.end_date} "
        f"report_dir={report_dir}"
    )
    print(f"feature validation markdown report: {markdown_path}")
    print(f"feature validation json report: {json_path}")
    if any(stage.status == "error" for stage in stages):
        return 1
    if any(stage.status == "fail" for stage in stages):
        return 1
    if any(stage.status == "partial" for stage in stages):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
