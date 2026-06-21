import argparse
from pathlib import Path

from .main_contract import (
    build_main_contract_continuous_frame,
    build_main_contract_continuous_frame_for_date_range,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stitch local commodity futures contracts into a main-contract series"
    )
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--commodity_name", required=True)
    parser.add_argument("--year")
    parser.add_argument("--start_date")
    parser.add_argument("--end_date")
    parser.add_argument("--symbol", default="fu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    has_range = bool(args.start_date or args.end_date)
    if has_range and not (args.start_date and args.end_date):
        parser.error("--start_date and --end_date must be provided together")
    if not has_range and not args.year:
        parser.error("either --year or --start_date/--end_date is required")
    return args


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.start_date and args.end_date:
        stitched = build_main_contract_continuous_frame_for_date_range(
            Path(args.raw_root),
            args.commodity_name,
            args.start_date,
            args.end_date,
            args.symbol,
        )
    else:
        stitched = build_main_contract_continuous_frame(
            Path(args.raw_root), args.commodity_name, args.year, args.symbol
        )
    stitched.to_csv(output, index=False)


if __name__ == "__main__":
    main()
