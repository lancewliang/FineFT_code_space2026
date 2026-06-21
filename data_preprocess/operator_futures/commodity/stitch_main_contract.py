import argparse
import logging
from pathlib import Path
import time

from .main_contract import (
    build_main_contract_continuous_frame,
    build_main_contract_continuous_frame_for_date_range,
)


logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
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
    configure_logging()
    args = parse_args()
    started_at = time.monotonic()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Starting commodity main-contract stitch: raw_root=%s commodity=%s symbol=%s year=%s start_date=%s end_date=%s output=%s",
        args.raw_root,
        args.commodity_name,
        args.symbol,
        args.year,
        args.start_date,
        args.end_date,
        output,
    )
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
    stitched.write_csv(output)
    logger.info(
        "Wrote stitched commodity main-contract file: output=%s rows=%d elapsed_seconds=%.2f",
        output,
        stitched.height,
        time.monotonic() - started_at,
    )


if __name__ == "__main__":
    main()
