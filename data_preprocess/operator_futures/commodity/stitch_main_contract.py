import argparse
import logging
from pathlib import Path
import time

from .main_contract import write_main_contract_daily_files_for_date_range


logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stitch local commodity futures contracts into daily main-contract files"
    )
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--commodity_name", required=True)
    parser.add_argument("--start_date", required=True)
    parser.add_argument("--end_date", required=True)
    parser.add_argument("--symbol", default="fu")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    started_at = time.monotonic()
    output_dir = Path(args.output_dir)
    logger.info(
        "Starting commodity main-contract daily stitch: raw_root=%s commodity=%s symbol=%s start_date=%s end_date=%s output_dir=%s",
        args.raw_root,
        args.commodity_name,
        args.symbol,
        args.start_date,
        args.end_date,
        output_dir,
    )
    written = write_main_contract_daily_files_for_date_range(
        raw_root=Path(args.raw_root),
        commodity_name=args.commodity_name,
        output_dir=output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        symbol=args.symbol,
    )
    logger.info(
        "Wrote stitched commodity main-contract daily files: output_dir=%s files=%d elapsed_seconds=%.2f",
        output_dir,
        len(written),
        time.monotonic() - started_at,
    )


if __name__ == "__main__":
    main()
