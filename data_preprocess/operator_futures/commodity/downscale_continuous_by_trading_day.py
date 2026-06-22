import argparse
from datetime import datetime, timedelta
import logging
from pathlib import Path
import time

import polars as pl

from .downscale import (
    create_second_level_snapshots,
    downscale_base_features,
    downscale_derivative_reference,
    downscale_orderbook,
    downscale_quote_features,
)


logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _parse_date(value: str):
    return datetime.strptime(value, "%Y-%m-%d").date()


def _iter_iso_dates(start_date: str, end_date: str):
    current = _parse_date(start_date)
    end = _parse_date(end_date)
    if end <= current:
        raise ValueError(
            f"end_date must be greater than start_date for left-open range: "
            f"{start_date} -> {end_date}"
        )

    while current < end:
        yield current.isoformat()
        current += timedelta(days=1)


def _trading_day_output_name(trading_day: str) -> str:
    trading_day = str(trading_day)
    if len(trading_day) == 8 and trading_day.isdigit():
        return datetime.strptime(trading_day, "%Y%m%d").date().isoformat()
    return trading_day


def _write_downscaled_day(
    day_frame: pl.DataFrame,
    output_root: Path,
    target_freq: str,
    symbol: str,
    depth: int,
) -> str:
    trading_days = (
        day_frame.select(pl.col("TradingDay").cast(pl.Utf8).unique().sort())
        .to_series()
        .to_list()
    )
    if len(trading_days) != 1:
        raise ValueError(
            f"Daily continuous file must contain one TradingDay: {trading_days}"
        )
    trading_day = trading_days[0]
    second = create_second_level_snapshots(day_frame)
    outputs = {
        "DOWNSCALE_DERTIC": downscale_derivative_reference(
            second, target_freq, symbol
        ),
        "DOWNSCALE_ORDERBOOK_25": downscale_orderbook(
            second, target_freq, depth=depth
        ),
        "BASE_FEATURE": downscale_base_features(second, target_freq, symbol),
        "COMMODITY_QUOTE_FEATURE": downscale_quote_features(second, target_freq),
    }
    output_name = _trading_day_output_name(trading_day)
    for folder, frame in outputs.items():
        path = output_root / folder / symbol / target_freq
        path.mkdir(parents=True, exist_ok=True)
        frame.write_ipc(path / f"{output_name}.feather")
    return trading_day


def downscale_continuous_by_trading_day(
    input_dir: Path,
    output_root: Path,
    target_freq: str,
    symbol: str,
    start_date: str,
    end_date: str,
    depth: int = 5,
) -> None:
    started_at = time.monotonic()
    logger.info(
        "Starting commodity continuous downscale: input_dir=%s output_root=%s target_freq=%s symbol=%s start_date=%s end_date=%s depth=%d",
        input_dir,
        output_root,
        target_freq,
        symbol,
        start_date,
        end_date,
        depth,
    )
    processed = []
    skipped = []
    for date in _iter_iso_dates(start_date, end_date):
        daily_file = input_dir / f"{date}.csv"
        if not daily_file.exists():
            logger.warning(
                "Missing commodity continuous daily file: date=%s input=%s",
                date,
                daily_file,
            )
            skipped.append(date)
            continue

        raw = pl.read_csv(daily_file)
        logger.info(
            "Downscaling commodity continuous daily file: date=%s input=%s rows=%d",
            date,
            daily_file,
            raw.height,
        )
        trading_day = _write_downscaled_day(raw, output_root, target_freq, symbol, depth)
        processed.append(trading_day)

    if skipped:
        logger.warning(
            "Skipped commodity continuous daily files: dates=%s",
            ",".join(skipped),
        )
    logger.info(
        "Finished commodity continuous downscale: trading_days=%d skipped_dates=%d elapsed_seconds=%.2f",
        len(processed),
        len(skipped),
        time.monotonic() - started_at,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downscale continuous commodity main-contract daily files by TradingDay"
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--start_date", required=True)
    parser.add_argument("--end_date", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--target_freq", default="5min")
    parser.add_argument("--symbol", default="fu")
    parser.add_argument("--depth", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    downscale_continuous_by_trading_day(
        input_dir=Path(args.input_dir),
        output_root=Path(args.output_root),
        target_freq=args.target_freq,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        depth=args.depth,
    )


if __name__ == "__main__":
    main()
