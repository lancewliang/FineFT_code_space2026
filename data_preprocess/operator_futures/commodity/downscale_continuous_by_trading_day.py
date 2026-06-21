import argparse
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


def downscale_continuous_by_trading_day(
    continuous_file: Path,
    output_root: Path,
    target_freq: str,
    symbol: str,
    depth: int = 5,
) -> None:
    started_at = time.monotonic()
    logger.info(
        "Starting commodity continuous downscale: input=%s output_root=%s target_freq=%s symbol=%s depth=%d",
        continuous_file,
        output_root,
        target_freq,
        symbol,
        depth,
    )
    raw = pl.read_csv(continuous_file)
    trading_days = (
        raw.select(pl.col("TradingDay").cast(pl.Utf8).unique().sort())
        .to_series()
        .to_list()
    )
    logger.info(
        "Loaded continuous commodity file: rows=%d trading_days=%d",
        raw.height,
        len(trading_days),
    )
    for index, trading_day in enumerate(trading_days, start=1):
        day_frame = raw.filter(pl.col("TradingDay").cast(pl.Utf8) == trading_day)
        logger.info(
            "Downscaling TradingDay %s (%d/%d): rows=%d",
            trading_day,
            index,
            len(trading_days),
            day_frame.height,
        )
        second = create_second_level_snapshots(day_frame)
        outputs = {
            "DOWNSCALE_DERTIC": downscale_derivative_reference(
                second, target_freq, symbol
            ),
            "DOWNSCALE_ORDERBOOK_25": downscale_orderbook(
                second, target_freq, depth=depth
            ),
            "BASE_FEATURE": downscale_base_features(second, target_freq),
            "COMMODITY_QUOTE_FEATURE": downscale_quote_features(second, target_freq),
        }
        for folder, frame in outputs.items():
            path = output_root / folder / symbol / target_freq
            path.mkdir(parents=True, exist_ok=True)
            frame.write_ipc(path / f"{trading_day}.feather")
    logger.info(
        "Finished commodity continuous downscale: trading_days=%d elapsed_seconds=%.2f",
        len(trading_days),
        time.monotonic() - started_at,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downscale a continuous commodity main-contract file by TradingDay"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--target_freq", default="5min")
    parser.add_argument("--symbol", default="fu")
    parser.add_argument("--depth", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    downscale_continuous_by_trading_day(
        continuous_file=Path(args.input),
        output_root=Path(args.output_root),
        target_freq=args.target_freq,
        symbol=args.symbol,
        depth=args.depth,
    )


if __name__ == "__main__":
    main()
