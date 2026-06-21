import argparse
from pathlib import Path

import pandas as pd

from .downscale import (
    create_second_level_snapshots,
    downscale_base_features,
    downscale_derivative_reference,
    downscale_orderbook,
)


def downscale_continuous_by_trading_day(
    continuous_file: Path,
    output_root: Path,
    target_freq: str,
    symbol: str,
    depth: int = 5,
) -> None:
    raw = pd.read_csv(continuous_file)
    for trading_day, day_frame in raw.groupby(raw["TradingDay"].astype(str), sort=True):
        second = create_second_level_snapshots(day_frame)
        outputs = {
            "DOWNSCALE_DERTIC": downscale_derivative_reference(
                second, target_freq, symbol
            ),
            "DOWNSCALE_ORDERBOOK_25": downscale_orderbook(
                second, target_freq, depth=depth
            ),
            "BASE_FEATURE": downscale_base_features(second, target_freq),
        }
        for folder, frame in outputs.items():
            path = output_root / folder / symbol / target_freq
            path.mkdir(parents=True, exist_ok=True)
            frame.to_feather(path / f"{trading_day}.feather")


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
