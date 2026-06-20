import argparse
from pathlib import Path

import pandas as pd

from .downscale import (
    create_second_level_snapshots,
    downscale_base_features,
    downscale_derivative_reference,
    downscale_orderbook,
    downscale_quote_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Downscale one commodity futures day")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--symbol", default="fu")
    parser.add_argument("--target_freq", default="5min")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.input)
    second = create_second_level_snapshots(raw)
    downscale_derivative_reference(second, args.target_freq, args.symbol).to_feather(
        output_dir / "derivative_reference.feather"
    )
    downscale_orderbook(second, args.target_freq, depth=5).to_feather(
        output_dir / "orderbook_5.feather"
    )
    downscale_base_features(second, args.target_freq).to_feather(
        output_dir / "base_feature.feather"
    )
    downscale_quote_features(second, args.target_freq).to_feather(
        output_dir / "quote_feature.feather"
    )


if __name__ == "__main__":
    main()
