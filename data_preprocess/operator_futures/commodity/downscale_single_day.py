import argparse
from pathlib import Path

import polars as pl

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
    parser.add_argument("--output_dir", "--output_root", dest="output_dir", required=True)
    parser.add_argument("--symbol", default="fu")
    parser.add_argument("--target_freq", default="5min")
    parser.add_argument(
        "--date",
        default=None,
        help="Optional output date name. When set, files are written to the shared pipeline layout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pl.read_csv(args.input)
    second = create_second_level_snapshots(raw)
    der = downscale_derivative_reference(second, args.target_freq, args.symbol)
    orderbook = downscale_orderbook(second, args.target_freq, depth=5)
    base = downscale_base_features(second, args.target_freq, args.symbol).with_columns(
        pl.lit(args.symbol).alias("symbol")
    )
    quote = downscale_quote_features(second, args.target_freq)
    if args.date:
        der_path = output_dir / "DOWNSCALE_DERTIC" / args.symbol / args.target_freq
        orderbook_path = (
            output_dir / "DOWNSCALE_ORDERBOOK_25" / args.symbol / args.target_freq
        )
        base_path = output_dir / "BASE_FEATURE" / args.symbol / args.target_freq
        quote_path = output_dir / "COMMODITY_QUOTE_FEATURE" / args.symbol / args.target_freq
        for path in [der_path, orderbook_path, base_path, quote_path]:
            path.mkdir(parents=True, exist_ok=True)
        der.write_ipc(der_path / f"{args.date}.feather")
        orderbook.write_ipc(orderbook_path / f"{args.date}.feather")
        base.write_ipc(base_path / f"{args.date}.feather")
        quote.write_ipc(quote_path / f"{args.date}.feather")
    else:
        der.write_ipc(output_dir / "derivative_reference.feather")
        orderbook.write_ipc(output_dir / "orderbook_5.feather")
        base.write_ipc(output_dir / "base_feature.feather")
        quote.write_ipc(output_dir / "quote_feature.feather")


if __name__ == "__main__":
    main()
