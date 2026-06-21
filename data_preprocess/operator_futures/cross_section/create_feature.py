import polars as pl
import os
import argparse
import sys

sys.path.append(".")
from operator_futures.util import match_strings_in_range
from memory_profiler import profile
from operator_futures.cross_section.base_feature_util import *

parser = argparse.ArgumentParser()
# data path
parser.add_argument(
    "--root_path",
    type=str,
    default=".",
    help="the path of storing the data",
)
parser.add_argument(
    "--data_path",
    type=str,
    default="PREPROCESS_DATASET/binance-futures/",
    help="the path of storing the data",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="PREPROCESS_DATASET/binance-futures/CROSS_SECTION",
    help="the path of storing the data",
)
parser.add_argument(
    "--symbols", type=str, default="BTCUSDT", help="the name of the ticker"
)
# date
parser.add_argument(
    "--date",
    type=str,
    default="2023-01-01",
    help="the path to save the data",
)
# freq
parser.add_argument(
    "--target_freq",
    type=str,
    default="10s",
    help="the date of start",
    choices=["10s", "1min", "5min","10min", "30min", "1H", "1D"],
)
parser.add_argument(
    "--market_type",
    type=str,
    default="crypto_futures",
    choices=["crypto_futures", "commodity_futures"],
    help="the market type of the preprocessed data",
)
parser.add_argument(
    "--orderbook_depth",
    type=int,
    default=25,
    help="the available orderbook depth",
)


def main(args):
    args.data_path = os.path.join(args.root_path, args.data_path)
    args.save_path = os.path.join(args.root_path, args.save_path)
    base_feature = pl.read_ipc(
        os.path.join(
            args.data_path,
            "BASE_FEATURE",
            args.symbols,
            args.target_freq,
            args.date + ".feather",
        )
    )
    snapshot = pl.read_ipc(
        os.path.join(
            args.data_path,
            "DOWNSCALE_ORDERBOOK_25",
            args.symbols,
            args.target_freq,
            args.date + ".feather",
        )
    )
    kline_feature = process_k_line_feature(base_feature)
    quotes_feature = process_quotes_n_feature(base_feature)
    snapshot_feature = process_snapshot_features(
        snapshot, depth=args.orderbook_depth
    )

    if not os.path.exists(
        os.path.join(args.save_path, "KLINE_FEATURE", args.symbols, args.target_freq)
    ):
        os.makedirs(
            os.path.join(
                args.save_path, "KLINE_FEATURE", args.symbols, args.target_freq
            )
        )
    if not os.path.exists(
        os.path.join(args.save_path, "QUOTES_FEATURE", args.symbols, args.target_freq)
    ):
        os.makedirs(
            os.path.join(
                args.save_path, "QUOTES_FEATURE", args.symbols, args.target_freq
            )
        )
    if not os.path.exists(
        os.path.join(args.save_path, "SNAPSHOT_FEATURE", args.symbols, args.target_freq)
    ):
        os.makedirs(
            os.path.join(
                args.save_path, "SNAPSHOT_FEATURE", args.symbols, args.target_freq
            )
        )
    kline_feature.write_ipc(
        os.path.join(
            args.save_path,
            "KLINE_FEATURE",
            args.symbols,
            args.target_freq,
            args.date + ".feather",
        )
    )
    quotes_feature.write_ipc(
        os.path.join(
            args.save_path,
            "QUOTES_FEATURE",
            args.symbols,
            args.target_freq,
            args.date + ".feather",
        )
    )
    snapshot_feature.write_ipc(
        os.path.join(
            args.save_path,
            "SNAPSHOT_FEATURE",
            args.symbols,
            args.target_freq,
            args.date + ".feather",
        )
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
