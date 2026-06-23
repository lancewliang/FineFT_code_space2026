import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
import argparse
import sys

sys.path.append(".")
from operator_futures.feature_validation.pandas_reference.util import match_strings_in_range
try:
    from memory_profiler import profile
except ModuleNotFoundError:  # pragma: no cover
    def profile(func):
        return func
from operator_futures.feature_validation.pandas_reference.cross_section.base_feature_util import *

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


def main(args):
    args.data_path = os.path.join(args.root_path, args.data_path)
    args.save_path = os.path.join(args.root_path, args.save_path)
    base_feature = pd.read_feather(
        os.path.join(
            args.data_path,
            "BASE_FEATURE",
            args.symbols,
            args.target_freq,
            args.date + ".feather",
        )
    )
    base_feature.index = base_feature["timestamp"]
    snapshot = pd.read_feather(
        os.path.join(
            args.data_path,
            "DOWNSCALE_ORDERBOOK_25",
            args.symbols,
            args.target_freq,
            args.date + ".feather",
        )
    )
    snapshot.index = snapshot["timestamp"]
    kline_feature = process_k_line_feature(base_feature)
    kline_feature.reset_index(inplace=True)
    quotes_feature = process_quotes_n_feature(base_feature)
    quotes_feature.reset_index(inplace=True)
    snapshot_feature = process_snapshot_features(snapshot)
    snapshot_feature.index.name = "timestamp"
    snapshot_feature.reset_index(inplace=True)

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
    kline_feature.to_feather(
        os.path.join(
            args.save_path,
            "KLINE_FEATURE",
            args.symbols,
            args.target_freq,
            args.date + ".feather",
        )
    )
    quotes_feature.to_feather(
        os.path.join(
            args.save_path,
            "QUOTES_FEATURE",
            args.symbols,
            args.target_freq,
            args.date + ".feather",
        )
    )
    snapshot_feature.to_feather(
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
