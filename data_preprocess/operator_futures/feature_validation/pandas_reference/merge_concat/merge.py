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
    default="PREPROCESS_DATASET/binance-futures/MERGE_CONCAT",
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
    snapshot = pd.read_feather(
        "{}/DOWNSCALE_ORDERBOOK_25/{}/{}/{}.feather".format(
            args.data_path, args.symbols, args.target_freq, args.date
        )
    )
    der = pd.read_feather(
        "{}/DOWNSCALE_DERTIC/{}/{}/{}.feather".format(
            args.data_path, args.symbols, args.target_freq, args.date
        )
    )
    base_feature = pd.read_feather(
        "{}/BASE_FEATURE/{}/{}/{}.feather".format(
            args.data_path, args.symbols, args.target_freq, args.date
        )
    )
    snapshot_feature = pd.read_feather(
        "{}/CROSS_SECTION/SNAPSHOT_FEATURE/{}/{}/{}.feather".format(
            args.data_path, args.symbols, args.target_freq, args.date
        )
    )
    quotes_feature = pd.read_feather(
        "{}/CROSS_SECTION/QUOTES_FEATURE/{}/{}/{}.feather".format(
            args.data_path, args.symbols, args.target_freq, args.date
        )
    )
    kline_feature = pd.read_feather(
        "{}/CROSS_SECTION/KLINE_FEATURE/{}/{}/{}.feather".format(
            args.data_path, args.symbols, args.target_freq, args.date
        )
    )
    snapshot.set_index("timestamp", inplace=True)
    der.set_index("timestamp", inplace=True)
    base_feature.set_index("timestamp", inplace=True)
    snapshot_feature.set_index("timestamp", inplace=True)
    quotes_feature.set_index("timestamp", inplace=True)
    kline_feature.set_index("timestamp", inplace=True)

    der.drop(columns=["symbol"], inplace=True)
    # base_feature.drop(columns=["symbol"], inplace=True)
    reward_feature = pd.concat(
        [
            snapshot,
            der,
            snapshot_feature,
        ],
        axis=1,
    )
    reward_feature.reset_index(inplace=True)
    base_feature = pd.concat(
        [
            base_feature,
            quotes_feature,
            kline_feature,
        ],
        axis=1,
    )
    base_feature.reset_index(inplace=True)

    # merged_feature = pd.concat(
    #     [
    #         snapshot,
    #         der,
    #         base_feature,
    #         snapshot_feature,
    #         quotes_feature,
    #         kline_feature,
    #     ],
    #     axis=1,
    # )
    # merged_feature.reset_index(inplace=True)

    if not os.path.exists(
        os.path.join(
            args.save_path,
            "MERGED_FEATURE",
            args.symbols,
            args.target_freq,
            "CONCURRENT_FEATURE",
        )
    ):
        os.makedirs(
            os.path.join(
                args.save_path,
                "MERGED_FEATURE",
                args.symbols,
                args.target_freq,
                "CONCURRENT_FEATURE",
            )
        )
    if not os.path.exists(
        os.path.join(
            args.save_path,
            "MERGED_FEATURE",
            args.symbols,
            args.target_freq,
            "FUTURE_FEATURE",
        )
    ):
        os.makedirs(
            os.path.join(
                args.save_path,
                "MERGED_FEATURE",
                args.symbols,
                args.target_freq,
                "FUTURE_FEATURE",
            )
        )
    reward_feature.to_feather(
        os.path.join(
            args.save_path,
            "MERGED_FEATURE",
            args.symbols,
            args.target_freq,
            "CONCURRENT_FEATURE",
            args.date + ".feather",
        )
    )
    base_feature.to_feather(
        os.path.join(
            args.save_path,
            "MERGED_FEATURE",
            args.symbols,
            args.target_freq,
            "FUTURE_FEATURE",
            args.date + ".feather",
        )
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
