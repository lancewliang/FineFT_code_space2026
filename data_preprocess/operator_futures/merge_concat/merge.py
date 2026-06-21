import polars as pl
import numpy as np
import os
import re
from datetime import datetime
import argparse
import sys

sys.path.append(".")
from operator_futures.util import match_strings_in_range
from memory_profiler import profile

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


def build_daily_feature_frames(
    snapshot: pl.DataFrame,
    der: pl.DataFrame,
    base_feature: pl.DataFrame,
    snapshot_feature: pl.DataFrame,
    quotes_feature: pl.DataFrame,
    kline_feature: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    der_without_symbol = der.drop("symbol") if "symbol" in der.columns else der
    reward_feature = (
        snapshot.join(der_without_symbol, on="timestamp", how="left")
        .join(snapshot_feature, on="timestamp", how="left")
    )
    future_feature = (
        base_feature.join(quotes_feature, on="timestamp", how="left")
        .join(kline_feature, on="timestamp", how="left")
    )
    return reward_feature, future_feature


def main(args):
    args.data_path = os.path.join(args.root_path, args.data_path)
    args.save_path = os.path.join(args.root_path, args.save_path)
    snapshot = pl.read_ipc(
        "{}/DOWNSCALE_ORDERBOOK_25/{}/{}/{}.feather".format(
            args.data_path, args.symbols, args.target_freq, args.date
        )
    )
    der = pl.read_ipc(
        "{}/DOWNSCALE_DERTIC/{}/{}/{}.feather".format(
            args.data_path, args.symbols, args.target_freq, args.date
        )
    )
    base_feature = pl.read_ipc(
        "{}/BASE_FEATURE/{}/{}/{}.feather".format(
            args.data_path, args.symbols, args.target_freq, args.date
        )
    )
    snapshot_feature = pl.read_ipc(
        "{}/CROSS_SECTION/SNAPSHOT_FEATURE/{}/{}/{}.feather".format(
            args.data_path, args.symbols, args.target_freq, args.date
        )
    )
    quotes_feature = pl.read_ipc(
        "{}/CROSS_SECTION/QUOTES_FEATURE/{}/{}/{}.feather".format(
            args.data_path, args.symbols, args.target_freq, args.date
        )
    )
    kline_feature = pl.read_ipc(
        "{}/CROSS_SECTION/KLINE_FEATURE/{}/{}/{}.feather".format(
            args.data_path, args.symbols, args.target_freq, args.date
        )
    )

    reward_feature, base_feature = build_daily_feature_frames(
        snapshot,
        der,
        base_feature,
        snapshot_feature,
        quotes_feature,
        kline_feature,
    )

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
    reward_feature.write_ipc(
        os.path.join(
            args.save_path,
            "MERGED_FEATURE",
            args.symbols,
            args.target_freq,
            "CONCURRENT_FEATURE",
            args.date + ".feather",
        )
    )
    base_feature.write_ipc(
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
