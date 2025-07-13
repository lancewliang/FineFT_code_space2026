import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
import argparse
import sys
from multiprocessing import Pool

sys.path.append(".")
from operator_futures.util import find_ohlcv_groups, find_ohlc_groups
from memory_profiler import profile
from operator_futures.time_operator.multi_processing_util import (
    get_multi_window_ohlcv,
    get_multi_window_ohlc,
    get_multi_feature_window_price,
)

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
    default="PREPROCESS_DATASET/binance-futures/MERGE_CONCAT/CONCAT_FEATURE/",
    help="the path of storing the data",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="PREPROCESS_DATASET/binance-futures/TIME_FEATURE/",
    help="the path of storing the data",
)
parser.add_argument(
    "--symbols", type=str, default="BTCUSDT", help="the name of the ticker"
)
# date
parser.add_argument(
    "--start_date",
    type=str,
    default="2023-01-01",
    help="the path to save the data",
)
parser.add_argument(
    "--end_date",
    type=str,
    default="2023-02-01",
    help="the path to save the data",
)

# freq
parser.add_argument(
    "--target_freq",
    type=str,
    default="10s",
    help="the date of start",
    choices=["10s", "1min", "5min", "10min", "30min", "1H", "1D"],
)
parser.add_argument(
    "--windows",
    type=str,
    default="2,6,12,16,24,48",
    help="List of windows sizes as comma-separated values",
)


def main(args):
    time_feature_list_all = []
    windows = list(map(int, args.windows.split(",")))
    args.data_path = os.path.join(args.root_path, args.data_path)
    args.save_path = os.path.join(args.root_path, args.save_path)
    original_df = pd.read_feather(
        os.path.join(
            args.data_path,
            args.symbols,
            args.target_freq,
            args.start_date + "-" + args.end_date + ".feather",
        )
    )
    original_df.set_index("timestamp", inplace=True)
    ohlcv_features, _ = find_ohlcv_groups(original_df)
    ohlc_features, _ = find_ohlc_groups(original_df)
    price_features = [
        *[f"bid{l+1}_price" for l in range(25)],
        *[f"ask{l+1}_price" for l in range(25)],
        "buy_spread_oe_max",
        "sell_spread_oe_max",
        "wap_1",
        "wap_2",
        "buy_wap",
        "sell_wap",
        "mark_price",
        "buy_volume_oe",
        "sell_volume_oe",
        "imblance_volume_oe",
        *[f"ask{l+1}_size_n" for l in range(25)],
        *[f"bid{l+1}_size_n" for l in range(25)],
    ]
    df_time = get_multi_feature_window_price(original_df, windows, price_features)
    time_feature_list_all.append(df_time)
    for key in ohlcv_features:
        ohlc_features.pop(key, None)

    for ffuixes in ohlcv_features:
        (prefix, suffix) = ffuixes
        # print("prefix",prefix,"suffix",suffix)
        after_name = prefix + suffix
        converted_strings = "_origin" if after_name == "" else after_name
        feature_names = ohlcv_features[ffuixes]
        df_ohlcv = original_df[feature_names].copy()
        df_ohlcv.rename(
            columns={
                prefix + key + suffix: key
                for key in ["open", "high", "low", "close", "volume"]
            },
            inplace=True,
        )
        p_process_ohlcv = get_multi_window_ohlcv(df_ohlcv, windows)
        p_process_ohlcv.rename(
            columns={key: key + converted_strings for key in p_process_ohlcv.columns},
            inplace=True,
        )
        time_feature_list_all.append(p_process_ohlcv)
    for ffuixes in ohlc_features:
        (prefix, suffix) = ffuixes
        # print("prefix",prefix,"suffix",suffix)
        after_name = prefix + suffix
        converted_strings = "_origin" if after_name == "" else after_name
        print(converted_strings)
        feature_names = ohlc_features[ffuixes]
        df_ohlc = original_df[feature_names].copy()
        df_ohlc.rename(
            columns={
                prefix + key + suffix: key
                for key in [
                    "open",
                    "high",
                    "low",
                    "close",
                ]
            },
            inplace=True,
        )
        p_process_ohlc = get_multi_window_ohlc(df_ohlc, windows)
        p_process_ohlc.rename(
            columns={key: key + converted_strings for key in p_process_ohlc.columns},
            inplace=True,
        )
        time_feature_list_all.append(p_process_ohlc)

    time_df = pd.concat(time_feature_list_all, axis=1, join="inner")
    time_df.reset_index(inplace=True)
    if not os.path.exists(os.path.join(args.save_path, args.symbols, args.target_freq)):
        os.makedirs(os.path.join(args.save_path, args.symbols, args.target_freq))
    time_df.to_feather(
        os.path.join(
            args.save_path,
            args.symbols,
            args.target_freq,
            args.start_date + "-" + args.end_date + ".feather",
        )
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    print("Done!")
