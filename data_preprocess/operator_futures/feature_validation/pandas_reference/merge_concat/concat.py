import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
import argparse
import sys

sys.path.append(".")
try:
    from memory_profiler import profile
except ModuleNotFoundError:  # pragma: no cover
    def profile(func):
        return func
from concurrent.futures import ThreadPoolExecutor
from operator_futures.feature_validation.pandas_reference.util import find_strings_in_range

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
    default="PREPROCESS_DATASET/binance-futures/MERGE_CONCAT",
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
    choices=["10s", "1min", "5min","10min", "30min", "1H", "1D"],
)


def read_feather_file(file):
    return pd.read_feather(file)


def main(args):
    args.data_path = os.path.join(args.root_path, args.data_path)
    args.save_path = os.path.join(args.root_path, args.save_path)
    cocurrent_path = "{}/MERGED_FEATURE/{}/{}/CONCURRENT_FEATURE".format(
        args.data_path,
        args.symbols,
        args.target_freq,
    )
    future_path = "{}/MERGED_FEATURE/{}/{}/FUTURE_FEATURE".format(
        args.data_path,
        args.symbols,
        args.target_freq,
    )
    file_list = os.listdir(cocurrent_path)
    file_list.sort()
    filter_file_list = find_strings_in_range(file_list, args.start_date, args.end_date)
    cocurrent_path_files = [
        os.path.join(cocurrent_path, file) for file in filter_file_list
    ]
    future_path_files = [os.path.join(future_path, file) for file in filter_file_list]
    with ThreadPoolExecutor() as executor:
        cocurrent_df_list = list(executor.map(read_feather_file, cocurrent_path_files))

    with ThreadPoolExecutor() as executor:
        future_df_list = list(executor.map(read_feather_file, future_path_files))
    # concat seperately
    cocurrent_df = pd.concat(cocurrent_df_list, axis=0)
    future_df = pd.concat(future_df_list, axis=0)
    # index
    cocurrent_df.set_index("timestamp", inplace=True)
    future_df.set_index("timestamp", inplace=True)
    # sort and resample
    cocurrent_df.sort_index(inplace=True)
    cocurrent_df = cocurrent_df.groupby(cocurrent_df.index).first()
    cocurrent_df = cocurrent_df.resample(args.target_freq).asfreq()

    future_df.sort_index(inplace=True)
    future_df = future_df.groupby(future_df.index).first()
    future_df = future_df.resample(args.target_freq).asfreq()
    future_df.drop(columns=["symbol", "exchange"], inplace=True)
    future_df = future_df.shift(+1)
    future_df = future_df.iloc[1:]

    # now merge them together
    df = pd.concat([cocurrent_df, future_df], axis=1, join="inner")
    df.reset_index(inplace=True)
    df.fillna(method="ffill", inplace=True)
    save_path = "{}/CONCAT_FEATURE/{}/{}".format(
        args.save_path, args.symbols, args.target_freq
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df.to_feather(
        os.path.join(save_path, "{}-{}.feather".format(args.start_date, args.end_date))
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    print("Concatenation is done.")
