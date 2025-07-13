import pandas as pd
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
    default="PREPROCESS_DATASET/binance-futures/DOWNSCALE_DERTIC",
    help="the path of storing the data",
)

# dataset name
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
    default="1min",
    help="the date of start",
    choices=["10s", "1min", "5min","10min", "30min", "1H", "1D"],
)
parser.add_argument(
    "--base_freq",
    type=str,
    default="10s",
    help="the date of start",
    choices=["10s", "1min", "5min", "30min", "1H", "1D"],
)


def down_scale_single_oe_snapshot(ticker_df: pd.DataFrame, agg_freq: str):
    ticker_df = ticker_df.set_index("timestamp")
    ticker_df_target = ticker_df.resample(agg_freq).agg(["first"])
    del ticker_df
    ticker_df_target.columns = ticker_df_target.columns.droplevel(1)
    ticker_df_target.reset_index(inplace=True)
    return ticker_df_target


@profile
def main(args):
    args.data_path = os.path.join(args.root_path, args.data_path)
    ticker_dir = "{}/{}/{}/{}".format(args.data_path, args.symbols, args.base_freq)
    single_df = pd.read_feather(os.path.join(ticker_dir, args.date + ".feather"))
    orderbook_df = down_scale_single_oe_snapshot(single_df, args.target_freq)
    orderbook_df = orderbook_df.ffill()
    if not os.path.exists(os.path.join(args.data_path, args.symbols, args.target_freq)):
        os.makedirs(os.path.join(args.data_path, args.symbols, args.target_freq))
    orderbook_df.to_feather(
        os.path.join(
            args.data_path,
            args.symbols,
            args.target_freq,
            args.date + ".feather",
        )
    )


if __name__ == "__main__":
    args = parser.parse_args()
    print("initiate down scale for {}".format(args.date))
    main(args)
