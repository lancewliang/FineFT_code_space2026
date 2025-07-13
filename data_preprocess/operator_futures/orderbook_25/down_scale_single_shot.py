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
    default="DOWNLOAD_DATASET/binance-futures",
    help="the path of storing the data",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="PREPROCESS_DATASET/binance-futures/DOWNSCALE_ORDERBOOK_25",
    help="the path to save the data",
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
    default="10s",
    help="the date of start",
    choices=["10s", "1min", "5min", "10min","30min", "1H", "1D"],
)


def down_scale_single_oe_snapshot(orderbook_df: pd.DataFrame, agg_freq: str):
    orderbook_df = orderbook_df.drop(columns=["local_timestamp", "exchange"])
    orderbook_df["timestamp"] = pd.to_datetime(orderbook_df["timestamp"] * 1000)
    orderbook_df = orderbook_df.set_index("timestamp")
    orderbook_df = orderbook_df.resample(agg_freq).agg(["first"])
    return orderbook_df


@profile
def main(args):
    args.data_path = os.path.join(args.root_path, args.data_path)
    args.save_path = os.path.join(args.root_path, args.save_path)
    orderbook_dir = "{}/{}/book_snapshot_25".format(args.data_path, args.symbols)
    order_book_file = os.listdir(orderbook_dir)
    order_book_file.sort()
    date = match_strings_in_range(order_book_file, args.date)

    single_df = pd.read_csv(os.path.join(orderbook_dir, date), engine="python")
    orderbook_df = down_scale_single_oe_snapshot(single_df, args.target_freq)
    del single_df
    orderbook_df.columns = orderbook_df.columns.droplevel(1)
    new_column_names = {}
    for i in range(25):
        new_column_names["asks[{}].price".format(i)] = "ask{}_price".format(i + 1)
        new_column_names["asks[{}].amount".format(i)] = "ask{}_size".format(i + 1)
        new_column_names["bids[{}].price".format(i)] = "bid{}_price".format(i + 1)
        new_column_names["bids[{}].amount".format(i)] = "bid{}_size".format(i + 1)

    orderbook_df.rename(columns=new_column_names, inplace=True)

    orderbook_df.reset_index(inplace=True)
    orderbook_df = orderbook_df.ffill()
    if not os.path.exists(os.path.join(args.save_path, args.symbols, args.target_freq)):
        os.makedirs(os.path.join(args.save_path, args.symbols, args.target_freq))
    orderbook_df.to_feather(
        os.path.join(
            args.save_path,
            args.symbols,
            args.target_freq,
            args.date + ".feather",
        )
    )


if __name__ == "__main__":
    args = parser.parse_args()
    print("initiate down scale for {}".format(args.date))
    main(args)
