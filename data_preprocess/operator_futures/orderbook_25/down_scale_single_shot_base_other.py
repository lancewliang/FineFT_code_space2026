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
    default="PREPROCESS_DATASET/binance-futures/DOWNSCALE_ORDERBOOK_25",
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
    choices=["10s", "1min", "5min","10min", "30min", "1H", "1D"],
)


def down_scale_single_oe_snapshot(orderbook_df: pl.DataFrame, agg_freq: str):
    return (
        orderbook_df.sort("timestamp")
        .group_by_dynamic("timestamp", every=agg_freq, closed="left", label="left")
        .agg(pl.all().first())
    )


@profile
def main(args):
    args.data_path = os.path.join(args.root_path, args.data_path)
    orderbook_dir = "{}/{}/{}".format(args.data_path, args.symbols, args.base_freq)
    single_df = pl.read_ipc(os.path.join(orderbook_dir, args.date + ".feather"))
    orderbook_df = down_scale_single_oe_snapshot(single_df, args.target_freq)
    orderbook_df = orderbook_df.fill_null(strategy="forward")
    if not os.path.exists(os.path.join(args.data_path, args.symbols, args.target_freq)):
        os.makedirs(os.path.join(args.data_path, args.symbols, args.target_freq))
    orderbook_df.write_ipc(
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
