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
    default="DOWNLOAD_DATASET/binance-futures",
    help="the path of storing the data",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="PREPROCESS_DATASET/binance-futures/DOWNSCALE_DERTIC",
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
    choices=["10s", "1min", "5min","10min", "30min", "1H", "1D"],
)


def down_scale_single_dertick(derivative_ticker: pl.DataFrame, agg_freq: str):
    return (
        derivative_ticker.select(
            [
                "timestamp",
                "symbol",
                "funding_timestamp",
                "funding_rate",
                "index_price",
                "mark_price",
            ]
        )
        .with_columns(
            pl.from_epoch("timestamp", time_unit="us").alias("timestamp"),
            pl.from_epoch("funding_timestamp", time_unit="us").alias(
                "funding_timestamp"
            ),
        )
        .sort("timestamp")
        .group_by_dynamic("timestamp", every=agg_freq, closed="left", label="left")
        .agg(pl.all().first())
    )


def main(args):
    args.data_path = os.path.join(args.root_path, args.data_path)
    args.save_path = os.path.join(args.root_path, args.save_path)
    derivative_ticker_path = "{}/{}/derivative_ticker".format(
        args.data_path, args.symbols
    )
    dates_list = os.listdir(derivative_ticker_path)
    dates_list.sort()
    date = match_strings_in_range(dates_list, args.date)
    single_df = pl.read_csv(os.path.join(derivative_ticker_path, date))
    derivative_ticker_target = down_scale_single_dertick(single_df, args.target_freq)
    del single_df
    derivative_ticker_target = derivative_ticker_target.fill_null(strategy="forward")
    if not os.path.exists(os.path.join(args.save_path, args.symbols, args.target_freq)):
        os.makedirs(os.path.join(args.save_path, args.symbols, args.target_freq))
    derivative_ticker_target.write_ipc(
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
