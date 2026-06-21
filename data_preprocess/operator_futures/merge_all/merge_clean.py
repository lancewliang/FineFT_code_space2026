import polars as pl
import numpy as np
import os
import re
from datetime import datetime
import argparse
import sys

sys.path.append(".")


parser = argparse.ArgumentParser()
# data path
parser.add_argument(
    "--root_path",
    type=str,
    default=".",
    help="the path of storing the data",
)
parser.add_argument(
    "--data_path_1",
    type=str,
    default="PREPROCESS_DATASET/binance-futures/MERGE_CONCAT/CONCAT_FEATURE",
    help="the path of cross section data",
)
parser.add_argument(
    "--data_path_2",
    type=str,
    default="PREPROCESS_DATASET/binance-futures/TIME_FEATURE",
    help="the path of time feature data",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="PREPROCESS_DATASET/binance-futures/ALL_FEATURE",
    help="the path of time feature data",
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
    default="2023-12-31",
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
    args.data_path_1= os.path.join(args.root_path, args.data_path_1)
    args.data_path_2 = os.path.join(args.root_path, args.data_path_2)
    args.save_path = os.path.join(args.root_path, args.save_path)
    time_feature_df = pl.read_ipc(
        os.path.join(
            args.data_path_2,
            args.symbols,
            args.target_freq,
            "{}-{}.feather".format(args.start_date, args.end_date),
        )
    )
    cross_section_df = pl.read_ipc(
        os.path.join(
            args.data_path_1,
            args.symbols,
            args.target_freq,
            "{}-{}.feather".format(args.start_date, args.end_date),
        )
    )
    all_feature_df = cross_section_df.join(time_feature_df, on="timestamp", how="inner")

    if not os.path.exists(os.path.join(args.save_path, args.symbols, args.target_freq)):
        os.makedirs(os.path.join(args.save_path, args.symbols, args.target_freq))
    all_feature_df.write_ipc(
        os.path.join(
            args.save_path,
            args.symbols,
            args.target_freq,
            "{}-{}.feather".format(args.start_date, args.end_date),
        )
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
