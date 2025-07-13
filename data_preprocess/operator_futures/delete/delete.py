import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
import argparse
import sys

sys.path.append(".")
from over_view.checkout_create_feature import get_date_list

parser = argparse.ArgumentParser()
total_process_list = [
    "BASE_FEATURE",
    *[
        "CROSS_SECTION/{}".format(cat)
        for cat in ["KLINE_FEATURE", "QUOTES_FEATURE", "SNAPSHOT_FEATURE"]
    ],
    "DOWNSCALE_DERTIC",
    "DOWNSCALE_ORDERBOOK_25",
]
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
    default="PREPROCESS_DATASET/binance-futures",
    help="the path of storing the data",
)
parser.add_argument(
    "--symbols", type=str, default="BTCUSDT", help="the name of the ticker"
)
parser.add_argument(
    "--start_date", type=str, default="2021-04-01", help="the name of the ticker"
)
parser.add_argument(
    "--end_date", type=str, default="2024-04-30", help="the name of the ticker"
)
parser.add_argument(
    "--target_freq",
    type=str,
    default="5min",
    help="the date of start",
    choices=["10s", "1min", "5min", "10min", "30min", "1H", "1D"],
)


def delete(args):
    for process in total_process_list:
        for date in get_date_list(args.start_date, args.end_date):
            df_path = os.path.join(
                args.root_path,
                args.data_path,
                process,
                args.symbols,
                args.target_freq,
                "{}.feather".format(date),
            )
            print(df_path)
            if os.path.exists(df_path):
                os.remove(df_path)
                print(f"delete {df_path}")


if __name__ == "__main__":
    args = parser.parse_args()
    delete(args)
