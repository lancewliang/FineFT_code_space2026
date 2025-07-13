import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
import argparse
import sys
from datetime import datetime, timedelta

sys.path.append(".")
from memory_profiler import profile
from concurrent.futures import ThreadPoolExecutor
from operator_futures.util import find_strings_in_range
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
# data path

"./DOWNLOAD_DATASET/binance-futures/BTCUSDT/derivative_ticker/binance-futures_derivative_ticker_2022-09-11_BTCUSDT.csv"
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
    default="DOWNLOAD_DATASET/binance-futures_overview",
    help="the path of storing the data",
)
parser.add_argument(
    "--symbols", type=str, default="BTCUSDT", help="the name of the ticker"
)
# date
parser.add_argument(
    "--start_date",
    type=str,
    default="2021-04-01",
    help="the path to save the data",
)
parser.add_argument(
    "--end_date",
    type=str,
    default="2024-05-01",
    help="the path to save the data",
)
# freq
parser.add_argument(
    "--target_freq",
    type=str,
    default="5min",
    help="the date of start",
    choices=["10s", "1min", "5min","10min", "30min", "1H", "1D"],
)


def get_date_list(start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # 生成日期列表
    date_list = []
    current_date = start
    while current_date <= end:
        date_list.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    return date_list


def read_feather_file(file):
    return pd.read_csv(file)


def main(args):
    args.data_path = os.path.join(args.root_path, args.data_path)
    args.save_path = os.path.join(args.root_path, args.save_path)
    date_list = get_date_list(args.start_date, args.end_date)
    file_path_list = [
        os.path.join(
            args.data_path,
            args.symbols,
            "derivative_ticker",
            f"binance-futures_derivative_ticker_{date}_{args.symbols}.csv",
        )
        for date in date_list
    ]
    with ThreadPoolExecutor() as executor:
        df_list = list(executor.map(read_feather_file, file_path_list))
    df_all = pd.concat(df_list, axis=0)
    df_all.set_index("timestamp", inplace=True)
    df_all.index = pd.to_datetime(df_all.index, unit="us")

    df_all.sort_index(inplace=True)
    df_all = df_all.groupby(df_all.index).first()
    df_all = df_all.resample(args.target_freq).asfreq()
    df_all = df_all.groupby(df_all.index).first()
    df_all.reset_index(inplace=True)
    save_path = os.path.join(args.save_path, args.symbols, args.target_freq)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plot(df_all, save_path, args.start_date, args.end_date)
    df_all.to_feather(
        os.path.join(save_path, "{}-{}.feather".format(args.start_date, args.end_date))
    )


def plot(df: pd.DataFrame, save_path, start_date, end_date):
    df.bfill(inplace=True)
    df.ffill(inplace=True)
    timestamp = df["timestamp"].values
    mark_price = df["mark_price"].values
    test_buy_hold_df_net_curve = np.array(mark_price / (mark_price[0])) - 1
    test_buy_hold_df_net_curve = 100 * test_buy_hold_df_net_curve
    plt.figure(figsize=(40, 10))
    plt.plot(timestamp, test_buy_hold_df_net_curve)
    plt.savefig(
        os.path.join(save_path, "overview_{}-{}.pdf".format(start_date, end_date))
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
