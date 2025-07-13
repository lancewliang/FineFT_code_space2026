import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
import argparse
import sys
from memory_profiler import profile

sys.path.append(".")
from operator_futures.util import match_strings_in_range
from memory_profiler import profile
from operator_futures.features_related.feature_util import *
import time

parser = argparse.ArgumentParser()
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
    default="PREPROCESS_DATASET/binance-futures/BASE_FEATURE",
    help="the path to save the data",
)
# date
parser.add_argument(
    "--date",
    type=str,
    default="2023-01-01",
    help="the path to save the data",
)
# symbol
parser.add_argument(
    "--symbols", type=str, default="BTCUSDT", help="the name of the ticker"
)
# freq
parser.add_argument(
    "--target_freq",
    type=str,
    default="10s",
    help="the date of start",
    choices=["10s", "1min", "5min","10min", "30min", "1H", "1D"],
)


@profile
def main(args):
    args.data_path = os.path.join(args.root_path, args.data_path)
    args.save_path = os.path.join(args.root_path, args.save_path)
    quotes_path = "{}/{}/quotes".format(args.data_path, args.symbols)
    dates_list = os.listdir(quotes_path)
    dates_list.sort()
    date = match_strings_in_range(dates_list, args.date)
    quotes = pd.read_csv(os.path.join(quotes_path, date), engine="python")
    quotes = preprocess_quotes(quotes)

    trades_path = "{}/{}/trades".format(args.data_path, args.symbols)
    dates_list = os.listdir(trades_path)
    dates_list.sort()
    date = match_strings_in_range(dates_list, args.date)
    trades = pd.read_csv(os.path.join(trades_path, date), engine="python")
    trades = preprocess_trades(trades)

    # liq_path = "{}/{}/liquidations".format(args.data_path, args.symbols)
    # dates_list = os.listdir(liq_path)
    # dates_list.sort()
    # date = match_strings_in_range(dates_list, args.date)
    # liq = pd.read_csv(os.path.join(liq_path, date), engine="python")
    # liq = preprocess_trades(liq)

    target_freq = args.target_freq
    quotes_df = create_quotes_feature(quotes, target_freq)
    quotes_df_ = create_ohlc_quotes_feature(quotes, target_freq)
    quotes_df = pd.concat([quotes_df, quotes_df_], axis=1)

    trades_df, trades = intial_process_trades(trades, target_freq)
    buy_sell_df = side_group_trades(trades, target_freq)
    trades_df = pd.concat([trades_df, buy_sell_df], axis=1)

    # liq_df, liq = intial_process_trades(liq, target_freq)
    # buy_sell_liq_df = side_group_trades(liq, target_freq)
    # liq_df = pd.concat([liq_df, buy_sell_liq_df], axis=1)
    # liq_df.columns = [i + "_liq" for i in liq_df.columns]

    # indicators_df = pd.concat([trades_df, quotes_df, liq_df], axis=1)
    indicators_df = pd.concat([trades_df, quotes_df], axis=1)
    indicators_df["exchange"] = trades["exchange"][0]
    indicators_df["symbol"] = trades["symbol"][0]
    indicators_df = move_column_in_position(indicators_df, "exchange", 0)
    indicators_df = move_column_in_position(indicators_df, "symbol", 1)

    indicators_df.reset_index(inplace=True)
    indicators_df = indicators_df.ffill()
    if not os.path.exists(os.path.join(args.save_path, args.symbols, args.target_freq)):
        os.makedirs(os.path.join(args.save_path, args.symbols, args.target_freq))
    indicators_df.to_feather(
        os.path.join(
            args.save_path,
            args.symbols,
            args.target_freq,
            args.date + ".feather",
        )
    )


if __name__ == "__main__":
    start_time = time.time()  # 记录开始时间

    args = parser.parse_args()
    main(args)
    end_time = time.time()  # 记录结束时间
    print(f"程序运行时间: {end_time - start_time} 秒")
