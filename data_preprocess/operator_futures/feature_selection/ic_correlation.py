# utlize the correlation between the return of the mark price and feature.
# select the features with high correlation with markprice return and low correlation with existing features.
import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
import argparse
import sys
from multiprocessing import Pool
import json

sys.path.append(".")
from operator_futures.feature_selection.cor_util import select_feature

# TODO add multi-labelling: the target is calculated as a list of window lengths
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
    default="PREPROCESS_DATASET/binance-futures/ALL_FEATURE/",
    help="the path of storing the data",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="PREPROCESS_DATASET/binance-futures/IC_RESULT/",
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
#
parser.add_argument(
    "--ic_theshold",
    type=float,
    default=0.01,
    help="the date of start",
)
parser.add_argument(
    "--cor_theshold",
    type=float,
    default=0.7,
    help="the date of start",
)

parser.add_argument(
    "--windows_list",
    type=int,  # 指定每个元素应转换为浮点数
    nargs="*",  # '*' 表示接受零个或多个参数
    default=[
        1,
        6,
        12,
    ],  # 设置默认值为一个列表
    help="List of threshold values",
)
parser.add_argument(
    "--market_type",
    type=str,
    default="crypto_futures",
    choices=["crypto_futures", "commodity_futures"],
    help="the market type of the preprocessed data",
)
parser.add_argument(
    "--orderbook_depth",
    type=int,
    default=25,
    help="the available orderbook depth",
)


def calculate_cor(column, target):
    return column.corr(target)
def remove_duplicates_preserve_order(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def calculate_target(df, reward_feature, window_length):
    # the length of the target = len(df)-1
    target = df[reward_feature].shift(-window_length) - df[reward_feature]
    target = target[:-window_length]
    return target


def select_reward_state_features(df, market_type="crypto_futures", orderbook_depth=25):
    if market_type == "commodity_futures":
        from operator_futures.commodity.schema import get_reward_execution_columns

        reward_features = [
            col for col in get_reward_execution_columns(orderbook_depth) if col in df.columns
        ]
    else:
        reward_features = list(df.columns[:106])
    state_feature = [col for col in df.columns if col not in reward_features]
    return reward_features, state_feature


def main(args):
    args.data_path = os.path.join(args.root_path, args.data_path)
    args.save_path = os.path.join(args.root_path, args.save_path)
    df = pd.read_feather(
        os.path.join(
            args.data_path,
            args.symbols,
            args.target_freq,
            "{}-{}.feather".format(args.start_date, args.end_date),
        )
    )
    reward_features, state_feature = select_reward_state_features(
        df, args.market_type, args.orderbook_depth
    )
    df.set_index("timestamp", inplace=True)
    cpu_count = int(max(os.cpu_count() - 10, os.cpu_count() / 2))

    ic_selection_key_all = []
    for window_length in args.windows_list:
        target = calculate_target(df, "mark_price", window_length)
        df_ic = df.iloc[:-window_length]

        with Pool(processes=min(len(state_feature), cpu_count)) as pool:
            results = [
                pool.apply_async(calculate_cor, args=(df_ic[feature], target))
                for feature in state_feature
            ]
            # 等待所有结果完成，并获取结果
            ic_result = [result.get() for result in results]
        sorted_pairs = sorted(
            zip(state_feature, ic_result), key=lambda x: abs(x[1]), reverse=True
        )

        cor = {feature: result for feature, result in sorted_pairs}
        if not os.path.exists(
            os.path.join(
                args.save_path,
                args.symbols,
                args.target_freq,
                "{}-{}".format(args.start_date, args.end_date),
            )
        ):
            os.makedirs(
                os.path.join(
                    args.save_path,
                    args.symbols,
                    args.target_freq,
                    "{}-{}".format(args.start_date, args.end_date),
                )
            )
        with open(
            os.path.join(
                args.save_path,
                args.symbols,
                args.target_freq,
                "{}-{}".format(args.start_date, args.end_date),
                "ic_window_{}.json".format(window_length),
            ),
            "w",
        ) as f:
            json.dump(cor, f)
        cor_abs = {}
        for key in cor.keys():
            cor_abs[key] = abs(cor[key])
        ic_selection_key = [
            key for key in cor_abs.keys() if cor_abs[key] > args.ic_theshold
        ]
        ic_selection_key_all.extend(ic_selection_key)
    # ic_selection_key = list(set(ic_selection_key_all))
    ic_selection_key=remove_duplicates_preserve_order(ic_selection_key_all)
    df_cor = df[ic_selection_key].corr()
    df_cor.to_csv(
        os.path.join(
            args.save_path,
            args.symbols,
            args.target_freq,
            "{}-{}".format(args.start_date, args.end_date),
            "correlation.csv",
        )
    )
    selected_feature_names = select_feature(corre_df=df_cor, theshold=args.cor_theshold)
    state_feature = selected_feature_names
    df.reset_index(inplace=True)
    df = df[reward_features + state_feature]
    df.to_feather(
        os.path.join(
            args.save_path,
            args.symbols,
            args.target_freq,
            "{}-{}".format(args.start_date, args.end_date),
            "df.feather".format(args.start_date, args.end_date),
        )
    )
    np.save(
        os.path.join(
            args.save_path,
            args.symbols,
            args.target_freq,
            "{}-{}".format(args.start_date, args.end_date),
            "state_features.npy",
        ),
        state_feature,
    )
    return df


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    print("Done!")
