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
    default="2022-01-01",
    help="the path to save the data",
)
parser.add_argument(
    "--end_date",
    type=str,
    default="2024-01-01",
    help="the path to save the data",
)

# freq
parser.add_argument(
    "--target_freq",
    type=str,
    default="5min",
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
    "--ic_choice",
    type=str,
    default="catboost",
    choices=["ic", "rank_ic", "catboost"],
    help="the way of choosing features",
)


def remove_duplicates_preserve_order(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def main(args):
    windows_list = args.windows_list
    windows_list.sort()
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
    reward_features = df.columns[:106]
    state_feature = [col for col in df.columns if col not in reward_features]
    df.set_index("timestamp", inplace=True)
    ic_selection_key_all = []
    assert args.ic_choice in ["ic", "rank_ic", "catboost"]
    if args.ic_choice == "ic":
        ic_file_name_list = ["ic_window_{}.json".format(w) for w in args.windows_list]
        cor_file_name = "correlation.csv"
        for ic_file_name in ic_file_name_list:
            with open(
                os.path.join(
                    args.save_path,
                    args.symbols,
                    args.target_freq,
                    "{}-{}".format(args.start_date, args.end_date),
                    ic_file_name,
                ),
                "r",
            ) as f:
                cor = json.load(f)
            cor_abs = {}
            for key in cor.keys():
                cor_abs[key] = abs(cor[key])
            ic_selection_key = [
                key for key in cor_abs.keys() if cor_abs[key] > args.ic_theshold
            ]
            ic_selection_key_all.extend(ic_selection_key)
        ic_selection_key = remove_duplicates_preserve_order(ic_selection_key_all)
        df_cor = pd.read_csv(
            os.path.join(
                args.save_path,
                args.symbols,
                args.target_freq,
                "{}-{}".format(args.start_date, args.end_date),
                cor_file_name,
            ),
            index_col=0,
        )
        df_cor = df_cor.loc[ic_selection_key, ic_selection_key]
    elif args.ic_choice == "rank_ic":
        ic_file_name_list = [
            "rank_ic_window_{}.json".format(w) for w in args.windows_list
        ]
        cor_file_name = "rank_correlation.csv"
        for ic_file_name in ic_file_name_list:
            with open(
                os.path.join(
                    args.save_path,
                    args.symbols,
                    args.target_freq,
                    "{}-{}".format(args.start_date, args.end_date),
                    ic_file_name,
                ),
                "r",
            ) as f:
                cor = json.load(f)
            cor_abs = {}
            for key in cor.keys():
                cor_abs[key] = abs(cor[key])
            ic_selection_key = [
                key for key in cor_abs.keys() if cor_abs[key] > args.ic_theshold
            ]
            ic_selection_key_all.extend(ic_selection_key)
        ic_selection_key = remove_duplicates_preserve_order(ic_selection_key_all)
        df_cor = pd.read_csv(
            os.path.join(
                args.save_path,
                args.symbols,
                args.target_freq,
                "{}-{}".format(args.start_date, args.end_date),
                cor_file_name,
            ),
            index_col=0,
        )
        df_cor = df_cor.loc[ic_selection_key, ic_selection_key]
    elif args.ic_choice == "catboost":
        ic_file_name_list = [
            "cat_boost_feature_importance_{}.csv".format(w) for w in args.windows_list
        ]
        cor_file_name = "correlation_catboost.csv"
        for ic_file_name in ic_file_name_list:
            feature_importance_df = pd.read_csv(
                os.path.join(
                    args.save_path,
                    args.symbols,
                    args.target_freq,
                    "{}-{}".format(args.start_date, args.end_date),
                    ic_file_name,
                )
            )
            catboost_selection_key = feature_importance_df[
                feature_importance_df["Importance"] > args.ic_theshold
            ]["Feature"].values.tolist()
            ic_selection_key_all.extend(catboost_selection_key)
        ic_selection_key = remove_duplicates_preserve_order(ic_selection_key_all)
        df_cor = pd.read_csv(
            os.path.join(
                args.save_path,
                args.symbols,
                args.target_freq,
                "{}-{}".format(args.start_date, args.end_date),
                cor_file_name,
            )
        )
        df_cor = df_cor.loc[ic_selection_key, ic_selection_key]
    else:
        print("wrong ic_choice")
    selected_feature_names = select_feature(corre_df=df_cor, theshold=args.cor_theshold)
    state_feature = selected_feature_names
    reward_features = reward_features.tolist()
    df.reset_index(inplace=True)
    df = df[reward_features + state_feature]
    if args.ic_choice == "ic":
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
    elif args.ic_choice == "rank_ic":
        df.to_feather(
            os.path.join(
                args.save_path,
                args.symbols,
                args.target_freq,
                "{}-{}".format(args.start_date, args.end_date),
                "df_rank.feather".format(args.start_date, args.end_date),
            )
        )
        np.save(
            os.path.join(
                args.save_path,
                args.symbols,
                args.target_freq,
                "{}-{}".format(args.start_date, args.end_date),
                "state_features_rank.npy",
            ),
            state_feature,
        )
    elif args.ic_choice == "catboost":
        df.to_feather(
            os.path.join(
                args.save_path,
                args.symbols,
                args.target_freq,
                "{}-{}".format(args.start_date, args.end_date),
                "df_catboost.feather".format(args.start_date, args.end_date),
            )
        )
        np.save(
            os.path.join(
                args.save_path,
                args.symbols,
                args.target_freq,
                "{}-{}".format(args.start_date, args.end_date),
                "state_features_catboost.npy",
            ),
            state_feature,
        )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
