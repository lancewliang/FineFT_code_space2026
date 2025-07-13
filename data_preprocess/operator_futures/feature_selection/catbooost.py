# utlize the correlation between the return of the mark price and feature.
# select the features with high correlation with markprice return and low correlation with existing features.
import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
import argparse
import sys
import json
from catboost import CatBoostRegressor, Pool

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
    reward_features = df.columns[:106]
    state_feature = [col for col in df.columns if col not in reward_features]
    df.set_index("timestamp", inplace=True)
    cpu_count = int(max(os.cpu_count() - 10, os.cpu_count() / 2))

    ic_selection_key_all = []
    for window_length in args.windows_list:
        target = calculate_target(df, "mark_price", window_length)
        df_ic = df.iloc[:-window_length]
        X = df_ic[state_feature].values
        y = target.values
        train_pool = Pool(X, y)
        test_pool = Pool(X, y)
        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            loss_function="MAE",
            task_type="GPU",  # 使用 GPU
            # devices="0,1,2,3",  # 指定 GPU 设备
            random_seed=42,
        )
        model.fit(train_pool, eval_set=test_pool, verbose=100)
        feature_importances = model.get_feature_importance(train_pool)
        feature_names = state_feature

        # 将特征重要性转换为 DataFrame
        feature_importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": feature_importances}
        )
        feature_importance_df = feature_importance_df.sort_values(
            by="Importance", ascending=False
        )
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
        feature_importance_df.reset_index(drop=True, inplace=True)
        feature_importance_df.to_csv(
            os.path.join(
                args.save_path,
                args.symbols,
                args.target_freq,
                "{}-{}".format(args.start_date, args.end_date),
                "cat_boost_feature_importance_{}.csv".format(window_length),
            ),
            index=False,
        )
        catboost_selection_key = feature_importance_df[
            feature_importance_df["Importance"] > args.ic_theshold
        ]["Feature"].values.tolist()
        ic_selection_key_all.extend(catboost_selection_key)
    #! huge problem use list(set) will change the order of the list, and therefore might throw away important features

    ic_selection_key = remove_duplicates_preserve_order(ic_selection_key_all)
    df_cor = df[ic_selection_key].corr()
    df_cor.to_csv(
        os.path.join(
            args.save_path,
            args.symbols,
            args.target_freq,
            "{}-{}".format(args.start_date, args.end_date),
            "correlation_catboost.csv",
        )
    )
    selected_feature_names = select_feature(corre_df=df_cor, theshold=args.cor_theshold)
    state_feature = selected_feature_names
    reward_features = reward_features.tolist()
    df.reset_index(inplace=True)
    df = df[reward_features + state_feature]
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
    return df


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    print("Done!")
