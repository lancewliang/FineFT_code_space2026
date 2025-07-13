# utlize the lasso and linear to determine the proper features to use.
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
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

sys.path.append(".")
from operator_futures.feature_selection.cor_util import select_feature

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
    choices=["10s", "1min", "5min", "10min","30min", "1H", "1D"],
)


def calculate_target(df, reward_feature):
    # the length of the target = len(df)-1
    target = df[reward_feature].shift(-1) - df[reward_feature]
    target = target[:-1]
    return target

def remove_duplicates_preserve_order(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
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
    target = calculate_target(df, "mark_price")
    df_lasso = df.iloc[:-1]
    df_state = df_lasso[state_feature]
    scaler = StandardScaler()
    df_state_trans = scaler.fit_transform(df_state)

    lasso_cv = LassoCV(cv=5, random_state=0, max_iter=10000).fit(df_state_trans, target)
    selected_feature_indices = np.where(lasso_cv.coef_ != 0)[0]
    selected_features_names = df_state.columns[selected_feature_indices]
    reward_features = reward_features.tolist()
    state_feature = selected_features_names.values.tolist()
    df.reset_index(inplace=True)
    df = df[reward_features + state_feature]
    np.save(
        os.path.join(
            args.save_path,
            args.symbols,
            args.target_freq,
            "{}-{}".format(args.start_date, args.end_date),
            "state_features_lasso.npy",
        ),
        state_feature,
    )
    df.to_feather(
        os.path.join(
            args.save_path,
            args.symbols,
            args.target_freq,
            "{}-{}".format(args.start_date, args.end_date),
            "df_lasso.feather".format(args.start_date, args.end_date),
        )
    )
    return df


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
