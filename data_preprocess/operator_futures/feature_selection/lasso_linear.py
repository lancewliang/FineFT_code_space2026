import argparse
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default=".", help="the path of storing the data")
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
parser.add_argument("--symbols", type=str, default="BTCUSDT", help="the name of the ticker")
parser.add_argument("--start_date", type=str, default="2023-01-01", help="the path to save the data")
parser.add_argument("--end_date", type=str, default="2023-02-01", help="the path to save the data")
parser.add_argument(
    "--target_freq",
    type=str,
    default="10s",
    help="the date of start",
    choices=["10s", "1min", "5min", "10min", "30min", "1H", "1D"],
)


def calculate_target(df, reward_feature):
    if isinstance(df, pl.DataFrame):
        target = df.select((pl.col(reward_feature).shift(-1) - pl.col(reward_feature)).alias(reward_feature))[reward_feature]
        return target.slice(0, max(target.len() - 1, 0))
    target = df[reward_feature].shift(-1) - df[reward_feature]
    return target[:-1]


def main(args):
    args.data_path = os.path.join(args.root_path, args.data_path)
    args.save_path = os.path.join(args.root_path, args.save_path)
    input_path = Path(args.data_path) / args.symbols / args.target_freq / f"{args.start_date}-{args.end_date}.feather"
    output_dir = Path(args.save_path) / args.symbols / args.target_freq / f"{args.start_date}-{args.end_date}"
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pl.read_ipc(input_path)
    reward_features = list(df.columns[:106])
    state_feature = [col for col in df.columns if col not in reward_features]
    target = calculate_target(df, "mark_price")
    df_lasso = df.slice(0, max(df.height - 1, 0))
    df_state = df_lasso.select(state_feature)
    scaler = StandardScaler()
    df_state_trans = scaler.fit_transform(df_state.to_numpy())

    lasso_cv = LassoCV(cv=5, random_state=0, max_iter=10000).fit(df_state_trans, target.to_numpy())
    selected_feature_indices = np.where(lasso_cv.coef_ != 0)[0]
    state_feature = [df_state.columns[i] for i in selected_feature_indices]
    out = df.select([*reward_features, *state_feature])
    np.save(output_dir / "state_features_lasso.npy", np.array(state_feature))
    out.write_ipc(output_dir / "df_lasso.feather")
    return out


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
