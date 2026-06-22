import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.append(".")
from operator_futures.feature_selection.cor_util import select_feature

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
parser.add_argument("--ic_theshold", type=float, default=0.01, help="the date of start")
parser.add_argument("--cor_theshold", type=float, default=0.7, help="the date of start")
parser.add_argument(
    "--windows_list",
    type=int,
    nargs="*",
    default=[1, 6, 12],
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
    if isinstance(df, pl.DataFrame):
        target = df.select((pl.col(reward_feature).shift(-window_length) - pl.col(reward_feature)).alias(reward_feature))[reward_feature]
        return target.slice(0, max(target.len() - window_length, 0))
    target = df[reward_feature].shift(-window_length) - df[reward_feature]
    return target[:-window_length]


def select_reward_state_features(df, market_type="crypto_futures", orderbook_depth=25):
    if market_type == "commodity_futures":
        from operator_futures.commodity.schema import get_reward_execution_columns

        reward_features = [col for col in get_reward_execution_columns(orderbook_depth) if col in df.columns]
    else:
        reward_features = list(df.columns[:106])
    state_feature = [col for col in df.columns if col not in reward_features]
    return reward_features, state_feature


def build_feature_importance_frame(feature_names, feature_importances):
    frame = pl.DataFrame({"Feature": feature_names, "Importance": feature_importances})
    return frame.sort("Importance", descending=True)


def main(args):
    args.data_path = os.path.join(args.root_path, args.data_path)
    args.save_path = os.path.join(args.root_path, args.save_path)
    input_path = Path(args.data_path) / args.symbols / args.target_freq / f"{args.start_date}-{args.end_date}.feather"
    output_dir = Path(args.save_path) / args.symbols / args.target_freq / f"{args.start_date}-{args.end_date}"
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pl.read_ipc(input_path)
    reward_features, state_feature = select_reward_state_features(df, "crypto_futures", 25)

    ic_selection_key_all = []
    for window_length in args.windows_list:
        target = calculate_target(df, "mark_price", window_length)
        df_ic = df.slice(0, max(df.height - window_length, 0))
        X = df_ic.select(state_feature).to_numpy()
        y = target.to_numpy()
        from catboost import CatBoostRegressor, Pool

        train_pool = Pool(X, y)
        test_pool = Pool(X, y)
        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            loss_function="MAE",
            task_type="GPU",
            random_seed=42,
        )
        model.fit(train_pool, eval_set=test_pool, verbose=100)
        feature_importance_df = build_feature_importance_frame(state_feature, model.get_feature_importance(train_pool))
        feature_importance_df.write_csv(output_dir / f"cat_boost_feature_importance_{window_length}.csv")
        ic_selection_key_all.extend(
            feature_importance_df.filter(pl.col("Importance") > args.ic_theshold)["Feature"].to_list()
        )

    ic_selection_key = remove_duplicates_preserve_order(ic_selection_key_all)
    df_cor = df.select(ic_selection_key).corr().with_columns(pl.Series("feature", ic_selection_key)) if ic_selection_key else pl.DataFrame({"feature": []})
    if ic_selection_key:
        df_cor = df_cor.select(["feature", *ic_selection_key])
    df_cor.write_csv(output_dir / "correlation_catboost.csv")
    selected_feature_names = select_feature(corre_df=df_cor, theshold=args.cor_theshold)
    out = df.select([*reward_features, *selected_feature_names])
    out.write_ipc(output_dir / "df_catboost.feather")
    np.save(output_dir / "state_features_catboost.npy", np.array(selected_feature_names))
    return out


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    print("Done!")
