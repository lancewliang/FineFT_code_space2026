# utlize the correlation between the return of the mark price and feature.
# select the features with high correlation with markprice return and low correlation with existing features.
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.append(".")
from operator_futures.feature_selection.cor_util import select_feature

# TODO add multi-labelling: the target is calculated as a list of window lengths
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


def calculate_cor(column, target):
    column = np.asarray(column, dtype=float)
    target = np.asarray(target, dtype=float)
    if column.size == 0 or target.size == 0 or np.nanstd(column) == 0 or np.nanstd(target) == 0:
        return 0.0
    value = np.corrcoef(np.argsort(np.argsort(column)), np.argsort(np.argsort(target)))[0, 1]
    return float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))


def calculate_target(df, reward_feature, window_length):
    if isinstance(df, pl.DataFrame):
        target = df.select(
            (pl.col(reward_feature).shift(-window_length) - pl.col(reward_feature)).alias(reward_feature)
        )[reward_feature]
        return target.slice(0, max(target.len() - window_length, 0))
    target = df[reward_feature].shift(-window_length) - df[reward_feature]
    return target[:-window_length]


def remove_duplicates_preserve_order(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _select_reward_state_features(df: pl.DataFrame, market_type="crypto_futures", orderbook_depth=25):
    if market_type == "commodity_futures":
        from operator_futures.commodity.schema import get_reward_execution_columns

        reward_features = [col for col in get_reward_execution_columns(orderbook_depth) if col in df.columns]
    else:
        reward_features = list(df.columns[:106])
    state_feature = [col for col in df.columns if col not in reward_features]
    return reward_features, state_feature


def main(args):
    args.data_path = os.path.join(args.root_path, args.data_path)
    args.save_path = os.path.join(args.root_path, args.save_path)
    input_path = Path(args.data_path) / args.symbols / args.target_freq / f"{args.start_date}-{args.end_date}.feather"
    output_dir = Path(args.save_path) / args.symbols / args.target_freq / f"{args.start_date}-{args.end_date}"
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pl.read_ipc(input_path)
    reward_features, state_feature = _select_reward_state_features(df, "crypto_futures", 25)
    if args.market_type == "commodity_futures":
        reward_features, state_feature = _select_reward_state_features(df, args.market_type, args.orderbook_depth)

    ic_selection_key_all = []
    for window_length in args.windows_list:
        target = calculate_target(df, "mark_price", window_length)
        df_ic = df.slice(0, max(df.height - window_length, 0))
        ic_result = [calculate_cor(df_ic[feature].to_numpy(), target.to_numpy()) for feature in state_feature]
        sorted_pairs = sorted(zip(state_feature, ic_result), key=lambda x: abs(x[1]), reverse=True)
        cor = {feature: result for feature, result in sorted_pairs}
        with open(output_dir / f"rank_ic_window_{window_length}.json", "w") as f:
            json.dump(cor, f)
        ic_selection_key_all.extend([key for key, value in cor.items() if abs(value) > args.ic_theshold])

    ic_selection_key = remove_duplicates_preserve_order(ic_selection_key_all)
    df_cor = df.select(ic_selection_key).corr().with_columns(pl.Series("feature", ic_selection_key)) if ic_selection_key else pl.DataFrame({"feature": []})
    if ic_selection_key:
        df_cor = df_cor.select(["feature", *ic_selection_key])
    df_cor.write_csv(output_dir / "rank_correlation.csv")
    selected_feature_names = select_feature(corre_df=df_cor, theshold=args.cor_theshold)
    out = df.select([*reward_features, *selected_feature_names])
    out.write_ipc(output_dir / "df_rank.feather")
    np.save(output_dir / "state_features_rank.npy", np.array(selected_feature_names))
    return out


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    print("Done!")
