# utlize the correlation between the return of the mark price and feature.
# select the features with high correlation with markprice return and low correlation with existing features.
import numpy as np
import os
import argparse
import sys
import json
from pathlib import Path

import polars as pl

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
    column = np.asarray(column, dtype=float)
    target = np.asarray(target, dtype=float)
    if column.size == 0 or target.size == 0 or np.nanstd(column) == 0 or np.nanstd(target) == 0:
        return 0.0
    value = np.corrcoef(column, target)[0, 1]
    return float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))


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
    if isinstance(df, pl.DataFrame):
        target = df.select(
            (pl.col(reward_feature).shift(-window_length) - pl.col(reward_feature)).alias(
                reward_feature
            )
        )[reward_feature]
        target = target.slice(0, max(target.len() - window_length, 0))
        return target
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
    input_path = Path(args.data_path) / args.symbols / args.target_freq / (
        "{}-{}.feather".format(args.start_date, args.end_date)
    )
    output_dir = Path(args.save_path) / args.symbols / args.target_freq / (
        "{}-{}".format(args.start_date, args.end_date)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pl.read_ipc(input_path)
    reward_features, state_feature = select_reward_state_features(
        df, args.market_type, args.orderbook_depth
    )

    ic_selection_key_all = []
    for window_length in args.windows_list:
        target = calculate_target(df, "mark_price", window_length)
        df_ic = df.slice(0, max(df.height - window_length, 0))
        ic_result = [
            calculate_cor(df_ic[feature].to_numpy(), target.to_numpy())
            for feature in state_feature
        ]
        sorted_pairs = sorted(
            zip(state_feature, ic_result), key=lambda x: abs(x[1]), reverse=True
        )

        cor = {feature: result for feature, result in sorted_pairs}
        with open(output_dir / "ic_window_{}.json".format(window_length), "w") as f:
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
    if ic_selection_key:
        df_cor = df.select(ic_selection_key).corr().with_columns(
            pl.Series("feature", ic_selection_key)
        )
        df_cor = df_cor.select(["feature", *ic_selection_key])
    else:
        df_cor = pl.DataFrame({"feature": []})
    df_cor.write_csv(output_dir / "correlation.csv")
    selected_feature_names = select_feature(corre_df=df_cor, theshold=args.cor_theshold)
    state_feature = selected_feature_names
    out = df.select([*reward_features, *state_feature])
    out.write_ipc(output_dir / "df.feather")
    np.save(output_dir / "state_features.npy", np.array(state_feature))
    return out


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    print("Done!")
