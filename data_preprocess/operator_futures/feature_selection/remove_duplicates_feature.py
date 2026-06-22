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
    windows_list = sorted(args.windows_list)
    args.data_path = os.path.join(args.root_path, args.data_path)
    args.save_path = os.path.join(args.root_path, args.save_path)
    input_path = Path(args.data_path) / args.symbols / args.target_freq / f"{args.start_date}-{args.end_date}.feather"
    output_dir = Path(args.save_path) / args.symbols / args.target_freq / f"{args.start_date}-{args.end_date}"
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pl.read_ipc(input_path)
    reward_features = list(df.columns[:106])
    ic_selection_key_all = []
    assert args.ic_choice in ["ic", "rank_ic", "catboost"]

    if args.ic_choice == "ic":
        ic_file_name_list = [f"ic_window_{w}.json" for w in windows_list]
        cor_file_name = "correlation.csv"
        for ic_file_name in ic_file_name_list:
            with open(output_dir / ic_file_name, "r", encoding="utf-8") as f:
                cor = json.load(f)
            ic_selection_key_all.extend(
                [key for key, value in cor.items() if abs(value) > args.ic_theshold]
            )
    elif args.ic_choice == "rank_ic":
        ic_file_name_list = [f"rank_ic_window_{w}.json" for w in windows_list]
        cor_file_name = "rank_correlation.csv"
        for ic_file_name in ic_file_name_list:
            with open(output_dir / ic_file_name, "r", encoding="utf-8") as f:
                cor = json.load(f)
            ic_selection_key_all.extend(
                [key for key, value in cor.items() if abs(value) > args.ic_theshold]
            )
    else:
        ic_file_name_list = [f"cat_boost_feature_importance_{w}.csv" for w in windows_list]
        cor_file_name = "correlation_catboost.csv"
        for ic_file_name in ic_file_name_list:
            feature_importance_df = pl.read_csv(output_dir / ic_file_name)
            ic_selection_key_all.extend(
                feature_importance_df.filter(pl.col("Importance") > args.ic_theshold)["Feature"].to_list()
            )

    ic_selection_key = remove_duplicates_preserve_order(ic_selection_key_all)
    df_cor = pl.read_csv(output_dir / cor_file_name)
    if "feature" not in df_cor.columns:
        df_cor = df_cor.rename({df_cor.columns[0]: "feature"})
    if ic_selection_key:
        keep_columns = ["feature", *ic_selection_key]
        df_cor = df_cor.select([column for column in keep_columns if column in df_cor.columns])
    selected_feature_names = select_feature(corre_df=df_cor, theshold=args.cor_theshold)
    out = df.select([*reward_features, *selected_feature_names])

    if args.ic_choice == "ic":
        out.write_ipc(output_dir / "df.feather")
        np.save(output_dir / "state_features.npy", np.array(selected_feature_names))
    elif args.ic_choice == "rank_ic":
        out.write_ipc(output_dir / "df_rank.feather")
        np.save(output_dir / "state_features_rank.npy", np.array(selected_feature_names))
    else:
        out.write_ipc(output_dir / "df_catboost.feather")
        np.save(output_dir / "state_features_catboost.npy", np.array(selected_feature_names))
    return out


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
