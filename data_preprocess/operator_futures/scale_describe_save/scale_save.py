import argparse
import os
from pathlib import Path

import numpy as np
import polars as pl

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default=".", help="the path of storing the data")
parser.add_argument(
    "--data_path",
    type=str,
    default="PREPROCESS_DATASET/binance-futures/IC_RESULT",
    help="the path of storing the data",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="PREPROCESS_DATASET/binance-futures/SCALE_SAVE/",
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
parser.add_argument("--clip_theshold", type=float, default=10, help="the date of start")
parser.add_argument("--base", type=float, default=10, help="the date of start")
parser.add_argument(
    "--ic_choice",
    type=str,
    default="catboost",
    choices=["ic", "rank_ic", "catboost"],
    help="the way of choosing features",
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


def scale_std(df: pl.DataFrame, log_base=10):
    exprs = []
    for column in df.columns:
        std = df.select(pl.col(column).std()).item()
        if std is None or std == 0:
            scale = 1.0
        else:
            scale = log_base ** np.floor(np.log10(std) * np.log10(log_base) / np.log10(10))
        exprs.append((pl.col(column) / scale).alias(column))
    return df.with_columns(exprs)


def scale_mean(df: pl.DataFrame, log_base=10, clip_theshold=10):
    exprs = []
    for column in df.columns:
        mean_value = df.select(pl.col(column).mean()).item()
        abs_mean = abs(mean_value) if mean_value is not None else 0.0
        if abs_mean > clip_theshold:
            adjustment = log_base ** np.round(np.log10(abs_mean) * np.log10(log_base) / np.log10(10))
            adjustment = -adjustment if mean_value > 0 else adjustment
        else:
            adjustment = 0.0
        exprs.append((pl.col(column) + adjustment).alias(column))
    return df.with_columns(exprs)


def main(args):
    args.data_path = os.path.join(args.root_path, args.data_path)
    args.save_path = os.path.join(args.root_path, args.save_path)
    assert args.ic_choice in ["ic", "rank_ic", "catboost"]
    if args.ic_choice == "ic":
        df_name = "df"
        state_name = "state_features"
    elif args.ic_choice == "rank_ic":
        df_name = "df_rank"
        state_name = "state_features_rank"
    else:
        df_name = "df_catboost"
        state_name = "state_features_catboost"

    input_dir = Path(args.data_path) / args.symbols / args.target_freq / f"{args.start_date}-{args.end_date}"
    output_dir = Path(args.save_path) / args.symbols / args.target_freq / f"{args.start_date}-{args.end_date}"
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pl.read_ipc(input_dir / f"{df_name}.feather")
    if args.market_type == "commodity_futures":
        from operator_futures.commodity.schema import get_reward_execution_columns

        reward_features = [col for col in get_reward_execution_columns(args.orderbook_depth) if col in df.columns]
    else:
        reward_features = list(df.columns[:106])
    state_feature = np.load(input_dir / f"{state_name}.npy", allow_pickle=True).tolist()
    df_reward = df.select(reward_features)
    df_state = df.select(state_feature)
    df_state = scale_std(df_state, args.base)
    df_state = scale_mean(df_state, args.base, args.clip_theshold)
    df_describe = df_state.describe()
    out = pl.concat([df_reward, df_state], how="horizontal")
    out.write_ipc(output_dir / "df.feather")
    np.save(output_dir / "state_features.npy", np.array(state_feature))
    df_describe.write_csv(output_dir / "df_describe.csv")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
