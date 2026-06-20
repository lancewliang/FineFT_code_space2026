import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
import argparse
import sys

sys.path.append(".")
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
    default="PREPROCESS_DATASET/binance-futures/IC_RESULT",
    help="the path of storing the data",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="PREPROCESS_DATASET/binance-futures/SCALE_SAVE/",
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
parser.add_argument(
    "--clip_theshold",
    type=float,
    default=10,
    help="the date of start",
)
parser.add_argument(
    "--base",
    type=float,
    default=10,
    help="the date of start",
)
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


def scale_std(df, log_base=10):
    df_state_std = np.log10(df.std()) * np.log10(log_base) / np.log10(10)
    clipped_values = np.floor(df_state_std)
    scale = log_base**clipped_values
    std_scaled = df / scale
    return std_scaled


def scale_mean(df, log_base=10, clip_theshold=10):
    mean_values = df.mean()

    # Calculate the absolute means to determine the magnitude of adjustment needed
    abs_mean_values = np.abs(mean_values)

    # Determine the power of 10 to adjust by, only for means outside the [-10, 10] range
    # We use np.where to filter out the means within the range (no adjustment needed)
    adjustment_powers = np.where(
        (abs_mean_values > clip_theshold),
        np.power(
            log_base,
            np.round(np.log10(abs_mean_values) * np.log10(log_base) / np.log10(10)),
        ),
        0,
    )

    # For positive means that are too large, we subtract the adjustment
    # For negative means that are too small, we add the adjustment
    adjustment_values = np.where(mean_values > 0, -adjustment_powers, adjustment_powers)

    # Adjust the DataFrame
    adjusted_values = df + adjustment_values

    return adjusted_values


def main(args):
    args.data_path = os.path.join(args.root_path, args.data_path)
    args.save_path = os.path.join(args.root_path, args.save_path)
    assert args.ic_choice in ["ic", "rank_ic", "catboost"]
    if args.ic_choice == "ic":
        df_name = "df"
        state_name='state_features'
    elif args.ic_choice == "rank_ic":
        df_name = "df_rank"
        state_name='state_features_rank'
    elif args.ic_choice == "catboost":
        df_name = "df_catboost"
        state_name='state_features_catboost'
    else:
        print("wrong ic_choice")

    df = pd.read_feather(
        os.path.join(
            args.data_path,
            args.symbols,
            args.target_freq,
            "{}-{}".format(args.start_date, args.end_date),
            "{}.feather".format(df_name),
        )
    )
    if args.market_type == "commodity_futures":
        from operator_futures.commodity.schema import get_reward_execution_columns

        reward_features = [
            col for col in get_reward_execution_columns(args.orderbook_depth)
            if col in df.columns
        ]
    else:
        reward_features = df.columns[:106]
    state_feature = np.load(
        os.path.join(
            args.data_path,
            args.symbols,
            args.target_freq,
            "{}-{}".format(args.start_date, args.end_date),
            "{}.npy".format(state_name),
        )
    )
    df_reward = df[reward_features]
    df_state = df[state_feature]
    df_state = scale_std(df_state, args.base)
    df_state = scale_mean(df_state, args.base, args.clip_theshold)
    df_describe = df_state.describe()
    df = pd.concat([df_reward, df_state], axis=1)
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
    df.to_feather(
        os.path.join(
            args.save_path,
            args.symbols,
            args.target_freq,
            "{}-{}".format(args.start_date, args.end_date),
            "df.feather",
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
    df_describe.to_csv(
        os.path.join(
            args.save_path,
            args.symbols,
            args.target_freq,
            "{}-{}".format(args.start_date, args.end_date),
            "df_describe.csv",
        )
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
