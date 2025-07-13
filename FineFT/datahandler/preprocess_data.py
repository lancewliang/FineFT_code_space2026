import pandas as pd
import numpy as np
import os
import sys

sys.path.append(".")
import argparse

parser = argparse.ArgumentParser()
# replay buffer coffient
parser.add_argument(
    "--data_path",
    type=str,
    default="dataset",
    help="data path",
)
parser.add_argument(
    "--trading_pair",
    type=str,
    default="BNBUSDT",
    help="trading pair",
)
parser.add_argument(
    "--chunk_length",
    type=int,
    default=8640,
    help="chunk length for one month",
)
parser.add_argument(
    "--early_stop",
    type=int,
    default=2160,
    help="early stop for 1 day",
)


# train valid test 3:1:1
def main(args):
    df = pd.read_feather(
        os.path.join(
            args.data_path,
            args.trading_pair,
            "df.feather",
        )
    )
    total_length = len(df)
    train_length = int(total_length / 2)
    valid_length = int(total_length / 5)
    train_df = df.iloc[:train_length].reset_index(drop=True)
    valid_df = df.iloc[train_length : train_length + valid_length].reset_index(
        drop=True
    )
    test_df = df.iloc[
        train_length + valid_length : train_length + 2 * valid_length
    ].reset_index(drop=True)
    train_df.to_feather(
        os.path.join(args.data_path, args.trading_pair, "train.feather")
    )
    valid_df.to_feather(
        os.path.join(args.data_path, args.trading_pair, "valid.feather")
    )
    test_df.to_feather(os.path.join(args.data_path, args.trading_pair, "test.feather"))
    train_df_chunk_num = int(len(train_df) / args.chunk_length)
    if not os.path.exists(
        os.path.join(
            args.data_path,
            args.trading_pair,
            "train",
        )
    ):
        os.makedirs(os.path.join(args.data_path, args.trading_pair, "train"))
    for i in range(train_df_chunk_num):
        train_df_chunk = train_df.iloc[
            i * args.chunk_length : (i + 1) * args.chunk_length + args.early_stop
        ].reset_index(drop=True)
        train_df_chunk.to_feather(
            os.path.join(
                args.data_path,
                args.trading_pair,
                "train",
                "df_{}.feather".format(i),
            )
        )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
