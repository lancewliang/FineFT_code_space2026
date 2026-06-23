import polars as pl
import numpy as np
import os
import re
from datetime import datetime
import argparse
import logging
import sys
import time

sys.path.append(".")
from memory_profiler import profile
from concurrent.futures import ThreadPoolExecutor
from operator_futures.util import find_strings_in_range


logger = logging.getLogger(__name__)


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

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
    default="PREPROCESS_DATASET/binance-futures/MERGE_CONCAT",
    help="the path of storing the data",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="PREPROCESS_DATASET/binance-futures/MERGE_CONCAT",
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
    choices=["10s", "1min", "5min","10min", "30min", "1H", "1D"],
)


def read_feather_file(file):
    return pl.read_ipc(file)


def _first_by_timestamp(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort("timestamp").group_by("timestamp", maintain_order=True).first()


def concat_concurrent_future_frames(
    concurrent_df: pl.DataFrame, future_df: pl.DataFrame
) -> pl.DataFrame:
    concurrent_df = _first_by_timestamp(concurrent_df)
    future_df = _first_by_timestamp(future_df)
    future_df = future_df.drop(
        [column for column in ["symbol", "exchange"] if column in future_df.columns]
    )
    value_columns = [column for column in future_df.columns if column != "timestamp"]
    future_df = future_df.with_columns(
        [pl.col(column).shift(1).alias(column) for column in value_columns]
    ).filter(pl.all_horizontal([pl.col(column).is_not_null() for column in value_columns]))
    return (
        concurrent_df.join(future_df, on="timestamp", how="inner")
        .sort("timestamp")
        .fill_null(strategy="forward")
    )


def main(args):
    started_at = time.monotonic()
    args.data_path = os.path.join(args.root_path, args.data_path)
    args.save_path = os.path.join(args.root_path, args.save_path)
    logger.info(
        "Starting concat process: symbol=%s start_date=%s end_date=%s target_freq=%s data_path=%s save_path=%s",
        args.symbols,
        args.start_date,
        args.end_date,
        args.target_freq,
        args.data_path,
        args.save_path,
    )
    cocurrent_path = "{}/MERGED_FEATURE/{}/{}/CONCURRENT_FEATURE".format(
        args.data_path,
        args.symbols,
        args.target_freq,
    )
    future_path = "{}/MERGED_FEATURE/{}/{}/FUTURE_FEATURE".format(
        args.data_path,
        args.symbols,
        args.target_freq,
    )
    file_list = os.listdir(cocurrent_path)
    file_list.sort()
    filter_file_list = find_strings_in_range(file_list, args.start_date, args.end_date)
    logger.info(
        "Matched concat files: available=%d selected=%d concurrent_dir=%s future_dir=%s",
        len(file_list),
        len(filter_file_list),
        cocurrent_path,
        future_path,
    )
    cocurrent_path_files = [
        os.path.join(cocurrent_path, file) for file in filter_file_list
    ]
    future_path_files = [os.path.join(future_path, file) for file in filter_file_list]
    logger.info("Reading concurrent concat files")
    with ThreadPoolExecutor() as executor:
        cocurrent_df_list = list(executor.map(read_feather_file, cocurrent_path_files))

    logger.info("Reading future concat files")
    with ThreadPoolExecutor() as executor:
        future_df_list = list(executor.map(read_feather_file, future_path_files))
    # concat seperately
    cocurrent_df = pl.concat(cocurrent_df_list, how="vertical")
    future_df = pl.concat(future_df_list, how="vertical")
    logger.info(
        "Loaded concat inputs: concurrent_rows=%d future_rows=%d",
        cocurrent_df.height,
        future_df.height,
    )
    df = concat_concurrent_future_frames(cocurrent_df, future_df)
    save_path = "{}/CONCAT_FEATURE/{}/{}".format(
        args.save_path, args.symbols, args.target_freq
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    output_path = os.path.join(save_path, "{}-{}.feather".format(args.start_date, args.end_date))
    logger.info("Writing concat output: output=%s rows=%d", output_path, df.height)
    df.write_ipc(output_path)
    logger.info(
        "Finished concat process: files=%d rows=%d elapsed_seconds=%.2f",
        len(filter_file_list),
        df.height,
        time.monotonic() - started_at,
    )


if __name__ == "__main__":
    configure_logging()
    args = parser.parse_args()
    main(args)
    logger.info("Concatenation is done.")
