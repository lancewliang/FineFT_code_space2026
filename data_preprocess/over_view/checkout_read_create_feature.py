import pandas as pd
import numpy as np
import sys

sys.path.append(".")
from over_view.checkout_create_feature import get_date_list
import os
from pyarrow.lib import ArrowInvalid
from concurrent.futures import ThreadPoolExecutor, as_completed


def read_feather_file(data_path, date):
    df_path = os.path.join(data_path, f"{date}.feather")
    error_log = {"date": date, "errors": []}

    try:
        df = pd.read_feather(df_path)
    except ArrowInvalid as e:
        error_log["errors"].append(
            f"Error reading current Arrow file for date {date}: {e}, the path is {df_path}"
        )

    return error_log


def check_read_file_list_single(
    root_path, target_freq, dataset_name, start_date, end_date
):
    date_list = get_date_list(start_date, end_date)
    path = os.path.join(root_path, dataset_name, target_freq)
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(read_feather_file, path, date): date for date in date_list
        }
        for future in as_completed(futures):
            result = future.result()
            date = result["date"]
            if result["errors"]:
                for error in result["errors"]:
                    print(error)


def check_read_file_list_double(
    root_path, target_freq, dataset_name, start_date, end_date
):
    # for merge data preprocessing
    date_list = get_date_list(start_date, end_date)
    path = os.path.join(root_path, dataset_name, target_freq, "CONCURRENT_FEATURE")
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(read_feather_file, path, date): date for date in date_list
        }
        for future in as_completed(futures):
            result = future.result()
            date = result["date"]
            if result["errors"]:
                for error in result["errors"]:
                    print(error)
    path = os.path.join(root_path, dataset_name, target_freq, "FUTURE_FEATURE")
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(read_feather_file, path, date): date for date in date_list
        }
        for future in as_completed(futures):
            result = future.result()
            date = result["date"]
            if result["errors"]:
                for error in result["errors"]:
                    print(error)


def main(base_path, process, target_freq, dataset_name, start_date, end_date):
    root_path = os.path.join(base_path, process)
    if process == "MERGE_CONCAT/MERGED_FEATURE":
        check_read_file_list_double(
            root_path, target_freq, dataset_name, start_date, end_date
        )
    else:
        check_read_file_list_single(
            root_path, target_freq, dataset_name, start_date, end_date
        )


if __name__ == "__main__":
    base_path = "./PREPROCESS_DATASET/binance-futures"
    process_list = [
        "DOWNSCALE_DERTIC",
        "DOWNSCALE_ORDERBOOK_25",
        "BASE_FEATURE",
        "CROSS_SECTION/KLINE_FEATURE",
        "CROSS_SECTION/QUOTES_FEATURE",
        "CROSS_SECTION/SNAPSHOT_FEATURE",
        # "MERGE_CONCAT/MERGED_FEATURE",
    ]
    for process in process_list:
        target_freq = "5min"
        start_date_list = [
            "2021-04-01",
            "2021-04-01",
            "2021-04-01",
            "2021-04-01",
        ]
        end_date_list = [
            "2024-07-15",
            "2024-07-15",
            "2024-07-15",
            "2024-07-15",
        ]
        dataset_name_list = [
            "BNBUSDT",
            "BTCUSDT",
            "ETHUSDT",
            "DOTUSDT",
        ]
        for dataset_name in dataset_name_list:
            main(
                base_path,
                process,
                target_freq,
                dataset_name,
                start_date_list[0],
                end_date_list[0],
            )
