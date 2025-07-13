import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


# 罗列所有需要的日期
# 开始日期到结束日期
def get_date_list(start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # 生成日期列表
    date_list = []
    current_date = start
    while current_date <= end:
        date_list.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    return date_list


# 找每个文件夹里面的是不是有对应的文件


def check_file_list(root_path, dataset_name, categories, start_date, end_date):
    has_bad_case = False
    date_list = get_date_list(start_date, end_date)
    root_path = os.path.join(
        root_path,
        dataset_name,
    )
    for cat in categories:
        path = os.path.join(
            root_path,
            cat,
        )

        for date in date_list:
            match = None
            for file in os.listdir(path):
                if date in file:
                    match = True
                    break
            if not match:
                print(f"{dataset_name} missing file for {date} in {path}")
                has_bad_case = True

    return has_bad_case


if __name__ == "__main__":
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
    root_path_list = [
        "./DOWNLOAD_DATASET/binance-futures",
    ]
    cat = ["book_snapshot_25", "trades", "derivative_ticker", "quotes"]
    for dataset_name, start_date, end_date in zip(
        dataset_name_list, start_date_list, end_date_list
    ):
        print(f"checking {dataset_name}")
        for root_path in root_path_list:
            has_bad_case = check_file_list(
                root_path, dataset_name, cat, start_date, end_date
            )
            if has_bad_case:
                break
