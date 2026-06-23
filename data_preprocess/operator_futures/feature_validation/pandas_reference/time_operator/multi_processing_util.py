import numpy as np
import pandas as pd
from multiprocessing import Pool
import time
from datetime import datetime
from multiprocessing import cpu_count

min_value = 1e-12


def my_rank(x):
    return pd.Series(x).rank(pct=True).iloc[-1]


def process_ohlcv_single_window(df: pd.DataFrame, w: int):
    columns = [
        "log_volume",
        # 对于每个w in window的列名
        # 用close做normalization
        *[f"roc_{w}"],
        *[f"ma_{w}"],
        *[f"std_{w}"],
        *[f"beta_{w}"],
        *[f"max_{w}"],
        *[f"min_{w}"],
        *[f"qtlu_{w}"],
        *[f"qtld_{w}"],
        *[f"rank_{w}"],
        *[f"imax_{w}"],
        *[f"imin_{w}"],
        *[f"imxd_{w}"],
        *[f"rsv_{w}"],
        *[f"cntp_{w}"],
        *[f"cntn_{w}"],
        *[f"cntd_{w}"],
        *[f"corr_{w}"],
        *[f"cord_{w}"],
        *[f"sump_{w}"],
        *[f"sumn_{w}"],
        *[f"sumd_{w}"],
        *[f"vma_{w}"],
        *[f"vstd_{w}"],
        *[f"wvma_{w}"],
        *[f"vsump_{w}"],
        *[f"vsumn_{w}"],
        *[f"vsumd_{w}"],
        # 用std做normalization
        *[f"roc_{w}_std_norm"],
        *[f"ma_{w}_std_norm"],
        *[f"beta_{w}_std_norm"],
        *[f"max_{w}_std_norm"],
        *[f"min_{w}_std_norm"],
        *[f"qtlu_{w}_std_norm"],
        *[f"qtld_{w}_std_norm"],
        *[f"rsv_{w}_std_norm"],
        *[f"vma_{w}_std_norm"],
        # 最初添加的列
        "ret1",
        "abs_ret1",
        "pos_ret1",
        "vchg1",
        "abs_vchg1",
        "pos_vchg1",
    ]
    df_feature = pd.DataFrame(columns=columns, index=df.index)
    df_feature["ret1"] = df["close"].pct_change(1, fill_method=None)
    df_feature["abs_ret1"] = np.abs(df_feature["ret1"])
    df_feature["pos_ret1"] = df_feature["ret1"]
    df_feature.loc[df_feature["pos_ret1"].lt(0), "pos_ret1"] = 0
    df_feature["vchg1"] = df["volume"] - df["volume"].shift(1)
    df_feature["abs_vchg1"] = np.abs(df_feature["vchg1"])
    df_feature["pos_vchg1"] = df_feature["vchg1"]
    df_feature.loc[df_feature["pos_vchg1"].lt(0), "pos_vchg1"] = 0
    df_feature["log_volume"] = np.log(df["volume"] + 1)

    close_shift = df["close"].shift(w)
    close_rolling = df["close"].rolling(w)
    close_std = close_rolling.std() + min_value
    volume_rolling = np.log(df["volume"] + 1).rolling(w)
    close_shift_1 = df["close"].shift(1)
    volume_shift_1 = df["volume"].shift(1)
    ori_volume_rolling = df["volume"].rolling(w)
    ori_volume_std = ori_volume_rolling.std() + min_value
    volume_std = volume_rolling.std() + min_value

    df_feature["roc_{}".format(w)] = close_shift / df["close"]
    df_feature["roc_{}_std_norm".format(w)] = close_shift / close_std

    df_feature["ma_{}".format(w)] = close_rolling.mean() / df["close"]
    df_feature["ma_{}_std_norm".format(w)] = close_rolling.mean() / close_std

    df_feature["std_{}".format(w)] = close_rolling.std() / df["close"]

    df_feature["beta_{}".format(w)] = (close_shift - df["close"]) / (w * df["close"])
    df_feature["beta_{}_std_norm".format(w)] = (close_shift - df["close"]) / (
        w * close_std
    )

    df_feature["max_{}".format(w)] = close_rolling.max() / df["close"]
    df_feature["max_{}_std_norm".format(w)] = close_rolling.max() / close_std
    df_feature["min_{}".format(w)] = close_rolling.min() / df["close"]
    df_feature["min_{}_std_norm".format(w)] = close_rolling.min() / close_std
    df_feature["qtlu_{}".format(w)] = close_rolling.quantile(0.8) / df["close"]
    df_feature["qtlu_{}_std_norm".format(w)] = close_rolling.quantile(0.8) / close_std
    df_feature["qtld_{}".format(w)] = close_rolling.quantile(0.2) / df["close"]
    df_feature["qtld_{}_std_norm".format(w)] = close_rolling.quantile(0.2) / close_std
    df_feature["rank_{}".format(w)] = close_rolling.apply(my_rank) / w
    df_feature["imax_{}".format(w)] = df["high"].rolling(w).apply(np.argmax) / w
    df_feature["imin_{}".format(w)] = df["low"].rolling(w).apply(np.argmin) / w
    df_feature["imxd_{}".format(w)] = (
        df["high"].rolling(w).apply(np.argmax) - df["low"].rolling(w).apply(np.argmin)
    ) / w
    # 前几日收盘价与当前low的最小值
    min = df["low"].where(df["low"] < close_shift, close_shift)
    # 前几日收盘价与当前high的最大值
    max = df["high"].where(df["high"] > close_shift, close_shift)
    df_feature["rsv_{}".format(w)] = (df["close"] - min) / (max - min + min_value)
    df_feature["rsv_{}_std_norm".format(w)] = (df["close"] - min) / (
        close_std + min_value
    )
    # 统计过去收益率大于0小于0的情况
    df_feature["cntp_{}".format(w)] = (df_feature["ret1"].gt(0)).rolling(w).sum() / w
    df_feature["cntn_{}".format(w)] = (df_feature["ret1"].lt(0)).rolling(w).sum() / w
    df_feature["cntd_{}".format(w)] = (
        df_feature["cntp_{}".format(w)] - df_feature["cntn_{}".format(w)]
    )
    df_feature["corr_{}".format(w)] = close_rolling.corr(pairwise=volume_rolling)
    previous_returns = df["close"] / close_shift_1
    previous_volume = np.log(df["volume"] / volume_shift_1 + 1)
    df_feature["cord_{}".format(w)] = previous_returns.rolling(w).corr(
        pairwise=previous_volume.rolling(w)
    )
    df_feature["sump_{}".format(w)] = df_feature["pos_ret1"].rolling(w).sum() / (
        df_feature["abs_ret1"].rolling(w).sum() + min_value
    )
    df_feature["sumn_{}".format(w)] = 1 - df_feature["sump_{}".format(w)]
    df_feature["sumd_{}".format(w)] = 2 * df_feature["sump_{}".format(w)] - 1
    df_feature["vma_{}".format(w)] = ori_volume_rolling.mean() / (
        df["volume"] + min_value
    )
    df_feature["vma_{}_std_norm".format(w)] = ori_volume_rolling.mean() / (
        ori_volume_std + min_value
    )
    df_feature["vstd_{}".format(w)] = ori_volume_rolling.std() / (
        df["volume"] + min_value
    )
    shift = np.abs((df["close"] / close_shift_1 - 1)) * df["volume"]
    df1 = shift.rolling(w).std()
    df2 = shift.rolling(w).mean()
    df_feature["wvma_{}".format(w)] = df1 / (df2 + min_value)
    df_feature["vsump_{}".format(w)] = df_feature["pos_vchg1"].rolling(w).sum() / (
        df_feature["abs_vchg1"].rolling(w).sum() + 1e-12
    )
    df_feature["vsumn_{}".format(w)] = 1 - df_feature["vsump_{}".format(w)]
    df_feature["vsumd_{}".format(w)] = 2 * df_feature["vsump_{}".format(w)] - 1

    df_feature.drop(
        columns=[
            "ret1",
            "abs_ret1",
            "pos_ret1",
            "vchg1",
            "abs_vchg1",
            "pos_vchg1",
        ],
        inplace=True,
    )

    df_feature.replace([np.inf, -np.inf], np.nan, inplace=True)
    # 原地用0填充所有NaN值
    df_feature.fillna(0, inplace=True)
    df_feature = df_feature.iloc[w + 10 :]

    return df_feature


def process_ohlc_single_window(df: pd.DataFrame, w: int):
    df_feature = pd.DataFrame(index=df.index)
    df_feature["ret1"] = df["close"].pct_change(1, fill_method=None)
    df_feature["abs_ret1"] = np.abs(df_feature["ret1"])
    df_feature["pos_ret1"] = df_feature["ret1"]
    df_feature.loc[df_feature["pos_ret1"].lt(0), "pos_ret1"] = 0

    close_shift = df["close"].shift(w)
    close_rolling = df["close"].rolling(w)
    close_std = close_rolling.std() + min_value
    df_feature["roc_{}".format(w)] = close_shift / df["close"]
    df_feature["roc_{}_std_norm".format(w)] = close_shift / close_std

    df_feature["ma_{}".format(w)] = close_rolling.mean() / df["close"]
    df_feature["ma_{}_std_norm".format(w)] = close_rolling.mean() / close_std

    df_feature["std_{}".format(w)] = close_rolling.std() / df["close"]

    df_feature["beta_{}".format(w)] = (close_shift - df["close"]) / (w * df["close"])
    df_feature["beta_{}_std_norm".format(w)] = (close_shift - df["close"]) / (
        w * close_std
    )
    df_feature["max_{}".format(w)] = close_rolling.max() / df["close"]
    df_feature["max_{}_std_norm".format(w)] = close_rolling.max() / close_std

    df_feature["min_{}".format(w)] = close_rolling.min() / df["close"]
    df_feature["min_{}_std_norm".format(w)] = close_rolling.min() / close_std

    df_feature["qtlu_{}".format(w)] = close_rolling.quantile(0.8) / df["close"]
    df_feature["qtlu_{}_std_norm".format(w)] = close_rolling.quantile(0.8) / close_std

    df_feature["qtld_{}".format(w)] = close_rolling.quantile(0.2) / df["close"]
    df_feature["qtld_{}_std_norm".format(w)] = close_rolling.quantile(0.2) / close_std

    df_feature["rank_{}".format(w)] = close_rolling.apply(my_rank) / w
    df_feature["imax_{}".format(w)] = df["high"].rolling(w).apply(np.argmax) / w
    df_feature["imin_{}".format(w)] = df["low"].rolling(w).apply(np.argmin) / w
    df_feature["imxd_{}".format(w)] = (
        df["high"].rolling(w).apply(np.argmax) - df["low"].rolling(w).apply(np.argmin)
    ) / w
    # 前几日收盘价与当前low的最小值
    min = df["low"].where(df["low"] < close_shift, close_shift)
    # 前几日收盘价与当前high的最大值
    max = df["high"].where(df["high"] > close_shift, close_shift)
    df_feature["rsv_{}".format(w)] = (df["close"] - min) / (max - min + 1e-12)
    # 统计过去收益率大于0小于0的情况
    df_feature["cntp_{}".format(w)] = (df_feature["ret1"].gt(0)).rolling(w).sum() / w
    df_feature["cntn_{}".format(w)] = (df_feature["ret1"].lt(0)).rolling(w).sum() / w
    df_feature["cntd_{}".format(w)] = (
        df_feature["cntp_{}".format(w)] - df_feature["cntn_{}".format(w)]
    )

    df_feature["sump_{}".format(w)] = df_feature["pos_ret1"].rolling(w).sum() / (
        df_feature["abs_ret1"].rolling(w).sum() + 1e-12
    )
    df_feature["sumn_{}".format(w)] = 1 - df_feature["sump_{}".format(w)]
    df_feature["sumd_{}".format(w)] = 2 * df_feature["sump_{}".format(w)] - 1

    df_feature.drop(
        columns=[
            "ret1",
            "abs_ret1",
            "pos_ret1",
        ],
        inplace=True,
    )

    df_feature.replace([np.inf, -np.inf], np.nan, inplace=True)
    # 原地用0填充所有NaN值
    df_feature.fillna(0, inplace=True)
    df_feature = df_feature.iloc[w + 1 :]

    return df_feature


def process_single_price_single_window(df: pd.Series, w: int):
    # the rename has been done in the function
    # create feature for none ohlcv form information
    assert len(df.shape) == 1
    feature_name = df.name
    columns = [
        *[f"log_return_{w}"],
        *[f"rolling_mean_{w}"],
        *[f"std_{w}"],
        *[f"trend_{w}"],
    ]
    df_time_feature = pd.DataFrame(columns=columns, index=df.index)
    rolling = df.rolling(w)
    df_time_feature[f"log_return_{w}"] = np.log(df / (df.shift(1)+min_value)) * 1000
    if w != 1:
        df_time_feature[f"rolling_mean_{w}"] = rolling.mean()
        df_time_feature[f"std_{w}"] = rolling.std()
        df_time_feature[f"trend_{w}"] = (df - df_time_feature[f"rolling_mean_{w}"]) / (
            df_time_feature[f"std_{w}"] + min_value
        )
        df_time_feature.drop(columns=[f"rolling_mean_{w}", f"std_{w}"], inplace=True)
    else:
        df_time_feature.drop(
            columns=[f"rolling_mean_{w}", f"std_{w}", f"trend_{w}"], inplace=True
        )
    df_time_feature = df_time_feature.iloc[w + 1 :]
    df_time_feature.rename(
        columns={
            column: feature_name + "_" + column for column in df_time_feature.columns
        },
        inplace=True,
    )
    return df_time_feature


def remove_duplicate_columns(df):
    # 转置DataFrame，因为drop_duplicates作用于行
    df_transposed = df.T
    # 删除重复的行（现在的行是原始的列）
    df_transposed = df_transposed.drop_duplicates()
    # 再次转置，回到原始的行列布局
    df_unique = df_transposed.T
    return df_unique


def get_multi_window_ohlcv(df, windows):
    with Pool(processes=len(windows)) as pool:
        results = [
            pool.apply_async(process_ohlcv_single_window, args=(df, w)) for w in windows
        ]

        # 等待所有结果完成，并获取结果
        processed_dfs = [result.get() for result in results]
    df_final = pd.concat(processed_dfs, axis=1, join="inner")
    df_final = remove_duplicate_columns(df_final)
    return df_final


def get_multi_window_ohlc(df, windows):
    with Pool(processes=len(windows)) as pool:
        results = [
            pool.apply_async(process_ohlc_single_window, args=(df, w)) for w in windows
        ]

        # 等待所有结果完成，并获取结果
        processed_dfs = [result.get() for result in results]
    df_final = pd.concat(processed_dfs, axis=1, join="inner")
    df_final = remove_duplicate_columns(df_final)
    return df_final


def get_multi_feature_window_price(df, windows, feature_name_list):
    df_list = [df[feature_name] for feature_name in feature_name_list]
    max_workers = int(min(cpu_count() / 2, len(df_list) * len(windows)))
    with Pool(processes=max_workers) as pool:
        results = [
            pool.apply_async(process_single_price_single_window, args=(df_single, w))
            for w in windows
            for df_single in df_list
        ]
        processed_dfs = [result.get() for result in results]
    df_final = pd.concat(processed_dfs, axis=1, join="inner")
    df_final = remove_duplicate_columns(df_final)
    return df_final


if __name__ == "__main__":
    start_time = time.time()
    windows = [5, 10]
    price_features = [
        *[f"bid{l+1}_price" for l in range(25)],
        *[f"ask{l+1}_price" for l in range(25)],
        "buy_spread_oe_max",
        "sell_spread_oe_max",
        "wap_1",
        "wap_2",
        "buy_wap",
        "sell_wap",
        "mark_price",
    ]
    # df = pd.read_feather("outlook/demo_df/df_ohlcv.feather")
    # df.set_index("timestamp", inplace=True)
    # df_final = get_multi_window_ohlcv(df, windows=windows)
    # df_final.reset_index(inplace=True)
    # end_time = time.time()
    # print("cost time", end_time - start_time)
    # df_final.to_feather("outlook/demo_df/multi_processing_p_df_ohlcv.feather")

    # df = pd.read_feather("outlook/demo_df/df_ohlc.feather")
    # df.set_index("timestamp", inplace=True)
    # df_final = get_multi_window_ohlc(df, windows=windows)
    # df_final.reset_index(inplace=True)
    # end_time = time.time()
    # print("cost time", end_time - start_time)
    # df_final.to_feather("outlook/demo_df/multi_processing_p_df_ohlc.feather")

    df = pd.read_feather(
        "./PREPROCESS_DATASET/binance-futures/MERGE_CONCAT/CONCAT_FEATURE/BNBUSDT/5min/2021-04-01-2024-05-01.feather"
    )
    df.set_index("timestamp", inplace=True)
    df_time=get_multi_feature_window_price(df, windows, price_features)
    df_time.reset_index(inplace=True)
    end_time = time.time()
    print("cost time", end_time - start_time)
    df_time.to_feather("outlook/demo_df/multi_processing_p_df_time.feather")