import numpy as np
import pandas as pd

import time


# 程序开始前的时间
def my_rank(x):
    return pd.Series(x).rank(pct=True).iloc[-1]


def process_ohlcv(df: pd.DataFrame, window: list):
    max_window = np.max(window)
    columns = [
        "log_volume",
        # 对于每个w in window的列名
        *[f"roc_{w}" for w in window],
        *[f"ma_{w}" for w in window],
        *[f"std_{w}" for w in window],
        *[f"beta_{w}" for w in window],
        *[f"max_{w}" for w in window],
        *[f"min_{w}" for w in window],
        *[f"qtlu_{w}" for w in window],
        *[f"qtld_{w}" for w in window],
        *[f"rank_{w}" for w in window],
        *[f"imax_{w}" for w in window],
        *[f"imin_{w}" for w in window],
        *[f"imxd_{w}" for w in window],
        *[f"rsv_{w}" for w in window],
        *[f"cntp_{w}" for w in window],
        *[f"cntn_{w}" for w in window],
        *[f"cntd_{w}" for w in window],
        *[f"corr_{w}" for w in window],
        *[f"cord_{w}" for w in window],
        *[f"sump_{w}" for w in window],
        *[f"sumn_{w}" for w in window],
        *[f"sumd_{w}" for w in window],
        *[f"vma_{w}" for w in window],
        *[f"vstd_{w}" for w in window],
        *[f"wvma_{w}" for w in window],
        *[f"vsump_{w}" for w in window],
        *[f"vsumn_{w}" for w in window],
        *[f"vsumd_{w}" for w in window],
        # 最初添加的列
        "ret1",
        "abs_ret1",
        "pos_ret1",
        "vchg1",
        "abs_vchg1",
        "pos_vchg1",
    ]
    df_feature = pd.DataFrame(columns=columns,index=df.index)
    df_feature["ret1"] = df["close"].pct_change(1, fill_method=None)
    df_feature["abs_ret1"] = np.abs(df_feature["ret1"])
    df_feature["pos_ret1"] = df_feature["ret1"]
    df_feature.loc[df_feature["pos_ret1"].lt(0), "pos_ret1"] = 0
    df_feature["vchg1"] = df["volume"] - df["volume"].shift(1)
    df_feature["abs_vchg1"] = np.abs(df_feature["vchg1"])
    df_feature["pos_vchg1"] = df_feature["vchg1"]
    df_feature.loc[df_feature["pos_vchg1"].lt(0), "pos_vchg1"] = 0
    df_feature["log_volume"] = np.log(df["volume"] + 1)
    for w in window:
        close_shift = df["close"].shift(w)
        close_rolling = df["close"].rolling(w)
        volume_rolling = np.log(df["volume"] + 1).rolling(w)
        close_shift_1 = df["close"].shift(1)
        volume_shift_1 = df["volume"].shift(1)
        ori_volume_rolling = df["volume"].rolling(w)

        df_feature["roc_{}".format(w)] = close_shift / df["close"]

        df_feature["ma_{}".format(w)] = close_rolling.mean() / df["close"]

        df_feature["std_{}".format(w)] = close_rolling.std() / df["close"]

        df_feature["beta_{}".format(w)] = (close_shift - df["close"]) / (
            w * df["close"]
        )
        df_feature["max_{}".format(w)] = close_rolling.max() / df["close"]
        df_feature["min_{}".format(w)] = close_rolling.min() / df["close"]
        df_feature["qtlu_{}".format(w)] = close_rolling.quantile(0.8) / df["close"]
        df_feature["qtld_{}".format(w)] = close_rolling.quantile(0.2) / df["close"]
        df_feature["rank_{}".format(w)] = close_rolling.apply(my_rank) / w
        df_feature["imax_{}".format(w)] = df["high"].rolling(w).apply(np.argmax) / w
        df_feature["imin_{}".format(w)] = df["low"].rolling(w).apply(np.argmin) / w
        df_feature["imxd_{}".format(w)] = (
            df["high"].rolling(w).apply(np.argmax)
            - df["low"].rolling(w).apply(np.argmin)
        ) / w
        # 前几日收盘价与当前low的最小值
        min = df["low"].where(df["low"] < close_shift, close_shift)
        # 前几日收盘价与当前high的最大值
        max = df["high"].where(df["high"] > close_shift, close_shift)
        df_feature["rsv_{}".format(w)] = (df["close"] - min) / (max - min + 1e-12)
        # 统计过去收益率大于0小于0的情况
        df_feature["cntp_{}".format(w)] = (df_feature["ret1"].gt(0)).rolling(
            w
        ).sum() / w
        df_feature["cntn_{}".format(w)] = (df_feature["ret1"].lt(0)).rolling(
            w
        ).sum() / w
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
            df_feature["abs_ret1"].rolling(w).sum() + 1e-12
        )
        df_feature["sumn_{}".format(w)] = 1 - df_feature["sump_{}".format(w)]
        df_feature["sumd_{}".format(w)] = 2 * df_feature["sump_{}".format(w)] - 1
        df_feature["vma_{}".format(w)] = ori_volume_rolling.mean() / (
            df["volume"] + 1e-12
        )
        df_feature["vstd_{}".format(w)] = ori_volume_rolling.std() / (
            df["volume"] + 1e-12
        )
        shift = np.abs((df["close"] / close_shift_1 - 1)) * df["volume"]
        df1 = shift.rolling(w).std()
        df2 = shift.rolling(w).mean()
        df_feature["wvma_{}".format(w)] = df1 / (df2 + 1e-12)
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
    df_feature = df_feature.iloc[max_window+1:]

    return df_feature


def process_ohlc(df: pd.DataFrame, window: list):
    max_window = np.max(window)
    df_feature = pd.DataFrame(index=df.index)
    df_feature["ret1"] = df["close"].pct_change(1, fill_method=None)
    df_feature["abs_ret1"] = np.abs(df_feature["ret1"])
    df_feature["pos_ret1"] = df_feature["ret1"]
    df_feature.loc[df_feature["pos_ret1"].lt(0), "pos_ret1"] = 0

    for w in window:
        close_shift = df["close"].shift(w)
        close_rolling = df["close"].rolling(w)

        df_feature["roc_{}".format(w)] = close_shift / df["close"]

        df_feature["ma_{}".format(w)] = close_rolling.mean() / df["close"]

        df_feature["std_{}".format(w)] = close_rolling.std() / df["close"]

        df_feature["beta_{}".format(w)] = (close_shift - df["close"]) / (
            w * df["close"]
        )
        df_feature["max_{}".format(w)] = close_rolling.max() / df["close"]
        df_feature["min_{}".format(w)] = close_rolling.min() / df["close"]
        df_feature["qtlu_{}".format(w)] = close_rolling.quantile(0.8) / df["close"]
        df_feature["qtld_{}".format(w)] = close_rolling.quantile(0.2) / df["close"]
        df_feature["rank_{}".format(w)] = close_rolling.apply(my_rank) / w
        df_feature["imax_{}".format(w)] = df["high"].rolling(w).apply(np.argmax) / w
        df_feature["imin_{}".format(w)] = df["low"].rolling(w).apply(np.argmin) / w
        df_feature["imxd_{}".format(w)] = (
            df["high"].rolling(w).apply(np.argmax)
            - df["low"].rolling(w).apply(np.argmin)
        ) / w
        # 前几日收盘价与当前low的最小值
        min = df["low"].where(df["low"] < close_shift, close_shift)
        # 前几日收盘价与当前high的最大值
        max = df["high"].where(df["high"] > close_shift, close_shift)
        df_feature["rsv_{}".format(w)] = (df["close"] - min) / (max - min + 1e-12)
        # 统计过去收益率大于0小于0的情况
        df_feature["cntp_{}".format(w)] = (df_feature["ret1"].gt(0)).rolling(
            w
        ).sum() / w
        df_feature["cntn_{}".format(w)] = (df_feature["ret1"].lt(0)).rolling(
            w
        ).sum() / w
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
    df_feature = df_feature.iloc[max_window+1:]

    return df_feature




if __name__ == "__main__":
    start_time = time.time()
    df_ohlcv = pd.read_feather("outlook/demo_df/df_ohlcv.feather")
    p_df_ohlcv = process_ohlcv(df_ohlcv, [5,10])
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"程序运行时间: {elapsed_time} 秒")
    p_df_ohlcv.to_feather("outlook/demo_df/p_df_ohlcv.feather")

    start_time = time.time()
    df_ohlc = pd.read_feather("outlook/demo_df/df_ohlc.feather")
    p_df_ohlc = process_ohlc(df_ohlc, [5,10])
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"程序运行时间: {elapsed_time} 秒")
    p_df_ohlc.to_feather("outlook/demo_df/p_df_ohlc.feather")
