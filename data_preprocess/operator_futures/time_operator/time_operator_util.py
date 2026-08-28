import numpy as np
import pandas as pd
import polars as pl

import time


_MARKET_STATE_ANCHOR_EPSILON = 1e-8


def _rolling_log_price_anchors(
    close_values: np.ndarray,
) -> dict[str, np.ndarray]:
    invalid_close = (~np.isfinite(close_values)) | (close_values <= 0.0)
    if invalid_close.any():
        first_invalid_index = int(np.flatnonzero(invalid_close)[0])
        raise ValueError(
            "close must contain only finite, strictly positive prices: "
            f"row={first_invalid_index} value={close_values[first_invalid_index]}"
        )

    log_prices = np.log(close_values)
    row_count = len(log_prices)
    anchors = {
        "log_price_slope_48": np.zeros(row_count, dtype=float),
        "log_price_slope_96": np.zeros(row_count, dtype=float),
        "trend_to_noise_48": np.zeros(row_count, dtype=float),
        "trend_to_noise_96": np.zeros(row_count, dtype=float),
        "signed_efficiency_48": np.zeros(row_count, dtype=float),
        "trend_r2_48": np.zeros(row_count, dtype=float),
        "log_return_vol_quantile_192": np.zeros(row_count, dtype=float),
    }
    log_returns = np.diff(log_prices)
    volatility_by_window: dict[int, np.ndarray] = {}

    for window in (48, 96):
        if row_count < window:
            continue

        centered_steps = np.arange(window, dtype=float) - (window - 1) / 2.0
        centered_step_sum_squares = float(np.square(centered_steps).sum())
        rolling_log_prices = np.lib.stride_tricks.sliding_window_view(
            log_prices, window
        )
        slopes = rolling_log_prices @ centered_steps / centered_step_sum_squares

        rolling_returns = np.lib.stride_tricks.sliding_window_view(
            log_returns, window - 1
        )
        constant_windows = np.all(rolling_returns == 0.0, axis=1)
        slopes[constant_windows] = 0.0
        return_volatility = rolling_returns.std(axis=1, ddof=0)
        volatility_by_window[window] = return_volatility

        slope_column = anchors[f"log_price_slope_{window}"]
        slope_column[window - 1 :] = slopes
        trend_to_noise = anchors[f"trend_to_noise_{window}"]
        trend_to_noise[window - 1 :] = slopes / np.maximum(
            return_volatility, _MARKET_STATE_ANCHOR_EPSILON
        )

        if window == 48:
            absolute_return_sum = np.abs(rolling_returns).sum(axis=1)
            efficiency = (rolling_log_prices[:, -1] - rolling_log_prices[:, 0]) / np.maximum(
                absolute_return_sum, _MARKET_STATE_ANCHOR_EPSILON
            )
            anchors["signed_efficiency_48"][window - 1 :] = np.clip(
                efficiency, -1.0, 1.0
            )

            centered_log_prices = rolling_log_prices - rolling_log_prices.mean(
                axis=1, keepdims=True
            )
            total_sum_squares = np.square(centered_log_prices).sum(axis=1)
            r_squared = np.zeros_like(slopes)
            non_constant = ~constant_windows
            r_squared[non_constant] = (
                np.square(slopes[non_constant]) * centered_step_sum_squares
            ) / total_sum_squares[non_constant]
            anchors["trend_r2_48"][window - 1 :] = np.clip(
                r_squared, 0.0, 1.0
            )

    volatility_48 = volatility_by_window.get(48)
    rank_window = 192
    if volatility_48 is not None and len(volatility_48) >= rank_window:
        quantiles = anchors["log_return_vol_quantile_192"]
        for end_index in range(rank_window - 1, len(volatility_48)):
            history = volatility_48[end_index - rank_window + 1 : end_index + 1]
            current = history[-1]
            less_count = float(np.count_nonzero(history < current))
            equal_count = float(np.count_nonzero(history == current))
            average_rank = less_count + (equal_count + 1.0) / 2.0
            price_index = 47 + end_index
            quantiles[price_index] = average_rank / rank_window

    return anchors


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
    if isinstance(df, pl.DataFrame):
        max_window = int(np.max(window))
        frame = df.sort("timestamp").with_columns(
            (pl.col("close") / pl.col("close").shift(1) - 1).alias("ret1")
        )
        frame = frame.with_columns(
            pl.col("ret1").abs().alias("abs_ret1"),
            pl.when(pl.col("ret1") > 0)
            .then(pl.col("ret1"))
            .otherwise(0)
            .alias("pos_ret1"),
        )

        expressions = []
        for w in window:
            close_shift = pl.col("close").shift(w)
            close_rolling = pl.col("close").rolling_mean(w)
            close_std = pl.col("close").rolling_std(w)
            close_max = pl.col("close").rolling_max(w)
            close_min = pl.col("close").rolling_min(w)
            close_q80 = pl.col("close").rolling_quantile(0.8, window_size=w)
            close_q20 = pl.col("close").rolling_quantile(0.2, window_size=w)
            close_rank = pl.col("close").rolling_map(
                lambda values: float((np.asarray(values) <= values[-1]).sum()) / len(values),
                window_size=w,
            )
            high_rank = pl.col("high").rolling_map(
                lambda values: float(np.argmax(np.asarray(values))) / w,
                window_size=w,
            )
            low_rank = pl.col("low").rolling_map(
                lambda values: float(np.argmin(np.asarray(values))) / w,
                window_size=w,
            )

            expressions.extend(
                [
                    (close_shift / pl.col("close")).alias(f"roc_{w}"),
                    (close_rolling / pl.col("close")).alias(f"ma_{w}"),
                    (close_std / pl.col("close")).alias(f"std_{w}"),
                    ((close_shift - pl.col("close")) / (w * pl.col("close"))).alias(
                        f"beta_{w}"
                    ),
                    (close_max / pl.col("close")).alias(f"max_{w}"),
                    (close_min / pl.col("close")).alias(f"min_{w}"),
                    (close_q80 / pl.col("close")).alias(f"qtlu_{w}"),
                    (close_q20 / pl.col("close")).alias(f"qtld_{w}"),
                    close_rank.alias(f"rank_{w}"),
                    high_rank.alias(f"imax_{w}"),
                    low_rank.alias(f"imin_{w}"),
                    (high_rank - low_rank).alias(f"imxd_{w}"),
                    (
                        (pl.col("close") - pl.min_horizontal(pl.col("low"), close_shift))
                        / (
                            pl.max_horizontal(pl.col("high"), close_shift)
                            - pl.min_horizontal(pl.col("low"), close_shift)
                            + 1e-12
                        )
                    ).alias(f"rsv_{w}"),
                    ((pl.col("ret1") > 0).rolling_sum(w) / w).alias(f"cntp_{w}"),
                    ((pl.col("ret1") < 0).rolling_sum(w) / w).alias(f"cntn_{w}"),
                    (
                        ((pl.col("ret1") > 0).rolling_sum(w) / w)
                        - ((pl.col("ret1") < 0).rolling_sum(w) / w)
                    ).alias(f"cntd_{w}"),
                    (
                        pl.col("pos_ret1").rolling_sum(w)
                        / (pl.col("abs_ret1").rolling_sum(w) + 1e-12)
                    ).alias(f"sump_{w}"),
                    (
                        1
                        - (
                            pl.col("pos_ret1").rolling_sum(w)
                            / (pl.col("abs_ret1").rolling_sum(w) + 1e-12)
                        )
                    ).alias(f"sumn_{w}"),
                    (
                        2
                        * (
                            pl.col("pos_ret1").rolling_sum(w)
                            / (pl.col("abs_ret1").rolling_sum(w) + 1e-12)
                        )
                        - 1
                    ).alias(f"sumd_{w}"),
                ]
            )

        return (
            frame.with_columns(expressions)
            .select(["timestamp"] + [expr.meta.output_name() for expr in expressions])
            .slice(max_window + 1)
            .fill_nan(0)
            .fill_null(0)
        )

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


def process_enhanced_state_features(df: pl.DataFrame) -> pl.DataFrame:
    if not isinstance(df, pl.DataFrame):
        raise ValueError("df must be a polars DataFrame")

    frame = df.sort("timestamp")
    exprs = []

    # Ticket 03: Trade Direction & Spread Z-Score
    up = pl.col("ntrade_up_estimated") if "ntrade_up_estimated" in frame.columns else pl.lit(0.0)
    down = pl.col("ntrade_down_estimated") if "ntrade_down_estimated" in frame.columns else pl.lit(0.0)
    net_ratio = (up - down) / (up + down + 1e-8)
    exprs.append(net_ratio.clip(-1.0, 1.0).alias("trade_direction_net_ratio_5m"))
    exprs.append(net_ratio.ewm_mean(span=20).fill_null(0.0).alias("trade_direction_persistence_20m"))

    if "relative_bid_ask_spread" in frame.columns:
        spread = pl.col("relative_bid_ask_spread")
        spread_mean = spread.rolling_mean(48)
        spread_std = spread.rolling_std(48).fill_null(0.0)
        zscore = pl.when(spread_std > 1e-8).then((spread - spread_mean) / spread_std).otherwise(0.0)
        exprs.append(zscore.fill_null(0.0).fill_nan(0.0).alias("spread_widening_zscore_48"))
    elif "relative_spread" in frame.columns:
        spread = pl.col("relative_spread")
        spread_mean = spread.rolling_mean(48)
        spread_std = spread.rolling_std(48).fill_null(0.0)
        zscore = pl.when(spread_std > 1e-8).then((spread - spread_mean) / spread_std).otherwise(0.0)
        exprs.append(zscore.fill_null(0.0).fill_nan(0.0).alias("spread_widening_zscore_48"))

    # Ticket 04: Trend Acceleration & Volatility Regime
    if "close" in frame.columns:
        close_values = frame.get_column("close").cast(pl.Float64).to_numpy()
        if "contract" in frame.columns:
            contracts = frame.get_column("contract").to_numpy()
            anchors = None
            for contract in dict.fromkeys(contracts.tolist()):
                contract_rows = contracts == contract
                contract_anchors = _rolling_log_price_anchors(
                    close_values[contract_rows]
                )
                if anchors is None:
                    anchors = {
                        name: np.zeros(frame.height, dtype=float)
                        for name in contract_anchors
                    }
                for name, values in contract_anchors.items():
                    anchors[name][contract_rows] = values
            if anchors is None:
                anchors = _rolling_log_price_anchors(close_values)
        else:
            anchors = _rolling_log_price_anchors(close_values)
        exprs.extend(pl.Series(name, values) for name, values in anchors.items())

        close = pl.col("close")
        v10_raw = close.ewm_mean(span=10) - close.ewm_mean(span=20)
        v10 = v10_raw / close
        close_std10 = (close / close.shift(1) - 1).rolling_std(10).fill_null(0.0)
        acc10 = pl.when(close_std10 > 1e-8).then((v10 - v10.shift(1)) / close_std10).otherwise(0.0)
        exprs.append(v10.fill_null(0.0).alias("price_velocity_10m"))
        exprs.append(acc10.fill_null(0.0).fill_nan(0.0).alias("price_acceleration_10m_norm"))

        ema_base = close.ewm_mean(span=20)
        for window in (96, 192):
            ema_prev = ema_base.shift(window)
            ema_slope = (
                pl.when(ema_prev.abs() > 1e-8)
                .then((ema_base - ema_prev) / ema_prev.abs() / window)
                .otherwise(0.0)
            )
            exprs.append(
                ema_slope.fill_null(0.0).fill_nan(0.0).alias(f"ema_slope_{window}")
            )

    if {"high", "low", "close"}.issubset(frame.columns):
        high = pl.col("high")
        low = pl.col("low")
        close = pl.col("close")
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = (
            pl.when((up_move > down_move) & (up_move > 0.0))
            .then(up_move)
            .otherwise(0.0)
        )
        minus_dm = (
            pl.when((down_move > up_move) & (down_move > 0.0))
            .then(down_move)
            .otherwise(0.0)
        )
        true_range = pl.max_horizontal(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ]
        )
        wilder_alpha = 1.0 / 14.0
        atr = true_range.ewm_mean(
            alpha=wilder_alpha,
            adjust=False,
            min_samples=14,
        )
        plus_dm_smoothed = plus_dm.ewm_mean(
            alpha=wilder_alpha,
            adjust=False,
            min_samples=14,
        )
        minus_dm_smoothed = minus_dm.ewm_mean(
            alpha=wilder_alpha,
            adjust=False,
            min_samples=14,
        )
        plus_di = (
            pl.when(atr > 1e-8)
            .then(100.0 * plus_dm_smoothed / atr)
            .otherwise(0.0)
        )
        minus_di = (
            pl.when(atr > 1e-8)
            .then(100.0 * minus_dm_smoothed / atr)
            .otherwise(0.0)
        )
        dx = (
            pl.when((plus_di + minus_di) > 1e-8)
            .then(100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di))
            .otherwise(0.0)
        )
        exprs.append(plus_di.fill_null(0.0).fill_nan(0.0).alias("plus_di_14"))
        exprs.append(minus_di.fill_null(0.0).fill_nan(0.0).alias("minus_di_14"))
        exprs.append(
            dx.ewm_mean(alpha=wilder_alpha, adjust=False, min_samples=14)
            .fill_null(0.0)
            .fill_nan(0.0)
            .alias("adx_14")
        )

    vwap = None
    if "vwap" in frame.columns:
        vwap = pl.col("vwap")
    elif "tradeval" in frame.columns and "volume" in frame.columns:
        volume = pl.col("volume")
        vwap = pl.when(volume.abs() > 1e-8).then(pl.col("tradeval") / volume).otherwise(0.0)
    elif "amount" in frame.columns and "volume" in frame.columns:
        volume = pl.col("volume")
        vwap = pl.when(volume.abs() > 1e-8).then(pl.col("amount") / volume).otherwise(0.0)
    if vwap is not None:
        for window in (96, 192):
            vwap_prev = vwap.shift(window)
            vwap_slope = (
                pl.when(vwap_prev.abs() > 1e-8)
                .then((vwap - vwap_prev) / vwap_prev.abs() / window)
                .otherwise(0.0)
            )
            exprs.append(
                vwap_slope.fill_null(0.0).fill_nan(0.0).alias(f"vwap_slope_{window}")
            )

    if {"volume", "ntrade_up_estimated", "ntrade_down_estimated"}.issubset(frame.columns):
        up = pl.col("ntrade_up_estimated")
        down = pl.col("ntrade_down_estimated")
        direction_ratio = ((up - down) / (up + down + 1e-8)).clip(-1.0, 1.0)
        signed_volume = pl.col("volume") * direction_ratio
        for window in (96, 192):
            volume_sum = pl.col("volume").rolling_sum(window)
            cvd_slope = (
                pl.when(volume_sum.abs() > 1e-8)
                .then(signed_volume.rolling_sum(window) / volume_sum.abs())
                .otherwise(0.0)
            )
            exprs.append(
                cvd_slope.fill_null(0.0).fill_nan(0.0).alias(f"cvd_slope_{window}")
            )

    if "garman_klass_volatility" in frame.columns:
        gk_vol = pl.col("garman_klass_volatility")
        q192 = gk_vol.rolling_quantile(0.5, window_size=192).fill_null(0.0)
        exprs.append(q192.alias("garman_klass_vol_quantile_192"))
    elif "garman_klass_volatility_12" in frame.columns:
        gk_vol = pl.col("garman_klass_volatility_12")
        q192 = gk_vol.rolling_quantile(0.5, window_size=192).fill_null(0.0)
        exprs.append(q192.alias("garman_klass_vol_quantile_192"))

    if "parkinson_volatility" in frame.columns:
        pv_vol = pl.col("parkinson_volatility")
        pv_mean = pv_vol.rolling_mean(192)
        pv_std = pv_vol.rolling_std(192).fill_null(0.0)
        pv_z = pl.when(pv_std > 1e-8).then((pv_vol - pv_mean) / pv_std).otherwise(0.0)
        exprs.append(pv_z.fill_null(0.0).fill_nan(0.0).alias("parkinson_vol_zscore_192"))
    elif "parkinson_volatility_12" in frame.columns:
        pv_vol = pl.col("parkinson_volatility_12")
        pv_mean = pv_vol.rolling_mean(192)
        pv_std = pv_vol.rolling_std(192).fill_null(0.0)
        pv_z = pl.when(pv_std > 1e-8).then((pv_vol - pv_mean) / pv_std).otherwise(0.0)
        exprs.append(pv_z.fill_null(0.0).fill_nan(0.0).alias("parkinson_vol_zscore_192"))

    # Ticket 05: Volume / OI Regime & Cross-Month Dynamics
    if "close" in frame.columns and "open_interest" in frame.columns and "volume" in frame.columns:
        p_diff = frame["close"].diff(2).sign().fill_null(0.0)
        oi_diff = frame["open_interest"].diff(2).sign().fill_null(0.0)
        vol_mean = pl.col("volume").rolling_mean(10).fill_null(0.0)
        rel_vol = pl.when(vol_mean > 1e-8).then(pl.col("volume") / vol_mean).otherwise(0.0)
        interaction = (p_diff * oi_diff * rel_vol).fill_null(0.0).fill_nan(0.0)
        exprs.append(interaction.alias("price_oi_vol_interaction_10m"))

        oi_prev10 = pl.col("open_interest").shift(10)
        oi_chg = pl.when(oi_prev10 > 1e-8).then((pl.col("open_interest") - oi_prev10) / oi_prev10).otherwise(0.0)
        exprs.append(oi_chg.fill_null(0.0).fill_nan(0.0).alias("oi_change_rate_norm_10m"))

    if "cm_main_sub_log_price_ratio" in frame.columns:
        cm_v = pl.col("cm_main_sub_log_price_ratio").diff(10).fill_null(0.0)
        exprs.append(cm_v.alias("cm_main_sub_log_price_spread_velocity_10m"))

    if "cm_main_sub_open_interest_share_sub" in frame.columns:
        cm_shift = pl.col("cm_main_sub_open_interest_share_sub").diff(10).fill_null(0.0)
        exprs.append(cm_shift.alias("cm_open_interest_shift_speed_10m"))

    if not exprs:
        return frame
    
    return frame.with_columns(exprs)
