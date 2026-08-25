from __future__ import annotations

from functools import reduce
import math

import numpy as np
import polars as pl

min_value = 1e-12


def get_multi_feature_window_price(df, windows, feature_name_list):
    df = _ensure_polars_with_timestamp(df)
    windows = list(windows)
    pieces = []
    for feature_name in feature_name_list:
        if feature_name not in df.columns:
            continue
        for window_index, window in enumerate(windows):
            exprs = []
            if window_index == 0:
                current = pl.col(feature_name)
                previous = current.shift(1)
                exprs.append(
                    pl.when((current > 0) & (previous > 0))
                    .then((current / (previous + min_value)).log() * 1000)
                    .otherwise(0.0)
                    .alias(f"{feature_name}_log_return_{window}")
                )
            if window != 1:
                mean = pl.col(feature_name).rolling_mean(window)
                std = pl.col(feature_name).rolling_std(window)
                exprs.append(
                    ((pl.col(feature_name) - mean) / (std + min_value)).alias(
                        f"{feature_name}_trend_{window}"
                    )
                )
            if exprs:
                pieces.append(df.select("timestamp", *exprs).slice(window + 1))
    return _inner_join_on_timestamp(pieces)


def _ensure_polars_with_timestamp(df) -> pl.DataFrame:
    if isinstance(df, pl.DataFrame):
        if "timestamp" in df.columns:
            return df
        return df.with_row_index("timestamp")
    raise TypeError("time operator helpers require a Polars DataFrame")


def _inner_join_on_timestamp(frames: list[pl.DataFrame]) -> pl.DataFrame:
    frames = [frame for frame in frames if frame is not None and frame.height > 0]
    if not frames:
        return pl.DataFrame({"timestamp": []})
    seen = {"timestamp"}
    deduped_frames = []
    for frame in frames:
        columns = [name for name in frame.columns if name == "timestamp" or name not in seen]
        deduped_frames.append(frame.select(columns))
        seen.update(name for name in frame.columns if name != "timestamp")
    result = reduce(lambda left, right: left.join(right, on="timestamp", how="inner"), deduped_frames)
    return result


def _clean_numeric(df: pl.DataFrame) -> pl.DataFrame:
    exprs = []
    for name, dtype in zip(df.columns, df.dtypes):
        if name == "timestamp" or not dtype.is_numeric():
            continue
        col = pl.col(name)
        exprs.append(
            pl.when(col.is_nan() | col.is_infinite() | col.is_null())
            .then(0.0)
            .otherwise(col)
            .alias(name)
        )
    return df.with_columns(exprs).fill_null(0) if exprs else df.fill_null(0)


def _rolling_rank_pct(column: str, window: int, alias: str) -> pl.Expr:
    return pl.col(column).rolling_map(
        lambda values: _pandas_rank_pct_last(values) / window,
        window_size=window,
    ).alias(alias)


def _pandas_rank_pct_last(values) -> float:
    values = np.asarray(values)
    last = values[-1]
    less_count = float((values < last).sum())
    equal_count = float((values == last).sum())
    average_rank = less_count + (equal_count + 1.0) / 2.0
    return average_rank / len(values)


def _rolling_arg(column: str, window: int, alias: str, fn) -> pl.Expr:
    return (
        pl.col(column)
        .rolling_map(lambda values: float(fn(np.asarray(values))) / window, window_size=window)
        .alias(alias)
    )


def _rolling_corr_expr(x: pl.Expr, y: pl.Expr, window: int, alias: str) -> pl.Expr:
    # The pandas reference passes the second rolling object as `pairwise`, so it
    # computes self-correlation rather than correlation against `y`.
    return pl.rolling_corr(x, x, window_size=window, ddof=1).alias(alias)


def _window_bounds(length: int, window: int) -> tuple[np.ndarray, np.ndarray]:
    indexes = np.arange(length)
    start = np.maximum(indexes + 1 - window, 0)
    end = indexes + 1
    return start, end


def _pandas_rolling_mean_array(values: np.ndarray, window: int) -> np.ndarray:
    values = values.astype(float, copy=False)
    start, end = _window_bounds(len(values), window)
    output = np.empty(len(values), dtype=float)
    nobs = neg_ct = 0
    sum_x = compensation_add = compensation_remove = 0.0
    prev_value = math.nan
    num_consecutive_same_value = 0
    previous_start = previous_end = 0

    for index, (s, e) in enumerate(zip(start, end)):
        if index == 0 or s >= previous_end:
            nobs = neg_ct = 0
            sum_x = compensation_add = compensation_remove = 0.0
            prev_value = values[s] if s < e else math.nan
            num_consecutive_same_value = 0
            for value in values[s:e]:
                (
                    nobs,
                    neg_ct,
                    sum_x,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,
                ) = _pandas_add_mean(
                    value,
                    nobs,
                    neg_ct,
                    sum_x,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,
                )
        else:
            for value in values[previous_start:s]:
                nobs, neg_ct, sum_x, compensation_remove = _pandas_remove_mean(
                    value, nobs, neg_ct, sum_x, compensation_remove
                )
            for value in values[previous_end:e]:
                (
                    nobs,
                    neg_ct,
                    sum_x,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,
                ) = _pandas_add_mean(
                    value,
                    nobs,
                    neg_ct,
                    sum_x,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,
                )

        if nobs >= window and nobs > 0:
            result = sum_x / float(nobs)
            if num_consecutive_same_value >= nobs:
                result = prev_value
            elif neg_ct == 0 and result < 0:
                result = 0.0
            elif neg_ct == nobs and result > 0:
                result = 0.0
            output[index] = result
        else:
            output[index] = math.nan
        previous_start = s
        previous_end = e
    return output


def _pandas_add_mean(
    value: float,
    nobs: int,
    neg_ct: int,
    sum_x: float,
    compensation: float,
    num_consecutive_same_value: int,
    prev_value: float,
) -> tuple[int, int, float, float, int, float]:
    if math.isnan(value):
        return nobs, neg_ct, sum_x, compensation, num_consecutive_same_value, prev_value
    nobs += 1
    y = value - compensation
    t = sum_x + y
    compensation = t - sum_x - y
    sum_x = t
    if np.signbit(value):
        neg_ct += 1
    if value == prev_value:
        num_consecutive_same_value += 1
    else:
        num_consecutive_same_value = 1
    prev_value = value
    return nobs, neg_ct, sum_x, compensation, num_consecutive_same_value, prev_value


def _pandas_remove_mean(
    value: float,
    nobs: int,
    neg_ct: int,
    sum_x: float,
    compensation: float,
) -> tuple[int, int, float, float]:
    if math.isnan(value):
        return nobs, neg_ct, sum_x, compensation
    nobs -= 1
    y = -value - compensation
    t = sum_x + y
    compensation = t - sum_x - y
    sum_x = t
    if np.signbit(value):
        neg_ct -= 1
    return nobs, neg_ct, sum_x, compensation


def _pandas_rolling_var_array(values: np.ndarray, window: int, ddof: int = 1) -> np.ndarray:
    values = values.astype(float, copy=False)
    start, end = _window_bounds(len(values), window)
    output = np.empty(len(values), dtype=float)
    nobs = 0.0
    mean_x = ssqdm_x = compensation_add = compensation_remove = 0.0
    prev_value = math.nan
    num_consecutive_same_value = 0
    previous_start = previous_end = 0

    for index, (s, e) in enumerate(zip(start, end)):
        if index == 0 or s >= previous_end:
            nobs = 0.0
            mean_x = ssqdm_x = compensation_add = compensation_remove = 0.0
            prev_value = values[s] if s < e else math.nan
            num_consecutive_same_value = 0
            for value in values[s:e]:
                (
                    nobs,
                    mean_x,
                    ssqdm_x,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,
                ) = _pandas_add_var(
                    value,
                    nobs,
                    mean_x,
                    ssqdm_x,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,
                )
        else:
            for value in values[previous_start:s]:
                nobs, mean_x, ssqdm_x, compensation_remove = _pandas_remove_var(
                    value, nobs, mean_x, ssqdm_x, compensation_remove
                )
            for value in values[previous_end:e]:
                (
                    nobs,
                    mean_x,
                    ssqdm_x,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,
                ) = _pandas_add_var(
                    value,
                    nobs,
                    mean_x,
                    ssqdm_x,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,
                )

        if nobs >= window and nobs > ddof:
            if nobs == 1 or num_consecutive_same_value >= nobs:
                output[index] = 0.0
            else:
                output[index] = ssqdm_x / (nobs - float(ddof))
        else:
            output[index] = math.nan
        previous_start = s
        previous_end = e
    return output


def _pandas_add_var(
    value: float,
    nobs: float,
    mean_x: float,
    ssqdm_x: float,
    compensation: float,
    num_consecutive_same_value: int,
    prev_value: float,
) -> tuple[float, float, float, float, int, float]:
    if math.isnan(value):
        return nobs, mean_x, ssqdm_x, compensation, num_consecutive_same_value, prev_value
    nobs += 1.0
    if value == prev_value:
        num_consecutive_same_value += 1
    else:
        num_consecutive_same_value = 1
    prev_value = value
    prev_mean = mean_x - compensation
    y = value - compensation
    t = y - mean_x
    compensation = t + mean_x - y
    delta = t
    mean_x = mean_x + delta / nobs if nobs else 0.0
    ssqdm_x = ssqdm_x + (value - prev_mean) * (value - mean_x)
    return nobs, mean_x, ssqdm_x, compensation, num_consecutive_same_value, prev_value


def _pandas_remove_var(
    value: float,
    nobs: float,
    mean_x: float,
    ssqdm_x: float,
    compensation: float,
) -> tuple[float, float, float, float]:
    if math.isnan(value):
        return nobs, mean_x, ssqdm_x, compensation
    nobs -= 1.0
    if nobs:
        prev_mean = mean_x - compensation
        y = value - compensation
        t = y - mean_x
        compensation = t + mean_x - y
        delta = t
        mean_x = mean_x - delta / nobs
        ssqdm_x = ssqdm_x - (value - prev_mean) * (value - mean_x)
    else:
        mean_x = 0.0
        ssqdm_x = 0.0
    return nobs, mean_x, ssqdm_x, compensation


def _pandas_rolling_count_array(values: np.ndarray, window: int) -> np.ndarray:
    valid = (~np.isnan(values)).astype(float)
    result = np.empty(len(values), dtype=float)
    for index in range(len(values)):
        start = max(index + 1 - window, 0)
        result[index] = valid[start : index + 1].sum()
    return result


def _pandas_rolling_self_corr_array(values: np.ndarray, window: int) -> np.ndarray:
    values = values.astype(float, copy=False)
    mean_x_y = _pandas_rolling_mean_array(values * values, window)
    mean_x = _pandas_rolling_mean_array(values, window)
    var_x = _pandas_rolling_var_array(values, window)
    count_x_y = _pandas_rolling_count_array(values, window)
    with np.errstate(all="ignore"):
        count_adjustment = np.divide(
            count_x_y,
            count_x_y - 1.0,
            out=np.full(len(values), np.nan, dtype=float),
            where=count_x_y > 1.0,
        )
        numerator = (mean_x_y - mean_x * mean_x) * count_adjustment
        return numerator / var_x


def _shifted_ratio(values: np.ndarray) -> np.ndarray:
    values = values.astype(float, copy=False)
    result = np.full(len(values), np.nan, dtype=float)
    result[1:] = values[1:] / values[:-1]
    return result


def _process_ohlcv_single_window_polars(df: pl.DataFrame, window: int) -> pl.DataFrame:
    df = _ensure_polars_with_timestamp(df)
    close_values = df.get_column("close").to_numpy()
    volume_values = df.get_column("volume").to_numpy()
    previous_returns_values = _shifted_ratio(close_values)
    shift_values = np.abs(previous_returns_values - 1.0) * volume_values
    close_std_raw = np.sqrt(_pandas_rolling_var_array(close_values, window))
    volume_std_raw = np.sqrt(_pandas_rolling_var_array(volume_values, window))
    shift_std_raw = np.sqrt(_pandas_rolling_var_array(shift_values, window))
    close = pl.col("close")
    volume = pl.col("volume")
    ret1 = (close / close.shift(1) - 1).alias("__ret1")
    vchg1 = (volume - volume.shift(1)).alias("__vchg1")
    base = df.with_columns(
        ret1,
        vchg1,
        (volume + 1).log().alias("__log_volume"),
    ).with_columns(
        pl.col("__ret1").abs().alias("__abs_ret1"),
        pl.when(pl.col("__ret1") > 0)
        .then(pl.col("__ret1"))
        .otherwise(0)
        .alias("__pos_ret1"),
        pl.col("__vchg1").abs().alias("__abs_vchg1"),
        pl.when(pl.col("__vchg1") > 0)
        .then(pl.col("__vchg1"))
        .otherwise(0)
        .alias("__pos_vchg1"),
        pl.Series("__close_mean", _pandas_rolling_mean_array(close_values, window)),
        pl.Series("__close_std_raw", close_std_raw),
        pl.Series("__close_std", close_std_raw + min_value),
        pl.Series("__volume_mean", _pandas_rolling_mean_array(volume_values, window)),
        pl.Series("__volume_std", volume_std_raw + min_value),
        pl.Series("__shift_mean", _pandas_rolling_mean_array(shift_values, window)),
        pl.Series("__shift_std", shift_std_raw),
        pl.Series("__corr", _pandas_rolling_self_corr_array(close_values, window)),
        pl.Series("__cord", _pandas_rolling_self_corr_array(previous_returns_values, window)),
    )
    close_shift = close.shift(window)
    close_std = pl.col("__close_std")
    volume_std = pl.col("__volume_std")
    log_volume = pl.col("__log_volume")
    min_price = pl.min_horizontal(pl.col("low"), close_shift)
    max_price = pl.max_horizontal(pl.col("high"), close_shift)
    previous_returns = close / close.shift(1)
    previous_volume = (volume / volume.shift(1) + 1).log()
    shift = (previous_returns - 1).abs() * volume
    high_w = pl.col("high").rolling_max(window)
    low_w = pl.col("low").rolling_min(window)
    pivot_pp = (high_w + low_w + close) / 3.0
    pivot_r1 = 2.0 * pivot_pp - low_w
    pivot_s1 = 2.0 * pivot_pp - high_w
    pivot_r2 = pivot_pp + (high_w - low_w)
    pivot_s2 = pivot_pp - (high_w - low_w)
    bollinger_upper = pl.col("__close_mean") + 2.0 * pl.col("__close_std_raw")
    bollinger_lower = pl.col("__close_mean") - 2.0 * pl.col("__close_std_raw")
    high_arg = _rolling_arg("high", window, f"imax_{window}", np.argmax)
    low_arg = _rolling_arg("low", window, f"imin_{window}", np.argmin)

    out = base.select(
        "timestamp",
        log_volume.alias("log_volume"),
        (close_shift / (close + min_value)).alias(f"roc_{window}"),
        (pl.col("__close_mean") / (close + min_value)).alias(f"ma_{window}"),
        (pl.col("__close_std_raw") / (close + min_value)).alias(f"std_{window}"),
        (4.0 * pl.col("__close_std_raw") / (pl.col("__close_mean") + min_value)).alias(f"bollinger_bandwidth_{window}"),
        (bollinger_upper / (close + min_value)).alias(f"bollinger_upper_{window}"),
        (bollinger_lower / (close + min_value)).alias(f"bollinger_lower_{window}"),
        (pivot_pp / (close + min_value)).alias(f"pivot_pp_{window}"),
        (pivot_r1 / (close + min_value)).alias(f"pivot_r1_{window}"),
        (pivot_s1 / (close + min_value)).alias(f"pivot_s1_{window}"),
        (pivot_r2 / (close + min_value)).alias(f"pivot_r2_{window}"),
        (pivot_s2 / (close + min_value)).alias(f"pivot_s2_{window}"),
        ((close_shift - close) / (window * (close + min_value))).alias(f"beta_{window}"),
        (close.rolling_max(window) / (close + min_value)).alias(f"max_{window}"),
        (close.rolling_min(window) / (close + min_value)).alias(f"min_{window}"),
        (close.rolling_quantile(0.8, interpolation="linear", window_size=window) / (close + min_value)).alias(f"qtlu_{window}"),
        (close.rolling_quantile(0.2, interpolation="linear", window_size=window) / (close + min_value)).alias(f"qtld_{window}"),
        _rolling_rank_pct("close", window, f"rank_{window}"),
        high_arg,
        low_arg,
        (high_arg - low_arg).alias(f"imxd_{window}"),
        ((close - min_price) / (max_price - min_price + min_value)).alias(f"rsv_{window}"),
        (pl.col("__ret1").gt(0).cast(pl.Float64).rolling_sum(window) / window).alias(f"cntp_{window}"),
        (pl.col("__ret1").lt(0).cast(pl.Float64).rolling_sum(window) / window).alias(f"cntn_{window}"),
        (
            pl.col("__ret1").gt(0).cast(pl.Float64).rolling_sum(window) / window
            - pl.col("__ret1").lt(0).cast(pl.Float64).rolling_sum(window) / window
        ).alias(f"cntd_{window}"),
        pl.col("__corr").alias(f"corr_{window}"),
        pl.col("__cord").alias(f"cord_{window}"),
        (pl.col("__pos_ret1").rolling_sum(window) / (pl.col("__abs_ret1").rolling_sum(window) + min_value)).alias(f"sump_{window}"),
        (1 - pl.col("__pos_ret1").rolling_sum(window) / (pl.col("__abs_ret1").rolling_sum(window) + min_value)).alias(f"sumn_{window}"),
        (2 * pl.col("__pos_ret1").rolling_sum(window) / (pl.col("__abs_ret1").rolling_sum(window) + min_value) - 1).alias(f"sumd_{window}"),
        (pl.col("__volume_mean") / (volume + min_value)).alias(f"vma_{window}"),
        ((pl.col("__volume_std") - min_value) / (volume + min_value)).alias(f"vstd_{window}"),
        (pl.col("__shift_std") / (pl.col("__shift_mean") + min_value)).alias(f"wvma_{window}"),
        (pl.col("__pos_vchg1").rolling_sum(window) / (pl.col("__abs_vchg1").rolling_sum(window) + min_value)).alias(f"vsump_{window}"),
        (1 - pl.col("__pos_vchg1").rolling_sum(window) / (pl.col("__abs_vchg1").rolling_sum(window) + min_value)).alias(f"vsumn_{window}"),
        (2 * pl.col("__pos_vchg1").rolling_sum(window) / (pl.col("__abs_vchg1").rolling_sum(window) + min_value) - 1).alias(f"vsumd_{window}"),
        (close_shift / close_std).alias(f"roc_{window}_std_norm"),
        (pl.col("__close_mean") / close_std).alias(f"ma_{window}_std_norm"),
        ((close_shift - close) / (window * close_std)).alias(f"beta_{window}_std_norm"),
        (close.rolling_max(window) / close_std).alias(f"max_{window}_std_norm"),
        (close.rolling_min(window) / close_std).alias(f"min_{window}_std_norm"),
        (close.rolling_quantile(0.8, interpolation="linear", window_size=window) / close_std).alias(f"qtlu_{window}_std_norm"),
        (close.rolling_quantile(0.2, interpolation="linear", window_size=window) / close_std).alias(f"qtld_{window}_std_norm"),
        ((close - min_price) / (close_std + min_value)).alias(f"rsv_{window}_std_norm"),
        (pl.col("__volume_mean") / volume_std).alias(f"vma_{window}_std_norm"),
    ).slice(window + 10)
    return _clean_numeric(out)


def _process_ohlc_single_window_polars(df: pl.DataFrame, window: int) -> pl.DataFrame:
    df = _ensure_polars_with_timestamp(df)
    close_values = df.get_column("close").to_numpy()
    close_std_raw = np.sqrt(_pandas_rolling_var_array(close_values, window))
    close = pl.col("close")
    ret1 = (close / close.shift(1) - 1).alias("__ret1")
    base = df.with_columns(ret1).with_columns(
        pl.col("__ret1").abs().alias("__abs_ret1"),
        pl.when(pl.col("__ret1") > 0).then(pl.col("__ret1")).otherwise(0).alias("__pos_ret1"),
        pl.Series("__close_mean", _pandas_rolling_mean_array(close_values, window)),
        pl.Series("__close_std_raw", close_std_raw),
        pl.Series("__close_std", close_std_raw + min_value),
    )

    close_shift = close.shift(window)
    close_rolling = pl.col("__close_mean")
    close_std = pl.col("__close_std")
    close_max = close.rolling_max(window)
    close_min = close.rolling_min(window)
    close_q80 = close.rolling_quantile(0.8, interpolation="linear", window_size=window)
    close_q20 = close.rolling_quantile(0.2, interpolation="linear", window_size=window)
    close_rank = _rolling_rank_pct("close", window, f"rank_{window}")
    high_w = pl.col("high").rolling_max(window)
    low_w = pl.col("low").rolling_min(window)
    pivot_pp = (high_w + low_w + close) / 3.0
    pivot_r1 = 2.0 * pivot_pp - low_w
    pivot_s1 = 2.0 * pivot_pp - high_w
    pivot_r2 = pivot_pp + (high_w - low_w)
    pivot_s2 = pivot_pp - (high_w - low_w)
    bollinger_upper = pl.col("__close_mean") + 2.0 * pl.col("__close_std_raw")
    bollinger_lower = pl.col("__close_mean") - 2.0 * pl.col("__close_std_raw")
    high_rank = pl.col("high").rolling_map(
        lambda values: float(np.argmax(np.asarray(values))) / window,
        window_size=window,
    )
    low_rank = pl.col("low").rolling_map(
        lambda values: float(np.argmin(np.asarray(values))) / window,
        window_size=window,
    )

    out = base.select(
        "timestamp",
        (close_shift / close).alias(f"roc_{window}"),
        (close_shift / close_std).alias(f"roc_{window}_std_norm"),
        (close_rolling / close).alias(f"ma_{window}"),
        (close_rolling / close_std).alias(f"ma_{window}_std_norm"),
        (pl.col("__close_std_raw") / close).alias(f"std_{window}"),
        (4.0 * pl.col("__close_std_raw") / (pl.col("__close_mean") + min_value)).alias(f"bollinger_bandwidth_{window}"),
        (bollinger_upper / (close + min_value)).alias(f"bollinger_upper_{window}"),
        (bollinger_lower / (close + min_value)).alias(f"bollinger_lower_{window}"),
        (pivot_pp / (close + min_value)).alias(f"pivot_pp_{window}"),
        (pivot_r1 / (close + min_value)).alias(f"pivot_r1_{window}"),
        (pivot_s1 / (close + min_value)).alias(f"pivot_s1_{window}"),
        (pivot_r2 / (close + min_value)).alias(f"pivot_r2_{window}"),
        (pivot_s2 / (close + min_value)).alias(f"pivot_s2_{window}"),
        ((close_shift - close) / (window * close)).alias(f"beta_{window}"),
        ((close_shift - close) / (window * close_std)).alias(f"beta_{window}_std_norm"),
        (close_max / close).alias(f"max_{window}"),
        (close_max / close_std).alias(f"max_{window}_std_norm"),
        (close_min / close).alias(f"min_{window}"),
        (close_min / close_std).alias(f"min_{window}_std_norm"),
        (close_q80 / close).alias(f"qtlu_{window}"),
        (close_q80 / close_std).alias(f"qtlu_{window}_std_norm"),
        (close_q20 / close).alias(f"qtld_{window}"),
        (close_q20 / close_std).alias(f"qtld_{window}_std_norm"),
        close_rank,
        high_rank.alias(f"imax_{window}"),
        low_rank.alias(f"imin_{window}"),
        (high_rank - low_rank).alias(f"imxd_{window}"),
        ((close - pl.min_horizontal(pl.col("low"), close_shift)) / (pl.max_horizontal(pl.col("high"), close_shift) - pl.min_horizontal(pl.col("low"), close_shift) + min_value)).alias(f"rsv_{window}"),
        ((pl.col("__ret1") > 0).rolling_sum(window) / window).alias(f"cntp_{window}"),
        ((pl.col("__ret1") < 0).rolling_sum(window) / window).alias(f"cntn_{window}"),
        (
            ((pl.col("__ret1") > 0).rolling_sum(window) / window)
            - ((pl.col("__ret1") < 0).rolling_sum(window) / window)
        ).alias(f"cntd_{window}"),
        (
            pl.col("__pos_ret1").rolling_sum(window) / (pl.col("__abs_ret1").rolling_sum(window) + min_value)
        ).alias(f"sump_{window}"),
        (
            1
            - (
                pl.col("__pos_ret1").rolling_sum(window) / (pl.col("__abs_ret1").rolling_sum(window) + min_value)
            )
        ).alias(f"sumn_{window}"),
        (
            2
            * (
                pl.col("__pos_ret1").rolling_sum(window) / (pl.col("__abs_ret1").rolling_sum(window) + min_value)
            )
            - 1
        ).alias(f"sumd_{window}"),
    )
    return _clean_numeric(out.slice(window + 1))


def get_multi_window_ohlcv(df, windows):
    df = _ensure_polars_with_timestamp(df)
    pieces = [_process_ohlcv_single_window_polars(df, window) for window in windows]
    return _clean_numeric(_inner_join_on_timestamp(pieces))


def get_multi_window_ohlc(df, windows):
    df = _ensure_polars_with_timestamp(df)
    pieces = [_process_ohlc_single_window_polars(df, window) for window in windows]
    return _clean_numeric(_inner_join_on_timestamp(pieces))


def compute_bars_per_day(symbol: str, target_freq: str) -> float:
    from operator_futures.commodity.config import get_commodity_config
    freq_norm = target_freq.strip().lower()
    if freq_norm in ("1d", "d"):
        return 1.0
    config = get_commodity_config(symbol)
    trading_sessions = config.trading_sessions
    total_minutes = sum(
        (s.end.hour * 60 + s.end.minute + s.end.second / 60.0)
        - (s.start.hour * 60 + s.start.minute + s.start.second / 60.0)
        for s in trading_sessions
    )
    if freq_norm.endswith("min") or freq_norm.endswith("m"):
        val_str = freq_norm.rstrip("min").rstrip("m")
        minutes = float(val_str)
    elif freq_norm.endswith("s"):
        val_str = freq_norm.rstrip("s")
        minutes = float(val_str) / 60.0
    elif freq_norm.endswith("h"):
        val_str = freq_norm.rstrip("h")
        minutes = float(val_str) * 60.0
    else:
        minutes = 5.0
    return total_minutes / minutes


def get_risk_and_liquidity_state_features(
    df: pl.DataFrame,
    windows: list[int],
    symbol: str = "fu",
    target_freq: str = "5min",
) -> pl.DataFrame:
    df = _ensure_polars_with_timestamp(df)

    required_prices = ("open", "high", "low", "close")
    for price_col in required_prices:
        if price_col not in df.columns:
            raise ValueError(f"Missing required price column: {price_col}")
        col = df.get_column(price_col)
        if col.null_count() > 0:
            invalid_row = df.filter(pl.col(price_col).is_null()).row(0, named=True)
            raise ValueError(
                f"Invalid price in column {price_col!r}: null value at timestamp={invalid_row.get('timestamp')}"
            )
        try:
            col_float = col.cast(pl.Float64, strict=False)
        except Exception as err:
            raise ValueError(f"Price column {price_col!r} is non-numeric: {err}")
        invalid_mask = (
            col_float.is_null()
            | col_float.is_nan()
            | col_float.is_infinite()
            | (col_float <= 0.0)
        )
        if invalid_mask.any():
            invalid_row = df.filter(invalid_mask).row(0, named=True)
            val = invalid_row.get(price_col)
            ts = invalid_row.get("timestamp")
            raise ValueError(
                f"Invalid price in column {price_col!r}: value={val!r} at timestamp={ts}"
            )

    required_liquidity = ("volume", "tradeval", "open_interest")
    for liq_col in required_liquidity:
        if liq_col not in df.columns:
            raise ValueError(f"Missing required column for liquidity state features: {liq_col}")
        col = df.get_column(liq_col)
        if col.null_count() > 0:
            invalid_row = df.filter(pl.col(liq_col).is_null()).row(0, named=True)
            raise ValueError(f"Column {liq_col!r} contains null at timestamp {invalid_row.get('timestamp')}")

    bars_per_day = compute_bars_per_day(symbol, target_freq)
    sqrt_bpd = math.sqrt(bars_per_day)

    close = pl.col("close").cast(pl.Float64, strict=False)
    open_p = pl.col("open").cast(pl.Float64, strict=False)
    high = pl.col("high").cast(pl.Float64, strict=False)
    low = pl.col("low").cast(pl.Float64, strict=False)
    volume = pl.col("volume").cast(pl.Float64, strict=False)
    tradeval = pl.col("tradeval").cast(pl.Float64, strict=False)
    open_interest = pl.col("open_interest").cast(pl.Float64, strict=False)

    close_prev = close.shift(1)
    r = (close / close_prev).log()
    r_sq = r ** 2

    tr_1 = high - low
    tr_2 = (high - close_prev).abs()
    tr_3 = (low - close_prev).abs()
    tr = pl.max_horizontal(tr_1, tr_2, tr_3)

    log_hl_sq = (high / low).log() ** 2
    log_co_sq = (close / open_p).log() ** 2
    gk_term = 0.5 * log_hl_sq - (2.0 * math.log(2.0) - 1.0) * log_co_sq

    all_exprs: list[pl.Expr] = []
    max_w = max(windows)

    for w in windows:
        atr_mean = tr.rolling_mean(w)
        atr_pct = (atr_mean / close) * 100.0
        all_exprs.append(atr_pct.alias(f"atr_pct_{w}"))

        r_std = r.rolling_std(w, ddof=1)
        hv = r_std * sqrt_bpd
        all_exprs.append(hv.alias(f"historical_volatility_{w}"))

        rv = r.ewm_std(span=w)
        all_exprs.append(rv.alias(f"rolling_volatility_{w}"))

        pv_mean = log_hl_sq.rolling_mean(w)
        pv_clipped = pl.when(pv_mean < 0.0).then(0.0).otherwise(pv_mean)
        pv = (pv_clipped / (4.0 * math.log(2.0))).sqrt()
        all_exprs.append(pv.alias(f"parkinson_volatility_{w}"))

        gk_mean = gk_term.rolling_mean(w)
        gk_clipped = pl.when(gk_mean < 0.0).then(0.0).otherwise(gk_mean)
        gkv = gk_clipped.sqrt()
        all_exprs.append(gkv.alias(f"garman_klass_volatility_{w}"))

        realized_v = r_sq.rolling_sum(w).sqrt()
        all_exprs.append(realized_v.alias(f"realized_volatility_{w}"))

        vol_mean = volume.rolling_mean(w)
        rel_vol = pl.when(vol_mean > 0.0).then(volume / vol_mean).otherwise(0.0)
        all_exprs.append(rel_vol.alias(f"relative_volume_{w}"))

        amt_mean = tradeval.rolling_mean(w)
        rel_amt = pl.when(amt_mean > 0.0).then(tradeval / amt_mean).otherwise(0.0)
        all_exprs.append(rel_amt.alias(f"relative_amount_{w}"))

        oi_mean = open_interest.rolling_mean(w)
        rel_oi = pl.when(oi_mean > 0.0).then(open_interest / oi_mean).otherwise(0.0)
        all_exprs.append(rel_oi.alias(f"relative_open_interest_{w}"))

        oi_prev = open_interest.shift(w)
        oi_change = (
            pl.when(oi_prev > 0.0)
            .then((open_interest - oi_prev) / oi_prev)
            .otherwise(0.0)
        )
        all_exprs.append(oi_change.alias(f"open_interest_change_ratio_{w}"))

    res = df.select("timestamp", *all_exprs).slice(max_w + 1)
    return _clean_numeric(res)
