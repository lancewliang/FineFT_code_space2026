from __future__ import annotations

from functools import reduce

import numpy as np
import polars as pl

min_value = 1e-12


def get_multi_feature_window_price(df, windows, feature_name_list):
    df = _ensure_polars_with_timestamp(df)
    pieces = []
    for feature_name in feature_name_list:
        if feature_name not in df.columns:
            continue
        for window in windows:
            exprs = [
                (
                    (pl.col(feature_name) / (pl.col(feature_name).shift(1) + min_value))
                    .log()
                    * 1000
                ).alias(f"{feature_name}_log_return_{window}")
            ]
            if window != 1:
                mean = pl.col(feature_name).rolling_mean(window)
                std = pl.col(feature_name).rolling_std(window)
                exprs.append(
                    ((pl.col(feature_name) - mean) / (std + min_value)).alias(
                        f"{feature_name}_trend_{window}"
                    )
                )
            pieces.append(df.select("timestamp", *exprs).slice(window + 1))
    return _clean_numeric(_inner_join_on_timestamp(pieces))


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
    return reduce(lambda left, right: left.join(right, on="timestamp", how="inner"), frames)


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
        lambda values: float((np.asarray(values) <= values[-1]).sum()) / len(values),
        window_size=window,
    ).alias(alias)


def _rolling_arg(column: str, window: int, alias: str, fn) -> pl.Expr:
    return (
        pl.col(column)
        .rolling_map(lambda values: float(fn(np.asarray(values))) / window, window_size=window)
        .alias(alias)
    )


def _rolling_corr_expr(x: pl.Expr, y: pl.Expr, window: int, alias: str) -> pl.Expr:
    mean_x = x.rolling_mean(window)
    mean_y = y.rolling_mean(window)
    cov = (x * y).rolling_mean(window) - mean_x * mean_y
    return (cov / (x.rolling_std(window) * y.rolling_std(window) + min_value)).alias(alias)


def _process_ohlcv_single_window_polars(df: pl.DataFrame, window: int) -> pl.DataFrame:
    df = _ensure_polars_with_timestamp(df)
    close = pl.col("close")
    volume = pl.col("volume")
    ret1 = (close / (close.shift(1) + min_value) - 1).alias("__ret1")
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
    )
    close_shift = close.shift(window)
    close_std = close.rolling_std(window) + min_value
    volume_std = volume.rolling_std(window) + min_value
    log_volume = pl.col("__log_volume")
    min_price = pl.min_horizontal(pl.col("low"), close_shift)
    max_price = pl.max_horizontal(pl.col("high"), close_shift)
    previous_returns = close / (close.shift(1) + min_value)
    previous_volume = (volume / (volume.shift(1) + min_value) + 1).log()
    shift = (previous_returns - 1).abs() * volume
    high_arg = _rolling_arg("high", window, f"imax_{window}", np.argmax)
    low_arg = _rolling_arg("low", window, f"imin_{window}", np.argmin)

    out = base.select(
        "timestamp",
        log_volume.alias("log_volume"),
        (close_shift / (close + min_value)).alias(f"roc_{window}"),
        (close.rolling_mean(window) / (close + min_value)).alias(f"ma_{window}"),
        (close.rolling_std(window) / (close + min_value)).alias(f"std_{window}"),
        ((close_shift - close) / (window * (close + min_value))).alias(f"beta_{window}"),
        (close.rolling_max(window) / (close + min_value)).alias(f"max_{window}"),
        (close.rolling_min(window) / (close + min_value)).alias(f"min_{window}"),
        (close.rolling_quantile(0.8, window_size=window) / (close + min_value)).alias(f"qtlu_{window}"),
        (close.rolling_quantile(0.2, window_size=window) / (close + min_value)).alias(f"qtld_{window}"),
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
        _rolling_corr_expr(close, log_volume, window, f"corr_{window}"),
        _rolling_corr_expr(previous_returns, previous_volume, window, f"cord_{window}"),
        (pl.col("__pos_ret1").rolling_sum(window) / (pl.col("__abs_ret1").rolling_sum(window) + min_value)).alias(f"sump_{window}"),
        (1 - pl.col("__pos_ret1").rolling_sum(window) / (pl.col("__abs_ret1").rolling_sum(window) + min_value)).alias(f"sumn_{window}"),
        (2 * pl.col("__pos_ret1").rolling_sum(window) / (pl.col("__abs_ret1").rolling_sum(window) + min_value) - 1).alias(f"sumd_{window}"),
        (volume.rolling_mean(window) / (volume + min_value)).alias(f"vma_{window}"),
        (volume.rolling_std(window) / (volume + min_value)).alias(f"vstd_{window}"),
        (shift.rolling_std(window) / (shift.rolling_mean(window) + min_value)).alias(f"wvma_{window}"),
        (pl.col("__pos_vchg1").rolling_sum(window) / (pl.col("__abs_vchg1").rolling_sum(window) + min_value)).alias(f"vsump_{window}"),
        (1 - pl.col("__pos_vchg1").rolling_sum(window) / (pl.col("__abs_vchg1").rolling_sum(window) + min_value)).alias(f"vsumn_{window}"),
        (2 * pl.col("__pos_vchg1").rolling_sum(window) / (pl.col("__abs_vchg1").rolling_sum(window) + min_value) - 1).alias(f"vsumd_{window}"),
        (close_shift / close_std).alias(f"roc_{window}_std_norm"),
        (close.rolling_mean(window) / close_std).alias(f"ma_{window}_std_norm"),
        ((close_shift - close) / (window * close_std)).alias(f"beta_{window}_std_norm"),
        (close.rolling_max(window) / close_std).alias(f"max_{window}_std_norm"),
        (close.rolling_min(window) / close_std).alias(f"min_{window}_std_norm"),
        (close.rolling_quantile(0.8, window_size=window) / close_std).alias(f"qtlu_{window}_std_norm"),
        (close.rolling_quantile(0.2, window_size=window) / close_std).alias(f"qtld_{window}_std_norm"),
        ((close - min_price) / (close_std + min_value)).alias(f"rsv_{window}_std_norm"),
        (volume.rolling_mean(window) / volume_std).alias(f"vma_{window}_std_norm"),
    ).slice(window + 10)
    return _clean_numeric(out)


def _process_ohlc_single_window_polars(df: pl.DataFrame, window: int) -> pl.DataFrame:
    ohlcv = _process_ohlcv_single_window_polars(df.with_columns(pl.lit(0.0).alias("volume")), window)
    keep = [
        name
        for name in ohlcv.columns
        if name == "timestamp"
        or not name.startswith(("log_volume", "corr_", "cord_", "vma_", "vstd_", "wvma_", "vsump_", "vsumn_", "vsumd_"))
        and not name.endswith("vma_{}_std_norm".format(window))
    ]
    return ohlcv.select(keep).slice(max(0, (window + 1) - (window + 10)))


def get_multi_window_ohlcv(df, windows):
    df = _ensure_polars_with_timestamp(df)
    pieces = [_process_ohlcv_single_window_polars(df, window) for window in windows]
    return _clean_numeric(_inner_join_on_timestamp(pieces))


def get_multi_window_ohlc(df, windows):
    df = _ensure_polars_with_timestamp(df)
    pieces = [_process_ohlc_single_window_polars(df, window) for window in windows]
    return _clean_numeric(_inner_join_on_timestamp(pieces))
