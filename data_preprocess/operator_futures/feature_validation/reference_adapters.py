from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from operator_futures.feature_validation.io import (
    date_range_exclusive,
    read_feather_frame,
    read_state_features,
)
from operator_futures.feature_validation.models import ValidationConfig
from operator_futures.feature_validation.pandas_reference.cross_section.base_feature_util import (
    process_k_line_feature,
    process_quotes_n_feature,
    process_snapshot_features,
)
from operator_futures.feature_validation.pandas_reference.feature_selection.cor_util import (
    select_feature,
)
from operator_futures.feature_validation.pandas_reference.time_operator.multi_processing_util import (
    get_multi_feature_window_price,
    get_multi_window_ohlc,
    get_multi_window_ohlcv,
)
from operator_futures.feature_validation.pandas_reference.util import (
    find_ohlc_groups,
    find_ohlcv_groups,
)
from operator_futures.feature_validation.pandas_reference.scale_describe_save.scale_save import (
    scale_mean,
    scale_std,
)
from operator_futures.commodity.schema import get_reward_execution_columns


DEFAULT_TIME_WINDOWS = [2, 6, 12, 16, 24, 48]
DEFAULT_IC_WINDOWS = [1, 6, 12]
DEFAULT_IC_THRESHOLD = 0.01
DEFAULT_COR_THRESHOLD = 0.7


def normalize_reference_frame(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if result.index.name == "timestamp":
        result = result.reset_index()
    if "datetime" in result.columns and "timestamp" not in result.columns:
        result = result.rename(columns={"datetime": "timestamp"})
    if "timestamp" in result.columns:
        result["timestamp"] = result["timestamp"].astype(str)
    return result


def _commodity_frame_path(config: ValidationConfig, *parts: str) -> Path:
    return config.root_path / "PREPROCESS_DATASET" / "commodity-futures" / Path(*parts)


def load_cross_section_input(config: ValidationConfig, date: str) -> pd.DataFrame:
    return read_feather_frame(
        _commodity_frame_path(
            config,
            "BASE_FEATURE",
            config.symbol,
            config.target_freq,
            f"{date}.feather",
        )
    )


def load_snapshot_input(config: ValidationConfig, date: str) -> pd.DataFrame:
    return read_feather_frame(
        _commodity_frame_path(
            config,
            "DOWNSCALE_ORDERBOOK_25",
            config.symbol,
            config.target_freq,
            f"{date}.feather",
        )
    )


def recompute_cross_section(config: ValidationConfig, date: str) -> dict[str, pd.DataFrame]:
    base_feature = load_cross_section_input(config, date)
    snapshot = load_snapshot_input(config, date)
    base_feature = base_feature.copy()
    snapshot = snapshot.copy()
    base_feature.set_index("timestamp", inplace=True)
    snapshot.set_index("timestamp", inplace=True)

    kline_feature = normalize_reference_frame(process_k_line_feature(base_feature))
    quotes_feature = normalize_reference_frame(process_quotes_n_feature(base_feature))
    snapshot_feature = normalize_reference_frame(process_snapshot_features(snapshot, topk=config.orderbook_depth))
    return {
        "kline": kline_feature,
        "quotes": quotes_feature,
        "snapshot": snapshot_feature,
    }


def _merge_concat_path(config: ValidationConfig) -> Path:
    return (
        config.root_path
        / "PREPROCESS_DATASET"
        / "commodity-futures"
        / "MERGE_CONCAT"
        / "CONCAT_FEATURE"
        / config.symbol
        / config.target_freq
    )


def recompute_merge_concat(config: ValidationConfig) -> pd.DataFrame:
    concurrent_frames: list[pd.DataFrame] = []
    future_frames: list[pd.DataFrame] = []
    for date in date_range_exclusive(config.start_date, config.end_date):
        root = (
            config.root_path
            / "PREPROCESS_DATASET"
            / "commodity-futures"
            / "MERGE_CONCAT"
            / "MERGED_FEATURE"
            / config.symbol
            / config.target_freq
        )
        concurrent_path = root / "CONCURRENT_FEATURE" / f"{date}.feather"
        future_path = root / "FUTURE_FEATURE" / f"{date}.feather"
        concurrent_frames.append(read_feather_frame(concurrent_path))
        future_frames.append(read_feather_frame(future_path))

    cocurrent_df = pd.concat(concurrent_frames, axis=0)
    future_df = pd.concat(future_frames, axis=0)
    cocurrent_df.set_index("timestamp", inplace=True)
    future_df.set_index("timestamp", inplace=True)
    cocurrent_df.sort_index(inplace=True)
    cocurrent_df = cocurrent_df.groupby(cocurrent_df.index).first()
    future_df.sort_index(inplace=True)
    future_df = future_df.groupby(future_df.index).first()
    future_df.drop(
        columns=[column for column in ["symbol", "exchange"] if column in future_df.columns],
        inplace=True,
    )
    future_df = future_df.shift(+1)
    future_df = future_df.iloc[1:]
    df = pd.concat([cocurrent_df, future_df], axis=1, join="inner")
    df.reset_index(inplace=True)
    df.fillna(method="ffill", inplace=True)
    return normalize_reference_frame(df)


def recompute_time_feature(config: ValidationConfig, start_date: str, end_date: str) -> pd.DataFrame:
    input_frame = read_feather_frame(
        _commodity_frame_path(
            config,
            "MERGE_CONCAT",
            "CONCAT_FEATURE",
            config.symbol,
            config.target_freq,
            f"{start_date}-{end_date}.feather",
        )
    )
    input_frame = input_frame.copy()
    input_frame.set_index("timestamp", inplace=True)
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
        "buy_volume_oe",
        "sell_volume_oe",
        "imblance_volume_oe",
        *[f"ask{l+1}_size_n" for l in range(25)],
        *[f"bid{l+1}_size_n" for l in range(25)],
    ]
    price_features = [name for name in price_features if name in input_frame.columns]
    time_feature_parts = [
        get_multi_feature_window_price(input_frame, DEFAULT_TIME_WINDOWS, price_features)
    ]

    ohlcv_features, _ = find_ohlcv_groups(input_frame)
    ohlc_features, _ = find_ohlc_groups(input_frame)
    for key in ohlcv_features:
        ohlc_features.pop(key, None)

    for prefix, suffix in ohlcv_features:
        feature_names = ohlcv_features[(prefix, suffix)]
        output_suffix = "_origin" if prefix + suffix == "" else prefix + suffix
        df_ohlcv = input_frame[feature_names].copy()
        df_ohlcv.rename(
            columns={
                prefix + key + suffix: key
                for key in ["open", "high", "low", "close", "volume"]
            },
            inplace=True,
        )
        part = get_multi_window_ohlcv(df_ohlcv, DEFAULT_TIME_WINDOWS)
        part.rename(columns={column: column + output_suffix for column in part.columns}, inplace=True)
        time_feature_parts.append(part)

    for prefix, suffix in ohlc_features:
        feature_names = ohlc_features[(prefix, suffix)]
        output_suffix = "_origin" if prefix + suffix == "" else prefix + suffix
        df_ohlc = input_frame[feature_names].copy()
        df_ohlc.rename(
            columns={
                prefix + key + suffix: key
                for key in ["open", "high", "low", "close"]
            },
            inplace=True,
        )
        part = get_multi_window_ohlc(df_ohlc, DEFAULT_TIME_WINDOWS)
        part.rename(columns={column: column + output_suffix for column in part.columns}, inplace=True)
        time_feature_parts.append(part)

    result = pd.concat(time_feature_parts, axis=1, join="inner")
    return normalize_reference_frame(result.reset_index())


def recompute_merge_clean(config: ValidationConfig, start_date: str, end_date: str) -> pd.DataFrame:
    root = config.root_path / "PREPROCESS_DATASET" / "commodity-futures"
    time_feature = read_feather_frame(root / "TIME_FEATURE" / config.symbol / config.target_freq / f"{start_date}-{end_date}.feather")
    cross_section = read_feather_frame(root / "MERGE_CONCAT" / "CONCAT_FEATURE" / config.symbol / config.target_freq / f"{start_date}-{end_date}.feather")
    time_feature = time_feature.copy()
    cross_section = cross_section.copy()
    time_feature.set_index("timestamp", inplace=True)
    cross_section.set_index("timestamp", inplace=True)
    result = pd.concat([cross_section, time_feature], axis=1, join="inner")
    return normalize_reference_frame(result.reset_index())


def recompute_ic_correlation(config: ValidationConfig, start_date: str, end_date: str) -> pd.DataFrame:
    all_feature = read_feather_frame(
        _commodity_frame_path(
            config,
            "ALL_FEATURE",
            config.symbol,
            config.target_freq,
            f"{start_date}-{end_date}.feather",
        )
    )
    all_feature = all_feature.copy()
    all_feature.set_index("timestamp", inplace=True)
    reward_features = _commodity_reward_columns(all_feature, config.orderbook_depth)
    ic_selection_key = _compute_ic_selection(all_feature, reward_features)
    df_cor = all_feature[ic_selection_key].corr()
    selected_feature_names = select_feature(corre_df=df_cor, theshold=DEFAULT_COR_THRESHOLD)
    result = all_feature[list(reward_features) + list(selected_feature_names)].reset_index()
    return normalize_reference_frame(result)


def recompute_ic_state_features(config: ValidationConfig, start_date: str, end_date: str) -> list[str]:
    all_feature = read_feather_frame(
        _commodity_frame_path(
            config,
            "ALL_FEATURE",
            config.symbol,
            config.target_freq,
            f"{start_date}-{end_date}.feather",
        )
    )
    all_feature = all_feature.copy()
    all_feature.set_index("timestamp", inplace=True)
    reward_features = _commodity_reward_columns(all_feature, config.orderbook_depth)
    state_feature = [col for col in all_feature.columns if col not in reward_features]
    ic_selection_key = _compute_ic_selection(all_feature, reward_features)
    df_cor = all_feature[ic_selection_key].corr()
    return select_feature(corre_df=df_cor, theshold=DEFAULT_COR_THRESHOLD)


def recompute_scale_save(config: ValidationConfig, start_date: str, end_date: str) -> pd.DataFrame:
    root = config.root_path / "PREPROCESS_DATASET" / "commodity-futures" / "IC_RESULT" / config.symbol / config.target_freq / f"{start_date}-{end_date}"
    df = read_feather_frame(root / "df.feather")
    state_feature = read_state_features(root / "state_features.npy")
    reward_features = _commodity_reward_columns(df, config.orderbook_depth)
    state_feature = [column for column in state_feature if column in df.columns]
    df_reward = df[reward_features]
    df_state = df[state_feature]
    df_state = scale_std(df_state, 10)
    df_state = scale_mean(df_state, 10, 10)
    result = pd.concat([df_reward, df_state], axis=1)
    result["symbol"] = config.symbol
    return normalize_reference_frame(result)


def load_reference_report_data(config: ValidationConfig, start_date: str, end_date: str) -> dict[str, str]:
    root = config.root_path / "PREPROCESS_DATASET" / "commodity-futures" / "IC_RESULT" / config.symbol / config.target_freq / f"{start_date}-{end_date}"
    return {
        "df": str(root / "df.feather"),
        "state_features": str(root / "state_features.npy"),
        "describe": str(root / "df_describe.csv"),
    }


def _commodity_reward_columns(df: pd.DataFrame, orderbook_depth: int) -> list[str]:
    return [
        column
        for column in get_reward_execution_columns(orderbook_depth)
        if column in df.columns
    ]


def _compute_ic_selection(all_feature: pd.DataFrame, reward_features: list[str]) -> list[str]:
    state_feature = [col for col in all_feature.columns if col not in reward_features]
    ic_selection_key_all: list[str] = []
    for window_length in DEFAULT_IC_WINDOWS:
        target = all_feature["mark_price"].shift(-window_length) - all_feature["mark_price"]
        target = target[:-window_length]
        df_ic = all_feature.iloc[:-window_length]
        ic_result = [df_ic[feature].corr(target) for feature in state_feature]
        sorted_pairs = sorted(
            zip(state_feature, ic_result), key=lambda x: abs(x[1]), reverse=True
        )
        cor = {feature: result for feature, result in sorted_pairs}
        ic_selection_key_all.extend(
            [key for key, value in cor.items() if abs(value) > DEFAULT_IC_THRESHOLD]
        )
    ic_selection_key = []
    seen = set()
    for key in ic_selection_key_all:
        if key not in seen:
            seen.add(key)
            ic_selection_key.append(key)
    return ic_selection_key
