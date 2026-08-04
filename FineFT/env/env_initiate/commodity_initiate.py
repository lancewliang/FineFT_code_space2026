import sys

import pandas as pd

sys.path.append(".")
from env.env_class.commodity_env import Commodity_Env


def _depth_columns(depth):
    bid_prices_names = [f"bid{i}_price" for i in range(1, depth + 1)]
    ask_prices_names = [f"ask{i}_price" for i in range(1, depth + 1)]
    bid_sizes_names = [f"bid{i}_size" for i in range(1, depth + 1)]
    ask_sizes_names = [f"ask{i}_size" for i in range(1, depth + 1)]
    return bid_prices_names, ask_prices_names, bid_sizes_names, ask_sizes_names


def _validate_columns(df, required_columns):
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"commodity env input missing columns: {missing}")


def initiate_commodity_env(
    df: pd.DataFrame,
    feature_list: list,
    max_holding_number=8,
    position_choices=9,
    leverage_choice=[5],
    long_estimated_rate=0.0005,
    short_estimated_rate=0,
    commission_rate=0.0002,
    maintenance_margin_ratio_dict={
        "50000": [0.004, 0],
        "500000": [0.005, 50],
        "10000000": [0.01, 2550],
    },
    early_stop=0,
    initial_state=(1e5, 0, 0, 0, 5),
    buy_fee_rate=0.0001,
    sell_fee_rate=0.0003,
    depth=5,
    enable_limit_reward=False,
    limit_hold_bonus=1.0,
    limit_stay_bonus=0.5,
    limit_reverse_penalty=1.5,
    near_limit_threshold=0.003,
):
    bid_prices_names, ask_prices_names, bid_sizes_names, ask_sizes_names = (
        _depth_columns(depth)
    )
    _validate_columns(
        df,
        [
            "mark_price",
            "timestamp",
            *feature_list,
            *bid_prices_names,
            *ask_prices_names,
            *bid_sizes_names,
            *ask_sizes_names,
        ],
    )

    markprice_array = df["mark_price"].values
    timestamp_array = df["timestamp"].values
    ask_prices_array = df[ask_prices_names].values
    bid_prices_array = df[bid_prices_names].values
    ask_qtys_array = df[ask_sizes_names].values
    bid_qtys_array = df[bid_sizes_names].values
    state_array = df[feature_list].values

    if enable_limit_reward:
        missing_limit_cols = [c for c in [
            "limit_up_single_sided_ratio",
            "limit_down_single_sided_ratio",
            "limit_up_ask_depth_ratio_5",
            "limit_down_bid_depth_ratio_5",
            "UpperLimitPrice",
            "LowerLimitPrice",
        ] if c not in df.columns]
        if missing_limit_cols:
            raise ValueError(f"enable_limit_reward=True 但 DataFrame 缺少必须的涨跌停列: {missing_limit_cols}")

    is_limit_up_array = (df["limit_up_single_sided_ratio"].values > 0) if "limit_up_single_sided_ratio" in df.columns else None
    is_limit_down_array = (df["limit_down_single_sided_ratio"].values > 0) if "limit_down_single_sided_ratio" in df.columns else None
    limit_up_ask_depth_ratio_5_array = df["limit_up_ask_depth_ratio_5"].values if "limit_up_ask_depth_ratio_5" in df.columns else None
    limit_down_bid_depth_ratio_5_array = df["limit_down_bid_depth_ratio_5"].values if "limit_down_bid_depth_ratio_5" in df.columns else None
    upper_limit_prices_array = df["UpperLimitPrice"].values if "UpperLimitPrice" in df.columns else None
    lower_limit_prices_array = df["LowerLimitPrice"].values if "LowerLimitPrice" in df.columns else None

    return Commodity_Env(
        state_array,
        ask_prices_array,
        bid_prices_array,
        ask_qtys_array,
        bid_qtys_array,
        markprice_array,
        timestamp_array,
        max_holding_number=max_holding_number,
        position_choices=position_choices,
        leverage_choice=leverage_choice,
        long_estimated_rate=long_estimated_rate,
        short_estimated_rate=short_estimated_rate,
        commission_rate=commission_rate,
        maintenance_margin_ratio_dict=maintenance_margin_ratio_dict,
        early_stop=early_stop,
        initial_state=initial_state,
        buy_fee_rate=buy_fee_rate,
        sell_fee_rate=sell_fee_rate,
        is_limit_up_array=is_limit_up_array,
        is_limit_down_array=is_limit_down_array,
        limit_up_ask_depth_ratio_5_array=limit_up_ask_depth_ratio_5_array,
        limit_down_bid_depth_ratio_5_array=limit_down_bid_depth_ratio_5_array,
        upper_limit_prices_array=upper_limit_prices_array,
        lower_limit_prices_array=lower_limit_prices_array,
        enable_limit_reward=enable_limit_reward,
        limit_hold_bonus=limit_hold_bonus,
        limit_stay_bonus=limit_stay_bonus,
        limit_reverse_penalty=limit_reverse_penalty,
        near_limit_threshold=near_limit_threshold,
    )
