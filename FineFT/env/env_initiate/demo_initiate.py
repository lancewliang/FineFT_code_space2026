import pandas as pd
import numpy as np
import sys

sys.path.append(".")
from env.env_class.demo_env import Demo_Env


def initiate_demo_env(
    df: pd.DataFrame,
    feature_list: list,
    max_holding_number=8,
    position_choices=9,  # (must be an odd number, the minum of trading equals to (max_holder_number)/((action_dim-1)/2)s))
    leverage_choice=[
        5
    ],  # recommend only use one leverage choice, because the leverage does not influence the return directly, the position
    # itself is enough to show the risk preference
    long_estimated_rate=0.0005,
    short_estimated_rate=0,
    commission_rate=0.0002,
    # maten_mar_ratio_dict varies among different perpertual contracts, need to perform a config file for different perpertual
    # the default is for btcusdt perpetual contract
    maintenance_margin_ratio_dict={
        "50000": [0.004, 0],
        "500000": [0.005, 50],
        "10000000": [0.01, 2550],
    },
    early_stop=0,
    # initial_personal_state
    initial_state=(1e5, 0, 0, 0, 5),
    gamma=1,
    max_punishment=1e10,
    order_book_depth=25,
    allow_reverse_position=False,
    holding_duration_norm_steps=180,
    enable_limit_reward=False,
    limit_hold_bonus=1.0,
    limit_stay_bonus=0.5,
    limit_reverse_penalty=1.5,
    near_limit_threshold=0.003,
):

    # 对应钱包余额，起始保证金，未实现盈亏，持仓量，对应的杠杆):
    bid_prices_names = ["bid{}_price".format(i) for i in range(1, order_book_depth + 1)]
    ask_prices_names = ["ask{}_price".format(i) for i in range(1, order_book_depth + 1)]
    bid_sizes_names = ["bid{}_size".format(i) for i in range(1, order_book_depth + 1)]
    ask_sizes_names = ["ask{}_size".format(i) for i in range(1, order_book_depth + 1)]

    markprice_array = df["mark_price"].values
    timestamp_array = df["timestamp"].values
    funding_rate_array = df["funding_rate"].values
    funding_timestamp_array = df["funding_timestamp"].values
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

    env = Demo_Env(
        state_array,
        ask_prices_array,
        bid_prices_array,
        ask_qtys_array,
        bid_qtys_array,
        markprice_array,
        timestamp_array,
        funding_rate_array,
        funding_timestamp_array,
        max_holding_number=max_holding_number,
        position_choices=position_choices,  # (must be an odd number, the minum of trading equals to (max_holder_number)/((action_dim-1)/2)s))
        leverage_choice=leverage_choice,  # recommend only use one leverage choice, because the leverage does not influence the return directly, the position
        # itself is enough to show the risk preference
        long_estimated_rate=long_estimated_rate,
        short_estimated_rate=short_estimated_rate,
        commission_rate=commission_rate,
        # maten_mar_ratio_dict varies among different perpertual contracts, need to perform a config file for different perpertual
        # the default is for btcusdt perpetual contract
        maintenance_margin_ratio_dict=maintenance_margin_ratio_dict,
        early_stop=early_stop,
        # initial_personal_state
        initial_state=initial_state,
        max_punishment=max_punishment,
        gamma=gamma,
        allow_reverse_position=allow_reverse_position,
        holding_duration_norm_steps=holding_duration_norm_steps,
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
    return env
