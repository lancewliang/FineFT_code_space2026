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
    )
    return env
