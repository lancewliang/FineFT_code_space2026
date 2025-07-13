import pandas as pd
import numpy as np
import sys

sys.path.append(".")
from env.env_class.agg_env import Agg_Env
import os


def initiate_high_level_earnhft_env(
    df: pd.DataFrame,
    adjust_len: int,
    potential_model_path: str,
    dynamics_num: int,
    low_level_hidden_nodes: int,
    high_level_feature_list: list,
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
    # the device for the low level model
    device="cpu",
    time_info_dim=2,
):
    bid_prices_names = ["bid{}_price".format(i) for i in range(1, 26)]
    ask_prices_names = ["ask{}_price".format(i) for i in range(1, 26)]
    bid_sizes_names = ["bid{}_size".format(i) for i in range(1, 26)]
    ask_sizes_names = ["ask{}_size".format(i) for i in range(1, 26)]

    markprice_array = df["mark_price"].values
    timestamp_array = df["timestamp"].values
    funding_rate_array = df["funding_rate"].values
    funding_timestamp_array = df["funding_timestamp"].values
    ask_prices_array = df[ask_prices_names].values
    bid_prices_array = df[bid_prices_names].values
    ask_qtys_array = df[ask_sizes_names].values
    bid_qtys_array = df[bid_sizes_names].values
    state_array = df[feature_list].values
    # high level earnhft unique
    # high level feature
    high_level_state_array = df[high_level_feature_list].values
    low_level_dicts = {}
    low_level_action_num = (position_choices - 1) * len(leverage_choice) + 1
    for i in range(low_level_action_num):
        models_list = []
        initial_action_path = f"{potential_model_path}/initial_action_{i}"
        for j in range(dynamics_num):
            model_path = f"{initial_action_path}/model_{j}.pth"
            models_list.append(model_path)
        low_level_dicts[i] = models_list
    env = Agg_Env(
        adjust_len,
        low_level_dicts,
        low_level_hidden_nodes,
        high_level_state_array,
        state_array,
        ask_prices_array,
        bid_prices_array,
        ask_qtys_array,
        bid_qtys_array,
        markprice_array,
        timestamp_array,
        funding_rate_array,
        funding_timestamp_array,
        max_holding_number,
        position_choices,  # (must be an odd number, the minum of trading equals to (max_holder_number)/((action_dim-1)/2)s))
        leverage_choice,  # recommend only use one leverage choice, because the leverage does not influence the return directly, the position
        # itself is enough to show the risk preference
        long_estimated_rate,
        short_estimated_rate,
        commission_rate,
        maintenance_margin_ratio_dict,
        early_stop,
        # initial_personal_state
        initial_state,
        device,
        time_info_dim,
    )
    return env


if __name__ == "__main__":
    device = "cuda"
    dataset_name = "BTCUSDT"
    df = pd.read_feather(
        os.path.join("/data2/mlqin/FT_0618/dataset", dataset_name, "train.feather")
    )
    adjust_len = 48
    low_level_hidden_nodes = 128
    potential_model_path = os.path.join("result/EarnHFT/potential_model", dataset_name)
    dynamics_num = 5
    high_level_feature_list = np.load(
        os.path.join("dataset", dataset_name, "high_level_state_features.npy")
    )
    feature_list = np.load(os.path.join("dataset", dataset_name, "state_features.npy"))
    env = initiate_high_level_earnhft_env(
        df=df,
        adjust_len=adjust_len,
        potential_model_path=potential_model_path,
        dynamics_num=dynamics_num,
        low_level_hidden_nodes=low_level_hidden_nodes,
        high_level_feature_list=high_level_feature_list,
        feature_list=feature_list,
        device=device,  
    )
    s, info = env.reset()
    print("info key", info.keys())
    done = False
    while not done:
        a = 1
        s, r, done, info = env.step(a)
        print(r)
