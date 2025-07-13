import cProfile
import pstats
import sys

sys.path.append(".")
from env.env_initiate.agg_initiate import initiate_high_level_earnhft_env
import pandas as pd
import numpy as np
import os

def main():
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
    for i in range(1000):
        a = 1
        s, r, done, info = env.step(a)
        print(r)
    print('done !')


if __name__ == "__main__":
    cProfile.run("main()", "profile_stats")
    p = pstats.Stats("profile_stats")
    p.sort_stats("cumulative").print_stats(10)
