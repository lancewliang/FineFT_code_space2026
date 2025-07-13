import sys
sys.path.append(".")
import pandas as pd
import numpy as np
from env.env_initiate.base_initiate import initiate_base_env


#parameters
df=pd.read_feather('/data2/mlqin/HFT4Ind2/dataset/BNBUSDT/train/df_0.feather')
features=np.load('/data2/mlqin/HFT4Ind2/dataset/BNBUSDT/state_features.npy')


if __name__=="__main__":
    env=initiate_base_env(df,features)
    state, info = env.reset()
    done = False
    # while not done:
    for i in range(30):
        action = info['avaiable_action_list'][0]
        print(action)
        
        state, reward, done, info = env.step(action)
        
        print(info)
    for i in range(10):
        action=4
        state, reward, done, info = env.step(action)
        print(info)
    print("final balance", env.wallet_balance + env.unrealized_pnl)