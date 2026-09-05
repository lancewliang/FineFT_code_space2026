# Code reference: https://github.com/Lizhi-sjtu/DRL-code-pytorch/tree/main/3.Rainbow_DQN
# Evaluate every sub-agent (each qnet in the ensemble) on all valid dataset files.
# For each sub-model x dataset file x initial action, run one full episode and
# report the final balance (= initial margin balance + cumulative reward),
# the cumulative reward sum, and the final return rate.

import argparse
import os
import re
import sys
from typing import List

import numpy as np
import pandas as pd
import torch

sys.path.append(".")

# model
from model.low_level import ensemble_Qnet

# env
from env.env_initiate.base_initiate import initiate_base_env
from env.env_class.futures_util import map_action_to_position_leverage


os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"

   

class weighted_trader:
    def __init__(self, 
        logg_file_path: str,
        data_file_path:str,
        model_path: str,
        tech_indicator_list_path: str,
        maintenance_margin_ratio_dict_path: str,
        transcation_cost: float,
        max_holding_number: int,          
        position_choices: int,   
        N: int,        
        time_info_dim: int = 2,
        hidden_nodes: int=128,
        leverage_choices: List[float]=[1],
        initial_leverage: int = 1,
        initial_position: int = 0,        
        initial_wallet_balance: float = 10000,
        order_book_depth: int=5,
        early_stop:int=2,
        enable_limit_reward: bool=False,
        limit_hold_bonus: float =1.0,
        limit_stay_bonus: float =0.5,
        limit_reverse_penalty: float =1.5,
        near_limit_threshold: float =0.05,
        allow_reverse_position: bool = True,
        
                 ):

        # device
        if torch.cuda.is_available():
            self.device = "cuda"
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.allow_tf32 = True
        else:
            self.device = "cpu"

        # model file to evaluate
        self.model_file = model_path
        self.data_file_path = data_file_path
        self.tech_indicator_list = np.load(
            tech_indicator_list_path
        )
        self.maintenance_margin_ratio_dict = np.load(maintenance_margin_ratio_dict_path, allow_pickle=True).item()
        self.max_holding_number = max_holding_number
        self.order_book_depth = order_book_depth
        self.position_choices = position_choices
        self.single_side_action_num = int((self.position_choices - 1) / 2)
        self.position_list = (
            [
                self.max_holding_number / self.single_side_action_num * i
                for i in range(1, self.single_side_action_num + 1)
            ]
            + [0]
            + [
                self.max_holding_number / self.single_side_action_num * -i
                for i in range(1, self.single_side_action_num + 1)
            ]
        )
        self.position_list.sort()
        self.leverage_choices = leverage_choices
        self.long_estimated_rate = 0
        self.short_estimated_rate = 0
        self.transcation_cost = transcation_cost
        self.allow_reverse_position = allow_reverse_position
        self.enable_limit_reward = enable_limit_reward
        self.limit_hold_bonus = limit_hold_bonus
        self.limit_stay_bonus = limit_stay_bonus
        self.limit_reverse_penalty = limit_reverse_penalty
        self.near_limit_threshold = near_limit_threshold
        self.early_stop = early_stop
        self.initial_wallet_balance = initial_wallet_balance
        self.initial_margin = 0
        self.initial_unrealized_pnL = 0
        self.initial_position = initial_position
        self.initial_leverage = initial_leverage
        # margin balance before any action, cumulative reward accumulates from it
        self.initial_margin_balance = (
            self.initial_wallet_balance + self.initial_unrealized_pnL
        )

        # network: the ensemble size is inferred from the checkpoint so any
        # trained_model.pkl can be evaluated directly
        self.time_info_dim = time_info_dim
        self.hidden_nodes = hidden_nodes        
        self.N_ACTIONS = (self.position_choices - 1) * len(self.leverage_choices) + 1
        state_dict = torch.load(self.model_file, map_location=self.device)
        self.N = N
        self.eval_net = ensemble_Qnet(
            N_STATES=len(self.tech_indicator_list),
            N_ACTIONS=self.N_ACTIONS,
            hidden_nodes=self.hidden_nodes,
            TIME_INFO_DIM=self.time_info_dim,
            ensemble_number=self.N,
        ).to(self.device)
        self.eval_net.load_state_dict(state_dict)
        self.eval_net.eval()
        self.initial_action_list = range(self.N_ACTIONS)

         

    def act_test(self, state, info, context_index):
        assert context_index in range(self.N)
        state = torch.unsqueeze(torch.FloatTensor(state).reshape(-1), 0).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor([info["previous_action"]]).float().to(self.device), 0
        ).to(self.device)
        avaliable_action = torch.unsqueeze(
            torch.tensor(info["avaliable_action"]).to(self.device), 0
        ).to(self.device)
        hour_count_down = (
            torch.unsqueeze(torch.tensor([info["funding_count_down_hour"]]), 0)
            .to(self.device)
            .float()
        )
        minute_count_down = (
            torch.unsqueeze(torch.tensor([info["funding_count_down_minute"]]), 0)
            .to(self.device)
            .float()
        )
        time_input = torch.cat([hour_count_down, minute_count_down], dim=1).to(
            self.device
        )
        trading_info = torch.unsqueeze(
            torch.tensor(info["trading_info"]).float().to(self.device), 0
        )
        with torch.inference_mode():
            action_value_chosen_index = self.eval_net.qnet_list[context_index](
                state=state,
                time=time_input,
                previous_action=previous_action,
                avaliable_action=avaliable_action,
                trading_info=trading_info,
            )
            action = torch.max(action_value_chosen_index, 1)[1].data.cpu().numpy()
        action = action[0]

        return action

    def _run_episode(self, test_df, initial_action, bin_index):
        """Run one full episode with the given sub-model and initial action.

        Returns the cumulative reward, the final balance and the final return
        rate of the episode.
        """
        initial_position, initial_leverage = map_action_to_position_leverage(
            initial_action,
            self.leverage_choices,
            self.position_list,
        )
        current_markprice = test_df["mark_price"].values[0]
        initial_margin = np.abs(
            initial_position * current_markprice / initial_leverage
        )
        initial_state = (
            self.initial_wallet_balance,
            initial_margin,
            self.initial_unrealized_pnL,
            initial_position,
            initial_leverage,
        )
        enable_limit = (
            getattr(self, "enable_limit_reward", True)
            and "UpperLimitPrice" in test_df.columns
            and "limit_up_single_sided_ratio" in test_df.columns
        )
        test_env = initiate_base_env(
            df=test_df,
            feature_list=self.tech_indicator_list,
            max_holding_number=self.max_holding_number,
            order_book_depth=self.order_book_depth,
            position_choices=self.position_choices,
            leverage_choice=self.leverage_choices,
            long_estimated_rate=self.long_estimated_rate,
            short_estimated_rate=self.short_estimated_rate,
            commission_rate=self.transcation_cost,
            maintenance_margin_ratio_dict=self.maintenance_margin_ratio_dict,
            early_stop=0,
            # initial_personal_state
            initial_state=initial_state,
            allow_reverse_position=getattr(self, "allow_reverse_position", False),
            enable_limit_reward=enable_limit,
            limit_hold_bonus=getattr(self, "limit_hold_bonus", 1.0),
            limit_stay_bonus=getattr(self, "limit_stay_bonus", 0.5),
            limit_reverse_penalty=getattr(self, "limit_reverse_penalty", 1.5),
            near_limit_threshold=getattr(self, "near_limit_threshold", 0.003),
        )
        s, info = test_env.reset()
        done = False
        reward_sum = 0
        while not done:
            a = self.act_test(s, info, bin_index)
            s_, r, done, info = test_env.step(a)
            s = s_
            reward_sum += r
        final_balance = (
            test_env.wallet_balance_history[-1]
            + test_env.unrealized_pnl_history[-1]
        )
        final_return_rate = (
            final_balance - self.initial_margin_balance
        ) / self.initial_margin_balance
        return {
            "initial_position": initial_position,
            "initial_leverage": initial_leverage,
            "reward_sum": reward_sum,
            "final_balance": final_balance,
            "final_return_rate": final_return_rate,
        }
    
    def test(self):
        print("start")
        self.eval_net.eval()
        overall_result = [] 
        test_df = pd.read_feather(self.data_file_path)
        for initial_action in self.initial_action_list:
            for bin_index in range(self.N):
                single_result = {
                    "sub_model_index": bin_index,
                    "df_path":self.data_file_path,
                    "initial_action": initial_action,
                    "df_length": len(test_df),
                    **self._run_episode(test_df, initial_action, bin_index),
                }
                print(single_result)
                overall_result.append(single_result)
       
        return overall_result
    
def evaluates( 
        logg_file_path: str,
        data_file_paths:List[str],
        model_path: str,
        tech_indicator_list_path: str,
        maintenance_margin_ratio_dict_path: str,
        transcation_cost: float,
        max_holding_number: int,          
        position_choices: int,   
        N: int,        
        time_info_dim: int = 2,
        hidden_nodes: int=128,
        leverage_choices: List[float]=[1],
        initial_leverage: int = 1,
        initial_position: int = 0,        
        initial_wallet_balance: float = 10000,
        order_book_depth: int=5,
        early_stop:int=2,
        enable_limit_reward: bool=False,
        limit_hold_bonus: float =1.0,
        limit_stay_bonus: float =0.5,
        limit_reverse_penalty: float =1.5,
        near_limit_threshold: float =0.05,
        allow_reverse_position: bool = True):
    """Evaluate the sub-agent."""
    pass
