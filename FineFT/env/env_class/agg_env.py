import pandas as pd
import numpy as np
import os
import sys
try:
    import gym
    from gym import spaces
except ImportError:
    from env.env_class.base_env import gym, spaces
import random
from time import time
import torch

sys.path.append(".")
from env.env_class.base_env import Base_Env
from model.low_level import Qnet


class Agg_Env(Base_Env):
    def __init__(
        self,
        # high level setting
        adjust_len,  # the frequency of the high level agent
        low_level_dicts,  # the low level netowrk path dictionary
        low_level_hidden_nodes,  # the number of the hidden nodes of the low level agents
        high_level_state_array,  # the state of the hidden nodes
        # base env initiation
        state_array,
        ask_prices_array,
        bid_prices_array,
        ask_qtys_array,
        bid_qtys_array,
        markprice_array,
        timestamp_array,
        funding_rate_array,
        funding_timestamp_array,
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
        initial_state=(1e5, 0, 0, 0, 1),
        # the device for the low level model
        device="cpu",
        time_info_dim=2,
    ):
        super().__init__(
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
            position_choices,
            leverage_choice,
            long_estimated_rate,
            short_estimated_rate,
            commission_rate,
            maintenance_margin_ratio_dict,
            early_stop,
            initial_state,
        )
        self.adjust_freq = adjust_len
        self.state_dim = self.state_array.shape[1]
        self.high_level_state_array = high_level_state_array
        self.N_ACTIONS = (self.position_choices - 1) * len(self.leverage_choices) + 1
        # initializing the network
        self.device = device
        self.low_level_dicts = low_level_dicts
        self.low_level_hidden_nodes = low_level_hidden_nodes
        self.low_level_agent_list_dict = {}
        for key in low_level_dicts:
            self.low_level_agent_list_dict[key] = []
            for model_path in low_level_dicts[key]:
                model = Qnet(
                    N_STATES=self.state_dim,
                    N_ACTIONS=self.N_ACTIONS,
                    hidden_nodes=self.low_level_hidden_nodes,
                    TIME_INFO_DIM=time_info_dim,
                ).to(self.device)
                model.load_state_dict(
                    torch.load(
                        model_path,
                        map_location=self.device,
                    )
                )
                self.low_level_agent_list_dict[key].append(model)
        self.chosen_model_history = []

    def reset(self):
        self.macro_action_history = []
        self.timestamp_history = []
        self.state, self.info = super(Agg_Env, self).reset()
        high_level_state = self.high_level_state_array[self.day]
        self.info["high_level_state"] = high_level_state
        self.chosen_model_history = []

        return self.state, self.info

    def step(self, action):
        previous_action = self.env_map_position_leverage_to_action(
            self.position, self.leverage
        )
        self.chosen_model = self.low_level_agent_list_dict[previous_action][action]
        self.chosen_model_history.append(
            previous_action * len(self.low_level_agent_list_dict) + action
        )
        reward_macro = 0
        for i in range(self.adjust_freq):
            micro_action = self.pose_macro_action(self.state, self.info)
            self.state, reward, done, self.info = Base_Env.step(self, micro_action)
            reward_macro += reward
            if self.terminal == True:
                high_level_state = self.high_level_state_array[self.day]
                self.info["high_level_state"] = high_level_state
                return self.state, reward_macro, done, self.info
        high_level_state = self.high_level_state_array[self.day]
        self.info["high_level_state"] = high_level_state
        return self.state, reward_macro, done, self.info

    def pose_macro_action(self, state, info):
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
        actions_value = self.chosen_model(
            state=state,
            time=time_input,
            previous_action=previous_action,
            avaliable_action=avaliable_action,
        )
        action = torch.max(actions_value, 1)[1].data.cpu().numpy()
        action = action[0]
        return action
