import sys

sys.path.append(".")
import pandas as pd
import numpy as np
try:
    import gym
    from gym import spaces
except ImportError:
    from env.env_class.base_env import gym, spaces
from env.env_class.base_env import Base_Env
from env.env_class.futures_util import create_optimal_q_table


class Demo_Env(Base_Env):
    def __init__(
        self,
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
        # unique for demo_env
        max_punishment=1e10,
        gamma=1,
    ):

        super(Demo_Env, self).__init__(
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
        )
        self.q_table = create_optimal_q_table(
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
            # the default is for btcusdt perpetual contract
            max_punishment=max_punishment,
            gamma=gamma,
        )

    def reset(self):
        state, info = super(Demo_Env, self).reset()
        info["q_value"] = self.q_table[self.day][info["previous_action"]]
        return state, info

    def step(self, action):
        state, reward, terminal, info = super(Demo_Env, self).step(action)
        info["q_value"] = self.q_table[self.day][info["previous_action"]]
        return state, reward, terminal, info
