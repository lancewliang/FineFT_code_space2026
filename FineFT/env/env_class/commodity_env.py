import numpy as np

from env.env_class.base_env import Base_Env


FUNDING_INFO_KEYS = {
    "funding_count_down",
    "funding_count_down_hour",
    "funding_count_down_minute",
    "funding_count_down_second",
}


class Commodity_Env(Base_Env):
    def __init__(
        self,
        state_array,
        ask_prices_array,
        bid_prices_array,
        ask_qtys_array,
        bid_qtys_array,
        markprice_array,
        timestamp_array,
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
    ):
        funding_rate_array = np.zeros(len(timestamp_array), dtype=float)
        funding_timestamp_array = timestamp_array
        super(Commodity_Env, self).__init__(
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
        )

    @staticmethod
    def _strip_funding_info(info):
        for key in FUNDING_INFO_KEYS:
            info.pop(key, None)
        return info

    def reset(self):
        state, info = super(Commodity_Env, self).reset()
        return state, self._strip_funding_info(info)

    def step(self, action):
        state, reward, terminal, info = super(Commodity_Env, self).step(action)
        return state, reward, terminal, self._strip_funding_info(info)
