import pandas as pd
import numpy as np
import sys
import gym
from gym import spaces

sys.path.append(".")
from env.env_class.futures_util import (
    change_of_wallet,
    calculate_avaiable_action,
    judge_liquidation,
    calculate_maintenance_margin,
    map_action_to_position_leverage,
    map_position_leverage_to_action,
)


class Simple_Env(gym.Env):
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
    ):
        # trading setting
        self.max_holding_number = max_holding_number
        self.position_choices = position_choices
        self.leverage_choices = leverage_choice
        self.long_estimated_rate = long_estimated_rate
        self.short_estimated_rate = short_estimated_rate
        self.maintenance_margin_ratio_dict = maintenance_margin_ratio_dict
        self.commission_rate = commission_rate
        # RL setting
        self.single_side_action_num = int((position_choices - 1) / 2)
        self.action_space = spaces.Discrete(
            (position_choices - 1) * len(leverage_choice) + 1
        )
        feature_num = state_array.shape[-1]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=+np.inf,
            shape=(feature_num,),
        )
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
        # data setting
        self.state_array = state_array
        self.ask_prices_array = ask_prices_array
        self.bid_prices_array = bid_prices_array
        self.ask_qtys_array = ask_qtys_array
        self.bid_qtys_array = bid_qtys_array
        self.markprice_array = markprice_array
        self.timestamp_array = timestamp_array
        self.funding_rate_array = funding_rate_array
        self.funding_timestamp_array = funding_timestamp_array
        self.stack_length = 1

        # general setting
        self.early_stop = early_stop
        self.initial_state = initial_state

        # initialization
        self.terminal = False
        self.day = 0
        (
            self.wallet_balance,
            self.initial_margin,
            self.unrealized_pnl,
            self.position,
            self.leverage,
        ) = self.initial_state
        self.current_markprice = markprice_array[self.day]
        # history related
        # one per step
        self.micro_action_history = []
        self.margine_balance_history = []

        # reset one, two per step
        self.initial_margin_history = []
        self.wallet_balance_history = []
        self.unrealized_pnl_history = []
        self.maintain_marigine_history = []
        self.new_position_required_money_history = []
        self.slippage_sum = 0

    def env_map_action_to_position_leverage(self, action):
        return map_action_to_position_leverage(
            action, self.leverage_choices, self.position_list
        )

    def env_map_position_leverage_to_action(self, position, leverage):
        return map_position_leverage_to_action(
            position, leverage, self.leverage_choices, self.position_list
        )

    def reset(self):
        self.terminal = False
        self.day = 0
        (
            self.wallet_balance,
            self.initial_margin,
            self.unrealized_pnl,
            self.position,
            self.leverage,
        ) = self.initial_state
        self.current_markprice = self.markprice_array[self.day]
        state = self.state_array[self.day]
        self.ask_prices = self.ask_prices_array[self.day]
        self.bid_prices = self.bid_prices_array[self.day]
        self.ask_qtys = self.ask_qtys_array[self.day]
        self.bid_qtys = self.bid_qtys_array[self.day]
        avaiable_actions = []
        avaiable_position_choices, avaiable_leverage_choices = (
            calculate_avaiable_action(
                self.current_markprice,
                self.ask_prices,
                self.ask_qtys,
                self.bid_prices,
                self.bid_qtys,
                long_estimated_rate=self.long_estimated_rate,
                short_estimated_rate=self.short_estimated_rate,
                commission_rate=self.commission_rate,
                # before action
                leverage=self.leverage,
                position=self.position,
                initial_margine=self.initial_margin,
                unrealized_pnL=self.unrealized_pnl,
                wallet_balance=self.wallet_balance,
                # action space setting
                leverage_choices=self.leverage_choices,
                position_choices=self.position_list,
            )
        )

        for avaible_position, avaiable_leverage in zip(
            avaiable_position_choices, avaiable_leverage_choices
        ):
            avaiable_actions.append(
                self.env_map_position_leverage_to_action(
                    avaible_position, avaiable_leverage
                )
            )
        avaiable_action_mask = np.zeros(self.action_space.n)
        avaiable_action_mask[avaiable_actions] = 1
        current_funding_timestamp = self.funding_timestamp_array[self.day]
        current_timestamp = self.timestamp_array[self.day]
        funding_count_down = current_funding_timestamp - current_timestamp
        total_seconds = funding_count_down / np.timedelta64(1, "s")
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        # history related
        self.micro_action_history = []
        self.margine_balance_history = [self.wallet_balance + self.unrealized_pnl]

        # reset one, two per step
        self.initial_margin_history = [self.initial_margin]
        self.wallet_balance_history = [self.wallet_balance]
        self.unrealized_pnl_history = [self.unrealized_pnl]
        self.maintain_marigine_history = [
            calculate_maintenance_margin(np.abs(self.current_markprice * self.position))
        ]
        self.slippage_sum = 0
        self.new_position_required_money_history = [0]
        return (
            state,
            {
                "personal_state": self.initial_state,
                "avaiable_action_list": avaiable_actions,
                "avaliable_action": avaiable_action_mask,
                "funding_count_down": current_funding_timestamp - current_timestamp,
                "funding_count_down_hour": hours,
                "funding_count_down_minute": minutes,
                "funding_count_down_second": seconds,
                "previous_action": self.env_map_position_leverage_to_action(
                    self.position, self.leverage
                ),
            },
        )

    def step(self, action):
        target_position, target_leverage = self.env_map_action_to_position_leverage(
            action
        )
        previous_margine_balance = self.wallet_balance + self.unrealized_pnl
        previous_timestamp = self.timestamp_array[self.day]
        previous_funding_rate = self.funding_rate_array[self.day]
        previous_funding_timestamp = self.funding_timestamp_array[self.day]
        previous_markprice = self.current_markprice
        (
            leverage,
            position,
            initial_margin,
            unrealized_pnL,
            wallet_balance,
            slippage,
        ) = change_of_wallet(
            markprice=self.current_markprice,
            ask_prices=self.ask_prices,
            ask_qtys=self.ask_qtys,
            bid_prices=self.bid_prices,
            bid_qtys=self.bid_qtys,
            long_estimated_rate=self.long_estimated_rate,
            short_estimated_rate=self.short_estimated_rate,
            commission_rate=self.commission_rate,
            # before action
            previous_leverage=self.leverage,
            previous_position=self.position,
            previous_initial_margine=self.initial_margin,
            previous_unrealized_pnL=self.unrealized_pnl,
            previous_wallet_balance=self.wallet_balance,
            # target after the action
            current_leverage=target_leverage,
            current_position=target_position,
            silent=False,
        )
        self.slippage_sum += slippage
        ##history related
        self.micro_action_history.append(action)

        self.wallet_balance_history.append(wallet_balance)
        self.initial_margin_history.append(initial_margin)
        self.unrealized_pnl_history.append(unrealized_pnL)
        self.maintain_marigine_history.append(
            calculate_maintenance_margin(np.abs(self.current_markprice * position))
        )
        # record the requried money
        if np.abs(target_position) > np.abs(self.position):
            if target_leverage == self.leverage:
                self.new_position_required_money_history.append(
                    initial_margin - self.initial_margin
                )
            else:
                self.new_position_required_money_history.append(
                    max(initial_margin - self.initial_margin, 0)
                )
        else:
            self.new_position_required_money_history.append(0)
        
        if judge_liquidation(
            self.current_markprice,
            position,
            unrealized_pnL,
            wallet_balance,
            maintenance_margin_ratio_dict=self.maintenance_margin_ratio_dict,
        ):
            previous_funding_count_down = (
                previous_funding_timestamp - previous_timestamp
            )
            total_seconds = previous_funding_count_down / np.timedelta64(1, "s")
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            self.margine_balance_history.append(wallet_balance + unrealized_pnL)
            print(
                "liquidation happened right after the change of position and leverage, there might be something wrong with the calculate_avaliable_action"
            )
            print(
                "the previous position is {}".format(self.position),
                "the previous leverage is {}".format(self.leverage),
                "the target position is {}".format(target_position),
                "the target leverage is {}".format(target_leverage),
                "the previous wallet_balance is {}".format(self.wallet_balance),
                "the previous unrealized_pnl is {}".format(self.unrealized_pnl),
                "the current markprice is {}".format(self.current_markprice),
            )
            self.terminal = True
            reward = (wallet_balance + unrealized_pnL) - (
                self.wallet_balance + self.unrealized_pnl
            )
            avaiable_position_choices, avaiable_leverage_choices = (
                [0],
                [self.leverage_choices[0]],
            )
            avaiable_actions = []
            for avaible_position, avaiable_leverage in zip(
                avaiable_position_choices, avaiable_leverage_choices
            ):
                avaiable_actions.append(
                    self.env_map_position_leverage_to_action(
                        avaible_position, avaiable_leverage
                    )
                )
            avaiable_actions = list(set(avaiable_actions))

            avaiable_action_mask = np.zeros(self.action_space.n)
            avaiable_action_mask[avaiable_actions] = 1
            state = self.state_array[self.day]

            return (
                state,
                reward,
                self.terminal,
                {
                    "personal_state": {0, 0, 0, 0, self.leverage_choices[0]},
                    "avaiable_action_list": avaiable_actions,
                    "avaliable_action": avaiable_action_mask,
                    "funding_count_down": previous_funding_timestamp
                    - previous_timestamp,
                    "funding_count_down_hour": hours,
                    "funding_count_down_minute": minutes,
                    "funding_count_down_second": seconds,
                    "ask_qyts": self.ask_qtys,
                    "bid_qyts": self.bid_qtys,
                    "previous_action": self.env_map_position_leverage_to_action(
                        self.position, self.leverage
                    ),
                },
            )
        else:

            # 来到下一个timestmap
            self.day += 1
            current_timestamp = self.timestamp_array[self.day]
            current_funding_rate = self.funding_rate_array[self.day]
            current_funding_timestamp = self.funding_timestamp_array[self.day]
            self.current_markprice = self.markprice_array[self.day]
            state = self.state_array[self.day]
            self.ask_prices = self.ask_prices_array[self.day]
            self.bid_prices = self.bid_prices_array[self.day]
            self.ask_qtys = self.ask_qtys_array[self.day]
            self.bid_qtys = self.bid_qtys_array[self.day]
            self.leverage = leverage
            self.position = position

            self.initial_margin = np.abs(
                self.position * self.current_markprice / self.leverage
            )
            future_value_increment = self.position * (
                self.current_markprice - previous_markprice
            )
            self.unrealized_pnl = unrealized_pnL + future_value_increment
            self.single_holding_return = +future_value_increment
            self.wallet_balance = wallet_balance
            funding_count_down = current_funding_timestamp - current_timestamp
            total_seconds = funding_count_down / np.timedelta64(1, "s")
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            if current_timestamp == previous_funding_timestamp:
                funding_fee = self.position * previous_markprice * current_funding_rate
                self.wallet_balance -= funding_fee
                future_value_increment -= funding_fee
                self.single_holding_return -= funding_fee
            self.margine_balance_history.append(
                self.wallet_balance + self.unrealized_pnl
            )
            self.wallet_balance_history.append(self.wallet_balance)
            self.initial_margin_history.append(self.initial_margin)
            self.unrealized_pnl_history.append(self.unrealized_pnl)
            self.maintain_marigine_history.append(
                calculate_maintenance_margin(
                    np.abs(self.current_markprice * self.position)
                )
            )
            self.new_position_required_money_history.append(0)
            if judge_liquidation(
                self.current_markprice,
                position,
                self.unrealized_pnl,
                self.wallet_balance,
                maintenance_margin_ratio_dict=self.maintenance_margin_ratio_dict,
            ):
                self.terminal = True
                reward = (self.wallet_balance + self.unrealized_pnl) - (
                    wallet_balance + unrealized_pnL
                )
                avaiable_position_choices, avaiable_leverage_choices = (
                    [0],
                    [self.leverage_choices[0]],
                )
                avaiable_actions = []
                for avaible_position, avaiable_leverage in zip(
                    avaiable_position_choices, avaiable_leverage_choices
                ):
                    avaiable_actions.append(
                        self.env_map_position_leverage_to_action(
                            avaible_position, avaiable_leverage
                        )
                    )
                avaiable_actions = list(set(avaiable_actions))

                avaiable_action_mask = np.zeros(self.action_space.n)
                avaiable_action_mask[avaiable_actions] = 1

                return (
                    state,
                    reward,
                    self.terminal,
                    {
                        "personal_state": (
                            self.wallet_balance,
                            self.initial_margin,
                            self.unrealized_pnl,
                            self.position,
                            self.leverage,
                        ),
                        "avaiable_action_list": avaiable_actions,
                        "avaliable_action": avaiable_action_mask,
                        "funding_count_down": current_funding_timestamp
                        - current_timestamp,
                        "funding_count_down_hour": hours,
                        "funding_count_down_minute": minutes,
                        "funding_count_down_second": seconds,
                        "previous_action": self.env_map_position_leverage_to_action(
                            self.position, self.leverage
                        ),
                    },
                )
            else:
                self.person_state = (
                    self.wallet_balance,
                    self.initial_margin,
                    self.unrealized_pnl,
                    self.position,
                    self.leverage,
                )
                avaiable_position_choices, avaiable_leverage_choices = (
                    calculate_avaiable_action(
                        markprice=self.current_markprice,
                        ask_prices=self.ask_prices,
                        ask_qtys=self.ask_qtys,
                        bid_prices=self.bid_prices,
                        bid_qtys=self.bid_qtys,
                        long_estimated_rate=self.long_estimated_rate,
                        short_estimated_rate=self.short_estimated_rate,
                        commission_rate=self.commission_rate,
                        # current action
                        leverage=self.leverage,
                        position=self.position,
                        initial_margine=self.initial_margin,
                        unrealized_pnL=self.unrealized_pnl,
                        wallet_balance=self.wallet_balance,
                        # action space setting
                        leverage_choices=self.leverage_choices,
                        position_choices=self.position_list,
                    )
                )
                avaiable_actions = []
                for avaible_position, avaiable_leverage in zip(
                    avaiable_position_choices, avaiable_leverage_choices
                ):
                    avaiable_actions.append(
                        self.env_map_position_leverage_to_action(
                            avaible_position, avaiable_leverage
                        )
                    )
                avaiable_actions = list(set(avaiable_actions))

                avaiable_action_mask = np.zeros(self.action_space.n)
                avaiable_action_mask[avaiable_actions] = 1
                if self.day == len(self.state_array) - self.early_stop - 1:
                    self.terminal = True
                reward = (
                    self.wallet_balance + self.unrealized_pnl - previous_margine_balance
                )

                # 在step之后才对single holding进行重置
               

                return (
                    state,
                    reward,
                    self.terminal,
                    {
                        "personal_state": (
                            self.wallet_balance,
                            self.initial_margin,
                            self.unrealized_pnl,
                            self.position,
                            self.leverage,
                        ),
                        "avaiable_action_list": avaiable_actions,
                        "avaliable_action": avaiable_action_mask,
                        "funding_count_down": current_funding_timestamp
                        - current_timestamp,
                        "funding_count_down_hour": hours,
                        "funding_count_down_minute": minutes,
                        "funding_count_down_second": seconds,
                        "previous_action": self.env_map_position_leverage_to_action(
                            self.position, self.leverage
                        ),
                    },
                )
