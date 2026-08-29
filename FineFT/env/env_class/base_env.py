import pandas as pd
import numpy as np
import sys
try:
    import gym
    from gym import spaces
except ImportError:
    class _FallbackEnv:
        pass

    class _FallbackDiscrete:
        def __init__(self, n):
            self.n = n

    class _FallbackBox:
        def __init__(self, low, high, shape):
            self.low = low
            self.high = high
            self.shape = shape

    class _FallbackGym:
        Env = _FallbackEnv

    class _FallbackSpaces:
        Discrete = _FallbackDiscrete
        Box = _FallbackBox

    gym = _FallbackGym()
    spaces = _FallbackSpaces()

sys.path.append(".")
from env.env_class.futures_util import (
    change_of_wallet,
    calculate_avaiable_action,
    compute_limit_reward,
    judge_liquidation,
    calculate_maintenance_margin,
    map_action_to_position_leverage,
    map_position_leverage_to_action,
)
from analysis.calculate_metric.calculate_metric import (
    calculate_required_money,
    calculate_single_holsing_max_draw_down,
)

TRADING_INFO_KEYS = (
    "position_exposure",
    "single_holding_return_rate",
    "single_holding_max_drawdown",
    "current_holding_duration_norm",
)


class Base_Env(gym.Env):
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
        buy_fee_rate=None,
        sell_fee_rate=None,
        allow_reverse_position=False,
        holding_duration_norm_steps=180,
        is_limit_up_array=None,
        is_limit_down_array=None,
        limit_up_ask_depth_ratio_5_array=None,
        limit_down_bid_depth_ratio_5_array=None,
        upper_limit_prices_array=None,
        lower_limit_prices_array=None,
        enable_limit_reward=False,
        limit_hold_bonus=1.0,
        limit_stay_bonus=0.5,
        limit_reverse_penalty=1.5,
        near_limit_threshold=0.003,
    ):
        # trading setting
        self.max_holding_number = max_holding_number
        self.position_choices = position_choices
        self.leverage_choices = leverage_choice
        self.long_estimated_rate = long_estimated_rate
        self.short_estimated_rate = short_estimated_rate
        self.maintenance_margin_ratio_dict = maintenance_margin_ratio_dict
        self.commission_rate = commission_rate
        self.buy_fee_rate = buy_fee_rate
        self.sell_fee_rate = sell_fee_rate
        self.allow_reverse_position = allow_reverse_position
        if holding_duration_norm_steps <= 0:
            raise ValueError(f"holding_duration_norm_steps must be positive, got {holding_duration_norm_steps}")
        self.holding_duration_norm_steps = float(holding_duration_norm_steps)
        self.current_holding_duration = 0
        self.is_limit_up_array = is_limit_up_array
        self.is_limit_down_array = is_limit_down_array
        self.limit_up_ask_depth_ratio_5_array = limit_up_ask_depth_ratio_5_array
        self.limit_down_bid_depth_ratio_5_array = limit_down_bid_depth_ratio_5_array
        self.upper_limit_prices_array = upper_limit_prices_array
        self.lower_limit_prices_array = lower_limit_prices_array
        self.enable_limit_reward = enable_limit_reward
        self.limit_hold_bonus = float(limit_hold_bonus)
        self.limit_stay_bonus = float(limit_stay_bonus)
        self.limit_reverse_penalty = float(limit_reverse_penalty)
        self.near_limit_threshold = float(near_limit_threshold)
        # RL setting
        self.single_side_action_num = int((position_choices - 1) / 2)
        self.action_space = spaces.Discrete(
            (position_choices - 1) * len(leverage_choice) + 1
        )
        feature_num = state_array.shape[1]
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
        self._initialize_position_cost()
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
        self._reset_execution_metrics()
        # single_holding_return
        self.single_holding_return = 0
        self.single_holding_return_rate = 0
        # mdd is a rate
        self.single_holding_max_drawdown = 0
        # the history track the cash flow for a single holding
        self.single_holding_history = [0]
        if self.position == 0:
            self.current_holding_duration = 0
        else:
            self.current_holding_duration = 1


    def _zero_trading_info(self):
        return np.zeros(len(TRADING_INFO_KEYS), dtype=np.float32)

    def _calculate_trading_info(self, old_position=0):
        max_abs_position = max(abs(p) for p in self.position_list)
        position_exposure = 0.0 if max_abs_position == 0 else float(self.position / max_abs_position)
        if self.position == 0:
            return self._zero_trading_info()
        duration_norm = min(float(self.current_holding_duration) / float(self.holding_duration_norm_steps), 1.0)
        if self.allow_reverse_position and old_position * self.position < 0:
            return np.array([position_exposure, 0.0, 0.0, duration_norm], dtype=np.float32)
        return np.array(
            [
                position_exposure,
                float(self.single_holding_return_rate),
                float(self.single_holding_max_drawdown),
                duration_norm,
            ],
            dtype=np.float32,
        )

    def _compute_step_limit_reward(self, old_position):
        """Limit-aware reward shaping for the action just taken in step().

        Evaluated at the action-time index (self.day - 1), i.e. the market
        state that justified the action. Returns 0.0 when shaping is disabled
        or no limit state is active.
        """
        if not self.enable_limit_reward:
            return 0.0
        t = self.day - 1
        if t < 0:
            return 0.0

        def _at(arr):
            if arr is None or t >= len(arr):
                return None
            return arr[t]

        return compute_limit_reward(
            old_position=old_position,
            new_position=self.position,
            is_limit_up=_at(self.is_limit_up_array),
            is_limit_down=_at(self.is_limit_down_array),
            limit_up_ask_depth_ratio_5=_at(self.limit_up_ask_depth_ratio_5_array),
            limit_down_bid_depth_ratio_5=_at(self.limit_down_bid_depth_ratio_5_array),
            upper_limit_price=_at(self.upper_limit_prices_array),
            lower_limit_price=_at(self.lower_limit_prices_array),
            markprice=self.markprice_array[t],
            enable_limit_reward=self.enable_limit_reward,
            limit_hold_bonus=self.limit_hold_bonus,
            limit_stay_bonus=self.limit_stay_bonus,
            limit_reverse_penalty=self.limit_reverse_penalty,
            near_limit_threshold=self.near_limit_threshold,
        )

    def _reset_execution_metrics(self):
        self.commission_fee_step = 0
        self.realized_pnl_step = 0
        self.slippage_step = 0
        self.cumulative_commission_fee = 0
        self.cumulative_realized_pnl = 0
        self.cumulative_slippage = 0

    def _update_execution_metrics(self, wallet_change):
        self.commission_fee_step = wallet_change.commission_fee_step
        self.realized_pnl_step = wallet_change.realized_pnl_step
        self.slippage_step = wallet_change.slippage_step
        self.cumulative_commission_fee += self.commission_fee_step
        self.cumulative_realized_pnl += self.realized_pnl_step
        self.cumulative_slippage += self.slippage_step

    def _execution_metric_info(self):
        return {
            "commission_fee_step": self.commission_fee_step,
            "realized_pnl_step": self.realized_pnl_step,
            "slippage_step": self.slippage_step,
            "cumulative_commission_fee": self.cumulative_commission_fee,
            "cumulative_realized_pnl": self.cumulative_realized_pnl,
            "cumulative_slippage": self.cumulative_slippage,
        }

    def _position_cost_info(self):
        return {
            "current_holding_opening_price": self.current_holding_opening_price,
            "current_holding_average_price": self.current_holding_average_price,
        }

    def _initialize_position_cost(self):
        initial_holding_price = (
            float(self.current_markprice) if self.position != 0 else 0.0
        )
        self.current_holding_opening_price = initial_holding_price
        self.current_holding_average_price = initial_holding_price

    def _clear_position_cost(self):
        self.current_holding_opening_price = 0.0
        self.current_holding_average_price = 0.0

    def _update_position_cost(self, old_position, wallet_change):
        new_position = wallet_change.position
        if new_position == 0:
            self._clear_position_cost()
            return

        opened_quantity = wallet_change.opened_quantity
        if opened_quantity <= 0:
            return

        if new_position > 0:
            opened_price = (
                wallet_change.opened_value + wallet_change.opening_fee
            ) / opened_quantity
        else:
            opened_price = (
                wallet_change.opened_value - wallet_change.opening_fee
            ) / opened_quantity

        if old_position == 0 or old_position * new_position < 0:
            self.current_holding_opening_price = opened_price
            self.current_holding_average_price = opened_price
            return

        old_quantity = abs(old_position)
        self.current_holding_average_price = (
            old_quantity * self.current_holding_average_price
            + opened_quantity * opened_price
        ) / (old_quantity + opened_quantity)

    def env_map_action_to_position_leverage(self, action):
        return map_action_to_position_leverage(
            action, self.leverage_choices, self.position_list
        )

    def env_map_position_leverage_to_action(self, position, leverage):
        return map_position_leverage_to_action(
            position, leverage, self.leverage_choices, self.position_list
        )

    def _is_price_limit_blocked(self, target_position):
        is_limit_up = (
            self.is_limit_up_array is not None
            and bool(self.is_limit_up_array[self.day])
        )
        is_limit_down = (
            self.is_limit_down_array is not None
            and bool(self.is_limit_down_array[self.day])
        )
        return (is_limit_up and target_position > self.position) or (
            is_limit_down and target_position < self.position
        )

    def _filter_price_limit_actions(self, position_choices, leverage_choices):
        allowed_pairs = [
            (position, leverage)
            for position, leverage in zip(position_choices, leverage_choices)
            if not self._is_price_limit_blocked(position)
        ]
        return (
            [position for position, _ in allowed_pairs],
            [leverage for _, leverage in allowed_pairs],
        )

    def reset(self):
        self.day = 0
        self.terminal = self.day >= len(self.state_array) - self.early_stop - 1
        (
            self.wallet_balance,
            self.initial_margin,
            self.unrealized_pnl,
            self.position,
            self.leverage,
        ) = self.initial_state
        self.current_markprice = self.markprice_array[self.day]
        self._initialize_position_cost()
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
                buy_fee_rate=self.buy_fee_rate,
                sell_fee_rate=self.sell_fee_rate,
                # before action
                leverage=self.leverage,
                position=self.position,
                initial_margine=self.initial_margin,
                unrealized_pnL=self.unrealized_pnl,
                wallet_balance=self.wallet_balance,
                # action space setting
                leverage_choices=self.leverage_choices,
                position_choices=self.position_list,
                allow_reverse_position=self.allow_reverse_position,
            )
        )
        avaiable_position_choices, avaiable_leverage_choices = (
            self._filter_price_limit_actions(
                avaiable_position_choices, avaiable_leverage_choices
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
        self._reset_execution_metrics()
        self.new_position_required_money_history = [0]
        self.single_holding_return = 0
        self.single_holding_return_rate = 0
        # mdd is a rate
        self.single_holding_max_drawdown = 0
        # the history track the cash flow for a single holding
        self.single_holding_history = [0]
        self.current_holding_duration = 0 if self.position == 0 else 1
        return (
            state,
            {
                "current_timestamp": current_timestamp,
                "current_markprice": self.current_markprice,
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
                "ask_qyts": self.ask_qtys,
                "bid_qyts": self.bid_qtys,
                "single_holding_return_rate": self.single_holding_return_rate,
                "single_holding_max_drawdown": self.single_holding_max_drawdown,
                "trading_info": self._calculate_trading_info(0),
                **self._position_cost_info(),
                **self._execution_metric_info(),
            },
        )

    def step(self, action):
        old_position = self.position
        target_position, target_leverage = self.env_map_action_to_position_leverage(
            action
        )
        if self._is_price_limit_blocked(target_position):
            target_position = self.position
            target_leverage = self.leverage
        previous_margine_balance = self.wallet_balance + self.unrealized_pnl
        previous_timestamp = self.timestamp_array[self.day]
        previous_funding_rate = self.funding_rate_array[self.day]
        previous_funding_timestamp = self.funding_timestamp_array[self.day]
        previous_markprice = self.current_markprice
        wallet_change = change_of_wallet(
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
            buy_fee_rate=self.buy_fee_rate,
            sell_fee_rate=self.sell_fee_rate,
            allow_reverse_position=self.allow_reverse_position,
            position_list=self.position_list,
        )
        leverage = wallet_change.leverage
        position = wallet_change.position
        initial_margin = wallet_change.initial_margin
        unrealized_pnL = wallet_change.unrealized_pnl
        wallet_balance = wallet_change.wallet_balance
        slippage = wallet_change.slippage_step
        self._update_execution_metrics(wallet_change)
        self._update_position_cost(old_position, wallet_change)
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
        if self.position == target_position:
            self.single_holding_return = self.single_holding_return
        else:
            current_value_increment = (
                self.wallet_balance_history[-1] + self.unrealized_pnl_history[-1]
            ) - (self.wallet_balance_history[-2] + self.unrealized_pnl_history[-2])
            self.single_holding_return += current_value_increment
            self.single_holding_history.append(current_value_increment)
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
            require_money = calculate_required_money(
                np.array(self.initial_margin_history),
                np.array(self.maintain_marigine_history),
                np.array(self.new_position_required_money_history),
                np.array(self.unrealized_pnl_history),
                np.array(self.wallet_balance_history),
            )
            self.single_holding_return_rate = self.single_holding_return / (
                require_money + 1e-12
            )
            self.single_holding_max_drawdown = calculate_single_holsing_max_draw_down(
                self.single_holding_history,
                self.initial_margin_history,
                self.maintain_marigine_history,
                self.new_position_required_money_history,
                self.unrealized_pnl_history,
                self.wallet_balance_history,
            )
            self._clear_position_cost()
            return (
                state,
                reward,
                self.terminal,
                {
                    "personal_state": {0, 0, 0, 0, self.leverage_choices[0]},
                    "avaiable_action_list": avaiable_actions,
                    "avaliable_action": avaiable_action_mask,
                    "previous_timestamp": previous_timestamp,
                    "current_timestamp": current_timestamp,
                    "funding_count_down": previous_funding_timestamp - previous_timestamp,
                    "funding_count_down_hour": hours,
                    "funding_count_down_minute": minutes,
                    "funding_count_down_second": seconds,
                    "ask_qyts": self.ask_qtys,
                    "bid_qyts": self.bid_qtys,
                    "single_holding_return_rate": self.single_holding_return_rate,
                    "single_holding_max_drawdown": self.single_holding_max_drawdown,
                    "trading_info": self._zero_trading_info(),
                    "limit_reward": 0.0,
                    "previous_action": self.env_map_position_leverage_to_action(
                        self.position, self.leverage
                    ),
                    **self._position_cost_info(),
                    **self._execution_metric_info(),
                },
            )
        else:

            # 来到下一个timestmap
            if self.terminal or self.day >= len(self.state_array) - self.early_stop - 1:
                self.terminal = True
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

                require_money = calculate_required_money(
                    np.array(self.initial_margin_history),
                    np.array(self.maintain_marigine_history),
                    np.array(self.new_position_required_money_history),
                    np.array(self.unrealized_pnl_history),
                    np.array(self.wallet_balance_history),
                )
                self.single_holding_return_rate = self.single_holding_return / (
                    require_money + 1e-12
                )
                self.single_holding_max_drawdown = calculate_single_holsing_max_draw_down(
                    self.single_holding_history,
                    self.initial_margin_history,
                    self.maintain_marigine_history,
                    self.new_position_required_money_history,
                    self.unrealized_pnl_history,
                    self.wallet_balance_history,
                )
                return (
                    self.state_array[self.day],
                    self.wallet_balance + self.unrealized_pnl - previous_margine_balance,
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
                        "previous_timestamp": previous_timestamp,
                        "current_timestamp": previous_timestamp,
                        "funding_count_down": previous_funding_timestamp - previous_timestamp,
                        "funding_count_down_hour": 0,
                        "funding_count_down_minute": 0,
                        "funding_count_down_second": 0,
                        "ask_qyts": self.ask_qtys,
                        "bid_qyts": self.bid_qtys,
                        "single_holding_return_rate": self.single_holding_return_rate,
                        "single_holding_max_drawdown": self.single_holding_max_drawdown,
                        "trading_info": self._zero_trading_info(),
                        "limit_reward": 0.0,
                        "previous_action": self.env_map_position_leverage_to_action(
                            self.position, self.leverage
                        ),
                        **self._position_cost_info(),
                        **self._execution_metric_info(),
                    },
                )
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
            self.single_holding_return += future_value_increment
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
            self.single_holding_history.append(future_value_increment)
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

                require_money = calculate_required_money(
                    np.array(self.initial_margin_history),
                    np.array(self.maintain_marigine_history),
                    np.array(self.new_position_required_money_history),
                    np.array(self.unrealized_pnl_history),
                    np.array(self.wallet_balance_history),
                )
                self.single_holding_return_rate = self.single_holding_return / (
                    require_money + 1e-12
                )
                self.single_holding_max_drawdown = (
                    calculate_single_holsing_max_draw_down(
                        self.single_holding_history,
                        self.initial_margin_history,
                        self.maintain_marigine_history,
                        self.new_position_required_money_history,
                        self.unrealized_pnl_history,
                        self.wallet_balance_history,
                    )
                )
                self._clear_position_cost()
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
                        "previous_timestamp": previous_timestamp,
                        "current_timestamp": current_timestamp,
                        "funding_count_down": current_funding_timestamp - current_timestamp,
                        "funding_count_down_hour": hours,
                        "funding_count_down_minute": minutes,
                        "funding_count_down_second": seconds,
                        "ask_qyts": self.ask_qtys,
                        "bid_qyts": self.bid_qtys,
                        "single_holding_return_rate": self.single_holding_return_rate,
                        "single_holding_max_drawdown": self.single_holding_max_drawdown,
                        "trading_info": self._zero_trading_info(),
                    "limit_reward": 0.0,
                        "previous_action": self.env_map_position_leverage_to_action(
                            self.position, self.leverage
                        ),
                        **self._position_cost_info(),
                        **self._execution_metric_info(),
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
                        buy_fee_rate=self.buy_fee_rate,
                        sell_fee_rate=self.sell_fee_rate,
                        # current action
                        leverage=self.leverage,
                        position=self.position,
                        initial_margine=self.initial_margin,
                        unrealized_pnL=self.unrealized_pnl,
                        wallet_balance=self.wallet_balance,
                        # action space setting
                        leverage_choices=self.leverage_choices,
                        position_choices=self.position_list,
                        allow_reverse_position=self.allow_reverse_position,
                    )
                )
                avaiable_position_choices, avaiable_leverage_choices = (
                    self._filter_price_limit_actions(
                        avaiable_position_choices, avaiable_leverage_choices
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
                limit_reward = self._compute_step_limit_reward(old_position)
                reward += limit_reward

                require_money = calculate_required_money(
                    np.array(self.initial_margin_history),
                    np.array(self.maintain_marigine_history),
                    np.array(self.new_position_required_money_history),
                    np.array(self.unrealized_pnl_history),
                    np.array(self.wallet_balance_history),
                )
                self.single_holding_return_rate = self.single_holding_return / (
                    require_money + 1e-12
                )
                self.single_holding_max_drawdown = (
                    calculate_single_holsing_max_draw_down(
                        self.single_holding_history,
                        self.initial_margin_history,
                        self.maintain_marigine_history,
                        self.new_position_required_money_history,
                        self.unrealized_pnl_history,
                        self.wallet_balance_history,
                    )
                )
                if self.position == 0:
                    self.current_holding_duration = 0
                elif old_position == 0 or (self.allow_reverse_position and old_position * self.position < 0):
                    self.current_holding_duration = 1
                else:
                    self.current_holding_duration += 1
                trading_info = self._calculate_trading_info(old_position)
                # 在step之后才对single holding进行重置
                if self.position == 0 or (self.allow_reverse_position and old_position * self.position < 0):
                    self.single_holding_return = 0
                    self.single_holding_history = [0]
                    self.initial_margin_history = [self.initial_margin]
                    self.wallet_balance_history = [self.wallet_balance]
                    self.unrealized_pnl_history = [self.unrealized_pnl]
                    self.maintain_marigine_history = [
                        calculate_maintenance_margin(
                            np.abs(self.current_markprice * self.position)
                        )
                    ]
                    self.new_position_required_money_history = [0]

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
                        "previous_timestamp": previous_timestamp,
                        "current_timestamp": current_timestamp,
                        "avaiable_action_list": avaiable_actions,
                        "avaliable_action": avaiable_action_mask,
                        "funding_count_down": current_funding_timestamp - current_timestamp,
                        "funding_count_down_hour": hours,
                        "funding_count_down_minute": minutes,
                        "funding_count_down_second": seconds,
                        "previous_action": self.env_map_position_leverage_to_action(
                            self.position, self.leverage
                        ),
                        "ask_qyts": self.ask_qtys,
                        "bid_qyts": self.bid_qtys,
                        "single_holding_return_rate": self.single_holding_return_rate,
                        "single_holding_max_drawdown": self.single_holding_max_drawdown,
                        "trading_info": trading_info,
                        "limit_reward": limit_reward,
                        **self._position_cost_info(),
                        **self._execution_metric_info(),
                    },
                )
