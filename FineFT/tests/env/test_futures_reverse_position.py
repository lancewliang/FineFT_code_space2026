import pandas as pd
import numpy as np
import pytest
from FineFT.env.env_class.futures_util import (
    change_of_wallet,
    calculate_avaiable_action,
    create_optimal_q_table,
    WalletChangeResult,
)
from FineFT.env.env_class.base_env import Base_Env
from FineFT.env.env_class.demo_env import Demo_Env


def test_change_of_wallet_reverse_disabled_by_default():
    position_list = np.array([-4, -2, 0, 2, 4])
    ask_prices = np.array([100.0, 101.0, 102.0])
    ask_qtys = np.array([10.0, 10.0, 10.0])
    bid_prices = np.array([99.0, 98.0, 97.0])
    bid_qtys = np.array([10.0, 10.0, 10.0])
    
    result = change_of_wallet(
        markprice=100.0,
        ask_prices=ask_prices,
        ask_qtys=ask_qtys,
        bid_prices=bid_prices,
        bid_qtys=bid_qtys,
        long_estimated_rate=0.0,
        short_estimated_rate=0.0,
        commission_rate=0.0001,
        previous_leverage=1,
        previous_position=2,
        previous_initial_margine=200.0,
        previous_unrealized_pnL=0.0,
        previous_wallet_balance=1000.0,
        current_leverage=1,
        current_position=-2,
        silent=True,
    )
    
    # Default allow_reverse_position=False returns previous state
    assert result.position == 2
    assert result.leverage == 1
    assert result.wallet_balance == 1000.0


def test_change_of_wallet_reverse_long_to_short_success():
    ask_prices = np.array([100.0, 101.0, 102.0])
    ask_qtys = np.array([10.0, 10.0, 10.0])
    bid_prices = np.array([100.0, 99.0, 98.0])
    bid_qtys = np.array([10.0, 10.0, 10.0])
    position_list = np.array([-4, -2, 0, 2, 4])

    result = change_of_wallet(
        markprice=100.0,
        ask_prices=ask_prices,
        ask_qtys=ask_qtys,
        bid_prices=bid_prices,
        bid_qtys=bid_qtys,
        long_estimated_rate=0.0,
        short_estimated_rate=0.0,
        commission_rate=0.001,
        previous_leverage=1,
        previous_position=2,
        previous_initial_margine=200.0,
        previous_unrealized_pnL=0.0,
        previous_wallet_balance=1000.0,
        current_leverage=2,
        current_position=-2,
        silent=True,
        allow_reverse_position=True,
        position_list=position_list,
    )
    
    assert result.position == -2
    assert result.leverage == 2
    assert result.commission_fee_step > 0
    assert result.wallet_balance < 1000.0


def test_change_of_wallet_reverse_short_to_long_success():
    ask_prices = np.array([100.0, 101.0, 102.0])
    ask_qtys = np.array([10.0, 10.0, 10.0])
    bid_prices = np.array([100.0, 99.0, 98.0])
    bid_qtys = np.array([10.0, 10.0, 10.0])
    position_list = np.array([-4, -2, 0, 2, 4])

    result = change_of_wallet(
        markprice=100.0,
        ask_prices=ask_prices,
        ask_qtys=ask_qtys,
        bid_prices=bid_prices,
        bid_qtys=bid_qtys,
        long_estimated_rate=0.0,
        short_estimated_rate=0.0,
        commission_rate=0.001,
        previous_leverage=1,
        previous_position=-2,
        previous_initial_margine=200.0,
        previous_unrealized_pnL=0.0,
        previous_wallet_balance=1000.0,
        current_leverage=2,
        current_position=2,
        silent=True,
        allow_reverse_position=True,
        position_list=position_list,
    )
    
    assert result.position == 2
    assert result.leverage == 2
    assert result.commission_fee_step > 0


def test_change_of_wallet_reverse_insufficient_margin():
    ask_prices = np.array([100.0, 101.0, 102.0])
    ask_qtys = np.array([10.0, 10.0, 10.0])
    bid_prices = np.array([100.0, 99.0, 98.0])
    bid_qtys = np.array([10.0, 10.0, 10.0])
    position_list = np.array([-100, -2, 0, 2, 100])

    result = change_of_wallet(
        markprice=100.0,
        ask_prices=ask_prices,
        ask_qtys=ask_qtys,
        bid_prices=bid_prices,
        bid_qtys=bid_qtys,
        long_estimated_rate=0.0,
        short_estimated_rate=0.0,
        commission_rate=0.001,
        previous_leverage=1,
        previous_position=2,
        previous_initial_margine=200.0,
        previous_unrealized_pnL=0.0,
        previous_wallet_balance=10.0,
        current_leverage=1,
        current_position=-100,
        silent=True,
        allow_reverse_position=True,
        position_list=position_list,
    )
    
    assert result.position == 0


def test_change_of_wallet_reverse_insufficient_depth():
    ask_prices = np.array([100.0, 101.0, 102.0])
    ask_qtys = np.array([10.0, 10.0, 10.0])
    bid_prices = np.array([100.0, 99.0, 98.0])
    bid_qtys = np.array([2.0, 0.0, 0.0])
    position_list = np.array([-10, -4, -2, 0, 2, 4, 10])

    result = change_of_wallet(
        markprice=100.0,
        ask_prices=ask_prices,
        ask_qtys=ask_qtys,
        bid_prices=bid_prices,
        bid_qtys=bid_qtys,
        long_estimated_rate=0.0,
        short_estimated_rate=0.0,
        commission_rate=0.001,
        previous_leverage=1,
        previous_position=2,
        previous_initial_margine=200.0,
        previous_unrealized_pnL=0.0,
        previous_wallet_balance=10000.0,
        current_leverage=1,
        current_position=-10,
        silent=True,
        allow_reverse_position=True,
        position_list=position_list,
    )
    
    assert result.position in position_list
    assert result.position <= 0


def test_calculate_available_action_reverse_disabled():
    ask_prices = np.array([100.0, 101.0, 102.0])
    ask_qtys = np.array([10.0, 10.0, 10.0])
    bid_prices = np.array([100.0, 99.0, 98.0])
    bid_qtys = np.array([10.0, 10.0, 10.0])
    
    pos_choices, lev_choices = calculate_avaiable_action(
        markprice=100.0,
        ask_prices=ask_prices,
        ask_qtys=ask_qtys,
        bid_prices=bid_prices,
        bid_qtys=bid_qtys,
        long_estimated_rate=0.0,
        short_estimated_rate=0.0,
        commission_rate=0.001,
        leverage=1,
        position=2,
        initial_margine=200.0,
        unrealized_pnL=0.0,
        wallet_balance=1000.0,
        leverage_choices=[1, 2],
        position_choices=[-4, -2, 0, 2, 4],
        allow_reverse_position=False,
    )
    
    assert all(p >= 0 for p in pos_choices)


def test_calculate_available_action_reverse_enabled_sufficient_margin():
    ask_prices = np.array([100.0, 101.0, 102.0])
    ask_qtys = np.array([10.0, 10.0, 10.0])
    bid_prices = np.array([100.0, 99.0, 98.0])
    bid_qtys = np.array([10.0, 10.0, 10.0])
    
    pos_choices, lev_choices = calculate_avaiable_action(
        markprice=100.0,
        ask_prices=ask_prices,
        ask_qtys=ask_qtys,
        bid_prices=bid_prices,
        bid_qtys=bid_qtys,
        long_estimated_rate=0.0,
        short_estimated_rate=0.0,
        commission_rate=0.001,
        leverage=1,
        position=2,
        initial_margine=200.0,
        unrealized_pnL=0.0,
        wallet_balance=1000.0,
        leverage_choices=[1, 2],
        position_choices=[-4, -2, 0, 2, 4],
        allow_reverse_position=True,
    )
    
    assert -2 in pos_choices
    assert -4 in pos_choices


def test_calculate_available_action_reverse_enabled_insufficient_margin():
    ask_prices = np.array([100.0, 101.0, 102.0])
    ask_qtys = np.array([10.0, 10.0, 10.0])
    bid_prices = np.array([100.0, 99.0, 98.0])
    bid_qtys = np.array([10.0, 10.0, 10.0])
    
    pos_choices, lev_choices = calculate_avaiable_action(
        markprice=100.0,
        ask_prices=ask_prices,
        ask_qtys=ask_qtys,
        bid_prices=bid_prices,
        bid_qtys=bid_qtys,
        long_estimated_rate=0.0,
        short_estimated_rate=0.0,
        commission_rate=0.001,
        leverage=1,
        position=2,
        initial_margine=200.0,
        unrealized_pnL=0.0,
        wallet_balance=1.0,
        leverage_choices=[1, 2],
        position_choices=[-100, -2, 0, 2, 100],
        allow_reverse_position=True,
    )
    
    assert -100 not in pos_choices


def test_create_optimal_q_table_reverse_position():
    n = 3
    ask_prices = np.tile([100.0, 101.0], (n, 1))
    bid_prices = np.tile([99.0, 98.0], (n, 1))
    ask_qtys = np.tile([10.0, 10.0], (n, 1))
    bid_qtys = np.tile([10.0, 10.0], (n, 1))
    markprices = np.array([100.0, 101.0, 102.0])
    timestamps = np.array([1, 2, 3])
    funding_rates = np.array([0.0, 0.0, 0.0])
    funding_timestamps = np.array([0, 0, 0])
    
    q_table_disabled = create_optimal_q_table(
        ask_prices, bid_prices, ask_qtys, bid_qtys,
        markprices, timestamps, funding_rates, funding_timestamps,
        max_holding_number=2, position_choices=3, leverage_choice=[1],
        allow_reverse_position=False,
    )
    assert q_table_disabled[-2, 0, 2] == -1e10

    q_table_enabled = create_optimal_q_table(
        ask_prices, bid_prices, ask_qtys, bid_qtys,
        markprices, timestamps, funding_rates, funding_timestamps,
        max_holding_number=2, position_choices=3, leverage_choice=[1],
        allow_reverse_position=True,
    )
    assert q_table_enabled[-2, 0, 2] != -1e10


def test_base_env_reverse_position():
    n = 5
    state_array = np.zeros((n, 10))
    ask_prices = np.tile([100.0, 101.0], (n, 1))
    bid_prices = np.tile([99.0, 98.0], (n, 1))
    ask_qtys = np.tile([10.0, 10.0], (n, 1))
    bid_qtys = np.tile([10.0, 10.0], (n, 1))
    markprices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    timestamps = np.array(["2025-01-01T00:00:00", "2025-01-01T00:01:00", "2025-01-01T00:02:00", "2025-01-01T00:03:00", "2025-01-01T00:04:00"], dtype="datetime64[ns]")
    funding_rates = np.zeros(n)
    funding_timestamps = timestamps

    env = Base_Env(
        state_array, ask_prices, bid_prices, ask_qtys, bid_qtys,
        markprices, timestamps, funding_rates, funding_timestamps,
        max_holding_number=2, position_choices=3, leverage_choice=[1],
        allow_reverse_position=True,
    )
    
    state, info = env.reset()
    state, reward, done, info = env.step(2)
    assert env.position == 2.0
    
    state, reward, done, info = env.step(0)
    assert env.position == -2.0
    assert env.single_holding_return == 0.0 or abs(env.single_holding_return) < 10.0


def test_demo_env_reverse_position():
    n = 5
    state_array = np.zeros((n, 10))
    ask_prices = np.tile([100.0, 101.0], (n, 1))
    bid_prices = np.tile([99.0, 98.0], (n, 1))
    ask_qtys = np.tile([10.0, 10.0], (n, 1))
    bid_qtys = np.tile([10.0, 10.0], (n, 1))
    markprices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    timestamps = np.array(["2025-01-01T00:00:00", "2025-01-01T00:01:00", "2025-01-01T00:02:00", "2025-01-01T00:03:00", "2025-01-01T00:04:00"], dtype="datetime64[ns]")
    funding_rates = np.zeros(n)
    funding_timestamps = timestamps

    env = Demo_Env(
        state_array, ask_prices, bid_prices, ask_qtys, bid_qtys,
        markprices, timestamps, funding_rates, funding_timestamps,
        max_holding_number=2, position_choices=3, leverage_choice=[1],
        allow_reverse_position=True,
    )
    
    state, info = env.reset()
    assert "q_value" in info
    state, reward, done, info = env.step(2)
    assert env.position == 2.0
    state, reward, done, info = env.step(0)
    assert env.position == -2.0

from FineFT.env.env_initiate.base_initiate import initiate_base_env
from FineFT.env.env_initiate.demo_initiate import initiate_demo_env
from FineFT.env.env_class.commodity_env import Commodity_Env
from FineFT.env.env_class.agg_env import Agg_Env
from FineFT.RL.DiHFT.low_level.pretrain_qtable_diagnostics import create_demo_env

def test_initiate_base_env_reverse_position():
    df = pd.DataFrame({
        "mark_price": [100.0, 101.0],
        "timestamp": pd.to_datetime(["2025-01-01T00:00:00", "2025-01-01T00:01:00"]),
        "funding_rate": [0.0, 0.0],
        "funding_timestamp": pd.to_datetime(["2025-01-01T00:00:00", "2025-01-01T00:01:00"]),
        "ask1_price": [100.0, 101.0], "ask1_size": [10.0, 10.0],
        "bid1_price": [99.0, 98.0], "bid1_size": [10.0, 10.0],
        "feat": [0.0, 0.0],
    })
    for i in range(2, 26):
        df[f"ask{i}_price"] = 100.0
        df[f"ask{i}_size"] = 10.0
        df[f"bid{i}_price"] = 99.0
        df[f"bid{i}_size"] = 10.0

    env = initiate_base_env(df, ["feat"], max_holding_number=2, position_choices=3, leverage_choice=[1], allow_reverse_position=True)
    assert env.allow_reverse_position is True

def test_initiate_demo_env_reverse_position():
    df = pd.DataFrame({
        "mark_price": [100.0, 101.0],
        "timestamp": pd.to_datetime(["2025-01-01T00:00:00", "2025-01-01T00:01:00"]),
        "funding_rate": [0.0, 0.0],
        "funding_timestamp": pd.to_datetime(["2025-01-01T00:00:00", "2025-01-01T00:01:00"]),
        "ask1_price": [100.0, 101.0], "ask1_size": [10.0, 10.0],
        "bid1_price": [99.0, 98.0], "bid1_size": [10.0, 10.0],
        "feat": [0.0, 0.0],
    })
    for i in range(2, 26):
        df[f"ask{i}_price"] = 100.0
        df[f"ask{i}_size"] = 10.0
        df[f"bid{i}_price"] = 99.0
        df[f"bid{i}_size"] = 10.0

    env = initiate_demo_env(df, ["feat"], max_holding_number=2, position_choices=3, leverage_choice=[1], allow_reverse_position=True)
    assert env.allow_reverse_position is True
    assert env.q_table[-2, 0, 2] != -1e10

def test_pretrain_qtable_diagnostics_create_demo_env():
    df = pd.DataFrame({
        "mark_price": [100.0, 101.0],
        "timestamp": pd.to_datetime(["2025-01-01T00:00:00", "2025-01-01T00:01:00"]),
        "funding_rate": [0.0, 0.0],
        "funding_timestamp": pd.to_datetime(["2025-01-01T00:00:00", "2025-01-01T00:01:00"]),
        "ask1_price": [100.0, 101.0], "ask1_size": [10.0, 10.0],
        "bid1_price": [99.0, 98.0], "bid1_size": [10.0, 10.0],
        "feat": [0.0, 0.0],
    })
    for i in range(2, 26):
        df[f"ask{i}_price"] = 100.0
        df[f"ask{i}_size"] = 10.0
        df[f"bid{i}_price"] = 99.0
        df[f"bid{i}_size"] = 10.0

    env_kwargs = {
        "feature_list": ["feat"],
        "max_holding_number": 2,
        "order_book_depth": 25,
        "position_choices": 3,
        "leverage_choices": [1],
        "long_estimated_rate": 0.0,
        "short_estimated_rate": 0.0,
        "commission_rate": 0.0001,
        "maintenance_margin_ratio_dict": {"50000": [0.004, 0]},
        "early_stop": 0,
        "gamma": 1,
        "allow_reverse_position": True,
    }
    initial_state = (1e5, 0, 0, 0, 1)
    env = create_demo_env(df, env_kwargs, initial_state)
    assert env.allow_reverse_position is True
