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
from FineFT.env.env_initiate.base_initiate import initiate_base_env
from FineFT.env.env_initiate.demo_initiate import initiate_demo_env


def test_change_of_wallet_rejects_long_to_short():
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

    # Single-step reverse position is prohibited; stays in previous situation
    assert result.position == 2
    assert result.leverage == 1
    assert result.wallet_balance == 1000.0
    assert result.realized_pnl_step == 0
    assert result.commission_fee_step == 0


def test_change_of_wallet_rejects_short_to_long():
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
        previous_position=-2,
        previous_initial_margine=200.0,
        previous_unrealized_pnL=0.0,
        previous_wallet_balance=1000.0,
        current_leverage=1,
        current_position=2,
        silent=True,
    )

    # Single-step reverse position is prohibited; stays in previous situation
    assert result.position == -2
    assert result.leverage == 1
    assert result.wallet_balance == 1000.0
    assert result.realized_pnl_step == 0
    assert result.commission_fee_step == 0


def test_calculate_available_action_forbids_reverse_from_long():
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
    )

    # No negative (short) positions should be available directly from long
    assert all(p >= 0 for p in pos_choices)


def test_calculate_available_action_forbids_reverse_from_short():
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
        position=-2,
        initial_margine=200.0,
        unrealized_pnL=0.0,
        wallet_balance=1000.0,
        leverage_choices=[1, 2],
        position_choices=[-4, -2, 0, 2, 4],
    )

    # No positive (long) positions should be available directly from short
    assert all(p <= 0 for p in pos_choices)


def test_create_optimal_q_table_punishes_reverse_actions():
    n = 3
    ask_prices = np.tile([100.0, 101.0], (n, 1))
    bid_prices = np.tile([99.0, 98.0], (n, 1))
    ask_qtys = np.tile([10.0, 10.0], (n, 1))
    bid_qtys = np.tile([10.0, 10.0], (n, 1))
    markprices = np.array([100.0, 101.0, 102.0])
    timestamps = np.array(["2025-01-01T00:00:00", "2025-01-01T00:01:00", "2025-01-01T00:02:00"], dtype="datetime64[ns]")
    funding_rates = np.zeros(n)
    funding_timestamps = timestamps

    q_table = create_optimal_q_table(
        ask_prices, bid_prices, ask_qtys, bid_qtys, markprices,
        timestamps, funding_rates, funding_timestamps,
        max_holding_number=2, position_choices=3, leverage_choice=[1],
        max_punishment=1e10,
    )

    # Action 0 = position -2, Action 1 = position 0, Action 2 = position 2
    # Transition from position 2 (action 2) to position -2 (action 0) must be punished
    assert q_table[0, 2, 0] == -1e10
    # Transition from position -2 (action 0) to position 2 (action 2) must be punished
    assert q_table[0, 0, 2] == -1e10


def test_base_env_rejects_single_step_reverse():
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
    )

    state, info = env.reset()
    # Action 2 corresponds to position 2
    state, reward, done, info = env.step(2)
    assert env.position == 2.0

    # Attempting to directly switch to Action 0 (position -2) is rejected by change_of_wallet
    state, reward, done, info = env.step(0)
    assert env.position == 2.0


def test_demo_env_qtable_punishes_reverse():
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
    )

    state, info = env.reset()
    assert "q_value" in info
    # Reversal actions in Q table are punished
    assert env.q_table[0, 2, 0] == -1e10
    assert env.q_table[0, 0, 2] == -1e10


def test_initiate_base_env():
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

    env = initiate_base_env(df, ["feat"], max_holding_number=2, position_choices=3, leverage_choice=[1])
    with pytest.raises(AttributeError):
        _ = env.allow_reverse_position
