import numpy as np
import pandas as pd
import pytest

from FineFT.env.env_class.base_env import Base_Env


def _make_sample_env(*, allow_reverse_position=True):
    n_rows = 10
    state_array = np.zeros((n_rows, 5), dtype=np.float32)
    ask_prices = np.tile(np.array([100.0, 101.0, 102.0, 103.0, 104.0]), (n_rows, 1))
    bid_prices = np.tile(np.array([99.0, 98.0, 97.0, 96.0, 95.0]), (n_rows, 1))
    ask_qtys = np.tile(np.array([10.0, 20.0, 30.0, 40.0, 50.0]), (n_rows, 1))
    bid_qtys = np.tile(np.array([10.0, 20.0, 30.0, 40.0, 50.0]), (n_rows, 1))
    markprices = np.full(n_rows, 100.0)
    timestamps = pd.to_datetime(["2026-01-01 09:00:00"] * n_rows).to_numpy()
    funding_rates = np.zeros(n_rows)
    funding_timestamps = timestamps.copy()

    env = Base_Env(
        state_array=state_array,
        ask_prices_array=ask_prices,
        bid_prices_array=bid_prices,
        ask_qtys_array=ask_qtys,
        bid_qtys_array=bid_qtys,
        markprice_array=markprices,
        timestamp_array=timestamps,
        funding_rate_array=funding_rates,
        funding_timestamp_array=funding_timestamps,
        max_holding_number=4,
        position_choices=5,
        leverage_choice=[1],
        commission_rate=0.0004,
        allow_reverse_position=allow_reverse_position,
        initial_state=(1e5, 0, 0, 0, 1),
    )
    return env


def test_estimate_action_costs_side_effect_free():
    env = _make_sample_env()
    state, info = env.reset()

    initial_wallet = env.wallet_balance
    initial_pos = env.position
    initial_margin = env.initial_margin
    initial_pnl = env.unrealized_pnl
    initial_history_len = len(env.micro_action_history)

    costs = env.estimate_action_transaction_costs()

    assert len(costs) == env.action_space.n
    assert (costs >= 0.0).all()
    # Side-effect free verification:
    assert env.wallet_balance == initial_wallet
    assert env.position == initial_pos
    assert env.initial_margin == initial_margin
    assert env.unrealized_pnl == initial_pnl
    assert len(env.micro_action_history) == initial_history_len


def test_estimate_action_costs_current_action_is_zero():
    env = _make_sample_env()
    state, info = env.reset()
    current_act = info["previous_action"]

    costs = env.estimate_action_transaction_costs()
    assert costs[current_act] == 0.0


def test_estimate_action_costs_matches_step_execution_cost():
    env = _make_sample_env(allow_reverse_position=True)
    state, info = env.reset()

    estimated_costs = info["estimated_costs"]

    # Target action 4 (e.g. open long)
    target_action = 4
    est_cost = estimated_costs[target_action]

    # Perform step
    next_state, reward, done, next_info = env.step(target_action)
    actual_fee = next_info["commission_fee_step"]
    actual_slippage = max(0.0, next_info["slippage_step"])
    actual_cost = actual_fee + actual_slippage

    assert est_cost == pytest.approx(actual_cost, rel=1e-5, abs=1e-5)
