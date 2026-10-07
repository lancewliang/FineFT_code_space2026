import numpy as np
import pandas as pd
import pytest

from env.env_initiate.base_initiate import initiate_base_env
from env.env_initiate.demo_initiate import initiate_demo_env
from env.env_class.futures_util import (
    create_optimal_q_table,
    create_optimal_q_table_from_df,
)
from RL.DiHFT.low_level.qtable_config import build_optimal_qtable_kwargs


def _make_dummy_df(num_rows=10):
    data = {
        "mark_price": np.full(num_rows, 100.0),
        "timestamp": pd.date_range("2026-01-01", periods=num_rows, freq="1min"),
        "funding_rate": np.zeros(num_rows),
        "funding_timestamp": pd.date_range("2026-01-01", periods=num_rows, freq="1min"),
        "feat1": np.zeros(num_rows),
    }
    for i in range(1, 26):
        data[f"bid{i}_price"] = np.full(num_rows, 99.0)
        data[f"ask{i}_price"] = np.full(num_rows, 101.0)
        data[f"bid{i}_size"] = np.full(num_rows, 10.0)
        data[f"ask{i}_size"] = np.full(num_rows, 10.0)
    return pd.DataFrame(data)


def test_base_env_step_turnover_penalty_reward_shaping():
    df = _make_dummy_df(num_rows=10)
    # position_choices=3, max_holding_number=1, leverage=[1] -> actions:
    # 0: short (-1.0), 1: flat (0.0), 2: long (+1.0)
    penalty_rate = 0.0002

    env_plain = initiate_base_env(
        df,
        feature_list=["feat1"],
        max_holding_number=1,
        position_choices=3,
        leverage_choice=[1],
        initial_state=(1e5, 0.0, 0.0, 0.0, 1),
        turnover_penalty_rate=0.0,
    )
    env_penalized = initiate_base_env(
        df,
        feature_list=["feat1"],
        max_holding_number=1,
        position_choices=3,
        leverage_choice=[1],
        initial_state=(1e5, 0.0, 0.0, 0.0, 1),
        turnover_penalty_rate=penalty_rate,
    )

    env_plain.reset()
    env_penalized.reset()

    # Step 1: Open long position (pos 0 -> 1). |Delta pos| = 1.0.
    # Expected penalty = 0.0002 * 1.0 * 100.0 = 0.02
    _, r_plain_1, _, _ = env_plain.step(2)
    _, r_pen_1, _, _ = env_penalized.step(2)

    assert r_pen_1 == pytest.approx(r_plain_1 - 0.02)
    # Verify wallet_balance is untouched by turnover penalty (pure reward shaping)
    assert env_penalized.wallet_balance == pytest.approx(env_plain.wallet_balance)

    # Step 2: Maintain long position (pos 1 -> 1). |Delta pos| = 0.0.
    # Penalty must be 0.0!
    _, r_plain_2, _, _ = env_plain.step(2)
    _, r_pen_2, _, _ = env_penalized.step(2)

    assert r_pen_2 == pytest.approx(r_plain_2)
    assert env_penalized.wallet_balance == pytest.approx(env_plain.wallet_balance)

    # Step 3: Close long position to flat (pos 1 -> 0). |Delta pos| = 1.0.
    # Expected penalty = 0.0002 * 1.0 * 100.0 = 0.02
    _, r_plain_3, _, _ = env_plain.step(1)
    _, r_pen_3, _, _ = env_penalized.step(1)

    assert r_pen_3 == pytest.approx(r_plain_3 - 0.02)
    assert env_penalized.wallet_balance == pytest.approx(env_plain.wallet_balance)


def test_create_optimal_q_table_turnover_penalty_synchronization():
    n = 3
    ask_prices = np.tile([100.0, 101.0], (n, 1))
    bid_prices = np.tile([99.0, 98.0], (n, 1))
    ask_qtys = np.tile([10.0, 10.0], (n, 1))
    bid_qtys = np.tile([10.0, 10.0], (n, 1))
    markprices = np.full(n, 100.0)
    timestamps = np.arange(1, n + 1)
    funding_rates = np.zeros(n)
    funding_timestamps = np.zeros(n)

    common_kwargs = dict(
        max_holding_number=2,
        position_choices=3,
        leverage_choice=[1],
        commission_rate=0.0,
        long_estimated_rate=0.0,
        short_estimated_rate=0.0,
        gamma=1.0,
    )
    # action mapping: 0 -> short (-2), 1 -> flat (0), 2 -> long (+2)
    penalty_rate = 0.0002

    q_plain = create_optimal_q_table(
        ask_prices,
        bid_prices,
        ask_qtys,
        bid_qtys,
        markprices,
        timestamps,
        funding_rates,
        funding_timestamps,
        turnover_penalty_rate=0.0,
        **common_kwargs,
    )

    q_penalized = create_optimal_q_table(
        ask_prices,
        bid_prices,
        ask_qtys,
        bid_qtys,
        markprices,
        timestamps,
        funding_rates,
        funding_timestamps,
        turnover_penalty_rate=penalty_rate,
        **common_kwargs,
    )

    # At penultimate step (index 1), future step is terminal step (index 2, max Q future is 0)
    # 1) Transition 1 -> 1 (flat -> flat): delta pos = 0 -> Q should be identical
    q_plain_stay = q_plain[1, 1, 1]
    q_pen_stay = q_penalized[1, 1, 1]
    assert q_pen_stay == pytest.approx(q_plain_stay)

    # 2) Transition 1 -> 2 (flat -> long +2): delta pos = 2.0 -> penalty = 0.0002 * 2.0 * 100.0 = 0.04
    q_plain_turnover = q_plain[1, 1, 2]
    q_pen_turnover = q_penalized[1, 1, 2]
    assert q_pen_turnover == pytest.approx(q_plain_turnover - 0.04)


def test_create_optimal_q_table_from_df_forwards_turnover_penalty():
    df = _make_dummy_df(num_rows=5)
    q_table = create_optimal_q_table_from_df(
        df,
        max_holding_number=2,
        position_choices=3,
        leverage_choice=[1],
        turnover_penalty_rate=0.0002,
    )
    assert isinstance(q_table, np.ndarray)
    assert q_table.shape[0] == len(df)


def test_build_optimal_qtable_kwargs_includes_turnover_penalty():
    kwargs = build_optimal_qtable_kwargs(
        max_holding_number=1.0,
        order_book_depth=5,
        position_choices=3,
        leverage_choice=[1],
        long_estimated_rate=0.0,
        short_estimated_rate=0.0,
        commission_rate=0.002,
        gamma=0.99,
        allow_reverse_position=False,
        turnover_penalty_rate=0.0002,
    )
    assert kwargs["turnover_penalty_rate"] == pytest.approx(0.0002)


def test_demo_env_turnover_penalty():
    df = _make_dummy_df(num_rows=6)
    env = initiate_demo_env(
        df,
        feature_list=["feat1"],
        max_holding_number=1,
        position_choices=3,
        leverage_choice=[1],
        initial_state=(1e5, 0.0, 0.0, 0.0, 1),
        turnover_penalty_rate=0.0002,
    )
    env.reset()
    # Step with action 2 (open long):
    _, reward, _, info = env.step(2)
    assert "q_value" in info
    assert env.turnover_penalty_rate == pytest.approx(0.0002)
