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


from env.env_class.futures_util import compute_directional_turnover_penalty_rate


def test_compute_directional_turnover_penalty_rate_uptrend():
    base = 0.0001
    ratio = 6.0
    adverse = 0.0006
    # Regime grid 2, 5, 8 are uptrend (slope_bin == 2)
    for grid in [2, 5, 8]:
        # Open long: 0 -> 1 (aligned)
        assert compute_directional_turnover_penalty_rate(0.0, 1.0, grid, base, ratio) == pytest.approx(base)
        # Add long: 0.5 -> 1.0 (aligned)
        assert compute_directional_turnover_penalty_rate(0.5, 1.0, grid, base, ratio) == pytest.approx(base)
        # Reverse short to long: -1 -> 1 (aligned correction)
        assert compute_directional_turnover_penalty_rate(-1.0, 1.0, grid, base, ratio) == pytest.approx(base)
        # Close short: -1 -> 0 (aligned correction)
        assert compute_directional_turnover_penalty_rate(-1.0, 0.0, grid, base, ratio) == pytest.approx(base)
        # Premature exit / close long: 1 -> 0 (adverse)
        assert compute_directional_turnover_penalty_rate(1.0, 0.0, grid, base, ratio) == pytest.approx(adverse)
        # Counter-trend reverse: 1 -> -1 (adverse)
        assert compute_directional_turnover_penalty_rate(1.0, -1.0, grid, base, ratio) == pytest.approx(adverse)
        # Counter-trend open short: 0 -> -1 (adverse)
        assert compute_directional_turnover_penalty_rate(0.0, -1.0, grid, base, ratio) == pytest.approx(adverse)


def test_compute_directional_turnover_penalty_rate_downtrend():
    base = 0.0001
    ratio = 6.0
    adverse = 0.0006
    # Regime grid 0, 3, 6 are downtrend (slope_bin == 0)
    for grid in [0, 3, 6]:
        # Open short: 0 -> -1 (aligned)
        assert compute_directional_turnover_penalty_rate(0.0, -1.0, grid, base, ratio) == pytest.approx(base)
        # Add short: -0.5 -> -1.0 (aligned)
        assert compute_directional_turnover_penalty_rate(-0.5, -1.0, grid, base, ratio) == pytest.approx(base)
        # Reverse long to short: 1 -> -1 (aligned correction)
        assert compute_directional_turnover_penalty_rate(1.0, -1.0, grid, base, ratio) == pytest.approx(base)
        # Close long: 1 -> 0 (aligned correction)
        assert compute_directional_turnover_penalty_rate(1.0, 0.0, grid, base, ratio) == pytest.approx(base)
        # Premature exit / close short: -1 -> 0 (adverse)
        assert compute_directional_turnover_penalty_rate(-1.0, 0.0, grid, base, ratio) == pytest.approx(adverse)
        # Counter-trend reverse: -1 -> 1 (adverse)
        assert compute_directional_turnover_penalty_rate(-1.0, 1.0, grid, base, ratio) == pytest.approx(adverse)
        # Counter-trend open long: 0 -> 1 (adverse)
        assert compute_directional_turnover_penalty_rate(0.0, 1.0, grid, base, ratio) == pytest.approx(adverse)


def test_compute_directional_turnover_penalty_rate_range():
    base = 0.0001
    ratio = 6.0
    adverse = 0.0006
    # Regime grid 1, 4, 7 are range/flat (slope_bin == 1)
    for grid in [1, 4, 7]:
        # Open long into noise: 0 -> 1 (adverse)
        assert compute_directional_turnover_penalty_rate(0.0, 1.0, grid, base, ratio) == pytest.approx(adverse)
        # Open short into noise: 0 -> -1 (adverse)
        assert compute_directional_turnover_penalty_rate(0.0, -1.0, grid, base, ratio) == pytest.approx(adverse)
        # Reversal back and forth: 1 -> -1, -1 -> 1 (adverse)
        assert compute_directional_turnover_penalty_rate(1.0, -1.0, grid, base, ratio) == pytest.approx(adverse)
        assert compute_directional_turnover_penalty_rate(-1.0, 1.0, grid, base, ratio) == pytest.approx(adverse)
        # De-risking exit: 1 -> 0, -1 -> 0 (base_rate)
        assert compute_directional_turnover_penalty_rate(1.0, 0.0, grid, base, ratio) == pytest.approx(base)
        assert compute_directional_turnover_penalty_rate(-1.0, 0.0, grid, base, ratio) == pytest.approx(base)


def test_compute_directional_turnover_penalty_rate_fallbacks():
    base = 0.0002
    ratio = 5.0
    # No position change -> 0.0
    assert compute_directional_turnover_penalty_rate(1.0, 1.0, 8, base, ratio) == 0.0
    # Zero base rate -> 0.0
    assert compute_directional_turnover_penalty_rate(0.0, 1.0, 8, 0.0, ratio) == 0.0
    # Missing regime_grid_id -> fallback to base_rate
    assert compute_directional_turnover_penalty_rate(0.0, 1.0, None, base, ratio) == pytest.approx(base)
    # Negative regime_grid_id -> fallback to base_rate
    assert compute_directional_turnover_penalty_rate(0.0, 1.0, -1, base, ratio) == pytest.approx(base)
    # Ratio <= 1.0 -> fallback to base_rate
    assert compute_directional_turnover_penalty_rate(1.0, 0.0, 8, base, 1.0) == pytest.approx(base)


def test_base_env_step_asymmetric_directional_turnover_penalty():
    df = _make_dummy_df(num_rows=10)
    # Assign regime_grid_id: first 5 rows are uptrend (grid 8), last 5 rows are range (grid 4)
    df["regime_grid_id"] = np.array([8, 8, 8, 8, 8, 4, 4, 4, 4, 4], dtype=np.int64)

    base_rate = 0.0001
    adverse_ratio = 6.0  # adverse_rate = 0.0006

    env_plain = initiate_base_env(
        df,
        feature_list=["feat1"],
        max_holding_number=1,
        position_choices=3,
        leverage_choice=[1],
        initial_state=(1e5, 0.0, 0.0, 0.0, 1),
        turnover_base_rate=0.0,
    )
    env_asym = initiate_base_env(
        df,
        feature_list=["feat1"],
        max_holding_number=1,
        position_choices=3,
        leverage_choice=[1],
        initial_state=(1e5, 0.0, 0.0, 0.0, 1),
        turnover_base_rate=base_rate,
        turnover_adverse_ratio=adverse_ratio,
    )

    env_plain.reset()
    env_asym.reset()

    # Step 1 (Uptrend grid 8): Open long (0 -> 1). Aligned -> base_rate (0.0001)
    # Penalty = 0.0001 * 1.0 * 100.0 = 0.01
    _, r_plain_1, _, _ = env_plain.step(2)
    _, r_asym_1, _, _ = env_asym.step(2)
    assert r_asym_1 == pytest.approx(r_plain_1 - 0.01)

    # Step 2 (Uptrend grid 8): Close long (1 -> 0). Adverse early exit -> adverse_rate (0.0006)
    # Penalty = 0.0006 * 1.0 * 100.0 = 0.06
    _, r_plain_2, _, _ = env_plain.step(1)
    _, r_asym_2, _, _ = env_asym.step(1)
    assert r_asym_2 == pytest.approx(r_plain_2 - 0.06)


def test_create_optimal_q_table_asymmetric_directional_synchronization():
    n = 3
    ask_prices = np.tile([100.0, 101.0], (n, 1))
    bid_prices = np.tile([99.0, 98.0], (n, 1))
    ask_qtys = np.tile([10.0, 10.0], (n, 1))
    bid_qtys = np.tile([10.0, 10.0], (n, 1))
    markprices = np.full(n, 100.0)
    timestamps = np.arange(1, n + 1)
    funding_rates = np.zeros(n)
    funding_timestamps = np.zeros(n)
    # Grid 8 (uptrend)
    regimes = np.full(n, 8, dtype=np.int64)

    common_kwargs = dict(
        max_holding_number=2,
        position_choices=3,
        leverage_choice=[1],
        commission_rate=0.0,
        long_estimated_rate=0.0,
        short_estimated_rate=0.0,
        gamma=1.0,
        regime_grid_ids_array=regimes,
    )
    base_rate = 0.0001
    ratio = 6.0  # adverse = 0.0006

    q_plain = create_optimal_q_table(
        ask_prices,
        bid_prices,
        ask_qtys,
        bid_qtys,
        markprices,
        timestamps,
        funding_rates,
        funding_timestamps,
        turnover_base_rate=0.0,
        **common_kwargs,
    )

    q_asym = create_optimal_q_table(
        ask_prices,
        bid_prices,
        ask_qtys,
        bid_qtys,
        markprices,
        timestamps,
        funding_rates,
        funding_timestamps,
        turnover_base_rate=base_rate,
        turnover_adverse_ratio=ratio,
        **common_kwargs,
    )

    # In uptrend:
    # 1) Transition 1 -> 2 (flat -> long +2): aligned -> penalty = 0.0001 * 2 * 100 = 0.02
    q_plain_open_long = q_plain[1, 1, 2]
    q_asym_open_long = q_asym[1, 1, 2]
    assert q_asym_open_long == pytest.approx(q_plain_open_long - 0.02)

    # 2) Transition 1 -> 0 (flat -> short -2): counter-trend -> penalty = 0.0006 * 2 * 100 = 0.12
    q_plain_open_short = q_plain[1, 1, 0]
    q_asym_open_short = q_asym[1, 1, 0]
    assert q_asym_open_short == pytest.approx(q_plain_open_short - 0.12)
