import numpy as np
import pandas as pd
import pytest

from env.env_class.futures_util import (
    create_optimal_q_table,
    create_optimal_q_table_from_df,
    get_dp_action_from_qtable,
)
from RL.DiHFT.low_level.qtable_config import build_optimal_qtable_kwargs


def _make_dummy_df(num_rows=10, mark_price=100.0, regime_grid_id=2):
    data = {
        "mark_price": np.full(num_rows, mark_price),
        "timestamp": pd.date_range("2026-01-01", periods=num_rows, freq="1min"),
        "funding_rate": np.zeros(num_rows),
        "funding_timestamp": pd.date_range("2026-01-01", periods=num_rows, freq="1min"),
        "regime_grid_id": np.full(num_rows, regime_grid_id, dtype=int),
    }
    for i in range(1, 26):
        data[f"bid{i}_price"] = np.full(num_rows, mark_price - 1.0)
        data[f"ask{i}_price"] = np.full(num_rows, mark_price + 1.0)
        data[f"bid{i}_size"] = np.full(num_rows, 10.0)
        data[f"ask{i}_size"] = np.full(num_rows, 10.0)
    return pd.DataFrame(data)


def test_qtable_kwargs_builder_supports_trend_entry_lock():
    kwargs = build_optimal_qtable_kwargs(
        max_holding_number=1,
        order_book_depth=25,
        position_choices=3,
        leverage_choice=[1],
        long_estimated_rate=0.0,
        short_estimated_rate=0.0,
        commission_rate=0.0,
        gamma=1.0,
        allow_reverse_position=False,
        enable_trend_entry_lock=True,
    )
    assert kwargs["enable_trend_entry_lock"] is True


def test_create_optimal_q_table_bull_regime_locks_short_entry():
    # Actions: 0: short (-1), 1: flat (0), 2: long (+1)
    # regime_grid_id = 8 -> slope_bin = 8 % 3 = 2 (Bull)
    df = _make_dummy_df(num_rows=5, regime_grid_id=8)
    q_table = create_optimal_q_table_from_df(
        df,
        max_holding_number=1,
        position_choices=3,
        leverage_choice=[1],
        enable_trend_entry_lock=True,
    )

    t_idx = 0
    # From flat (1), opening short (0) must be locked out
    assert q_table[t_idx, 1, 0] <= -1e9
    # From long (2), reversing to short (0) must be locked out
    assert q_table[t_idx, 2, 0] <= -1e9
    # From long (2), closing to flat (1) must NOT be locked out
    assert q_table[t_idx, 2, 1] > -1e9
    # From flat (1), opening long (2) must NOT be locked out
    assert q_table[t_idx, 1, 2] > -1e9


def test_create_optimal_q_table_bear_regime_locks_long_entry():
    # Actions: 0: short (-1), 1: flat (0), 2: long (+1)
    # regime_grid_id = 6 -> slope_bin = 6 % 3 = 0 (Bear)
    df = _make_dummy_df(num_rows=5, regime_grid_id=6)
    q_table = create_optimal_q_table_from_df(
        df,
        max_holding_number=1,
        position_choices=3,
        leverage_choice=[1],
        enable_trend_entry_lock=True,
    )

    t_idx = 0
    # From flat (1), opening long (2) must be locked out
    assert q_table[t_idx, 1, 2] <= -1e9
    # From short (0), reversing to long (2) must be locked out
    assert q_table[t_idx, 0, 2] <= -1e9
    # From short (0), closing to flat (1) must NOT be locked out
    assert q_table[t_idx, 0, 1] > -1e9
    # From flat (1), opening short (0) must NOT be locked out
    assert q_table[t_idx, 1, 0] > -1e9


def test_dp_action_path_in_bull_produces_no_short_actions():
    # Prices dip slightly in step 2 then rise, but bull regime is locked
    df = _make_dummy_df(num_rows=5, regime_grid_id=2)
    # Give a minor dip: 100 -> 98 -> 105
    df.loc[1, "mark_price"] = 98.0
    df.loc[2, "mark_price"] = 105.0
    q_table = create_optimal_q_table_from_df(
        df,
        max_holding_number=1,
        position_choices=3,
        leverage_choice=[1],
        enable_trend_entry_lock=True,
    )
    # Initial action: 1 (flat)
    action_path = get_dp_action_from_qtable(q_table, initial_action=1)
    # Action 0 is short, it should never appear
    assert 0 not in action_path
