from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from env.env_initiate.base_initiate import initiate_base_env


def test_account_audit_history_survives_position_closure():
    n_steps = 10
    mark_prices = np.linspace(100.0, 110.0, n_steps + 1)
    timestamps = pd.date_range("2025-01-02 09:30:00", periods=n_steps + 1, freq="10min")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "funding_rate": np.zeros(n_steps + 1),
        "funding_timestamp": timestamps,
        "mark_price": mark_prices,
        "ask1_price": mark_prices + 0.1,
        "bid1_price": mark_prices - 0.1,
        "ask1_size": np.ones(n_steps + 1) * 100.0,
        "bid1_size": np.ones(n_steps + 1) * 100.0,
    })

    env = initiate_base_env(
        df=df,
        feature_list=["mark_price"],
        max_holding_number=1,
        position_choices=3,  # [-1, 0, 1]
        leverage_choice=[1],
        long_estimated_rate=0.0,
        short_estimated_rate=0.0,
        commission_rate=0.0001,
        initial_state=(10000.0, 0, 0, 0, 1),
        order_book_depth=1,
    )

    s, info = env.reset()
    assert hasattr(env, "account_audit_history")
    assert len(env.account_audit_history) == 1
    init_snap = env.account_audit_history[0]
    assert init_snap["snapshot_type"] == "initial"
    assert init_snap["wallet_balance"] == 10000.0
    assert init_snap["equity"] == 10000.0
    assert init_snap["position"] == 0.0

    # Step 0: Open Long (action 2)
    s, r, done, info = env.step(2)
    assert len(env.account_audit_history) == 2
    snap1 = env.account_audit_history[-1]
    assert snap1["snapshot_type"] == "post_step"
    assert snap1["position"] == 1.0

    # Step 1: Hold Long (action 2)
    s, r, done, info = env.step(2)
    assert len(env.account_audit_history) == 3
    snap2 = env.account_audit_history[-1]
    assert snap2["position"] == 1.0

    # Step 2: Close position to Flat (action 1) -> position becomes 0!
    s, r, done, info = env.step(1)
    assert len(env.account_audit_history) == 4
    snap3 = env.account_audit_history[-1]
    assert snap3["position"] == 0.0

    # Audit history MUST survive position closure (not reset to length 1!)
    assert len(env.account_audit_history) == 4

    # Step 3: Open Short (action 0)
    s, r, done, info = env.step(0)
    assert len(env.account_audit_history) == 5
    snap4 = env.account_audit_history[-1]
    assert snap4["position"] == -1.0

    # Step 4: Close short to Flat (action 1)
    s, r, done, info = env.step(1)
    assert len(env.account_audit_history) == 6
    snap5 = env.account_audit_history[-1]
    assert snap5["position"] == 0.0

    # Verify all snapshots have equity = wallet_balance + unrealized_pnl
    for snap in env.account_audit_history:
        assert snap["equity"] == pytest.approx(snap["wallet_balance"] + snap["unrealized_pnl"], abs=1e-6)
