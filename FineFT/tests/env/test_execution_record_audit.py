import numpy as np
import pytest
from env.env_class.futures_util import (
    WalletChangeResult,
    change_of_wallet,
)
from env.env_class.base_env import Base_Env
import pandas as pd
from RL.DiHFT.high_level.vae_routing_util import vae_risk_aware_routing
from common import ActionDecisionReasons


def test_wallet_change_result_tracks_execution_details():
    ask_prices = np.array([100.0, 101.0, 102.0])
    ask_qtys = np.array([10.0, 10.0, 10.0])
    bid_prices = np.array([99.0, 98.0, 97.0])
    bid_qtys = np.array([10.0, 10.0, 10.0])

    # 1. Open Long
    res_open = change_of_wallet(
        markprice=100.0,
        ask_prices=ask_prices,
        ask_qtys=ask_qtys,
        bid_prices=bid_prices,
        bid_qtys=bid_qtys,
        long_estimated_rate=0.0,
        short_estimated_rate=0.0,
        commission_rate=0.0003,
        previous_leverage=1,
        previous_position=0,
        previous_initial_margine=0.0,
        previous_unrealized_pnL=0.0,
        previous_wallet_balance=1000.0,
        current_leverage=1,
        current_position=1,
        silent=True,
    )
    assert res_open.position == 1
    assert res_open.opened_quantity == 1.0
    assert res_open.opened_value == 100.0
    assert np.isclose(res_open.opening_fee, 100.0 * 0.0003)
    assert res_open.reference_price == 100.0

    # 2. Close Long
    res_close = change_of_wallet(
        markprice=105.0,
        ask_prices=ask_prices + 5.0,
        ask_qtys=ask_qtys,
        bid_prices=bid_prices + 5.0,
        bid_qtys=bid_qtys,
        long_estimated_rate=0.0,
        short_estimated_rate=0.0,
        commission_rate=0.0003,
        previous_leverage=1,
        previous_position=1,
        previous_initial_margine=105.0,
        previous_unrealized_pnL=5.0,
        previous_wallet_balance=1000.0 - res_open.opening_fee,
        current_leverage=1,
        current_position=0,
        silent=True,
    )
    assert res_close.position == 0
    assert res_close.closed_quantity == 1.0
    # bid[0] is 99 + 5 = 104
    assert res_close.closed_value == 104.0
    assert np.isclose(res_close.closing_fee, 104.0 * 0.0003)
    assert res_close.reference_price == 105.0


def test_base_env_exposes_execution_snapshot():
    n = 5
    state_array = np.zeros((n, 10))
    ask_prices = np.tile([100.0, 101.0], (n, 1))
    bid_prices = np.tile([99.0, 98.0], (n, 1))
    ask_qtys = np.tile([10.0, 10.0], (n, 1))
    bid_qtys = np.tile([10.0, 10.0], (n, 1))
    markprices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    timestamps = np.array([f"2025-01-01T00:0{i}:00" for i in range(n)], dtype="datetime64[ns]")
    funding_rates = np.zeros(n)
    funding_timestamps = timestamps

    env = Base_Env(
        state_array, ask_prices, bid_prices, ask_qtys, bid_qtys,
        markprices, timestamps, funding_rates, funding_timestamps,
        max_holding_number=1, position_choices=3, leverage_choice=[1],
    )
    env.reset()
    # Action 2 is position 1.0 (open long)
    state, reward, done, info = env.step(2)
    assert hasattr(env, "last_wallet_change")
    assert hasattr(env, "last_reference_price")
    assert env.last_reference_price == 100.0
    assert env.last_wallet_change.opened_quantity == 1.0
    assert env.last_wallet_change.opened_value == 100.0
    assert np.isclose(env.last_wallet_change.opening_fee, 100.0 * env.commission_rate)


def test_record_trade_audit_real_execution_and_fee_conservation():
    n = 5
    state_array = np.zeros((n, 10))
    ask_prices = np.tile([100.0, 101.0], (n, 1))
    bid_prices = np.tile([99.0, 98.0], (n, 1))
    ask_qtys = np.tile([10.0, 10.0], (n, 1))
    bid_qtys = np.tile([10.0, 10.0], (n, 1))
    markprices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    timestamps = np.array([f"2025-01-01T00:0{i}:00" for i in range(n)], dtype="datetime64[ns]")
    funding_rates = np.zeros(n)
    funding_timestamps = timestamps

    env = Base_Env(
        state_array, ask_prices, bid_prices, ask_qtys, bid_qtys,
        markprices, timestamps, funding_rates, funding_timestamps,
        max_holding_number=1, position_choices=3, leverage_choice=[1],
        commission_rate=0.0003,
    )
    env.reset()

    router = object.__new__(vae_risk_aware_routing)
    router.run_id = "test_run"
    router.current_contract_name = "fu2509"
    router.audit_trades = []
    router.df = pd.DataFrame({
        "timestamp": timestamps,
        "trading_day": ["2025-01-01"] * n,
    })

    # Step 0: Open Long (action 2: position 1.0).
    # Decision markprice is 100.0, next markprice will be 101.0
    old_pos = float(env.position)
    env.step(2)
    router._record_trade_audit(
        env=env,
        step_index=0,
        action=2,
        old_position=old_pos,
        target_position=1.0,
        decision_reason=int(ActionDecisionReasons.POLICY_INFERENCE),
        prediction_id="pred_0",
    )

    assert len(router.audit_trades) == 1
    t_open = router.audit_trades[0]
    assert t_open["event_type"] == "open"
    assert t_open["reference_price"] == 100.0  # Decision-time price, NOT 101.0
    assert t_open["fill_price"] == 100.0      # Ask price executed
    assert t_open["executed_quantity"] == 1.0
    expected_fee_open = abs(t_open["fill_price"] * t_open["executed_quantity"]) * 0.0003
    assert np.isclose(t_open["commission"], expected_fee_open, atol=1e-7)

    # Step 1: Close Long (action 1: position 0.0).
    # Decision markprice is 101.0, next markprice will be 102.0
    old_pos = float(env.position)
    env.step(1)
    router._record_trade_audit(
        env=env,
        step_index=1,
        action=1,
        old_position=old_pos,
        target_position=0.0,
        decision_reason=int(ActionDecisionReasons.POLICY_INFERENCE),
        prediction_id="pred_1",
    )

    assert len(router.audit_trades) == 2
    t_close = router.audit_trades[1]
    assert t_close["event_type"] == "close"
    assert t_close["reference_price"] == 101.0  # Decision-time price, NOT 102.0
    assert t_close["fill_price"] == 99.0       # Bid price executed
    assert t_close["executed_quantity"] == 1.0
    expected_fee_close = abs(t_close["fill_price"] * t_close["executed_quantity"]) * 0.0003
    assert np.isclose(t_close["commission"], expected_fee_close, atol=1e-7)

    # Reversal trade test (mocked wallet change with split legs)
    wc_reversal = WalletChangeResult(
        leverage=1, position=-1.0, initial_margin=100.0,
        unrealized_pnl=0.0, wallet_balance=900.0,
        slippage_step=0.0, commission_fee_step=0.06, realized_pnl_step=5.0,
        closed_quantity=1.0, closed_value=99.0, closing_fee=99.0 * 0.0003,
        opened_quantity=1.0, opened_value=101.0, opening_fee=101.0 * 0.0003,
        reference_price=100.0,
    )
    env.last_wallet_change = wc_reversal
    env.last_reference_price = 100.0
    env.position = -1.0
    env.commission_fee_step = wc_reversal.commission_fee_step
    env.realized_pnl_step = wc_reversal.realized_pnl_step

    router._record_trade_audit(
        env=env,
        step_index=2,
        action=0,
        old_position=1.0,
        target_position=-1.0,
        decision_reason=int(ActionDecisionReasons.POLICY_INFERENCE),
        prediction_id="pred_2",
    )

    assert len(router.audit_trades) == 4
    rev_close = router.audit_trades[2]
    rev_open = router.audit_trades[3]
    assert rev_close["event_type"] == "reversal_close"
    assert rev_close["fill_price"] == 99.0
    assert np.isclose(rev_close["commission"], abs(99.0 * 1.0) * 0.0003, atol=1e-7)

    assert rev_open["event_type"] == "reversal_open"
    assert rev_open["fill_price"] == 101.0
    assert np.isclose(rev_open["commission"], abs(101.0 * 1.0) * 0.0003, atol=1e-7)
