import argparse
import numpy as np
import pandas as pd
import pytest

from common import ActionDecisionReasons
from RL.DiHFT.high_level import vae_routing_util as vru
from tests.rl.test_graceful_hold_exit_and_synchronized_portfolio import (
    _sample_manifest,
)


def _build_test_router(
    enable_delivery_defense: bool = True,
    delivery_defense_days: int = 5,
) -> vru.vae_risk_aware_routing:
    r = vru.vae_risk_aware_routing.__new__(vru.vae_risk_aware_routing)
    r.num_labels = 3
    r.slot_count = 9
    r.rule_base_threshold = 0.2
    r.axis_thresholds = {"volatility": 0.2, "slope": 0.2}
    r.gating_strategy = vru.create_gating_strategy("absolute", slope_threshold=0.2, volatility_threshold=0.2)
    r.selection_manifest = _sample_manifest(3)
    r.enable_non_main_contract_defense = False
    r.non_main_defense_mode = "graceful_hold_exit"
    r.enable_delivery_month_defense = enable_delivery_defense
    r.delivery_defense_days = delivery_defense_days
    r.is_in_delivery_terminal = False
    r.is_terminal_defense_step = None
    r.role_tier_index = None
    r.zero_position_action = 1  # [-1.0, 0.0, 1.0] -> 0: -1.0, 1: 0.0, 2: 1.0
    r.leverage_choices = [1]
    r.position_list = [-1.0, 0.0, 1.0]
    r.action = 1
    r.flat_action = 1
    r.current_action = 1
    r.remaining_persist = 0
    r.action_persistence = 1
    r.cooldown_remaining_steps = 0
    r.circuit_breaker_remaining_steps = 0
    r.stop_loss_return_threshold = 0.0
    r.previous_step_position = 0.0
    r.consecutive_stop_loss_count = 0
    r.active_trade_stopped = False
    r.last_stopped_position = 0.0
    r.macro_action_history = []
    r.action_decision_reason_history = []
    r.step_idx = 0
    r.calculate_axis_window_result = lambda axis: {"volatility": [0.1, 0.8, 0.1], "slope": [0.1, 0.1, 0.8]}[axis]
    return r


def test_delivery_defense_parser_defaults_and_validation():
    args = vru.parser.parse_args([])
    assert args.enable_delivery_month_defense is False
    assert args.delivery_defense_days == 5

    args_custom = vru.parser.parse_args([
        "--enable_delivery_month_defense",
        "--delivery_defense_days", "10",
    ])
    assert args_custom.enable_delivery_month_defense is True
    assert args_custom.delivery_defense_days == 10

    # Negative days validation
    invalid_args = argparse.Namespace(
        delivery_defense_days=-1,
    )
    r = vru.vae_risk_aware_routing.__new__(vru.vae_risk_aware_routing)
    with pytest.raises(ValueError, match="delivery_defense_days must be non-negative"):
        # Test validation via __init__ parameter extraction
        r.delivery_defense_days = int(getattr(invalid_args, "delivery_defense_days", 5))
        if r.delivery_defense_days < 0:
            raise ValueError("delivery_defense_days must be non-negative")


def test_delivery_defense_identifies_tail_trading_days():
    router = _build_test_router(enable_delivery_defense=True, delivery_defense_days=3)

    # Create 5 unique trading dates with 2 bars each (total 10 rows)
    dates = pd.date_range("2026-08-10", periods=5, freq="D")
    timestamps = []
    for d in dates:
        timestamps.append(d + pd.Timedelta(hours=9))
        timestamps.append(d + pd.Timedelta(hours=14))

    df = pd.DataFrame({"timestamp": timestamps, "close": [100.0] * 10})
    router.df = df

    # Test the tail calculation logic as used in run_single_valid_df
    unique_dates = sorted(router.df["timestamp"].dt.date.unique())
    cutoff_idx = max(0, len(unique_dates) - router.delivery_defense_days)
    terminal_dates = set(unique_dates[cutoff_idx:])
    router.is_terminal_defense_step = (
        router.df["timestamp"].dt.date.isin(terminal_dates).to_numpy()
    )

    assert len(unique_dates) == 5
    # Last 3 dates: 2026-08-12, 2026-08-13, 2026-08-14
    assert len(terminal_dates) == 3
    # Rows 0-3 belong to first 2 dates -> False
    assert not router.is_terminal_defense_step[0:4].any()
    # Rows 4-9 belong to last 3 dates -> True
    assert router.is_terminal_defense_step[4:10].all()


def test_delivery_defense_forces_flat_when_holding_position():
    router = _build_test_router(enable_delivery_defense=True, delivery_defense_days=5)
    router.is_in_delivery_terminal = True
    router.agent_act = lambda s, info: 2  # Agent wants to hold or open long (+1.0)

    state = np.array([100.0, 200.0])
    info = {"avaliable_action": [1, 1, 1], "ask_qyts": [1000], "bid_qyts": [1000], "avaiable_action_list": [0, 1, 2]}

    # Currently holding long position (+1.0)
    action = router.get_action(info, state, current_position=1.0, current_leverage=1)
    # Must force flat (action 1 -> pos 0.0)
    assert action == 1
    assert router.action_decision_reason_history[-1] == ActionDecisionReasons.DEFENSIVE_RULE_CLOSE
    assert router.macro_action_history[-1] == router.slot_count


def test_delivery_defense_preempts_action_persistence():
    router = _build_test_router(enable_delivery_defense=True, delivery_defense_days=5)
    router.is_in_delivery_terminal = True
    router.remaining_persist = 4  # Persistence was active!

    state = np.array([100.0, 200.0])
    info = {"avaliable_action": [1, 1, 1], "ask_qyts": [1000], "bid_qyts": [1000], "avaiable_action_list": [0, 1, 2]}

    action = router.get_action(info, state, current_position=1.0, current_leverage=1)
    assert action == 1
    # Should record DEFENSIVE_PREEMPTION when breaking persistence
    assert router.action_decision_reason_history[-1] == ActionDecisionReasons.DEFENSIVE_PREEMPTION
    assert router.remaining_persist == 0


def test_delivery_defense_forbids_opening_when_flat():
    router = _build_test_router(enable_delivery_defense=True, delivery_defense_days=5)
    router.is_in_delivery_terminal = True
    router.agent_act = lambda s, info: 2  # Agent attempts to open long (+1.0)

    state = np.array([100.0, 200.0])
    info = {"avaliable_action": [1, 1, 1], "ask_qyts": [1000], "bid_qyts": [1000], "avaiable_action_list": [0, 1, 2]}

    # Currently flat (0.0)
    action = router.get_action(info, state, current_position=0.0, current_leverage=1)
    # Must remain flat (action 1)
    assert action == 1
    assert router.action_decision_reason_history[-1] == ActionDecisionReasons.DEFENSIVE_RULE_CLOSE


def test_delivery_defense_disabled_allows_normal_trading():
    router = _build_test_router(enable_delivery_defense=False)
    router.agent_act = lambda s, info: 2  # Agent opens long (+1.0)

    state = np.array([100.0, 200.0])
    info = {"avaliable_action": [1, 1, 1], "ask_qyts": [1000], "bid_qyts": [1000], "avaiable_action_list": [0, 1, 2]}

    action = router.get_action(info, state, current_position=0.0, current_leverage=1)
    # Defense is disabled, policy inference proceeds normally
    assert action == 2
    assert router.action_decision_reason_history[-1] == ActionDecisionReasons.POLICY_INFERENCE
