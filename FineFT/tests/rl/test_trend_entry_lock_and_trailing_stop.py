import sys
from unittest.mock import MagicMock
import numpy as np
import pytest

if "optuna" not in sys.modules:
    sys.modules["optuna"] = MagicMock()

from common import ActionDecisionReasons
from RL.DiHFT.high_level import vae_routing_util as vru
from RL.DiHFT.high_level import vae_routing_optuna as vro
from analysis.pick_agent.FineFT_two_dimensional_agent_selector import (
    TwoDimensionalSelectionManifest,
)


def _sample_manifest_payload(num_labels: int = 3, **overrides) -> dict:
    labels = [f"label_{i}" for i in range(num_labels)]
    slot_count = num_labels * num_labels
    payload = {
        "schema_version": 1,
        "selection_method": "two_dimensional_marginal_and_dual_context_lcb",
        "candidate_root": "/path/to/candidates",
        "valid_root": "/path/to/valid",
        "axes": {
            "volatility": labels,
            "slope": labels,
        },
        "slot_count": slot_count,
        "slot_index_formula": "volatility_index * num_labels + slope_index",
        "null_policy": {
            "logical_kind": "empty_model",
            "intended_runtime_behavior": "flat_position",
            "model_assembly_status": "built_as_flat_qnet",
        },
        "candidate_scope": {
            "common_epochs": [1],
            "discovered_candidate_count": 1,
            "complete_candidate_count": 1,
            "excluded_incomplete_candidate_count": 0,
            "initial_actions": [0],
        },
        "selection_config": {},
        "metric_definition": {
            "return": "sum(reward) / transition_count",
            "aggregation": "mean",
            "lcb": "mean - z * se",
            "pair_score": "min(lcb)",
            "joint_context_note": "filtered by timestamp",
        },
        "artifacts": {
            "model_assembly": "/path/to/model.pth",
            "high_level_model_change": "not_performed",
        },
        "slots": [
            {"slot_id": slot_id, "kind": "model"}
            for slot_id in range(slot_count)
        ],
    }
    payload.update(overrides)
    return payload


def _create_test_router(
    slope_index: int = 2,
    volatility_index: int = 1,
    action_persistence: int = 1,
    enable_trend_entry_lock: bool = True,
    enable_trailing_stop: bool = True,
    trailing_stop_activation_threshold: float = 0.08,
    trailing_stop_retracement_ratio: float = 0.25,
    trailing_stop_profit_floor: float = 0.003,
    trailing_stop_cooldown_steps: int = 24,
    trailing_stop_require_peak_breakout: bool = True,
) -> vru.vae_risk_aware_routing:
    routing = vru.vae_risk_aware_routing.__new__(vru.vae_risk_aware_routing)
    routing.num_labels = 3
    routing.slot_count = 9
    routing.rule_base_threshold = 0.2
    routing.axis_window_lengths = {"volatility": 64, "slope": 64}
    routing.axis_thresholds = {"volatility": 0.2, "slope": 0.2}
    routing.selection_manifest = TwoDimensionalSelectionManifest.from_dict(
        _sample_manifest_payload(num_labels=3)
    )
    routing.enable_non_main_contract_defense = False
    routing.role_tier_index = None
    routing.zero_position_action = 1
    routing.flat_action = 1
    routing.action_persistence = action_persistence
    routing.remaining_persist = 0
    routing.current_action = 1
    routing.action_decision_reason_history = []
    routing.action = 1
    routing.macro_action_history = []
    routing.leverage_choices = [1]
    # Actions: 0 -> -1.0 (short), 1 -> 0.0 (flat), 2 -> +1.0 (long)
    routing.position_list = [-1.0, 0.0, 1.0]
    routing.calculate_axis_window_result = lambda axis: {
        "volatility": [0.1, 0.8, 0.1],
        "slope": [0.1, 0.1, 0.8] if slope_index == 2 else ([0.8, 0.1, 0.1] if slope_index == 0 else [0.1, 0.8, 0.1]),
    }[axis]
    routing.gating_strategy = vru.create_gating_strategy(
        "absolute", slope_threshold=0.2, volatility_threshold=0.2
    )
    routing._defensive_action = lambda info, pos, lev: 1
    routing.stop_loss_return_threshold = 0.0
    routing.stop_loss_cooldown_steps = 0
    routing.circuit_breaker_consecutive_stops = 0
    routing.circuit_breaker_cooling_steps = 0
    routing.circuit_breaker_remaining_steps = 0
    routing.cooldown_remaining_steps = 0
    routing.last_stopped_position = 0.0
    routing.consecutive_stop_loss_count = 0
    routing.previous_step_position = 0.0
    routing.hard_stop_loss_count = 0
    routing.cooldown_intercept_count = 0
    routing.circuit_breaker_suspension_count = 0
    routing._arm_persistence = lambda action: None

    # Trend lock & trailing stop configs
    routing.enable_trend_entry_lock = enable_trend_entry_lock
    routing.enable_trailing_stop = enable_trailing_stop
    routing.trailing_stop_activation_threshold = trailing_stop_activation_threshold
    routing.trailing_stop_retracement_ratio = trailing_stop_retracement_ratio
    routing.trailing_stop_profit_floor = trailing_stop_profit_floor
    routing.trailing_stop_cooldown_steps = trailing_stop_cooldown_steps
    routing.trailing_stop_require_peak_breakout = trailing_stop_require_peak_breakout

    # State
    routing.trailing_stop_active = False
    routing.trailing_peak_return = 0.0
    routing.trailing_peak_markprice = 0.0
    routing.trailing_stop_remaining_steps = 0
    routing.trailing_stop_last_position = 0.0
    routing.trailing_stop_hurdle_price = None
    routing.trailing_stop_hurdle_slope = None
    routing.trailing_stop_count = 0
    routing.trailing_stop_cooldown_intercept_count = 0
    routing.trend_entry_lock_count = 0

    return routing


def _make_info(current_action=1, markprice=100.0, unrealized_pnl=0.0):
    return {
        "avaiable_action_list": [0, 1, 2],
        "avaliable_action": np.array([1, 1, 1], dtype=np.int32),
        "previous_action": current_action,
        "trading_info": np.zeros(4, dtype=np.float32),
        "funding_count_down_hour": 0,
        "funding_count_down_minute": 0,
        "current_markprice": markprice,
        "unrealized_pnl": unrealized_pnl,
        "markprice": markprice,
    }


def test_trend_entry_lock_intercepts_short_in_bull():
    router = _create_test_router(slope_index=2)  # Bull
    # Agent wants to output action 0 (short) from flat
    router.agent_act = MagicMock(return_value=0)
    info = _make_info(current_action=1, markprice=100.0)
    state = np.zeros(10)

    action = router.get_action(info, state, current_position=0.0, current_leverage=1, current_unrealized_pnl=0.0, current_markprice=100.0)
    # Action must be intercepted to flat (1)
    assert action == 1
    assert router.action_decision_reason_history[-1] == ActionDecisionReasons.TREND_ENTRY_LOCK
    assert router.trend_entry_lock_count == 1


def test_trend_entry_lock_allows_closing_long_in_bull():
    router = _create_test_router(slope_index=2)  # Bull
    # Agent holds long (+1.0) and wants to output flat (1)
    router.agent_act = MagicMock(return_value=1)
    info = _make_info(current_action=2, markprice=100.0)
    state = np.zeros(10)

    action = router.get_action(info, state, current_position=1.0, current_leverage=1, current_unrealized_pnl=0.0, current_markprice=100.0)
    # Closing to flat must be allowed!
    assert action == 1
    assert router.action_decision_reason_history[-1] == ActionDecisionReasons.POLICY_INFERENCE
    assert router.trend_entry_lock_count == 0


def test_trailing_profit_stop_activation_and_execution():
    router = _create_test_router(slope_index=2)
    router.agent_act = MagicMock(return_value=2)  # keep holding long
    state = np.zeros(10)

    # 1. Price at 100, unrealized pnl = 5.0 (5% profit on notional 100). Activation is 8%.
    info1 = _make_info(current_action=2, markprice=100.0, unrealized_pnl=5.0)
    a1 = router.get_action(info1, state, current_position=1.0, current_leverage=1, current_unrealized_pnl=5.0, current_markprice=100.0)
    assert a1 == 2
    assert router.trailing_stop_active is False

    # 2. Price at 112, unrealized pnl = 12.0 (12% return). Crosses 8% activation hurdle!
    info2 = _make_info(current_action=2, markprice=112.0, unrealized_pnl=12.0)
    a2 = router.get_action(info2, state, current_position=1.0, current_leverage=1, current_unrealized_pnl=12.0, current_markprice=112.0)
    assert a2 == 2
    assert router.trailing_stop_active is True
    assert pytest.approx(router.trailing_peak_return, rel=1e-3) == 12.0 / (1.0 * 112.0)
    assert router.trailing_peak_markprice == 112.0

    # 3. Price drops to 108, unrealized pnl drops from peak.
    # Peak return is ~10.71%. Suppose return drops to 7.0%. Retracement ratio = (10.71 - 7.0) / 10.71 = 34.6% >= 25%.
    info3 = _make_info(current_action=2, markprice=108.0, unrealized_pnl=7.5)
    a3 = router.get_action(info3, state, current_position=1.0, current_leverage=1, current_unrealized_pnl=7.5, current_markprice=108.0)
    # Trailing stop must trigger!
    assert a3 == 1
    assert router.action_decision_reason_history[-1] == ActionDecisionReasons.TRAILING_PROFIT_STOP
    assert router.trailing_stop_count == 1
    assert router.trailing_stop_remaining_steps == 24
    assert router.trailing_stop_hurdle_price == 112.0


def test_trailing_stop_dual_spatiotemporal_guard():
    router = _create_test_router(slope_index=2)
    # Directly prime trailing stop state as just triggered on Long (+1.0)
    router.trailing_stop_remaining_steps = 24
    router.trailing_stop_last_position = 1.0
    router.trailing_stop_hurdle_price = 112.0
    router.trailing_stop_hurdle_slope = 2
    state = np.zeros(10)

    # Agent attempts to open Long again at price 108.0
    router.agent_act = MagicMock(return_value=2)
    info1 = _make_info(current_action=1, markprice=108.0)
    a1 = router.get_action(info1, state, current_position=0.0, current_leverage=1, current_unrealized_pnl=0.0, current_markprice=108.0)
    assert a1 == 1  # intercepted to flat
    assert router.action_decision_reason_history[-1] == ActionDecisionReasons.TRAILING_STOP_COOLDOWN
    assert router.trailing_stop_remaining_steps == 23

    # Cooldown timer expires to 0, but price is still 110.0 (< 112.0 hurdle)
    router.trailing_stop_remaining_steps = 0
    info2 = _make_info(current_action=1, markprice=110.0)
    a2 = router.get_action(info2, state, current_position=0.0, current_leverage=1, current_unrealized_pnl=0.0, current_markprice=110.0)
    assert a2 == 1  # STILL intercepted because price hasn't broken 112.0!
    assert router.action_decision_reason_history[-1] == ActionDecisionReasons.TRAILING_STOP_COOLDOWN

    # Price breaks out to 113.0 (>= 112.0 hurdle)
    info3 = _make_info(current_action=1, markprice=113.0)
    a3 = router.get_action(info3, state, current_position=0.0, current_leverage=1, current_unrealized_pnl=0.0, current_markprice=113.0)
    assert a3 == 2  # Allowed!
    assert router.action_decision_reason_history[-1] == ActionDecisionReasons.POLICY_INFERENCE


def test_trend_entry_lock_intercepts_long_in_bear():
    router = _create_test_router(slope_index=0)  # Bear
    router.agent_act = MagicMock(return_value=2)  # Agent wants long
    info = _make_info(current_action=1, markprice=100.0)
    state = np.zeros(10)

    action = router.get_action(info, state, current_position=0.0, current_leverage=1, current_unrealized_pnl=0.0, current_markprice=100.0)
    assert action == 1  # Intercepted to flat
    assert router.action_decision_reason_history[-1] == ActionDecisionReasons.TREND_ENTRY_LOCK
    assert router.trend_entry_lock_count == 1


def test_trailing_stop_hurdle_resets_on_regime_shift():
    router = _create_test_router(slope_index=2)  # Bull
    router.trailing_stop_remaining_steps = 24
    router.trailing_stop_last_position = 1.0
    router.trailing_stop_hurdle_price = 112.0
    router.trailing_stop_hurdle_slope = 2
    state = np.zeros(10)

    # Regime shifts to Shock (slope_index=1)
    router.calculate_axis_window_result = lambda axis: {
        "volatility": [0.1, 0.8, 0.1],
        "slope": [0.1, 0.8, 0.1],  # Shock
    }[axis]

    router.agent_act = MagicMock(return_value=2)
    info = _make_info(current_action=1, markprice=108.0)  # Below 112 hurdle
    a = router.get_action(info, state, current_position=0.0, current_leverage=1, current_unrealized_pnl=0.0, current_markprice=108.0)

    # Hurdle must be reset because regime shifted!
    assert router.trailing_stop_remaining_steps == 0
    assert router.trailing_stop_hurdle_price is None
    assert router.trailing_stop_last_position == 0.0
    assert a == 2
    assert router.action_decision_reason_history[-1] == ActionDecisionReasons.POLICY_INFERENCE


def test_cli_parser_defaults_and_optuna_propagation():
    args_default = vru.parser.parse_args([])
    assert args_default.enable_trend_entry_lock is True
    assert args_default.enable_trailing_stop is True
    assert args_default.trailing_stop_activation_threshold == 0.08
    assert args_default.trailing_stop_retracement_ratio == 0.25
    assert args_default.trailing_stop_profit_floor == 0.003
    assert args_default.trailing_stop_cooldown_steps == 24
    assert args_default.trailing_stop_require_peak_breakout is True

    args_optuna = vro.parser_all.parse_args([])
    assert args_optuna.enable_trend_entry_lock is True
    assert args_optuna.enable_trailing_stop is True
    assert args_optuna.trailing_stop_activation_threshold == 0.08
