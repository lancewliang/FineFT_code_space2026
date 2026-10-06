import sys
import types
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
    action_persistence: int = 3,
    stop_loss_abs_threshold: float = 50.0,
    stop_loss_cooldown_steps: int = 12,
    circuit_breaker_consecutive_stops: int = 2,
    circuit_breaker_cooling_steps: int = 72,
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
    routing.position_list = [-1.0, 0.0, 1.0]
    routing.calculate_axis_window_result = lambda axis: {
        "volatility": [0.1, 0.8, 0.2],
        "slope": [0.1, 0.2, 0.9],
    }[axis]
    routing.gating_strategy = vru.create_gating_strategy(
        "absolute", slope_threshold=0.2, volatility_threshold=0.2
    )
    routing._defensive_action = lambda info, pos, lev: 1

    # Risk controls
    routing.stop_loss_abs_threshold = stop_loss_abs_threshold
    routing.stop_loss_cooldown_steps = stop_loss_cooldown_steps
    routing.circuit_breaker_consecutive_stops = circuit_breaker_consecutive_stops
    routing.circuit_breaker_cooling_steps = circuit_breaker_cooling_steps

    routing.cooldown_remaining_steps = 0
    routing.last_stopped_position = 0.0
    routing.consecutive_stop_loss_count = 0
    routing.circuit_breaker_remaining_steps = 0
    routing.hard_stop_loss_count = 0
    routing.cooldown_intercept_count = 0
    routing.circuit_breaker_suspension_count = 0

    return routing


def test_action_decision_reasons_enum_extensions():
    assert ActionDecisionReasons.HARD_STOP_LOSS == 5
    assert ActionDecisionReasons.STOP_LOSS_COOLDOWN == 6
    assert ActionDecisionReasons.CIRCUIT_BREAKER_SUSPENSION == 7


def test_risk_controls_cli_arguments_and_validation(monkeypatch):
    monkeypatch.setattr(
        vro,
        "load_two_dimensional_selection_manifest",
        lambda p: MagicMock(slots=[{} for _ in range(9)]),
    )

    # 1. Base parser defaults
    args_default = vru.parser.parse_args([])
    assert args_default.stop_loss_abs_threshold == 50.0
    assert args_default.stop_loss_cooldown_steps == 12
    assert args_default.circuit_breaker_consecutive_stops == 2
    assert args_default.circuit_breaker_cooling_steps == 72

    # 2. Custom values in base parser
    args_custom = vru.parser.parse_args(
        [
            "--stop_loss_abs_threshold", "40.0",
            "--stop_loss_cooldown_steps", "6",
            "--circuit_breaker_consecutive_stops", "3",
            "--circuit_breaker_cooling_steps", "144",
        ]
    )
    assert args_custom.stop_loss_abs_threshold == 40.0
    assert args_custom.stop_loss_cooldown_steps == 6
    assert args_custom.circuit_breaker_consecutive_stops == 3
    assert args_custom.circuit_breaker_cooling_steps == 144

    # 3. Optuna parser defaults and custom
    args_optuna_default = vro.parser_all.parse_args([])
    assert args_optuna_default.stop_loss_abs_threshold == 50.0
    assert args_optuna_default.stop_loss_cooldown_steps == 12
    assert args_optuna_default.circuit_breaker_consecutive_stops == 2
    assert args_optuna_default.circuit_breaker_cooling_steps == 72

    base_args = vro.prepare_base_args(args_default, args_custom)
    assert base_args.stop_loss_abs_threshold == 40.0
    assert base_args.stop_loss_cooldown_steps == 6
    assert base_args.circuit_breaker_consecutive_stops == 3
    assert base_args.circuit_breaker_cooling_steps == 144

    # 4. Validation raises ValueError on negative thresholds/steps
    def _create_valid_args(**overrides):
        d = {
            "base_path": "dataset",
            "dataset_name": "BTCUSDT",
            "experiment_name": "default",
            "max_holding_number": 8,
            "position_choices": 3,
            "leverage_choices": [1],
            "action_persistence": 3,
            "gamma": 0.9,
            "rule_base_threshold": 0.2,
            "window_length": 64,
            "slope_window_length": 64,
            "volatility_window_length": 64,
            "slope_gamma": 0.9,
            "volatility_gamma": 0.9,
            "slope_rule_base_threshold": 0.2,
            "volatility_rule_base_threshold": 0.2,
            "gating_strategy": "absolute",
            "ood_threshold": 0.005,
            "slope_margin_threshold": 0.12,
            "volatility_margin_threshold": 0.12,
            "enable_non_main_contract_defense": False,
            "trial_number": None,
            "eval_stage": "valid",
            "save_artifacts": False,
            "stop_loss_abs_threshold": 50.0,
            "stop_loss_cooldown_steps": 12,
            "circuit_breaker_consecutive_stops": 2,
            "circuit_breaker_cooling_steps": 72,
        }
        d.update(overrides)
        return types.SimpleNamespace(**d)

    router = vru.vae_risk_aware_routing.__new__(vru.vae_risk_aware_routing)
    router.save_artifacts = False

    with pytest.raises(ValueError, match="stop_loss_abs_threshold must be non-negative"):
        router.reconfigure_routing(_create_valid_args(stop_loss_abs_threshold=-1.0))

    with pytest.raises(ValueError, match="stop_loss_cooldown_steps must be non-negative"):
        router.reconfigure_routing(_create_valid_args(stop_loss_cooldown_steps=-1))

    with pytest.raises(ValueError, match="circuit_breaker_consecutive_stops must be non-negative"):
        router.reconfigure_routing(_create_valid_args(circuit_breaker_consecutive_stops=-1))

    with pytest.raises(ValueError, match="circuit_breaker_cooling_steps must be -1 or non-negative"):
        router.reconfigure_routing(_create_valid_args(circuit_breaker_cooling_steps=-2))


def test_hard_stop_loss_triggers_flat_and_preempts_persistence():
    router = _create_test_router(
        action_persistence=3,
        stop_loss_abs_threshold=50.0,
        stop_loss_cooldown_steps=12,
    )
    info = {"avaliable_action": [1, 1, 1]}
    state = np.array([0.0])

    # Simulate active long position with persistence locked
    router.remaining_persist = 2
    router.current_action = 2  # long
    router.last_stopped_position = 0.0
    router.cooldown_remaining_steps = 0

    # Unrealized loss exceeds threshold (-50.1 <= -50.0)
    action = router.get_action(
        info, state, current_position=1.0, current_leverage=1, current_unrealized_pnl=-50.1
    )

    assert action == router.flat_action
    assert router.remaining_persist == 0
    assert router.last_stopped_position == 1.0
    assert router.cooldown_remaining_steps == 12
    assert router.consecutive_stop_loss_count == 1
    assert router.hard_stop_loss_count == 1
    assert router.action_decision_reason_history[-1] == ActionDecisionReasons.HARD_STOP_LOSS


def test_hard_stop_loss_does_not_trigger_when_within_tolerance_or_disabled():
    router = _create_test_router(
        action_persistence=3,
        stop_loss_abs_threshold=50.0,
    )
    info = {"avaliable_action": [1, 1, 1]}
    state = np.array([0.0])

    # 1. Within tolerance (-49.9 > -50.0)
    router.remaining_persist = 2
    router.current_action = 2
    action = router.get_action(
        info, state, current_position=1.0, current_leverage=1, current_unrealized_pnl=-49.9
    )
    assert action == 2
    assert router.remaining_persist == 1
    assert router.action_decision_reason_history[-1] == ActionDecisionReasons.ACTION_PERSISTENCE
    assert router.hard_stop_loss_count == 0

    # 2. Disabled when threshold == 0.0
    router.stop_loss_abs_threshold = 0.0
    router.remaining_persist = 1
    action = router.get_action(
        info, state, current_position=1.0, current_leverage=1, current_unrealized_pnl=-1000.0
    )
    assert action == 2
    assert router.hard_stop_loss_count == 0

    # 3. Flat position (position == 0) ignores unrealized pnl
    router.stop_loss_abs_threshold = 50.0
    router.remaining_persist = 0
    router.agent_act = lambda s, inf: 2
    action = router.get_action(
        info, state, current_position=0.0, current_leverage=1, current_unrealized_pnl=-100.0
    )
    assert action == 2
    assert router.action_decision_reason_history[-1] == ActionDecisionReasons.POLICY_INFERENCE


def test_directional_cooldown_intercepts_same_direction_and_allows_opposite():
    router = _create_test_router(
        action_persistence=1,
        stop_loss_abs_threshold=50.0,
        stop_loss_cooldown_steps=12,
    )
    info = {"avaliable_action": [1, 1, 1]}
    state = np.array([0.0])

    # 1. Trigger stop-loss on Long (position = 1.0)
    action0 = router.get_action(
        info, state, current_position=1.0, current_leverage=1, current_unrealized_pnl=-55.0
    )
    assert action0 == router.flat_action
    assert router.last_stopped_position == 1.0
    assert router.cooldown_remaining_steps == 12

    # 2. Next step: agent wants to go Long again (action 2 maps to position 1.0)
    router.agent_act = lambda s, inf: 2  # Long
    action1 = router.get_action(
        info, state, current_position=0.0, current_leverage=1, current_unrealized_pnl=0.0
    )
    assert action1 == router.flat_action
    assert router.action_decision_reason_history[-1] == ActionDecisionReasons.STOP_LOSS_COOLDOWN
    assert router.cooldown_intercept_count == 1
    assert router.cooldown_remaining_steps == 11

    # 3. Next step: agent wants to go Short (action 0 maps to position -1.0)
    router.agent_act = lambda s, inf: 0  # Short
    action2 = router.get_action(
        info, state, current_position=0.0, current_leverage=1, current_unrealized_pnl=0.0
    )
    assert action2 == 0  # Allowed!
    assert router.action_decision_reason_history[-1] == ActionDecisionReasons.POLICY_INFERENCE
    assert router.cooldown_remaining_steps == 10


def test_circuit_breaker_trips_after_consecutive_stops_and_suspends():
    router = _create_test_router(
        action_persistence=1,
        stop_loss_abs_threshold=50.0,
        stop_loss_cooldown_steps=2,
        circuit_breaker_consecutive_stops=2,
        circuit_breaker_cooling_steps=72,
    )
    info = {"avaliable_action": [1, 1, 1]}
    state = np.array([0.0])

    # Stop 1
    a0 = router.get_action(
        info, state, current_position=1.0, current_leverage=1, current_unrealized_pnl=-55.0
    )
    assert a0 == router.flat_action
    assert router.consecutive_stop_loss_count == 1
    assert router.circuit_breaker_remaining_steps == 0
    assert router.cooldown_remaining_steps == 2

    # Pass 2 cooldown steps with flat
    router.agent_act = lambda s, inf: router.flat_action
    router.get_action(info, state, current_position=0.0, current_leverage=1, current_unrealized_pnl=0.0)
    router.get_action(info, state, current_position=0.0, current_leverage=1, current_unrealized_pnl=0.0)
    assert router.cooldown_remaining_steps == 0

    # New trade opened (Short, position = -1.0)
    # Stop 2 hits!
    a3 = router.get_action(
        info, state, current_position=-1.0, current_leverage=1, current_unrealized_pnl=-60.0
    )
    assert a3 == router.flat_action
    assert router.consecutive_stop_loss_count == 2
    assert router.circuit_breaker_remaining_steps == 72

    # Next step: Circuit breaker suspension active
    router.agent_act = lambda s, inf: 0
    a4 = router.get_action(
        info, state, current_position=0.0, current_leverage=1, current_unrealized_pnl=0.0
    )
    assert a4 == router.flat_action
    assert router.action_decision_reason_history[-1] == ActionDecisionReasons.CIRCUIT_BREAKER_SUSPENSION
    assert router.circuit_breaker_suspension_count == 1
    assert router.circuit_breaker_remaining_steps == 71


def test_circuit_breaker_resets_streak_on_normal_holding_exit():
    router = _create_test_router(
        action_persistence=1,
        stop_loss_abs_threshold=50.0,
        stop_loss_cooldown_steps=1,
        circuit_breaker_consecutive_stops=2,
    )
    info = {"avaliable_action": [1, 1, 1]}
    state = np.array([0.0])

    # Stop 1
    router.get_action(
        info, state, current_position=1.0, current_leverage=1, current_unrealized_pnl=-55.0
    )
    assert router.consecutive_stop_loss_count == 1

    # Cooldown step
    router.agent_act = lambda s, inf: router.flat_action
    router.get_action(info, state, current_position=0.0, current_leverage=1, current_unrealized_pnl=0.0)

    # Open Trade 2 (Long, position = 1.0)
    router.agent_act = lambda s, inf: 2
    router.get_action(info, state, current_position=0.0, current_leverage=1, current_unrealized_pnl=0.0)

    # Trade 2 exits normally to Flat (profitable / rule close, NOT stop-loss)
    router.agent_act = lambda s, inf: router.flat_action
    router.get_action(info, state, current_position=1.0, current_leverage=1, current_unrealized_pnl=10.0)

    # Now position is 0.0, consecutive_stop_loss_count should have reset to 0!
    router.get_action(info, state, current_position=0.0, current_leverage=1, current_unrealized_pnl=0.0)
    assert router.consecutive_stop_loss_count == 0


def test_circuit_breaker_permanent_suspension():
    router = _create_test_router(
        action_persistence=1,
        stop_loss_abs_threshold=50.0,
        stop_loss_cooldown_steps=1,
        circuit_breaker_consecutive_stops=2,
        circuit_breaker_cooling_steps=-1,  # Permanent suspension
    )
    info = {"avaliable_action": [1, 1, 1]}
    state = np.array([0.0])

    # Stop 1
    router.get_action(
        info, state, current_position=1.0, current_leverage=1, current_unrealized_pnl=-55.0
    )
    # Cooldown 1 step
    router.agent_act = lambda s, inf: router.flat_action
    router.get_action(info, state, current_position=0.0, current_leverage=1, current_unrealized_pnl=0.0)

    # Stop 2 hits
    router.get_action(
        info, state, current_position=1.0, current_leverage=1, current_unrealized_pnl=-55.0
    )
    assert router.circuit_breaker_remaining_steps == -1

    # Indefinite suspension for next 5 steps
    for _ in range(5):
        a = router.get_action(
            info, state, current_position=0.0, current_leverage=1, current_unrealized_pnl=0.0
        )
        assert a == router.flat_action
        assert router.action_decision_reason_history[-1] == ActionDecisionReasons.CIRCUIT_BREAKER_SUSPENSION
        assert router.circuit_breaker_remaining_steps == -1


def test_reset_routing_state_clears_all_risk_counters_and_timers():
    router = _create_test_router()
    router.cooldown_remaining_steps = 10
    router.last_stopped_position = 1.0
    router.consecutive_stop_loss_count = 2
    router.circuit_breaker_remaining_steps = 50
    router.hard_stop_loss_count = 3
    router.cooldown_intercept_count = 4
    router.circuit_breaker_suspension_count = 5
    router.previous_step_position = 1.0
    router.active_trade_stopped = True

    router.reset_routing_state()

    assert router.cooldown_remaining_steps == 0
    assert router.last_stopped_position == 0.0
    assert router.consecutive_stop_loss_count == 0
    assert router.circuit_breaker_remaining_steps == 0
    assert router.hard_stop_loss_count == 0
    assert router.cooldown_intercept_count == 0
    assert router.circuit_breaker_suspension_count == 0
    assert router.previous_step_position == 0.0
    assert router.active_trade_stopped is False
