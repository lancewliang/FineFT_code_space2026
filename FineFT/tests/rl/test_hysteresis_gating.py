import pytest
import numpy as np
import types

from RL.DiHFT.high_level.gating.absolute_gating import AbsoluteThresholdGating
from RL.DiHFT.high_level.gating.factory import create_gating_strategy
from RL.DiHFT.high_level import vae_routing_util as vru
from RL.DiHFT.high_level import vae_routing_optuna as vro
from common.routing_params import RoutingParamColumns
from analysis.pick_agent.FineFT_two_dimensional_agent_selector import (
    TwoDimensionalSelectionManifest,
)
from tests.rl.test_vae_routing_non_main_defense import _sample_manifest_payload


def test_hysteresis_gating_initialization_and_validation():
    gating = AbsoluteThresholdGating(
        slope_threshold=0.40,
        volatility_threshold=0.50,
        hysteresis_exit_ratio=0.70,
    )
    assert gating.strategy_name == "absolute"
    assert gating.slope_threshold == pytest.approx(0.40)
    assert gating.volatility_threshold == pytest.approx(0.50)
    assert gating.hysteresis_exit_ratio == pytest.approx(0.70)

    with pytest.raises(ValueError, match="hysteresis_exit_ratio must be in"):
        AbsoluteThresholdGating(0.4, 0.5, hysteresis_exit_ratio=0.0)

    with pytest.raises(ValueError, match="hysteresis_exit_ratio must be in"):
        AbsoluteThresholdGating(0.4, 0.5, hysteresis_exit_ratio=1.2)


def test_hysteresis_gating_flat_position_requires_full_entry_threshold():
    # T_vol = 0.40, T_slope = 0.40, exit_ratio = 0.65 -> T_exit = 0.26
    gating = AbsoluteThresholdGating(
        slope_threshold=0.40,
        volatility_threshold=0.40,
        hysteresis_exit_ratio=0.65,
    )
    # Weights are 0.35 (above exit threshold 0.26, but below entry threshold 0.40)
    vol_weights = [0.10, 0.35, 0.10]
    slope_weights = [0.10, 0.35, 0.10]

    # When flat (current_position == 0.0), entry is rejected
    decision_flat = gating.decide(vol_weights, slope_weights, current_position=0.0)
    assert decision_flat.is_defensive is True
    assert decision_flat.reject_reason == "absolute_threshold"
    assert decision_flat.metrics["effective_volatility_threshold"] == pytest.approx(0.40)
    assert decision_flat.metrics["effective_slope_threshold"] == pytest.approx(0.40)


def test_hysteresis_gating_holding_position_tolerates_dip_above_exit_threshold():
    # T_vol = 0.40, T_slope = 0.40, exit_ratio = 0.65 -> T_exit = 0.26
    gating = AbsoluteThresholdGating(
        slope_threshold=0.40,
        volatility_threshold=0.40,
        hysteresis_exit_ratio=0.65,
    )
    # Weights are 0.35 (above exit threshold 0.26, below entry 0.40)
    vol_weights = [0.10, 0.35, 0.10]
    slope_weights = [0.10, 0.35, 0.10]

    # When holding long (+1.0) or short (-1.0), hysteresis permits holding
    decision_long = gating.decide(vol_weights, slope_weights, current_position=1.0)
    assert decision_long.is_defensive is False
    assert decision_long.reject_reason == "none"
    assert decision_long.volatility_index == 1
    assert decision_long.slope_index == 1
    assert decision_long.metrics["effective_volatility_threshold"] == pytest.approx(0.26)
    assert decision_long.metrics["effective_slope_threshold"] == pytest.approx(0.26)

    decision_short = gating.decide(vol_weights, slope_weights, current_position=-1.0)
    assert decision_short.is_defensive is False
    assert decision_short.volatility_index == 1
    assert decision_short.slope_index == 1


def test_hysteresis_gating_holding_position_triggers_defense_below_exit_threshold():
    # T_vol = 0.40, T_slope = 0.40, exit_ratio = 0.65 -> T_exit = 0.26
    gating = AbsoluteThresholdGating(
        slope_threshold=0.40,
        volatility_threshold=0.40,
        hysteresis_exit_ratio=0.65,
    )
    # Weights drop to 0.22 (below exit threshold 0.26)
    vol_weights = [0.10, 0.22, 0.10]
    slope_weights = [0.10, 0.35, 0.10]

    decision = gating.decide(vol_weights, slope_weights, current_position=1.0)
    assert decision.is_defensive is True
    assert decision.reject_reason == "absolute_threshold"
    assert decision.metrics["effective_volatility_threshold"] == pytest.approx(0.26)


def _make_mock_router(strategy_name: str, **kwargs):
    router = vru.vae_risk_aware_routing.__new__(vru.vae_risk_aware_routing)
    router.num_labels = 3
    router.slot_count = 9
    router.rule_base_threshold = 0.40
    router.axis_thresholds = {"volatility": 0.40, "slope": 0.40}
    router.hysteresis_exit_ratio = float(kwargs.get("hysteresis_exit_ratio", 0.65))
    router.enable_non_main_contract_defense = False
    router.role_tier_index = None
    router.zero_position_action = 4
    router.leverage_choices = [5]
    router.position_list = [-1.0, 0.0, 1.0]
    router.action = 4
    router.flat_action = 4
    router.current_action = 4
    router.remaining_persist = 0
    router.action_persistence = 1
    router.cooldown_remaining_steps = 0
    router.circuit_breaker_remaining_steps = 0
    router.stop_loss_return_threshold = 0.0
    router.consecutive_stop_loss_count = 0
    router.previous_step_position = 0.0
    router.macro_action_history = []
    router.action_decision_reason_history = []
    router.selection_manifest = TwoDimensionalSelectionManifest.from_dict(
        _sample_manifest_payload(num_labels=3)
    )
    router._defensive_action = lambda info, current_position, current_leverage: 4
    router.agent_act = lambda state, info: 10
    router.gating_strategy = create_gating_strategy(strategy_name, **kwargs)
    return router


def test_router_hysteresis_preserves_position_during_critical_fluctuation():
    router = _make_mock_router(
        "absolute",
        slope_threshold=0.40,
        volatility_threshold=0.40,
        hysteresis_exit_ratio=0.65,
    )
    # Weights: max is 0.35 (in hysteresis buffer [0.26, 0.40))
    router.calculate_axis_window_result = lambda axis: {
        "volatility": [0.05, 0.35, 0.05],
        "slope": [0.05, 0.35, 0.05],
    }[axis]

    # 1. When flat (current_position=0.0): entry is blocked, returns defensive flat action 4
    action_flat = router.get_action(
        info={"avaliable_action": [1] * 20},
        s=np.zeros(10),
        current_position=0.0,
        current_leverage=5,
    )
    assert action_flat == 4
    assert router.macro_action_history[-1] == 9  # defensive

    # 2. When holding a position (current_position=1.0): hysteresis allows agent inference
    action_holding = router.get_action(
        info={"avaliable_action": [1] * 20},
        s=np.zeros(10),
        current_position=1.0,
        current_leverage=5,
    )
    assert action_holding == 10  # routed to agent_act
    assert router.selected_agent_index == 1 * 3 + 1  # slot 4


def test_optuna_search_space_includes_hysteresis_exit_ratio():
    class MockTrial:
        def __init__(self):
            self.suggested = {}

        def suggest_int(self, name, low, high):
            self.suggested[name] = low
            return low

        def suggest_float(self, name, low, high, log=False):
            self.suggested[name] = (low + high) / 2.0
            return self.suggested[name]

        def suggest_categorical(self, name, choices):
            self.suggested[name] = choices[0]
            return choices[0]

    search_args = types.SimpleNamespace(
        window_length_min=10,
        window_length_max=50,
        gamma_min=0.9,
        gamma_max=0.99,
        gating_strategy="absolute",
        rule_base_threshold_min=0.2,
        rule_base_threshold_max=0.5,
        hysteresis_exit_ratio_min=0.50,
        hysteresis_exit_ratio_max=0.80,
    )
    trial_args = types.SimpleNamespace()
    trial = MockTrial()

    vro.suggest_trial_parameters(trial, trial_args, search_args)
    assert RoutingParamColumns.HYSTERESIS_EXIT_RATIO in trial.suggested
    assert trial_args.hysteresis_exit_ratio == pytest.approx(0.65)

    # Test apply_best_trial_parameters
    best_params = {
        RoutingParamColumns.SLOPE_WINDOW_LENGTH: 20,
        RoutingParamColumns.VOLATILITY_WINDOW_LENGTH: 30,
        RoutingParamColumns.SLOPE_GAMMA: 0.95,
        RoutingParamColumns.VOLATILITY_GAMMA: 0.96,
        RoutingParamColumns.SLOPE_RULE_BASE_THRESHOLD: 0.42,
        RoutingParamColumns.VOLATILITY_RULE_BASE_THRESHOLD: 0.38,
        RoutingParamColumns.HYSTERESIS_EXIT_RATIO: 0.68,
    }
    replayed_args = types.SimpleNamespace()
    vro.apply_best_trial_parameters(replayed_args, best_params, search_args)
    assert replayed_args.hysteresis_exit_ratio == pytest.approx(0.68)
