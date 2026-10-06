import pytest
import numpy as np

from RL.DiHFT.high_level.gating.base import GatingDecision
from RL.DiHFT.high_level.gating.absolute_gating import AbsoluteThresholdGating


def test_absolute_gating_strategy_name():
    gating = AbsoluteThresholdGating(slope_threshold=0.25, volatility_threshold=0.30)
    assert gating.strategy_name == "absolute"
    assert gating.slope_threshold == pytest.approx(0.25)
    assert gating.volatility_threshold == pytest.approx(0.30)


def test_absolute_gating_triggers_defensive_when_slope_below_threshold():
    gating = AbsoluteThresholdGating(slope_threshold=0.25, volatility_threshold=0.20)
    vol_weights = [0.10, 0.40, 0.20]   # max = 0.40 >= 0.20
    slope_weights = [0.10, 0.24, 0.05]  # max = 0.24 < 0.25

    decision = gating.decide(vol_weights, slope_weights)

    assert decision.is_defensive is True
    assert decision.reject_reason == "absolute_threshold"
    assert decision.volatility_index == -1
    assert decision.slope_index == -1
    assert decision.metrics["max_volatility_weight"] == pytest.approx(0.40)
    assert decision.metrics["max_slope_weight"] == pytest.approx(0.24)


def test_absolute_gating_triggers_defensive_when_volatility_below_threshold():
    gating = AbsoluteThresholdGating(slope_threshold=0.20, volatility_threshold=0.35)
    vol_weights = [0.10, 0.34, 0.20]   # max = 0.34 < 0.35
    slope_weights = [0.10, 0.50, 0.05]  # max = 0.50 >= 0.20

    decision = gating.decide(vol_weights, slope_weights)

    assert decision.is_defensive is True
    assert decision.reject_reason == "absolute_threshold"
    assert decision.volatility_index == -1
    assert decision.slope_index == -1


def test_absolute_gating_routes_normally_when_both_axes_pass():
    gating = AbsoluteThresholdGating(slope_threshold=0.20, volatility_threshold=0.25)
    vol_weights = [0.10, 0.45, 0.20]   # max index 1
    slope_weights = [0.10, 0.20, 0.60]  # max index 2

    decision = gating.decide(vol_weights, slope_weights)

    assert decision.is_defensive is False
    assert decision.reject_reason == "none"
    assert decision.volatility_index == 1
    assert decision.slope_index == 2
    assert decision.metrics["max_volatility_weight"] == pytest.approx(0.45)
    assert decision.metrics["max_slope_weight"] == pytest.approx(0.60)


def test_absolute_gating_accepts_numpy_arrays():
    gating = AbsoluteThresholdGating(slope_threshold=0.15, volatility_threshold=0.15)
    vol_weights = np.array([0.5, 0.2, 0.3])
    slope_weights = np.array([0.1, 0.8, 0.1])

    decision = gating.decide(vol_weights, slope_weights)

    assert decision.is_defensive is False
    assert decision.volatility_index == 0
    assert decision.slope_index == 1


from RL.DiHFT.high_level.gating.hierarchical_gating import HierarchicalDualGating


def test_hierarchical_gating_strategy_name():
    gating = HierarchicalDualGating(
        ood_threshold=0.005,
        slope_margin_threshold=0.12,
        volatility_margin_threshold=0.10,
    )
    assert gating.strategy_name == "hierarchical"
    assert gating.ood_threshold == pytest.approx(0.005)
    assert gating.slope_margin_threshold == pytest.approx(0.12)
    assert gating.volatility_margin_threshold == pytest.approx(0.10)


def test_hierarchical_gating_gate1_triggers_on_extreme_volatility_ood():
    gating = HierarchicalDualGating(
        ood_threshold=0.005,
        slope_margin_threshold=0.10,
        volatility_margin_threshold=0.10,
    )
    vol_weights = [0.001, 0.002, 0.004]   # max = 0.004 < 0.005
    slope_weights = [0.05, 0.20, 0.05]   # max = 0.20 >= 0.005

    decision = gating.decide(vol_weights, slope_weights)

    assert decision.is_defensive is True
    assert decision.reject_reason == "ood_circuit_breaker"
    assert decision.volatility_index == -1
    assert decision.slope_index == -1
    assert decision.metrics["max_volatility_weight"] == pytest.approx(0.004)


def test_hierarchical_gating_gate1_triggers_on_extreme_slope_ood():
    gating = HierarchicalDualGating(
        ood_threshold=0.005,
        slope_margin_threshold=0.10,
        volatility_margin_threshold=0.10,
    )
    vol_weights = [0.05, 0.20, 0.05]    # max = 0.20 >= 0.005
    slope_weights = [0.001, 0.003, 0.002] # max = 0.003 < 0.005

    decision = gating.decide(vol_weights, slope_weights)

    assert decision.is_defensive is True
    assert decision.reject_reason == "ood_circuit_breaker"
    assert decision.volatility_index == -1
    assert decision.slope_index == -1


def test_hierarchical_gating_gate2_triggers_on_slope_margin_ambiguity():
    gating = HierarchicalDualGating(
        ood_threshold=0.005,
        slope_margin_threshold=0.15,
        volatility_margin_threshold=0.10,
    )
    # Volatility passes Gate 1 and Gate 2: p = [0.1/0.6, 0.4/0.6, 0.1/0.6] -> [0.167, 0.667, 0.167], margin = 0.50 >= 0.10
    vol_weights = [0.10, 0.40, 0.10]
    # Slope passes Gate 1 (max 0.22 >= 0.005), but p = [0.20/0.50, 0.22/0.50, 0.08/0.50] = [0.40, 0.44, 0.16]
    # Slope margin = 0.44 - 0.40 = 0.04 < 0.15
    slope_weights = [0.20, 0.22, 0.08]

    decision = gating.decide(vol_weights, slope_weights)

    assert decision.is_defensive is True
    assert decision.reject_reason == "margin_ambiguity"
    assert decision.volatility_index == -1
    assert decision.slope_index == -1
    assert decision.metrics["slope_margin"] == pytest.approx(0.04, abs=1e-3)
    assert decision.metrics["volatility_margin"] == pytest.approx(0.50, abs=1e-3)


def test_hierarchical_gating_gate2_triggers_on_volatility_margin_ambiguity():
    gating = HierarchicalDualGating(
        ood_threshold=0.005,
        slope_margin_threshold=0.10,
        volatility_margin_threshold=0.15,
    )
    # Volatility passes Gate 1, but margin is small: [0.19, 0.20, 0.05], sum=0.44 -> p=[0.432, 0.455, 0.114], margin=0.023 < 0.15
    vol_weights = [0.19, 0.20, 0.05]
    slope_weights = [0.10, 0.40, 0.10]

    decision = gating.decide(vol_weights, slope_weights)

    assert decision.is_defensive is True
    assert decision.reject_reason == "margin_ambiguity"


def test_hierarchical_gating_passes_both_gates_and_routes_correctly():
    gating = HierarchicalDualGating(
        ood_threshold=0.005,
        slope_margin_threshold=0.12,
        volatility_margin_threshold=0.12,
    )
    # Volatility: [0.015, 0.005, 0.007], sum=0.027. p = [0.556, 0.185, 0.259], margin = 0.556 - 0.259 = 0.296 >= 0.12, argmax = 0
    vol_weights = [0.015, 0.005, 0.007]
    # Slope: [0.010, 0.030, 0.005], sum=0.045. p = [0.222, 0.667, 0.111], margin = 0.667 - 0.222 = 0.444 >= 0.12, argmax = 1
    slope_weights = [0.010, 0.030, 0.005]

    decision = gating.decide(vol_weights, slope_weights)

    assert decision.is_defensive is False
    assert decision.reject_reason == "none"
    assert decision.volatility_index == 0
    assert decision.slope_index == 1
    assert decision.metrics["volatility_margin"] > 0.12
    assert decision.metrics["slope_margin"] > 0.12
    assert "volatility_probs" in decision.metrics
    assert "slope_probs" in decision.metrics


from RL.DiHFT.high_level.gating.factory import create_gating_strategy


def test_create_gating_strategy_factory():
    strat_abs = create_gating_strategy("absolute", slope_threshold=0.2, volatility_threshold=0.3)
    assert strat_abs.strategy_name == "absolute"
    assert strat_abs.slope_threshold == 0.2
    assert strat_abs.volatility_threshold == 0.3

    strat_hier = create_gating_strategy(
        "hierarchical",
        ood_threshold=0.006,
        slope_margin_threshold=0.15,
        volatility_margin_threshold=0.14,
    )
    assert strat_hier.strategy_name == "hierarchical"
    assert strat_hier.ood_threshold == 0.006
    assert strat_hier.slope_margin_threshold == 0.15
    assert strat_hier.volatility_margin_threshold == 0.14

    with pytest.raises(ValueError, match="Unknown gating strategy type"):
        create_gating_strategy("invalid_strategy")


import types
from RL.DiHFT.high_level import vae_routing_util as vru
from analysis.pick_agent.FineFT_two_dimensional_agent_selector import TwoDimensionalSelectionManifest


from tests.rl.test_vae_routing_non_main_defense import _sample_manifest_payload


def _make_mock_router_for_gating(strategy_name: str, **kwargs):
    router = vru.vae_risk_aware_routing.__new__(vru.vae_risk_aware_routing)
    router.num_labels = 3
    router.slot_count = 9
    router.rule_base_threshold = 0.2
    router.axis_thresholds = {"volatility": 0.2, "slope": 0.2}
    router.enable_non_main_contract_defense = False
    router.role_tier_index = None
    router.zero_position_action = 4
    router.leverage_choices = [5]
    router.position_list = [-1.0, 0.0, 1.0]
    router.action = 4
    router.macro_action_history = []
    router.action_decision_reason_history = []
    router.selection_manifest = TwoDimensionalSelectionManifest.from_dict(
        _sample_manifest_payload(num_labels=3)
    )
    router._defensive_action = lambda info, current_position, current_leverage: 4
    router.agent_act = lambda state, info: 10
    router.gating_strategy = create_gating_strategy(strategy_name, **kwargs)
    return router


def test_router_integration_absolute_strategy_rejection():
    router = _make_mock_router_for_gating("absolute", slope_threshold=0.20, volatility_threshold=0.20)
    # Weights are deflated (like in fuel oil): max is 0.038 < 0.20
    router.calculate_axis_window_result = lambda axis: {
        "volatility": [0.015, 0.005, 0.007],
        "slope": [0.014, 0.023, 0.001],
    }[axis]

    action = router.get_action(info={}, s=np.zeros(10), current_position=0.0, current_leverage=5)
    # Under absolute strategy, deflated weights trigger flat defense (action = zero_position_action = 4)
    assert action == 4
    assert router.macro_action_history[-1] == 9  # slot_count represents defense


def test_router_integration_hierarchical_strategy_routes_deflated_weights():
    router = _make_mock_router_for_gating(
        "hierarchical",
        ood_threshold=0.005,
        slope_margin_threshold=0.12,
        volatility_margin_threshold=0.12,
    )
    # Volatility: [0.015, 0.005, 0.007], max=0.015 >= 0.005. p = [0.556, 0.185, 0.259], margin = 0.296 >= 0.12 -> idx 0
    # Slope: [0.005, 0.025, 0.005], max=0.025 >= 0.005. p = [0.143, 0.714, 0.143], margin = 0.571 >= 0.12 -> idx 1
    router.calculate_axis_window_result = lambda axis: {
        "volatility": [0.015, 0.005, 0.007],
        "slope": [0.005, 0.025, 0.005],
    }[axis]

    action = router.get_action(info={}, s=np.zeros(10), current_position=0.0, current_leverage=5)
    # Under hierarchical strategy, both gates pass and it routes to slot_id = 0 * 3 + 1 = 1
    assert action == 10  # returned by mock agent_act
    assert router.selected_agent_index == 1
    assert router.macro_action_history[-1] == 1


def test_resolve_routing_parameters_auto_detects_hierarchical_para_str(tmp_path):
    para_file = tmp_path / "high_level_agent_para.txt"
    para_file.write_text(
        "strat_hierarchical_trial_5_ws_70_wv_90_gs_0.94_gv_0.93_ood_0.0080_ms_0.1500_mv_0.1600\n",
        encoding="utf-8",
    )
    args = types.SimpleNamespace(
        para_file=str(para_file),
        optuna_csv=None,
        slope_window_length=None,
        volatility_window_length=None,
        slope_gamma=None,
        volatility_gamma=None,
        slope_rule_base_threshold=None,
        volatility_rule_base_threshold=None,
        window_length=64,
        gamma=0.9,
        rule_base_threshold=0.2,
        gating_strategy="absolute",
        ood_threshold=0.005,
        slope_margin_threshold=0.12,
        volatility_margin_threshold=0.12,
    )
    resolved = vru.resolve_routing_parameters(args)
    assert resolved.gating_strategy == "hierarchical"
    assert resolved.ood_threshold == pytest.approx(0.0080)
    assert resolved.slope_margin_threshold == pytest.approx(0.1500)
    assert resolved.volatility_margin_threshold == pytest.approx(0.1600)
    assert resolved.slope_window_length == 70
    assert resolved.volatility_window_length == 90
