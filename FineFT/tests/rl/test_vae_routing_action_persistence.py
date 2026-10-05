import os
import sys
import types
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

if "optuna" not in sys.modules:
    sys.modules["optuna"] = MagicMock()

from common import ActionDecisionReasons, HistoryArtifactNames, ArtifactNames
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
    enable_defense: bool = False,
    role_tier_index: int | None = None,
) -> vru.vae_risk_aware_routing:
    routing = vru.vae_risk_aware_routing.__new__(vru.vae_risk_aware_routing)
    routing.num_labels = 3
    routing.slot_count = 9
    routing.rule_base_threshold = 0.2
    routing.axis_thresholds = {"volatility": 0.2, "slope": 0.2}
    routing.selection_manifest = TwoDimensionalSelectionManifest.from_dict(
        _sample_manifest_payload(num_labels=3)
    )
    routing.enable_non_main_contract_defense = enable_defense
    routing.role_tier_index = role_tier_index
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
    return routing


def test_action_persistence_cli_arguments_and_validation(monkeypatch):
    monkeypatch.setattr(vro, "load_two_dimensional_selection_manifest", lambda p: MagicMock(slots=[{} for _ in range(9)]))
    # 1. Base parser default
    args_default = vru.parser.parse_args([])
    assert args_default.action_persistence == 3

    # 2. Custom value
    args_custom = vru.parser.parse_args(["--action_persistence", "5"])
    assert args_custom.action_persistence == 5

    # 3. Optuna parser default and custom
    args_optuna = vro.parser_all.parse_args([])
    assert args_optuna.action_persistence == 3
    args_optuna_custom = vro.parser_all.parse_args(["--action_persistence", "4"])
    assert args_optuna_custom.action_persistence == 4

    # 4. Optuna prepare_base_args propagation
    base_args = vro.prepare_base_args(args_default, args_optuna_custom)
    assert base_args.action_persistence == 4

    # 5. Non-positive validation raises ValueError
    args_invalid = types.SimpleNamespace(
        base_path="dataset",
        dataset_name="BTCUSDT",
        experiment_name="default",
        max_holding_number=8,
        position_choices=3,
        leverage_choices=[1],
        action_persistence=0,
    )
    with pytest.raises(ValueError, match="action_persistence must be positive"):
        router = vru.vae_risk_aware_routing.__new__(vru.vae_risk_aware_routing)
        router.reconfigure_routing(args_invalid)


def test_action_persistence_holds_non_flat_action_and_skips_inference():
    router = _create_test_router(action_persistence=3)
    info = {"avaliable_action": [1, 1, 1]}
    state = np.array([0.0])

    agent_calls = []

    def mock_agent_act(s, inf):
        agent_calls.append(len(agent_calls))
        return 2  # non-flat long action

    router.agent_act = mock_agent_act

    # Step 0: Agent called, returns 2. remaining_persist becomes 2. Reason = POLICY_INFERENCE (0)
    a0 = router.get_action(info, state, 0.0, 1)
    assert a0 == 2
    assert len(agent_calls) == 1
    assert router.remaining_persist == 2
    assert router.action_decision_reason_history[-1] == ActionDecisionReasons.POLICY_INFERENCE

    # Step 1: Persisted action 2 reused without calling agent_act. remaining_persist becomes 1. Reason = ACTION_PERSISTENCE (1)
    a1 = router.get_action(info, state, 1.0, 1)
    assert a1 == 2
    assert len(agent_calls) == 1  # No new call!
    assert router.remaining_persist == 1
    assert router.action_decision_reason_history[-1] == ActionDecisionReasons.ACTION_PERSISTENCE

    # Step 2: Persisted action 2 reused. remaining_persist becomes 0. Reason = ACTION_PERSISTENCE (1)
    a2 = router.get_action(info, state, 1.0, 1)
    assert a2 == 2
    assert len(agent_calls) == 1  # Still no new call!
    assert router.remaining_persist == 0
    assert router.action_decision_reason_history[-1] == ActionDecisionReasons.ACTION_PERSISTENCE

    # Step 3: Persistence expired. Agent called again! Reason = POLICY_INFERENCE (0)
    a3 = router.get_action(info, state, 1.0, 1)
    assert a3 == 2
    assert len(agent_calls) == 2  # New call executed!
    assert router.remaining_persist == 2
    assert router.action_decision_reason_history[-1] == ActionDecisionReasons.POLICY_INFERENCE


def test_action_persistence_does_not_persist_flat_action():
    router = _create_test_router(action_persistence=3)
    info = {"avaliable_action": [1, 1, 1]}
    state = np.array([0.0])

    agent_calls = []

    def mock_agent_act(s, inf):
        agent_calls.append(len(agent_calls))
        return 1  # Flat action (zero position)

    router.agent_act = mock_agent_act

    for step in range(5):
        action = router.get_action(info, state, 0.0, 1)
        assert action == 1
        assert router.remaining_persist == 0
        assert router.action_decision_reason_history[-1] == ActionDecisionReasons.POLICY_INFERENCE

    # Must be evaluated every single step
    assert len(agent_calls) == 5


def test_defensive_preemption_breaks_persistence():
    router = _create_test_router(action_persistence=3, enable_defense=True, role_tier_index=0)
    info = {"avaliable_action": [1, 1, 1]}

    # Step 0: State indicates main contract (tier=1.0) -> acts normally, takes Long (2)
    state_main = np.array([1.0])
    router.agent_act = lambda s, inf: 2
    a0 = router.get_action(info, state_main, 0.0, 1)
    assert a0 == 2
    assert router.remaining_persist == 2
    assert router.action_decision_reason_history[-1] == ActionDecisionReasons.POLICY_INFERENCE

    # Step 1: State switches to non-main contract (tier=0.0) -> triggers non-main defense!
    state_non_main = np.array([0.0])
    a1 = router.get_action(info, state_non_main, 1.0, 1)

    # Must trigger defensive action (flat=1), break persistence immediately
    assert a1 == 1
    assert router.remaining_persist == 0
    assert router.macro_action_history[-1] == router.slot_count
    assert router.action_decision_reason_history[-1] == ActionDecisionReasons.DEFENSIVE_PREEMPTION


def test_action_unavailable_breaks_persistence():
    router = _create_test_router(action_persistence=3)
    info_available = {"avaliable_action": [1, 1, 1]}
    state = np.array([0.0])

    call_count = 0

    def mock_agent_act(s, inf):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return 2  # First call selects Long (2)
        return 0  # Re-eval selects Short (0)

    router.agent_act = mock_agent_act

    # Step 0: Takes action 2
    a0 = router.get_action(info_available, state, 0.0, 1)
    assert a0 == 2
    assert router.remaining_persist == 2
    assert router.action_decision_reason_history[-1] == ActionDecisionReasons.POLICY_INFERENCE

    # Step 1: Action 2 becomes unavailable ([1, 1, 0])
    info_unavailable = {"avaliable_action": [1, 1, 0]}
    a1 = router.get_action(info_unavailable, state, 1.0, 1)

    # Must detect unavailable action, break persistence, and re-evaluate policy!
    assert a1 == 0
    assert call_count == 2
    assert router.action_decision_reason_history[-1] == ActionDecisionReasons.ACTION_UNAVAILABLE_BREAK


def test_diagnostic_artifacts_and_metrics_saved_to_disk(tmp_path, monkeypatch):
    valid_path = tmp_path / "valid.feather"
    pd.DataFrame({"price": [100.0, 101.0, 102.0, 103.0]}).to_feather(valid_path)

    class MultiStepFakeEnv:
        position = 0.0
        leverage = 1
        margine_balance_history = [100.0, 100.5, 101.0, 101.5]
        micro_action_history = []
        initial_margin_history = [0.0, 10.0, 10.0, 10.0]
        wallet_balance_history = [100.0, 100.0, 100.0, 100.0]
        unrealized_pnl_history = [0.0, 0.5, 1.0, 1.5]
        maintain_marigine_history = [0.0, 1.0, 1.0, 1.0]
        new_position_required_money_history = [0.0, 10.0, 0.0, 0.0]

        def __init__(self):
            self.step_idx = 0
            self.micro_action_history = []

        def reset(self):
            self.step_idx = 0
            return np.array([0.0]), {"previous_action": 1, "avaliable_action": [1, 1, 1]}

        def step(self, action):
            self.step_idx += 1
            self.micro_action_history.append(action)
            done = self.step_idx >= 3
            return np.array([0.0]), 0.5, done, {"previous_action": action, "avaliable_action": [1, 1, 1]}

    fake_env = MultiStepFakeEnv()
    monkeypatch.setattr(vru, "initiate_base_env", lambda **kwargs: fake_env)
    monkeypatch.setattr(vru, "calculate_required_money", lambda *args: 10.0)

    save_dir = tmp_path / "result_artifacts"
    router = _create_test_router(action_persistence=3)
    router.initial_wallet_balance = 100.0
    router.vae_slope_indicators = []
    router.vae_volatility_indicators = []
    router.tech_indicator_list = []
    router.max_holding_number = 1
    router.position_choices = 3
    router.leverage_choices = [1]
    router.long_estimated_rate = 0.0
    router.short_estimated_rate = 0.0
    router.transcation_cost = 0.0
    router.maintenance_margin_ratio_dict = {}
    router.early_stop = 0
    router.initial_state = (100.0, 0.0, 0.0, 0.0, 1.0)
    router.order_book_depth = 5
    router.allow_reverse_position = False
    router.initial_rollout = types.MethodType(
        lambda self, env, s, info: (env, s, 0.0, False, info),
        router,
    )
    router.get_quantiles = types.MethodType(lambda self, *args, **kwargs: None, router)
    router.agent_act = lambda s, inf: 2

    router.run_single_valid_df(pd.read_feather(valid_path), str(save_dir))

    # Assert artifacts exist
    reason_file = save_dir / HistoryArtifactNames.ACTION_DECISION_REASON_HISTORY_NPY
    micro_file = save_dir / HistoryArtifactNames.MICRO_ACTION_HISTORY_NPY
    trading_info_file = save_dir / ArtifactNames.TRADING_INFO_NPY

    assert reason_file.exists()
    assert micro_file.exists()
    assert trading_info_file.exists()

    reasons = np.load(reason_file)
    micro_actions = np.load(micro_file)
    trading_info = np.load(trading_info_file, allow_pickle=True).item()

    assert len(reasons) == len(micro_actions) == 3
    # Step 0: policy inference (0), Step 1: persisted (1), Step 2: persisted (1)
    assert reasons.tolist() == [0, 1, 1]
    assert trading_info["total_steps"] == 3
    assert trading_info["inference_steps"] == 1
    assert trading_info["persistence_held_steps"] == 2
    assert trading_info["skip_inference_ratio"] == pytest.approx(2.0 / 3.0)
    assert trading_info["defensive_preemptions"] == 0
    assert trading_info["action_unavailable_breaks"] == 0


def test_reconfigure_routing_resets_persistence_state():
    router = _create_test_router(action_persistence=3)
    router.remaining_persist = 2
    router.current_action = 2
    router.action_decision_reason_history = [0, 1]

    args = types.SimpleNamespace(
        gamma=0.95,
        rule_base_threshold=0.25,
        window_length=100,
        slope_window_length=100,
        volatility_window_length=100,
        slope_gamma=0.95,
        volatility_gamma=0.95,
        slope_rule_base_threshold=0.25,
        volatility_rule_base_threshold=0.25,
        trial_number=1,
        enable_non_main_contract_defense=False,
        action_persistence=4,
        gating_strategy="absolute",
        ood_threshold=0.005,
        slope_margin_threshold=0.12,
        volatility_margin_threshold=0.12,
    )
    router._resolve_test_path = lambda a: "/tmp/trial_test"

    router.reconfigure_routing(args)

    assert router.action_persistence == 4
    assert router.remaining_persist == 0
    assert router.current_action == router.flat_action
    assert router.action_decision_reason_history == []
