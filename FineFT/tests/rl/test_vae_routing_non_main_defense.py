import json
import argparse
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

if "optuna" not in sys.modules:
    sys.modules["optuna"] = MagicMock()

from analysis.pick_agent.FineFT_two_dimensional_agent_selector import (
    TwoDimensionalSelectionManifest,
)
from RL.DiHFT.high_level import vae_routing_optuna as vro
from RL.DiHFT.high_level import vae_routing_util as vru


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


def _create_mock_router(
    enable_defense: bool = True,
    role_tier_index: int | None = 0,
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
    routing.zero_position_action = 4
    routing.leverage_choices = [5]
    routing.position_list = [-1.0, 0.0, 1.0]
    routing.action = 4
    routing.macro_action_history = []
    routing.calculate_axis_window_result = lambda axis: {
        "volatility": [0.1, 0.8, 0.2],
        "slope": [0.1, 0.2, 0.9],
    }[axis]
    routing.agent_act = lambda state, info: 7
    return routing


def test_prepare_base_args_forwards_enable_non_main_contract_defense(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(_sample_manifest_payload(num_labels=3)),
        encoding="utf-8",
    )
    args_1 = types.SimpleNamespace(
        dataset_name="BTCUSDT",
        experiment_name="default",
        max_holding_number=8.0,
        order_book_depth=25,
        allow_reverse_position=False,
        selection_manifest=str(manifest_path),
        enable_non_main_contract_defense=False,
    )
    args_2 = types.SimpleNamespace(
        dataset_name="fu",
        experiment_name="10min_parallel",
        max_holding_number=1.0,
        order_book_depth=5,
        allow_reverse_position=True,
        selection_manifest=str(manifest_path),
        enable_non_main_contract_defense=True,
    )

    base_args = vro.prepare_base_args(args_1, args_2)

    assert base_args.enable_non_main_contract_defense is True


def test_non_main_contract_defense_triggers_rule_based_close(monkeypatch):
    routing = _create_mock_router(enable_defense=True, role_tier_index=0)
    monkeypatch.setattr(vru, "rule_based_close", lambda *args: 4)

    def fail_if_agent_act_called(state, info):
        raise AssertionError("agent_act must not be called on non-main contract")

    routing.agent_act = fail_if_agent_act_called

    # role_tier = 0.0 (non-main / cold-start)
    s = np.array([0.0, 100.0, 200.0])
    action = routing.get_action({}, s, 1.0, 5)

    assert action == 4
    assert routing.macro_action_history == [9]


@pytest.mark.parametrize("role_tier", [0.5, 1.0])
def test_main_and_sub_main_contracts_route_normally(role_tier):
    routing = _create_mock_router(enable_defense=True, role_tier_index=0)

    # role_tier = 0.5 (sub-main) or 1.0 (main)
    s = np.array([role_tier, 100.0, 200.0])
    action = routing.get_action({}, s, 0.0, 5)

    # normal routing selects slot 1 * 3 + 2 = 5, agent_act returns 7
    assert action == 7
    assert routing.selected_agent_index == 5
    assert routing.macro_action_history == [5]


def test_defense_disabled_by_default_routes_normally_on_non_main():
    routing = _create_mock_router(enable_defense=False, role_tier_index=0)

    # role_tier = 0.0, but defense is disabled (ablation baseline)
    s = np.array([0.0, 100.0, 200.0])
    action = routing.get_action({}, s, 0.0, 5)

    assert action == 7
    assert routing.selected_agent_index == 5
    assert routing.macro_action_history == [5]


def test_defense_gracefully_handles_none_role_tier_index():
    routing = _create_mock_router(enable_defense=True, role_tier_index=None)

    # role_tier_index is None, should route normally without error
    s = np.array([100.0, 200.0])
    action = routing.get_action({}, s, 0.0, 5)

    assert action == 7
    assert routing.selected_agent_index == 5
    assert routing.macro_action_history == [5]


def test_reconfigure_routing_preserves_defense_flag(monkeypatch):
    routing = _create_mock_router(enable_defense=False, role_tier_index=0)
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
        enable_non_main_contract_defense=True,
    )
    monkeypatch.setattr(routing, "_resolve_test_path", lambda a: "/tmp/test_trial_1")
    monkeypatch.setattr(routing, "reset_routing_state", lambda: None)

    routing.reconfigure_routing(args)

    assert routing.enable_non_main_contract_defense is True
