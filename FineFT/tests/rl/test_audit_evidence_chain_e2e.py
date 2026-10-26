from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest
import torch

if "optuna" not in sys.modules:
    sys.modules["optuna"] = MagicMock()

from common.artifacts import ActionDecisionReasons, AuditArtifactNames
from RL.DiHFT.high_level import vae_routing_util as vru
from analysis.diagnostics.daily_trading_audit import DailyTradingAuditPublisher
from analysis.pick_agent.FineFT_two_dimensional_agent_selector import (
    TwoDimensionalSelectionManifest,
)
from env.env_initiate.base_initiate import initiate_base_env


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


def _create_e2e_router(
    tmp_path: Path,
    action_persistence: int = 2,
) -> vru.vae_risk_aware_routing:
    router = vru.vae_risk_aware_routing.__new__(vru.vae_risk_aware_routing)
    router.device = "cpu"
    router.num_labels = 3
    router.slot_count = 9
    router.rule_base_threshold = 0.2
    router.axis_thresholds = {"volatility": 0.2, "slope": 0.2}
    router.selection_manifest = TwoDimensionalSelectionManifest.from_dict(
        _sample_manifest_payload(num_labels=3)
    )
    router.enable_non_main_contract_defense = False
    router.role_tier_index = None
    router.zero_position_action = 1
    router.flat_action = 1
    router.action_persistence = action_persistence
    router.remaining_persist = 0
    router.current_action = 1
    router.action_decision_reason_history = []
    router.action = 1
    router.macro_action_history = []
    router.leverage_choices = [1]
    router.position_list = [-1.0, 0.0, 1.0]
    router.calculate_axis_window_result = lambda axis: {
        "volatility": [0.1, 0.8, 0.2],
        "slope": [0.1, 0.2, 0.9],
    }[axis]
    router.axis_window_lengths = {"volatility": 1, "slope": 1}
    router.gating_strategy = vru.create_gating_strategy(
        "expert_hysteresis",
        slope_threshold=0.2,
        volatility_threshold=0.2,
    )
    router._defensive_action = lambda info, pos, lev: 1
    router.save_artifacts = True
    router.initial_wallet_balance = 10000.0
    router.vae_slope_indicators = []
    router.vae_volatility_indicators = []
    router.tech_indicator_list = ["mark_price"]
    router.max_holding_number = 1
    router.position_choices = 3
    router.N_ACTIONS = 3
    router.long_estimated_rate = 0.0
    router.short_estimated_rate = 0.0
    router.transcation_cost = 0.0001
    router.maintenance_margin_ratio_dict = {}
    router.early_stop = 0
    router.initial_state = (10000.0, 0.0, 0.0, 0.0, 1.0)
    router.order_book_depth = 1
    router.precomputed_quantiles_dir = None
    router.current_contract_quantiles = None
    router.stop_loss_return_threshold = 0.0
    router.stop_loss_cooldown_steps = 0
    router.circuit_breaker_consecutive_stops = 0
    router.circuit_breaker_cooling_steps = 0
    router.circuit_breaker_remaining_steps = 0
    router.cooldown_remaining_steps = 0
    router.consecutive_stop_loss_count = 0
    router.hard_stop_loss_count = 0
    router.cooldown_intercept_count = 0
    router.circuit_breaker_suspension_count = 0
    router.last_stopped_position = 0.0

    # Deterministic mock Q-network output
    class FakeLowLevelNetwork(torch.nn.Module):
        def forward(self, state, time, previous_action, avaliable_action, trading_info):
            # return shape (1, 9, 3)
            # slot 4 has action values [0.2, 0.5, 1.8] -> Long (2)
            out = torch.zeros(1, 9, 3)
            out[:, :, :] = torch.tensor([0.2, 0.5, 1.8])
            return out

    router.low_level_network = FakeLowLevelNetwork()
    router.run_id = "e2e_run_01"
    router.model_id = "test_model_v1"
    router.feature_version = "dual_stream_v2"
    router.parameter_version = "params_v1"
    router.quantiles = {"slope": [[] for _ in range(3)], "volatility": [[] for _ in range(3)]}
    router.get_quantiles = types.MethodType(lambda self, *args, **kwargs: None, router)
    router.initial_rollout_window_length = 1

    return router


def test_high_level_backtest_to_daily_audit_e2e(tmp_path: Path):
    contract = "fu2508"
    n_steps = 5
    mark_prices = np.linspace(3000.0, 3050.0, n_steps + 1)
    timestamps = pd.date_range("2025-01-02 09:30:00", periods=n_steps + 1, freq="10min")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "trading_day": ["2025-01-02"] * (n_steps + 1),
        "funding_rate": np.zeros(n_steps + 1),
        "funding_timestamp": timestamps,
        "mark_price": mark_prices,
        "ask1_price": mark_prices + 0.1,
        "bid1_price": mark_prices - 0.1,
        "ask1_size": np.ones(n_steps + 1) * 100.0,
        "bid1_size": np.ones(n_steps + 1) * 100.0,
    })

    market_dir = tmp_path / "market"
    market_dir.mkdir(parents=True, exist_ok=True)
    df.to_feather(market_dir / f"{contract}.feather")

    result_dir = tmp_path / "result"
    contract_result_dir = result_dir / "contracts" / contract

    router = _create_e2e_router(tmp_path, action_persistence=2)

    # Run the high-level backtest
    res = router.run_single_valid_df(df, str(contract_result_dir), contract_name=contract)
    assert res["rows"] == len(df)

    # Verify audit artifacts exist in the result directory
    assert (contract_result_dir / AuditArtifactNames.AUDIT_PREDICTIONS_JSON).is_file()
    assert (contract_result_dir / AuditArtifactNames.AUDIT_TRADES_JSON).is_file()
    assert (contract_result_dir / AuditArtifactNames.AUDIT_ACCOUNT_SNAPSHOTS_JSON).is_file()
    assert (contract_result_dir / AuditArtifactNames.AUDIT_METADATA_JSON).is_file()
    assert (contract_result_dir / AuditArtifactNames.AUDIT_ROUTING_JSON).is_file()

    # Read the audit artifacts
    with open(contract_result_dir / AuditArtifactNames.AUDIT_PREDICTIONS_JSON, encoding="utf-8") as f:
        preds = json.load(f)
    with open(contract_result_dir / AuditArtifactNames.AUDIT_TRADES_JSON, encoding="utf-8") as f:
        trades = json.load(f)
    with open(contract_result_dir / AuditArtifactNames.AUDIT_ACCOUNT_SNAPSHOTS_JSON, encoding="utf-8") as f:
        accounts = json.load(f)

    # 1. Warm-up step was executed at step 0 (initial_rollout window = 1)
    assert trades[0]["decision_reason"] == "WARM_UP"
    assert trades[0]["prediction_id"] == ""

    # 2. At step 1, genuine policy inference was performed
    assert preds[0]["step_index"] == 1
    assert preds[0]["action_value_long"] == pytest.approx(1.8)
    assert preds[0]["action_value_margin"] == pytest.approx(1.8 - 0.5)
    assert trades[1]["prediction_id"] == preds[0]["prediction_id"]

    # 3. The retained expert performs genuine fresh inference on every step.
    assert trades[2]["decision_reason"] == "POLICY_INFERENCE"
    pred_steps = [p["step_index"] for p in preds]
    assert pred_steps == [1, 2, 3, 4]
    assert trades[2]["prediction_id"] == preds[1]["prediction_id"]

    # 4. Account snapshots: 1 initial + 5 post_step = 6 snapshots
    assert len(accounts) == n_steps + 1
    assert accounts[0]["snapshot_type"] == "initial"

    # Now run the offline diagnostic publication command
    output_dir = tmp_path / "published_daily"
    publisher = DailyTradingAuditPublisher(
        result_dir=result_dir,
        market_data_dir=market_dir,
        output_dir=output_dir,
        run_id="e2e_run_01",
    )
    manifest = publisher.publish()

    assert manifest["validation_status"] == "passed"
    day_dir = output_dir / "2025-01-02"
    assert (day_dir / AuditArtifactNames.PREDICTION_RECORDS_CSV).is_file()
    assert (day_dir / AuditArtifactNames.TRADE_RECORDS_CSV).is_file()
    assert (day_dir / AuditArtifactNames.ACCOUNT_RECORDS_CSV).is_file()
    assert (day_dir / AuditArtifactNames.ROUTING_RECORDS_CSV).is_file()
    assert manifest["schema_version"] == "1.1.0"
