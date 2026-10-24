"""Public routing decisions survive full export and atomic daily publication."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from analysis.diagnostics.daily_trading_audit import DailyTradingAuditPublisher
from common.artifacts import AuditArtifactNames
from common.audit_records import build_decision_id
from RL.DiHFT.high_level import vae_routing_util as vru
from tests.rl.test_audit_evidence_chain_e2e import _create_e2e_router
from tests.rl.test_expert_hysteresis_routing import act, make_router


def market_frame(steps: int) -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-02 09:30:00", periods=steps + 1, freq="10min")
    prices = np.full(steps + 1, 3000.0)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "trading_day": ["2025-01-02"] * (steps + 1),
            "funding_rate": np.zeros(steps + 1),
            "funding_timestamp": timestamps,
            "mark_price": prices,
            "ask1_price": prices + 0.1,
            "bid1_price": prices - 0.1,
            "ask1_size": np.full(steps + 1, 100.0),
            "bid1_size": np.full(steps + 1, 100.0),
        }
    )


def replay(tmp_path: Path, *, save_artifacts=True):
    router = _create_e2e_router(tmp_path, action_persistence=5)
    router.gating_strategy = vru.create_gating_strategy(
        "expert_hysteresis",
        slope_threshold=0.6,
        volatility_threshold=0.8,
        hysteresis_exit_ratio=0.5,
        expert_confirm_steps=2,
    )
    router.axis_window_lengths = {"volatility": 1, "slope": 1}
    router.selection_manifest.slots[8]["kind"] = "empty_model"
    router.save_artifacts = save_artifacts
    baseline = ([0.1, 0.8, 0.2], [0.1, 0.2, 0.6])
    rival = ([0.95, 0.8, 0.1], [0.9, 0.1, 0.6])
    invalidation = ([0.95, 0.3, 0.1], [0.9, 0.1, 0.2])
    ineligible = ([0.2, 0.3, 0.1], [0.2, 0.1, 0.1])
    recovery = ([0.8, 0.1, 0.1], [0.1, 0.1, 0.6])
    empty = ([0.1, 0.2, 0.95], [0.1, 0.2, 0.95])
    sequence = {
        4: rival,
        5: invalidation,
        6: ineligible,
        7: recovery,
        8: recovery,
        **{step: empty for step in range(9, 14)},
    }
    router.calculate_axis_window_result = lambda axis: sequence.get(router.step_idx, baseline)[
        0 if axis == "volatility" else 1
    ]
    df = market_frame(15)
    result_dir = tmp_path / "result"
    contract_dir = result_dir / "contracts" / "fu2508"
    metrics = router.run_single_valid_df(df, str(contract_dir), "fu2508")
    return router, df, metrics, result_dir, contract_dir


def publish(tmp_path, df, result_dir):
    market = tmp_path / "market"
    market.mkdir(exist_ok=True)
    df.to_feather(market / "fu2508.feather")
    output = tmp_path / "published"
    manifest = DailyTradingAuditPublisher(result_dir, market, output).publish()
    return manifest, output


def test_full_routing_trace_preserves_completion_invalidation_recovery_and_empty(tmp_path):
    router, df, _, result_dir, contract_dir = replay(tmp_path)
    records = json.loads((contract_dir / AuditArtifactNames.AUDIT_ROUTING_JSON).read_text())
    predictions = json.loads((contract_dir / AuditArtifactNames.AUDIT_PREDICTIONS_JSON).read_text())
    trades = json.loads((contract_dir / AuditArtifactNames.AUDIT_TRADES_JSON).read_text())
    assert [record["step_index"] for record in records] == list(range(15))
    assert records[0]["routing_reason"] == "warm_up"
    assert records[0]["prediction_id"] == ""
    assert records[1]["candidate_count_after"] == 1
    assert records[2]["admission_kind"] == "first_entry"
    assert records[2]["candidate_count_before"] == 1
    assert records[2]["completed_candidate_slot"] == 5
    assert records[2]["completed_candidate_count"] == 2
    assert records[2]["candidate_slot_after"] is None
    assert records[3]["routing_reason"] == "expert_retained"
    assert records[4]["candidate_slot_after"] == 0
    direct = records[5]
    assert direct["active_slot_before"] == 5
    assert direct["active_slot_after"] == 0
    assert direct["incumbent_invalidated"]
    assert direct["completed_candidate_slot"] == 0
    assert direct["completed_candidate_count"] == 2
    assert direct["admission_kind"] == "replacement"
    assert records[6]["incumbent_invalidated"]
    assert records[6]["active_slot_after"] is None
    assert records[7]["routing_reason"] == "entry_confirmation"
    assert records[8]["admission_kind"] == "recovery"
    assert records[8]["active_slot_after"] == 2
    empty_records = records[9:14]
    assert all(record["dominant_slot"] == 8 for record in empty_records)
    assert all(record["active_slot_after"] is None for record in empty_records)
    assert all(record["queried_slot"] is None for record in empty_records)
    assert all(record["prediction_id"] == "" for record in empty_records)
    assert [record["final_action"] for record in empty_records] == [2, 2, 2, 2, 1]
    assert records[14]["candidate_count_after"] == 1
    assert [prediction["step_index"] for prediction in predictions] == [2, 3, 4, 5, 8]
    assert records[2]["volatility_entry_threshold"] == 0.8
    assert records[2]["volatility_holding_threshold"] == 0.4
    assert records[2]["slope_entry_threshold"] == 0.6
    assert records[2]["slope_holding_threshold"] == 0.3
    for record in records:
        assert record["decision_id"] == build_decision_id(
            router.run_id, "fu2508", record["step_index"]
        )
    assert {trade["decision_id"] for trade in trades} == {
        record["decision_id"] for record in records
    }
    # Replacement while the same action remains Long creates no synthetic fee.
    assert [trade["commission"] for trade in trades if trade["step_index"] == 5] == [0.0]
    manifest, output = publish(tmp_path, df, result_dir)
    assert manifest["validation_status"] == "passed"
    assert manifest["schema_version"] == "1.1.0"
    assert manifest["total_row_counts"]["routing_records"] == 15
    daily = pd.read_csv(output / "2025-01-02" / AuditArtifactNames.ROUTING_RECORDS_CSV)
    assert daily["decision_id"].tolist() == [record["decision_id"] for record in records]
    assert json.loads(daily.iloc[5]["volatility_scores"]) == [0.95, 0.3, 0.1]


@pytest.mark.parametrize("risk", ["cooldown", "unavailable", "non_main"])
def test_actual_intercepted_inference_and_final_action_remain_distinct(risk):
    router = make_router(confirm_steps=1)
    if risk == "cooldown":
        router.cooldown_remaining_steps = 3
        router.last_stopped_position = 1.0
    elif risk == "non_main":
        router.enable_non_main_contract_defense = True
        router.non_main_defense_mode = "graceful_hold_exit"
        router.role_tier_index = 0
        router.agent_act = lambda state, info: 0
    info = {"previous_action": 1, "avaliable_action": [1, 1, risk != "unavailable"]}
    if risk == "non_main":
        # Make a real held contract non-main without relying on a private route counter.
        router.step_idx = -1
    assert act(router, position=1.0 if risk == "non_main" else 0.0, info=info) == 1
    record = router.latest_routing_record
    prediction = router.audit_predictions[-1]
    assert record["active_slot_after"] is None  # Provisional admission was rolled back.
    assert record["queried_slot"] == 5
    assert record["final_action"] == 1
    assert (
        record["proposed_action"]
        == prediction["selected_action"]
        == (0 if risk == "non_main" else 2)
    )
    assert record["prediction_id"] == prediction["prediction_id"]
    assert record["decision_id"] == prediction["decision_id"]
    assert record["completed_candidate_count"] == 1
    assert record["transition_reason"] == "expert_entered"


def test_risk_before_query_keeps_truthful_rule_record():
    router = make_router(confirm_steps=1)
    act(router)
    router.circuit_breaker_remaining_steps = 3
    assert act(router) == 1
    record = router.latest_routing_record
    assert record["routing_reason"] == "circuit_breaker_suspension"
    assert record["queried_slot"] is None
    assert record["prediction_id"] == ""
    assert len(router.audit_predictions) == 1


def test_lean_rollout_retains_no_growing_audit_and_matches_full_metrics(tmp_path):
    full, _, full_metrics, _, _ = replay(tmp_path / "full")
    lean, _, lean_metrics, result_dir, _ = replay(tmp_path / "lean", save_artifacts=False)
    assert lean_metrics.keys() == full_metrics.keys()
    for key in lean_metrics:
        np.testing.assert_equal(lean_metrics[key], full_metrics[key])
    assert lean.action_decision_reason_history == full.action_decision_reason_history
    assert lean.audit_routing == []
    assert lean.audit_predictions == []
    assert lean.audit_trades == []
    assert lean.latest_routing_record["step_index"] == 14
    assert not result_dir.exists()


@pytest.mark.parametrize("fault", ["missing", "duplicate", "prediction", "action", "coverage"])
def test_corrupt_new_route_evidence_fails_before_atomic_publication(tmp_path, fault):
    _, df, _, result_dir, contract_dir = replay(tmp_path)
    path = contract_dir / AuditArtifactNames.AUDIT_ROUTING_JSON
    records = json.loads(path.read_text())
    if fault == "missing":
        path.unlink()
    else:
        if fault == "duplicate":
            records.append(records[0])
        elif fault == "prediction":
            records[3]["prediction_id"] = records[2]["prediction_id"]
        elif fault == "action":
            records[3]["final_action"] = 1
        elif fault == "coverage":
            records.pop()
        path.write_text(json.dumps(records))
    with pytest.raises(ValueError):
        publish(tmp_path, df, result_dir)
    assert not (tmp_path / "published").exists()


def test_full_replay_routing_events_and_join_keys_are_deterministic(tmp_path):
    first, _, _, _, first_dir = replay(tmp_path / "first")
    second, _, _, _, second_dir = replay(tmp_path / "second")
    assert first.run_id == second.run_id
    for artifact in (
        AuditArtifactNames.AUDIT_ROUTING_JSON,
        AuditArtifactNames.AUDIT_PREDICTIONS_JSON,
        AuditArtifactNames.AUDIT_TRADES_JSON,
        AuditArtifactNames.AUDIT_ACCOUNT_SNAPSHOTS_JSON,
    ):
        assert (first_dir / artifact).read_bytes() == (second_dir / artifact).read_bytes()


def test_candidate_changes_and_ties_keep_observable_before_state():
    router = make_router(confirm_steps=3)
    act(router)  # Slot 5 starts.
    first = router.latest_routing_record
    act(router, [0.8, 0.1, 0.2])  # Slot 2 starts, cannot pool Slot 5 evidence.
    changed = router.latest_routing_record
    act(router, [0.8, 0.8, 0.2])  # Existing Slot 2 wins the tie.
    tied = router.latest_routing_record
    act(router, [0.79, 0.1, 0.2])  # Entry threshold breaks confirmation.
    interrupted = router.latest_routing_record
    assert (first["candidate_slot_after"], first["candidate_count_after"]) == (5, 1)
    assert (changed["candidate_slot_before"], changed["candidate_count_before"]) == (5, 1)
    assert (changed["candidate_slot_after"], changed["candidate_count_after"]) == (2, 1)
    assert (tied["dominant_slot"], tied["candidate_count_after"]) == (2, 2)
    assert (interrupted["candidate_slot_before"], interrupted["candidate_count_before"]) == (2, 2)
    assert interrupted["candidate_slot_after"] is None
    assert interrupted["candidate_count_after"] == 0
    assert router.audit_predictions == []


def test_full_export_links_post_query_risk_prediction_to_actual_execution(tmp_path):
    router = _create_e2e_router(tmp_path, action_persistence=5)
    router.gating_strategy = vru.create_gating_strategy(
        "expert_hysteresis", slope_threshold=0.2, volatility_threshold=0.2
    )
    router.axis_window_lengths = {"volatility": 1, "slope": 1}
    original_query = router.agent_act

    def query(state, info):
        if router.step_idx == 2:
            router.cooldown_remaining_steps = 1
            router.last_stopped_position = 1.0
        return original_query(state, info)

    router.agent_act = query
    df = market_frame(4)
    result_dir = tmp_path / "result"
    contract_dir = result_dir / "contracts" / "fu2508"
    router.run_single_valid_df(df, str(contract_dir), "fu2508")
    record = router.audit_routing[2]
    prediction = next(item for item in router.audit_predictions if item["step_index"] == 2)
    trade = next(item for item in router.audit_trades if item["step_index"] == 2)
    assert record["routing_reason"] == "stop_loss_cooldown"
    assert record["final_action"] == 1
    assert record["proposed_action"] == prediction["selected_action"] == 2
    assert trade["requested_position"] == 0.0
    assert record["prediction_id"] == prediction["prediction_id"] == trade["prediction_id"]
    manifest, _ = publish(tmp_path, df, result_dir)
    assert manifest["validation_status"] == "passed"


def test_reconfiguration_identity_changes_only_with_effective_experiment_inputs():
    router = make_router()
    router.save_artifacts = False
    router._resolve_test_path = lambda args: "/unused"
    args = vru.parser.parse_args(["--gating_strategy", "expert_hysteresis"])
    vru.resolve_routing_parameters(args)
    router.reconfigure_routing(args)
    first = router.run_id
    router.reconfigure_routing(args)
    assert router.run_id == first
    args.expert_confirm_steps = 3
    router.reconfigure_routing(args)
    assert router.run_id != first
    args.frozen_configuration_identity = "frozen_config_sha256"
    router.reconfigure_routing(args)
    assert router.run_id == "run_expert_hysteresis_frozen_config_sha256"
