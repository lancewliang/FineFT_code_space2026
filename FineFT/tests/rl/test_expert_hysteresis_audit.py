"""Public expert-routing evidence stays linked through daily audit publication."""

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
from tests.rl.test_expert_hysteresis_routing import act, make_router


def _market_frame(steps: int) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2025-01-02 09:30:00",
        periods=steps + 1,
        freq="10min",
    )
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


def _replay(tmp_path: Path):
    router = make_router(confirm_steps=1)
    router.device = "cpu"
    router.axis_thresholds = {"volatility": 0.8, "slope": 0.6}
    router.axis_window_lengths = {"volatility": 1, "slope": 1}
    router.calculate_axis_window_result = lambda axis: {
        "volatility": [0.1, 0.8, 0.2],
        "slope": [0.1, 0.2, 0.6],
    }[axis]
    router.N_ACTIONS = 3
    router.save_artifacts = True
    router.initial_wallet_balance = 10000.0
    router.vae_slope_indicators = []
    router.vae_volatility_indicators = []
    router.tech_indicator_list = ["mark_price"]
    router.max_holding_number = 1
    router.position_choices = 3
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
    router.previous_step_position = 0.0
    router.active_trade_stopped = False
    router.run_id = "routing_audit_run_01"
    router.model_id = "test_model_v1"
    router.feature_version = "triple_stream_v1"
    router.parameter_version = "params_v1"
    router.quantiles = {
        "slope": [[] for _ in range(3)],
        "volatility": [[] for _ in range(3)],
    }
    router.get_quantiles = lambda *args, **kwargs: None
    router.initial_rollout_window_length = 1
    data = _market_frame(5)
    result_dir = tmp_path / "result"
    contract_dir = result_dir / "contracts" / "fu2508"
    router.run_single_valid_df(data, str(contract_dir), "fu2508")
    return router, data, result_dir, contract_dir


def _publish(tmp_path: Path, data: pd.DataFrame, result_dir: Path):
    market_dir = tmp_path / "market"
    market_dir.mkdir(exist_ok=True)
    data.to_feather(market_dir / "fu2508.feather")
    output_dir = tmp_path / "published"
    manifest = DailyTradingAuditPublisher(
        result_dir,
        market_dir,
        output_dir,
    ).publish()
    return manifest, output_dir


def test_completed_candidate_and_intercepted_inference_remain_observable():
    router = make_router(confirm_steps=1)
    unavailable = {
        "previous_action": 1,
        "avaliable_action": [1, 1, 0],
        "trading_info": [0.0],
    }

    assert act(router, info=unavailable) == 1

    record = router.latest_routing_record
    prediction = router.audit_predictions[-1]
    assert record["completed_candidate_slot"] == 5
    assert record["completed_candidate_count"] == 1
    assert record["candidate_slot_after"] is None
    assert record["candidate_count_after"] == 0
    assert record["active_slot_after"] is None
    assert record["queried_slot"] == 5
    assert record["proposed_action"] == prediction["selected_action"] == 2
    assert record["final_action"] == 1
    assert record["prediction_id"] == prediction["prediction_id"]
    assert record["routing_reason"] == "action_unavailable"
    assert record["transition_reason"] == "expert_entered"


def test_pre_inference_risk_has_no_fake_prediction():
    router = make_router(confirm_steps=1)
    assert act(router) == 2
    prediction_count = len(router.audit_predictions)
    router.circuit_breaker_remaining_steps = 1

    assert act(router) == 1

    record = router.latest_routing_record
    assert record["routing_reason"] == "circuit_breaker_suspension"
    assert record["queried_slot"] is None
    assert record["proposed_action"] is None
    assert record["prediction_id"] == ""
    assert len(router.audit_predictions) == prediction_count


def test_route_lifecycle_distinguishes_entry_replacement_invalidation_and_recovery():
    router = make_router(confirm_steps=2)
    rival = ([0.95, 0.8, 0.1], [0.9, 0.1, 0.6])
    invalidation = ([0.95, 0.3, 0.1], [0.9, 0.1, 0.2])
    ineligible = ([0.2, 0.3, 0.1], [0.2, 0.1, 0.1])
    recovery = ([0.8, 0.1, 0.1], [0.1, 0.1, 0.6])

    assert act(router) == 1
    waiting = router.latest_routing_record
    assert waiting["routing_reason"] == "entry_confirmation"
    assert (waiting["candidate_slot_after"], waiting["candidate_count_after"]) == (5, 1)

    assert act(router) == 2
    entered = router.latest_routing_record
    assert entered["admission_kind"] == "first_entry"
    assert entered["completed_candidate_slot"] == 5
    assert entered["completed_candidate_count"] == 2
    assert entered["candidate_slot_after"] is None

    assert act(router) == 2
    retained = router.latest_routing_record
    assert retained["routing_reason"] == "expert_retained"
    assert retained["active_slot_before"] == retained["active_slot_after"] == 5

    assert act(router, *rival) == 2
    assert router.latest_routing_record["routing_reason"] == "switch_confirmation"
    assert act(router, *invalidation) == 2
    replaced = router.latest_routing_record
    assert replaced["incumbent_invalidated"]
    assert replaced["active_slot_before"] == 5
    assert replaced["active_slot_after"] == 0
    assert replaced["completed_candidate_slot"] == 0
    assert replaced["admission_kind"] == "replacement"

    assert act(router, *ineligible) == 1
    invalidated = router.latest_routing_record
    assert invalidated["incumbent_invalidated"]
    assert invalidated["active_slot_after"] is None
    assert invalidated["routing_reason"] == "entry_threshold"

    assert act(router, *recovery) == 1
    assert act(router, *recovery) == 2
    recovered = router.latest_routing_record
    assert recovered["active_slot_after"] == 2
    assert recovered["admission_kind"] == "recovery"


def test_full_route_artifact_publishes_with_decision_and_prediction_joins(tmp_path):
    router, data, result_dir, contract_dir = _replay(tmp_path)
    routes = json.loads(
        (contract_dir / AuditArtifactNames.AUDIT_ROUTING_JSON).read_text()
    )
    predictions = json.loads(
        (contract_dir / AuditArtifactNames.AUDIT_PREDICTIONS_JSON).read_text()
    )
    trades = json.loads(
        (contract_dir / AuditArtifactNames.AUDIT_TRADES_JSON).read_text()
    )

    assert [record["step_index"] for record in routes] == list(range(5))
    assert routes[0]["routing_reason"] == "warm_up"
    assert routes[0]["prediction_id"] == ""
    assert all(record["queried_slot"] == 5 for record in routes[1:])
    assert [prediction["step_index"] for prediction in predictions] == [1, 2, 3, 4]
    for record in routes:
        assert record["decision_id"] == build_decision_id(
            router.run_id,
            "fu2508",
            record["step_index"],
        )
    assert {trade["decision_id"] for trade in trades} == {
        record["decision_id"] for record in routes
    }

    manifest, output_dir = _publish(tmp_path, data, result_dir)
    assert manifest["schema_version"] == "1.1.0"
    assert manifest["total_row_counts"]["routing_records"] == 5
    published = pd.read_csv(
        output_dir / "2025-01-02" / AuditArtifactNames.ROUTING_RECORDS_CSV
    )
    assert published["decision_id"].tolist() == [
        record["decision_id"] for record in routes
    ]
    assert json.loads(published.iloc[1]["volatility_scores"]) == [0.1, 0.8, 0.2]


@pytest.mark.parametrize("fault", ["missing", "schema", "prediction", "coverage"])
def test_corrupt_route_evidence_blocks_atomic_publication(tmp_path, fault):
    _, data, result_dir, contract_dir = _replay(tmp_path)
    route_path = contract_dir / AuditArtifactNames.AUDIT_ROUTING_JSON
    routes = json.loads(route_path.read_text())
    if fault == "missing":
        route_path.unlink()
    else:
        if fault == "schema":
            routes[0]["schema_version"] = "0.9.0"
        elif fault == "prediction":
            routes[1]["prediction_id"] = "pred_missing"
        else:
            routes.pop()
        route_path.write_text(json.dumps(routes))

    with pytest.raises((KeyError, ValueError)):
        _publish(tmp_path, data, result_dir)
    assert not (tmp_path / "published").exists()
