from __future__ import annotations

from dataclasses import dataclass

import pytest

from RL.DiHFT.high_level.gating.stack import create_high_level_gating_stack
from RL.DiHFT.high_level import vae_routing_util as vru
from common.audit_records import PredictionRecord


@dataclass
class _FakeStack:
    stack_name: str = "tri_module"


def test_high_level_gate_stack_defaults_to_legacy() -> None:
    args = vru.parser.parse_args([])

    assert args.high_level_gating_stack == "legacy"


def test_stack_factory_never_silently_falls_back_from_tri_module() -> None:
    legacy = object()

    selected = create_high_level_gating_stack(
        "legacy",
        legacy_stack=legacy,
    )
    assert selected is legacy

    with pytest.raises(ValueError, match="tri_module stack is not configured"):
        create_high_level_gating_stack("tri_module", legacy_stack=legacy)

    tri_module = _FakeStack()
    selected = create_high_level_gating_stack(
        "tri_module",
        legacy_stack=legacy,
        tri_module_stack=tri_module,
    )
    assert selected is tri_module


def test_prediction_record_keeps_legacy_utility_fields_nullable() -> None:
    record = PredictionRecord(
        prediction_id="pred_01",
        decision_id="dec_01",
        run_id="run_01",
        model_id="model_v1",
        feature_version="feat_v1",
        parameter_version="param_v1",
        contract="fu2508",
        step_index=1,
        event_time="2025-01-02 09:30:00",
        trading_day="2025-01-02",
        market_index=1,
        routing_slot=4,
        routing_evidence='{"vol": 0.1, "slope": 0.2}',
        action_value_short=0.5,
        action_value_flat=0.2,
        action_value_long=1.2,
        action_value_margin=0.7,
        available_actions="[1, 1, 1]",
        selected_action=2,
    )

    payload = record.to_dict()

    assert payload["gating_stack"] == "legacy"
    assert payload["candidate_action"] is None
    assert payload["utility_lower"] is None
    assert payload["utility_reject_reason"] is None
    assert PredictionRecord.from_dict(payload) == record
