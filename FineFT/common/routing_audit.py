"""Immutable expert routing evidence, separate from real inference and execution."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any

from common.audit_records import PredictionRecord, TradeRecord, build_decision_id


@dataclass(frozen=True, slots=True)
class RoutingRecord:
    schema_version: str
    decision_id: str
    run_id: str
    contract: str
    step_index: int
    event_time: str
    trading_day: str
    strategy_version: str
    dominant_slot: int | None
    active_slot_before: int | None
    active_slot_after: int | None
    candidate_slot_before: int | None
    candidate_slot_after: int | None
    candidate_count_before: int
    candidate_count_after: int
    completed_candidate_slot: int | None
    completed_candidate_count: int
    incumbent_invalidated: bool
    routing_reason: str
    transition_reason: str
    admission_kind: str
    action_reason: str
    queried_slot: int | None
    proposed_action: int | None
    final_action: int
    prediction_id: str
    volatility_scores: list[float]
    slope_scores: list[float]
    volatility_entry_threshold: float
    slope_entry_threshold: float
    volatility_holding_threshold: float
    slope_holding_threshold: float
    hysteresis_exit_ratio: float
    expert_switch_lead: float
    expert_confirm_steps: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize an owned snapshot for JSON/CSV artifact boundaries."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RoutingRecord:
        """Read only the versioned fields of a persisted routing record."""
        return cls(**{field.name: data[field.name] for field in fields(cls)})

    def validate(self) -> None:
        """Reject incomplete, non-finite or incompatible persisted evidence."""
        if self.schema_version != "1.0.0" or self.strategy_version != "expert_hysteresis.v1":
            raise ValueError("Unsupported routing audit schema or strategy version")
        if self.step_index < 0 or self.decision_id != build_decision_id(
            self.run_id, self.contract, self.step_index
        ):
            raise ValueError(f"Invalid routing decision identity: {self.decision_id}")
        if not self.event_time or not self.trading_day:
            raise ValueError(f"Missing routing event time or TradingDay: {self.decision_id}")
        scores = self.volatility_scores + self.slope_scores
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores):
            raise ValueError(f"Invalid routing score: {self.decision_id}")
        if self.routing_reason != "warm_up" and (
            not self.volatility_scores or not self.slope_scores
        ):
            raise ValueError(f"Missing routing scores: {self.decision_id}")
        slots = (
            self.dominant_slot,
            self.active_slot_before,
            self.active_slot_after,
            self.candidate_slot_before,
            self.candidate_slot_after,
            self.completed_candidate_slot,
            self.queried_slot,
        )
        slot_count = len(self.volatility_scores) * len(self.slope_scores)
        if any(
            slot is not None and (type(slot) is not int or not 0 <= slot < slot_count)
            for slot in slots
        ):
            raise ValueError(f"Invalid routing Slot identity: {self.decision_id}")
        for candidate, count in (
            (self.candidate_slot_before, self.candidate_count_before),
            (self.candidate_slot_after, self.candidate_count_after),
        ):
            if type(count) is not int or (candidate is None) != (count == 0):
                raise ValueError(f"Invalid pending routing candidate: {self.decision_id}")
        if self.incumbent_invalidated and self.active_slot_before is None:
            raise ValueError(f"Invalid incumbent invalidation fact: {self.decision_id}")
        thresholds = (
            (self.volatility_holding_threshold, self.volatility_entry_threshold),
            (self.slope_holding_threshold, self.slope_entry_threshold),
        )
        if any(not 0 <= hold < entry <= 1 for hold, entry in thresholds):
            raise ValueError(f"Invalid routing thresholds: {self.decision_id}")
        if not 0 < self.hysteresis_exit_ratio < 1 or any(
            not math.isclose(hold, entry * self.hysteresis_exit_ratio) for hold, entry in thresholds
        ):
            raise ValueError(f"Inconsistent routing exit ratio: {self.decision_id}")
        if (
            not math.isfinite(self.expert_switch_lead)
            or self.expert_switch_lead < 0
            or type(self.expert_confirm_steps) is not int
            or self.expert_confirm_steps < 1
            or min(self.candidate_count_before, self.candidate_count_after) < 0
        ):
            raise ValueError(f"Invalid routing confirmation configuration: {self.decision_id}")
        if type(self.final_action) is not int or (
            self.proposed_action is not None and type(self.proposed_action) is not int
        ):
            raise ValueError(f"Invalid routing action: {self.decision_id}")
        if (self.queried_slot is None) != (self.proposed_action is None):
            raise ValueError(f"Incomplete routing inference provenance: {self.decision_id}")
        if bool(self.prediction_id) != (self.queried_slot is not None):
            raise ValueError(f"Inconsistent routing prediction reference: {self.decision_id}")
        if self.completed_candidate_slot is None:
            if self.completed_candidate_count != 0:
                raise ValueError(f"Missing completed routing candidate: {self.decision_id}")
        elif self.completed_candidate_count != self.expert_confirm_steps:
            raise ValueError(f"Incomplete routing candidate confirmation: {self.decision_id}")


ROUTING_RECORD_COLUMNS = [field.name for field in fields(RoutingRecord)]


def validate_routing_evidence(
    records: list[RoutingRecord],
    predictions: list[PredictionRecord],
    trades: list[TradeRecord],
    action_positions: dict[str, list[float]],
) -> None:
    """Validate complete decision coverage and genuine inference/execution joins."""
    by_decision: dict[str, RoutingRecord] = {}
    for record in records:
        record.validate()
        if record.decision_id in by_decision:
            raise ValueError(f"Duplicate routing decision: {record.decision_id}")
        by_decision[record.decision_id] = record
    route_contracts = set(action_positions)
    relevant_trades = [trade for trade in trades if trade.contract in route_contracts]
    if set(by_decision) != {trade.decision_id for trade in relevant_trades}:
        raise ValueError("Routing decision coverage does not match executed steps")
    predictions_by_id = {prediction.prediction_id: prediction for prediction in predictions}
    for record in records:
        if record.prediction_id:
            prediction = predictions_by_id.get(record.prediction_id)
            if prediction is None or (
                prediction.decision_id != record.decision_id
                or prediction.routing_slot != record.queried_slot
                or prediction.selected_action != record.proposed_action
                or prediction.run_id != record.run_id
                or prediction.contract != record.contract
                or prediction.step_index != record.step_index
            ):
                raise ValueError(f"Routing prediction provenance mismatch: {record.decision_id}")
        positions = action_positions[record.contract]
        if not 0 <= record.final_action < len(positions):
            raise ValueError(f"Invalid routing final action: {record.decision_id}")
    final_legs: dict[str, TradeRecord] = {}
    for trade in relevant_trades:
        record = by_decision[trade.decision_id]
        if (
            trade.run_id != record.run_id
            or trade.contract != record.contract
            or trade.step_index != record.step_index
            or trade.event_time != record.event_time
            or trade.trading_day != record.trading_day
            or trade.prediction_id != record.prediction_id
            or trade.decision_reason != record.action_reason
        ):
            raise ValueError(f"Routing execution identity mismatch: {record.decision_id}")
        if trade.decision_id not in final_legs or (
            trade.leg_index > final_legs[trade.decision_id].leg_index
        ):
            final_legs[trade.decision_id] = trade
    for decision_id, trade in final_legs.items():
        record = by_decision[decision_id]
        if not math.isclose(
            trade.requested_position, action_positions[record.contract][record.final_action]
        ):
            raise ValueError(f"Routing final action/execution mismatch: {decision_id}")
    genuine_ids = {
        prediction.prediction_id
        for prediction in predictions
        if prediction.contract in route_contracts
    }
    if genuine_ids != {record.prediction_id for record in records if record.prediction_id}:
        raise ValueError("Routing evidence omits genuine expert inference")
