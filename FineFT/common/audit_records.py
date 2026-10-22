from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Sequence


def build_prediction_id(run_id: str, contract: str, step: int) -> str:
    return f"pred_{run_id}_{contract}_{step:06d}"


def build_decision_id(run_id: str, contract: str, step: int) -> str:
    return f"dec_{run_id}_{contract}_{step:06d}"


def build_trade_id(run_id: str, contract: str, step: int, leg: int = 0) -> str:
    return f"trade_{run_id}_{contract}_{step:06d}_leg{leg}"


def build_account_snapshot_id(run_id: str, contract: str, step: int, kind: str) -> str:
    return f"acc_{run_id}_{contract}_{step:06d}_{kind}"


def calculate_action_value_margin(action_values: Sequence[float]) -> float:
    if len(action_values) < 2:
        return 0.0
    sorted_values = sorted([float(x) for x in action_values], reverse=True)
    return float(sorted_values[0] - sorted_values[1])


PREDICTION_RECORD_COLUMNS = [
    "prediction_id",
    "decision_id",
    "run_id",
    "model_id",
    "feature_version",
    "parameter_version",
    "contract",
    "step_index",
    "event_time",
    "trading_day",
    "market_index",
    "routing_slot",
    "routing_evidence",
    "action_value_short",
    "action_value_flat",
    "action_value_long",
    "action_value_margin",
    "available_actions",
    "selected_action",
    "gating_stack",
    "ood_state",
    "ood_reason",
    "routing_volatility_slot",
    "routing_slope_slot",
    "candidate_id",
    "candidate_kind",
    "candidate_action",
    "denied_baseline_action",
    "final_action",
    "utility_raw",
    "utility_calibrated",
    "utility_lower",
    "win_probability",
    "channel_eligible",
    "utility_reject_stage",
    "utility_reject_reason",
    "utility_artifact_id",
]

TRADE_RECORD_COLUMNS = [
    "trade_id",
    "decision_id",
    "prediction_id",
    "run_id",
    "contract",
    "step_index",
    "event_time",
    "trading_day",
    "leg_index",
    "event_type",
    "execution_status",
    "decision_reason",
    "requested_position",
    "executed_position",
    "requested_quantity",
    "executed_quantity",
    "remaining_quantity",
    "reference_price",
    "fill_price",
    "commission",
    "realized_pnl",
    "slippage",
    "pre_snapshot_id",
    "post_snapshot_id",
]

ACCOUNT_RECORD_COLUMNS = [
    "snapshot_id",
    "run_id",
    "contract",
    "account_id",
    "account_scope",
    "step_index",
    "snapshot_type",
    "event_time",
    "trading_day",
    "wallet_balance",
    "unrealized_pnl",
    "equity",
    "margin",
    "available_balance",
    "position",
    "exposure",
    "leverage",
    "peak_equity",
    "drawdown",
    "reward",
]


@dataclass(frozen=True)
class PredictionRecord:
    prediction_id: str
    decision_id: str
    run_id: str
    model_id: str
    feature_version: str
    parameter_version: str
    contract: str
    step_index: int
    event_time: str
    trading_day: str
    market_index: int
    routing_slot: int
    routing_evidence: str
    action_value_short: float
    action_value_flat: float
    action_value_long: float
    action_value_margin: float
    available_actions: str
    selected_action: int
    gating_stack: str = "legacy"
    ood_state: str | None = None
    ood_reason: str | None = None
    routing_volatility_slot: int | None = None
    routing_slope_slot: int | None = None
    candidate_id: str | None = None
    candidate_kind: str | None = None
    candidate_action: int | None = None
    denied_baseline_action: int | None = None
    final_action: int | None = None
    utility_raw: float | None = None
    utility_calibrated: float | None = None
    utility_lower: float | None = None
    win_probability: float | None = None
    channel_eligible: bool | None = None
    utility_reject_stage: str | None = None
    utility_reject_reason: str | None = None
    utility_artifact_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PredictionRecord:
        kwargs = {k: data[k] for k in PREDICTION_RECORD_COLUMNS if k in data}
        return cls(
            prediction_id=str(kwargs["prediction_id"]),
            decision_id=str(kwargs["decision_id"]),
            run_id=str(kwargs["run_id"]),
            model_id=str(kwargs["model_id"]),
            feature_version=str(kwargs["feature_version"]),
            parameter_version=str(kwargs["parameter_version"]),
            contract=str(kwargs["contract"]),
            step_index=int(kwargs["step_index"]),
            event_time=str(kwargs["event_time"]),
            trading_day=str(kwargs["trading_day"]),
            market_index=int(kwargs["market_index"]),
            routing_slot=int(kwargs["routing_slot"]),
            routing_evidence=str(kwargs["routing_evidence"]),
            action_value_short=float(kwargs["action_value_short"]),
            action_value_flat=float(kwargs["action_value_flat"]),
            action_value_long=float(kwargs["action_value_long"]),
            action_value_margin=float(kwargs["action_value_margin"]),
            available_actions=str(kwargs["available_actions"]),
            selected_action=int(kwargs["selected_action"]),
            gating_stack=str(kwargs.get("gating_stack", "legacy")),
            ood_state=kwargs.get("ood_state"),
            ood_reason=kwargs.get("ood_reason"),
            routing_volatility_slot=(
                int(kwargs["routing_volatility_slot"])
                if kwargs.get("routing_volatility_slot") is not None
                else None
            ),
            routing_slope_slot=(
                int(kwargs["routing_slope_slot"])
                if kwargs.get("routing_slope_slot") is not None
                else None
            ),
            candidate_id=kwargs.get("candidate_id"),
            candidate_kind=kwargs.get("candidate_kind"),
            candidate_action=(
                int(kwargs["candidate_action"])
                if kwargs.get("candidate_action") is not None
                else None
            ),
            denied_baseline_action=(
                int(kwargs["denied_baseline_action"])
                if kwargs.get("denied_baseline_action") is not None
                else None
            ),
            final_action=(
                int(kwargs["final_action"])
                if kwargs.get("final_action") is not None
                else None
            ),
            utility_raw=(
                float(kwargs["utility_raw"])
                if kwargs.get("utility_raw") is not None
                else None
            ),
            utility_calibrated=(
                float(kwargs["utility_calibrated"])
                if kwargs.get("utility_calibrated") is not None
                else None
            ),
            utility_lower=(
                float(kwargs["utility_lower"])
                if kwargs.get("utility_lower") is not None
                else None
            ),
            win_probability=(
                float(kwargs["win_probability"])
                if kwargs.get("win_probability") is not None
                else None
            ),
            channel_eligible=(
                bool(kwargs["channel_eligible"])
                if kwargs.get("channel_eligible") is not None
                else None
            ),
            utility_reject_stage=kwargs.get("utility_reject_stage"),
            utility_reject_reason=kwargs.get("utility_reject_reason"),
            utility_artifact_id=kwargs.get("utility_artifact_id"),
        )


@dataclass(frozen=True)
class TradeRecord:
    trade_id: str
    decision_id: str
    prediction_id: str
    run_id: str
    contract: str
    step_index: int
    event_time: str
    trading_day: str
    leg_index: int
    event_type: str
    execution_status: str
    decision_reason: str
    requested_position: float
    executed_position: float
    requested_quantity: float
    executed_quantity: float
    remaining_quantity: float
    reference_price: float
    fill_price: float
    commission: float
    realized_pnl: float
    slippage: float
    pre_snapshot_id: str
    post_snapshot_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TradeRecord:
        kwargs = {k: data[k] for k in TRADE_RECORD_COLUMNS if k in data}
        pred_val = kwargs.get("prediction_id")
        return cls(
            trade_id=str(kwargs["trade_id"]),
            decision_id=str(kwargs["decision_id"]),
            prediction_id="" if pred_val is None or str(pred_val).lower() == "none" or str(pred_val).lower() == "nan" else str(pred_val),
            run_id=str(kwargs["run_id"]),
            contract=str(kwargs["contract"]),
            step_index=int(kwargs["step_index"]),
            event_time=str(kwargs["event_time"]),
            trading_day=str(kwargs["trading_day"]),
            leg_index=int(kwargs["leg_index"]),
            event_type=str(kwargs["event_type"]),
            execution_status=str(kwargs["execution_status"]),
            decision_reason=str(kwargs["decision_reason"]),
            requested_position=float(kwargs["requested_position"]),
            executed_position=float(kwargs["executed_position"]),
            requested_quantity=float(kwargs["requested_quantity"]),
            executed_quantity=float(kwargs["executed_quantity"]),
            remaining_quantity=float(kwargs["remaining_quantity"]),
            reference_price=float(kwargs["reference_price"]),
            fill_price=float(kwargs["fill_price"]),
            commission=float(kwargs["commission"]),
            realized_pnl=float(kwargs["realized_pnl"]),
            slippage=float(kwargs["slippage"]),
            pre_snapshot_id=str(kwargs["pre_snapshot_id"]),
            post_snapshot_id=str(kwargs["post_snapshot_id"]),
        )


@dataclass(frozen=True)
class AccountRecord:
    snapshot_id: str
    run_id: str
    contract: str
    account_id: str
    account_scope: str
    step_index: int
    snapshot_type: str
    event_time: str
    trading_day: str
    wallet_balance: float
    unrealized_pnl: float
    equity: float
    margin: float
    available_balance: float
    position: float
    exposure: float
    leverage: float
    peak_equity: float
    drawdown: float
    reward: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AccountRecord:
        kwargs = {k: data[k] for k in ACCOUNT_RECORD_COLUMNS if k in data}
        return cls(
            snapshot_id=str(kwargs["snapshot_id"]),
            run_id=str(kwargs["run_id"]),
            contract=str(kwargs["contract"]),
            account_id=str(kwargs["account_id"]),
            account_scope=str(kwargs.get("account_scope", "independent_contract_wallet")),
            step_index=int(kwargs["step_index"]),
            snapshot_type=str(kwargs["snapshot_type"]),
            event_time=str(kwargs["event_time"]),
            trading_day=str(kwargs["trading_day"]),
            wallet_balance=float(kwargs["wallet_balance"]),
            unrealized_pnl=float(kwargs["unrealized_pnl"]),
            equity=float(kwargs["equity"]),
            margin=float(kwargs["margin"]),
            available_balance=float(kwargs["available_balance"]),
            position=float(kwargs["position"]),
            exposure=float(kwargs["exposure"]),
            leverage=float(kwargs["leverage"]),
            peak_equity=float(kwargs["peak_equity"]),
            drawdown=float(kwargs["drawdown"]),
            reward=float(kwargs["reward"]),
        )


@dataclass
class AuditManifest:
    schema_version: str
    run_id: str
    model_id: str
    feature_version: str
    parameter_version: str
    source_data: str
    account_scope: str
    trading_day_source: str
    contracts: list[str]
    trading_days: list[str]
    partitions: dict[str, dict[str, int]]
    total_row_counts: dict[str, int]
    validation_status: str
    created_at: str
    future_horizon_predictor: str = "none (daily records do not claim to predict today's or next TradingDay's close)"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
