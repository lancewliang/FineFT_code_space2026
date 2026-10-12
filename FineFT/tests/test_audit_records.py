from __future__ import annotations

import pytest

from common.artifacts import ActionDecisionReasons, AuditArtifactNames
from common.audit_records import (
    AccountRecord,
    AuditManifest,
    PredictionRecord,
    TradeRecord,
    build_account_snapshot_id,
    build_decision_id,
    build_prediction_id,
    build_trade_id,
    calculate_action_value_margin,
)


def test_audit_artifact_names_and_reasons():
    assert AuditArtifactNames.PREDICTION_RECORDS_CSV == "prediction_records.csv"
    assert AuditArtifactNames.TRADE_RECORDS_CSV == "trade_records.csv"
    assert AuditArtifactNames.ACCOUNT_RECORDS_CSV == "account_records.csv"
    assert AuditArtifactNames.AUDIT_MANIFEST_JSON == "audit_manifest.json"
    assert AuditArtifactNames.AUDIT_PREDICTIONS_JSON == "audit_predictions.json"
    assert AuditArtifactNames.AUDIT_TRADES_JSON == "audit_trades.json"
    assert AuditArtifactNames.AUDIT_ACCOUNT_SNAPSHOTS_JSON == "audit_account_snapshots.json"
    assert AuditArtifactNames.AUDIT_METADATA_JSON == "audit_metadata.json"

    assert ActionDecisionReasons.WARM_UP == 8
    assert ActionDecisionReasons(8).name == "WARM_UP"


def test_deterministic_id_generation():
    run_id = "test_run_01"
    contract = "fu2508"
    step = 5

    pred_id = build_prediction_id(run_id, contract, step)
    dec_id = build_decision_id(run_id, contract, step)
    trade_id_0 = build_trade_id(run_id, contract, step, leg=0)
    trade_id_1 = build_trade_id(run_id, contract, step, leg=1)
    acc_id_init = build_account_snapshot_id(run_id, contract, 0, "initial")
    acc_id_post = build_account_snapshot_id(run_id, contract, step, "post")

    assert pred_id == "pred_test_run_01_fu2508_000005"
    assert dec_id == "dec_test_run_01_fu2508_000005"
    assert trade_id_0 == "trade_test_run_01_fu2508_000005_leg0"
    assert trade_id_1 == "trade_test_run_01_fu2508_000005_leg1"
    assert acc_id_init == "acc_test_run_01_fu2508_000000_initial"
    assert acc_id_post == "acc_test_run_01_fu2508_000005_post"

    # Stability across calls
    assert build_prediction_id(run_id, contract, step) == pred_id


def test_calculate_action_value_margin():
    # Clear winner
    assert calculate_action_value_margin([1.0, 0.5, 2.5]) == pytest.approx(1.5)
    # Tie between top two
    assert calculate_action_value_margin([2.0, 2.0, 0.1]) == pytest.approx(0.0)
    # Negative values
    assert calculate_action_value_margin([-1.5, -0.5, -2.0]) == pytest.approx(1.0)


def test_record_schemas_and_serialization():
    pred = PredictionRecord(
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
    d = pred.to_dict()
    assert d["prediction_id"] == "pred_01"
    assert d["action_value_margin"] == 0.7
    pred2 = PredictionRecord.from_dict(d)
    assert pred2 == pred

    trade = TradeRecord(
        trade_id="trade_01_leg0",
        decision_id="dec_01",
        prediction_id="pred_01",
        run_id="run_01",
        contract="fu2508",
        step_index=1,
        event_time="2025-01-02 09:30:00",
        trading_day="2025-01-02",
        leg_index=0,
        event_type="open",
        execution_status="filled",
        decision_reason="POLICY_INFERENCE",
        requested_position=1.0,
        executed_position=1.0,
        requested_quantity=1.0,
        executed_quantity=1.0,
        remaining_quantity=0.0,
        reference_price=3000.0,
        fill_price=3000.5,
        commission=1.5,
        realized_pnl=0.0,
        slippage=0.5,
        pre_snapshot_id="acc_init",
        post_snapshot_id="acc_post_1",
    )
    td = trade.to_dict()
    assert td["execution_status"] == "filled"
    assert TradeRecord.from_dict(td) == trade

    account = AccountRecord(
        snapshot_id="acc_post_1",
        run_id="run_01",
        contract="fu2508",
        account_id="acct_fu2508",
        account_scope="independent_contract_wallet",
        step_index=1,
        snapshot_type="post_step",
        event_time="2025-01-02 09:30:00",
        trading_day="2025-01-02",
        wallet_balance=10000.0,
        unrealized_pnl=50.0,
        equity=10050.0,
        margin=1500.0,
        available_balance=8500.0,
        position=1.0,
        exposure=30000.0,
        leverage=1.0,
        peak_equity=10050.0,
        drawdown=0.0,
        reward=50.0,
    )
    ad = account.to_dict()
    assert ad["equity"] == 10050.0
    assert AccountRecord.from_dict(ad) == account
