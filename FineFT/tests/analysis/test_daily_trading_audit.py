from __future__ import annotations
import shutil
import sys

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from analysis.diagnostics.daily_trading_audit import DailyTradingAuditPublisher
from common.artifacts import ActionDecisionReasons, AuditArtifactNames
from common.audit_records import (
    AccountRecord,
    PredictionRecord,
    TradeRecord,
    build_account_snapshot_id,
    build_decision_id,
    build_prediction_id,
    build_trade_id,
    calculate_action_value_margin,
)


def _setup_mock_run(
    tmp_path: Path,
    contract: str = "fu2508",
    run_id: str = "run_test_01",
    n_steps: int = 3,
    trading_day: str = "2025-01-02",
    include_trading_day_col: bool = True,
):
    market_dir = tmp_path / "market"
    result_dir = tmp_path / "result"
    contract_result_dir = result_dir / "contracts" / contract
    market_dir.mkdir(parents=True, exist_ok=True)
    contract_result_dir.mkdir(parents=True, exist_ok=True)

    # 1. Market data
    prices = np.linspace(3000.0, 3030.0, n_steps + 1)
    timestamps = pd.date_range("2025-01-02 09:30:00", periods=n_steps + 1, freq="10min")
    market_df = pd.DataFrame({
        "timestamp": timestamps,
        "mark_price": prices,
        "funding_rate": np.zeros(n_steps + 1),
        "funding_timestamp": timestamps,
    })
    if include_trading_day_col:
        market_df["trading_day"] = trading_day
    market_df.to_feather(market_dir / f"{contract}.feather")

    # 2. Metadata
    metadata = {
        "run_id": run_id,
        "model_id": "model_dihft_v1",
        "feature_version": "dual_stream_v2",
        "parameter_version": "param_fu_10min",
        "contract": contract,
    }
    with open(contract_result_dir / AuditArtifactNames.AUDIT_METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 3. Account snapshots (1 initial + n_steps post_step)
    snapshots = []
    # Initial snapshot
    snapshots.append({
        "snapshot_id": build_account_snapshot_id(run_id, contract, 0, "initial"),
        "run_id": run_id,
        "contract": contract,
        "account_id": f"acct_{contract}",
        "account_scope": "independent_contract_wallet",
        "step_index": 0,
        "snapshot_type": "initial",
        "event_time": str(timestamps[0]),
        "trading_day": trading_day,
        "wallet_balance": 10000.0,
        "unrealized_pnl": 0.0,
        "equity": 10000.0,
        "margin": 0.0,
        "available_balance": 10000.0,
        "position": 0.0,
        "exposure": 0.0,
        "leverage": 1.0,
        "peak_equity": 10000.0,
        "drawdown": 0.0,
        "reward": 0.0,
    })

    # Post-step snapshots
    positions = [1.0, 1.0, 0.0]  # open, hold, close
    rewards = [10.0, 10.0, -2.0]
    balances = [9998.5, 9998.5, 10018.0]
    unrealizeds = [11.5, 21.5, 0.0]
    equities = [10010.0, 10020.0, 10018.0]
    peak = 10000.0

    for i in range(n_steps):
        eq = equities[i]
        peak = max(peak, eq)
        snapshots.append({
            "snapshot_id": build_account_snapshot_id(run_id, contract, i, "post"),
            "run_id": run_id,
            "contract": contract,
            "account_id": f"acct_{contract}",
            "account_scope": "independent_contract_wallet",
            "step_index": i + 1,
            "snapshot_type": "post_step",
            "event_time": str(timestamps[i + 1]),
            "trading_day": trading_day,
            "wallet_balance": balances[i],
            "unrealized_pnl": unrealizeds[i],
            "equity": eq,
            "margin": 3000.0 if positions[i] != 0 else 0.0,
            "available_balance": balances[i] - (3000.0 if positions[i] != 0 else 0.0),
            "position": positions[i],
            "exposure": abs(positions[i]) * prices[i + 1],
            "leverage": 1.0,
            "peak_equity": peak,
            "drawdown": peak - eq,
            "reward": rewards[i],
        })

    with open(contract_result_dir / AuditArtifactNames.AUDIT_ACCOUNT_SNAPSHOTS_JSON, "w", encoding="utf-8") as f:
        json.dump(snapshots, f, indent=2)

    # 4. Predictions: genuine inference at step 0 (open) and step 2 (close)
    # Step 1 is action persistence (no new prediction)
    predictions = [
        {
            "prediction_id": build_prediction_id(run_id, contract, 0),
            "decision_id": build_decision_id(run_id, contract, 0),
            "run_id": run_id,
            "model_id": "model_dihft_v1",
            "feature_version": "dual_stream_v2",
            "parameter_version": "param_fu_10min",
            "contract": contract,
            "step_index": 0,
            "event_time": str(timestamps[0]),
            "trading_day": trading_day,
            "market_index": 0,
            "routing_slot": 2,
            "routing_evidence": json.dumps({"volatility": [0.1, 0.2], "slope": [0.3, 0.4]}),
            "action_value_short": 0.2,
            "action_value_flat": 0.5,
            "action_value_long": 1.8,
            "action_value_margin": 1.8 - 0.5,
            "available_actions": "[1, 1, 1]",
            "selected_action": 2,
        },
        {
            "prediction_id": build_prediction_id(run_id, contract, 2),
            "decision_id": build_decision_id(run_id, contract, 2),
            "run_id": run_id,
            "model_id": "model_dihft_v1",
            "feature_version": "dual_stream_v2",
            "parameter_version": "param_fu_10min",
            "contract": contract,
            "step_index": 2,
            "event_time": str(timestamps[2]),
            "trading_day": trading_day,
            "market_index": 2,
            "routing_slot": 2,
            "routing_evidence": json.dumps({"volatility": [0.1, 0.2], "slope": [0.3, 0.4]}),
            "action_value_short": 0.1,
            "action_value_flat": 1.5,
            "action_value_long": 0.4,
            "action_value_margin": 1.5 - 0.4,
            "available_actions": "[1, 1, 1]",
            "selected_action": 1,
        },
    ]
    predictions = [p for p in predictions if p["step_index"] < n_steps]
    with open(contract_result_dir / AuditArtifactNames.AUDIT_PREDICTIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    # 5. Trades
    trades = [
        # Step 0: Open Long
        {
            "trade_id": build_trade_id(run_id, contract, 0, leg=0),
            "decision_id": build_decision_id(run_id, contract, 0),
            "prediction_id": build_prediction_id(run_id, contract, 0),
            "run_id": run_id,
            "contract": contract,
            "step_index": 0,
            "event_time": str(timestamps[0]),
            "trading_day": trading_day,
            "leg_index": 0,
            "event_type": "open",
            "execution_status": "filled",
            "decision_reason": "POLICY_INFERENCE",
            "requested_position": 1.0,
            "executed_position": 1.0,
            "requested_quantity": 1.0,
            "executed_quantity": 1.0,
            "remaining_quantity": 0.0,
            "reference_price": prices[0],
            "fill_price": prices[0] + 0.1,
            "commission": 1.5,
            "realized_pnl": 0.0,
            "slippage": 0.1,
            "pre_snapshot_id": build_account_snapshot_id(run_id, contract, 0, "initial"),
            "post_snapshot_id": build_account_snapshot_id(run_id, contract, 0, "post"),
        },
        # Step 1: Hold (action persistence references step 0 prediction)
        {
            "trade_id": build_trade_id(run_id, contract, 1, leg=0),
            "decision_id": build_decision_id(run_id, contract, 1),
            "prediction_id": build_prediction_id(run_id, contract, 0),
            "run_id": run_id,
            "contract": contract,
            "step_index": 1,
            "event_time": str(timestamps[1]),
            "trading_day": trading_day,
            "leg_index": 0,
            "event_type": "hold",
            "execution_status": "filled",
            "decision_reason": "ACTION_PERSISTENCE",
            "requested_position": 1.0,
            "executed_position": 1.0,
            "requested_quantity": 0.0,
            "executed_quantity": 0.0,
            "remaining_quantity": 0.0,
            "reference_price": prices[1],
            "fill_price": prices[1],
            "commission": 0.0,
            "realized_pnl": 0.0,
            "slippage": 0.0,
            "pre_snapshot_id": build_account_snapshot_id(run_id, contract, 0, "post"),
            "post_snapshot_id": build_account_snapshot_id(run_id, contract, 1, "post"),
        },
        # Step 2: Close
        {
            "trade_id": build_trade_id(run_id, contract, 2, leg=0),
            "decision_id": build_decision_id(run_id, contract, 2),
            "prediction_id": build_prediction_id(run_id, contract, 2),
            "run_id": run_id,
            "contract": contract,
            "step_index": 2,
            "event_time": str(timestamps[2]),
            "trading_day": trading_day,
            "leg_index": 0,
            "event_type": "close",
            "execution_status": "filled",
            "decision_reason": "POLICY_INFERENCE",
            "requested_position": 0.0,
            "executed_position": 0.0,
            "requested_quantity": 1.0,
            "executed_quantity": 1.0,
            "remaining_quantity": 0.0,
            "reference_price": prices[2],
            "fill_price": prices[2] - 0.1,
            "commission": 1.5,
            "realized_pnl": 19.5,
            "slippage": 0.1,
            "pre_snapshot_id": build_account_snapshot_id(run_id, contract, 1, "post"),
            "post_snapshot_id": build_account_snapshot_id(run_id, contract, 2, "post"),
        },
    ]
    trades = [t for t in trades if t["step_index"] < n_steps]
    with open(contract_result_dir / AuditArtifactNames.AUDIT_TRADES_JSON, "w", encoding="utf-8") as f:
        json.dump(trades, f, indent=2)

    return result_dir, market_dir


def test_tracer_bullet_open_hold_close_single_day(tmp_path: Path):
    result_dir, market_dir = _setup_mock_run(tmp_path)
    output_dir = tmp_path / "audit_published"

    publisher = DailyTradingAuditPublisher(
        result_dir=result_dir,
        market_data_dir=market_dir,
        output_dir=output_dir,
        run_id="run_test_01",
    )
    manifest = publisher.publish()

    # 1. Output directory layout
    assert output_dir.is_dir()
    manifest_path = output_dir / AuditArtifactNames.AUDIT_MANIFEST_JSON
    assert manifest_path.is_file()

    day_dir = output_dir / "2025-01-02"
    assert day_dir.is_dir()
    assert (day_dir / AuditArtifactNames.PREDICTION_RECORDS_CSV).is_file()
    assert (day_dir / AuditArtifactNames.TRADE_RECORDS_CSV).is_file()
    assert (day_dir / AuditArtifactNames.ACCOUNT_RECORDS_CSV).is_file()

    # 2. Verify CSV contents
    pred_df = pd.read_csv(day_dir / AuditArtifactNames.PREDICTION_RECORDS_CSV)
    trade_df = pd.read_csv(day_dir / AuditArtifactNames.TRADE_RECORDS_CSV)
    acc_df = pd.read_csv(day_dir / AuditArtifactNames.ACCOUNT_RECORDS_CSV)

    assert len(pred_df) == 2  # 2 genuine inferences (open and close)
    assert len(trade_df) == 3  # open, hold, close
    assert len(acc_df) == 4   # 1 initial + 3 post_step

    # Predictions
    assert "action_value_short" in pred_df.columns
    assert "action_value_margin" in pred_df.columns
    assert pred_df.iloc[0]["selected_action"] == 2
    assert pred_df.iloc[0]["action_value_margin"] == pytest.approx(1.3)

    # Trades
    assert trade_df.iloc[0]["event_type"] == "open"
    assert trade_df.iloc[1]["event_type"] == "hold"
    assert trade_df.iloc[1]["decision_reason"] == "ACTION_PERSISTENCE"
    assert trade_df.iloc[1]["prediction_id"] == pred_df.iloc[0]["prediction_id"]
    assert trade_df.iloc[2]["event_type"] == "close"

    # Accounts
    assert acc_df.iloc[0]["snapshot_type"] == "initial"
    assert acc_df.iloc[0]["position"] == 0.0
    assert (acc_df["equity"] == acc_df["wallet_balance"] + acc_df["unrealized_pnl"]).all()

    # Manifest
    assert manifest["validation_status"] == "passed"
    assert manifest["account_scope"] == "independent_contract_wallet"
    assert "none" in manifest["future_horizon_predictor"].lower()
    assert manifest["total_row_counts"]["prediction_records"] == 2
    assert manifest["total_row_counts"]["trade_records"] == 3
    assert manifest["total_row_counts"]["account_records"] == 4


def test_action_provenance_and_rules(tmp_path: Path):
    market_dir = tmp_path / "market"
    result_dir = tmp_path / "result"
    contract_result_dir = result_dir / "contracts" / "fu2508"
    market_dir.mkdir(parents=True, exist_ok=True)
    contract_result_dir.mkdir(parents=True, exist_ok=True)

    n_steps = 6
    timestamps = pd.date_range("2025-01-02 09:30:00", periods=n_steps + 1, freq="10min")
    prices = np.full(n_steps + 1, 3000.0)
    market_df = pd.DataFrame({
        "timestamp": timestamps,
        "mark_price": prices,
        "trading_day": "2025-01-02",
    })
    market_df.to_feather(market_dir / "fu2508.feather")

    run_id = "run_rules_01"
    contract = "fu2508"
    trading_day = "2025-01-02"

    metadata = {
        "run_id": run_id,
        "model_id": "model_test",
        "feature_version": "v1",
        "parameter_version": "v1",
        "contract": contract,
    }
    with open(contract_result_dir / AuditArtifactNames.AUDIT_METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    # 1. Accounts: 1 initial + 6 post-steps (flat all along, equity constant)
    snapshots = [
        {
            "snapshot_id": build_account_snapshot_id(run_id, contract, 0, "initial"),
            "run_id": run_id,
            "contract": contract,
            "account_id": f"acct_{contract}",
            "account_scope": "independent_contract_wallet",
            "step_index": 0,
            "snapshot_type": "initial",
            "event_time": str(timestamps[0]),
            "trading_day": trading_day,
            "wallet_balance": 10000.0,
            "unrealized_pnl": 0.0,
            "equity": 10000.0,
            "margin": 0.0,
            "available_balance": 10000.0,
            "position": 0.0,
            "exposure": 0.0,
            "leverage": 1.0,
            "peak_equity": 10000.0,
            "drawdown": 0.0,
            "reward": 0.0,
        }
    ]
    for i in range(n_steps):
        snapshots.append({
            "snapshot_id": build_account_snapshot_id(run_id, contract, i, "post"),
            "run_id": run_id,
            "contract": contract,
            "account_id": f"acct_{contract}",
            "account_scope": "independent_contract_wallet",
            "step_index": i + 1,
            "snapshot_type": "post_step",
            "event_time": str(timestamps[i + 1]),
            "trading_day": trading_day,
            "wallet_balance": 10000.0,
            "unrealized_pnl": 0.0,
            "equity": 10000.0,
            "margin": 0.0,
            "available_balance": 10000.0,
            "position": 0.0,
            "exposure": 0.0,
            "leverage": 1.0,
            "peak_equity": 10000.0,
            "drawdown": 0.0,
            "reward": 0.0,
        })
    with open(contract_result_dir / AuditArtifactNames.AUDIT_ACCOUNT_SNAPSHOTS_JSON, "w", encoding="utf-8") as f:
        json.dump(snapshots, f)

    # 2. Prediction: Only step 1 has a genuine policy inference
    # Step 0 is WARM_UP (no prediction)
    # Step 1 is POLICY_INFERENCE (arms persistence)
    # Step 2 is ACTION_PERSISTENCE (points to Step 1 prediction)
    # Step 3 is HARD_STOP_LOSS (rule, no prediction)
    # Step 4 is STOP_LOSS_COOLDOWN (rule, no prediction)
    # Step 5 is CIRCUIT_BREAKER_SUSPENSION (rule, no prediction)
    predictions = [
        {
            "prediction_id": build_prediction_id(run_id, contract, 1),
            "decision_id": build_decision_id(run_id, contract, 1),
            "run_id": run_id,
            "model_id": "model_test",
            "feature_version": "v1",
            "parameter_version": "v1",
            "contract": contract,
            "step_index": 1,
            "event_time": str(timestamps[1]),
            "trading_day": trading_day,
            "market_index": 1,
            "routing_slot": 0,
            "routing_evidence": "{}",
            "action_value_short": 0.1,
            "action_value_flat": 1.0,
            "action_value_long": 0.2,
            "action_value_margin": 0.8,
            "available_actions": "[1, 1, 1]",
            "selected_action": 1,
        }
    ]
    with open(contract_result_dir / AuditArtifactNames.AUDIT_PREDICTIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(predictions, f)

    reasons = [
        "WARM_UP",
        "POLICY_INFERENCE",
        "ACTION_PERSISTENCE",
        "HARD_STOP_LOSS",
        "STOP_LOSS_COOLDOWN",
        "CIRCUIT_BREAKER_SUSPENSION",
    ]
    pred_refs = [
        "",  # warm-up has no prediction
        build_prediction_id(run_id, contract, 1),
        build_prediction_id(run_id, contract, 1),  # persistence references step 1
        "",  # hard stop loss has no prediction
        "",  # cooldown has no prediction
        "",  # circuit breaker has no prediction
    ]

    trades = []
    for step_i in range(n_steps):
        pre_id = build_account_snapshot_id(run_id, contract, 0, "initial") if step_i == 0 else build_account_snapshot_id(run_id, contract, step_i - 1, "post")
        post_id = build_account_snapshot_id(run_id, contract, step_i, "post")
        trades.append({
            "trade_id": build_trade_id(run_id, contract, step_i, 0),
            "decision_id": build_decision_id(run_id, contract, step_i),
            "prediction_id": pred_refs[step_i],
            "run_id": run_id,
            "contract": contract,
            "step_index": step_i,
            "event_time": str(timestamps[step_i]),
            "trading_day": trading_day,
            "leg_index": 0,
            "event_type": "hold",
            "execution_status": "filled",
            "decision_reason": reasons[step_i],
            "requested_position": 0.0,
            "executed_position": 0.0,
            "requested_quantity": 0.0,
            "executed_quantity": 0.0,
            "remaining_quantity": 0.0,
            "reference_price": 3000.0,
            "fill_price": 3000.0,
            "commission": 0.0,
            "realized_pnl": 0.0,
            "slippage": 0.0,
            "pre_snapshot_id": pre_id,
            "post_snapshot_id": post_id,
        })
    with open(contract_result_dir / AuditArtifactNames.AUDIT_TRADES_JSON, "w", encoding="utf-8") as f:
        json.dump(trades, f)

    output_dir = tmp_path / "out_rules"
    pub = DailyTradingAuditPublisher(result_dir, market_dir, output_dir, run_id=run_id)
    manifest = pub.publish()

    trade_df = pd.read_csv(output_dir / "2025-01-02" / AuditArtifactNames.TRADE_RECORDS_CSV)
    assert len(trade_df) == 6
    # Warm-up has empty prediction_id and reason WARM_UP
    assert pd.isna(trade_df.iloc[0]["prediction_id"]) or trade_df.iloc[0]["prediction_id"] == ""
    assert trade_df.iloc[0]["decision_reason"] == "WARM_UP"

    # Action persistence references step 1 prediction
    assert trade_df.iloc[2]["decision_reason"] == "ACTION_PERSISTENCE"
    assert trade_df.iloc[2]["prediction_id"] == build_prediction_id(run_id, contract, 1)

    # Risk rules have empty prediction_id
    for idx, rule_reason in [(3, "HARD_STOP_LOSS"), (4, "STOP_LOSS_COOLDOWN"), (5, "CIRCUIT_BREAKER_SUSPENSION")]:
        assert trade_df.iloc[idx]["decision_reason"] == rule_reason
        assert pd.isna(trade_df.iloc[idx]["prediction_id"]) or trade_df.iloc[idx]["prediction_id"] == ""


def test_complex_execution_events(tmp_path: Path):
    # Tests reversal split into 2 legs sharing decision_id, partial fill, and rejection
    result_dir, market_dir = _setup_mock_run(tmp_path, n_steps=3)
    contract = "fu2508"
    run_id = "run_test_01"
    contract_result_dir = result_dir / "contracts" / contract

    # Replace trades with:
    # Step 0: reversal split into Leg 0 (reversal_close) and Leg 1 (reversal_open) sharing decision_id
    # Step 1: partial fill
    # Step 2: price limit rejection
    trades = [
        # Step 0 Leg 0: close long
        {
            "trade_id": build_trade_id(run_id, contract, 0, leg=0),
            "decision_id": build_decision_id(run_id, contract, 0),
            "prediction_id": build_prediction_id(run_id, contract, 0),
            "run_id": run_id,
            "contract": contract,
            "step_index": 0,
            "event_time": "2025-01-02 09:30:00",
            "trading_day": "2025-01-02",
            "leg_index": 0,
            "event_type": "reversal_close",
            "execution_status": "filled",
            "decision_reason": "POLICY_INFERENCE",
            "requested_position": 0.0,
            "executed_position": 0.0,
            "requested_quantity": 1.0,
            "executed_quantity": 1.0,
            "remaining_quantity": 0.0,
            "reference_price": 3000.0,
            "fill_price": 2999.0,
            "commission": 1.0,
            "realized_pnl": 5.0,
            "slippage": 1.0,
            "pre_snapshot_id": build_account_snapshot_id(run_id, contract, 0, "initial"),
            "post_snapshot_id": build_account_snapshot_id(run_id, contract, 0, "post"),
        },
        # Step 0 Leg 1: open short
        {
            "trade_id": build_trade_id(run_id, contract, 0, leg=1),
            "decision_id": build_decision_id(run_id, contract, 0),
            "prediction_id": build_prediction_id(run_id, contract, 0),
            "run_id": run_id,
            "contract": contract,
            "step_index": 0,
            "event_time": "2025-01-02 09:30:00",
            "trading_day": "2025-01-02",
            "leg_index": 1,
            "event_type": "reversal_open",
            "execution_status": "filled",
            "decision_reason": "POLICY_INFERENCE",
            "requested_position": -1.0,
            "executed_position": -1.0,
            "requested_quantity": 1.0,
            "executed_quantity": 1.0,
            "remaining_quantity": 0.0,
            "reference_price": 3000.0,
            "fill_price": 2999.0,
            "commission": 1.0,
            "realized_pnl": 0.0,
            "slippage": 1.0,
            "pre_snapshot_id": build_account_snapshot_id(run_id, contract, 0, "initial"),
            "post_snapshot_id": build_account_snapshot_id(run_id, contract, 0, "post"),
        },
        # Step 1: partial fill
        {
            "trade_id": build_trade_id(run_id, contract, 1, leg=0),
            "decision_id": build_decision_id(run_id, contract, 1),
            "prediction_id": build_prediction_id(run_id, contract, 0),
            "run_id": run_id,
            "contract": contract,
            "step_index": 1,
            "event_time": "2025-01-02 09:40:00",
            "trading_day": "2025-01-02",
            "leg_index": 0,
            "event_type": "open",
            "execution_status": "partial_fill",
            "decision_reason": "ACTION_PERSISTENCE",
            "requested_position": -2.0,
            "executed_position": -1.0,
            "requested_quantity": 1.0,
            "executed_quantity": 0.5,
            "remaining_quantity": 0.5,
            "reference_price": 3010.0,
            "fill_price": 3010.0,
            "commission": 0.5,
            "realized_pnl": 0.0,
            "slippage": 0.0,
            "pre_snapshot_id": build_account_snapshot_id(run_id, contract, 0, "post"),
            "post_snapshot_id": build_account_snapshot_id(run_id, contract, 1, "post"),
        },
        # Step 2: Price limit rejection
        {
            "trade_id": build_trade_id(run_id, contract, 2, leg=0),
            "decision_id": build_decision_id(run_id, contract, 2),
            "prediction_id": build_prediction_id(run_id, contract, 2),
            "run_id": run_id,
            "contract": contract,
            "step_index": 2,
            "event_time": "2025-01-02 09:50:00",
            "trading_day": "2025-01-02",
            "leg_index": 0,
            "event_type": "rejection",
            "execution_status": "rejected",
            "decision_reason": "POLICY_INFERENCE",
            "requested_position": 1.0,
            "executed_position": 0.0,
            "requested_quantity": 1.0,
            "executed_quantity": 0.0,
            "remaining_quantity": 1.0,
            "reference_price": 3020.0,
            "fill_price": 0.0,
            "commission": 0.0,
            "realized_pnl": 0.0,
            "slippage": 0.0,
            "pre_snapshot_id": build_account_snapshot_id(run_id, contract, 1, "post"),
            "post_snapshot_id": build_account_snapshot_id(run_id, contract, 2, "post"),
        },
    ]
    with open(contract_result_dir / AuditArtifactNames.AUDIT_TRADES_JSON, "w", encoding="utf-8") as f:
        json.dump(trades, f)

    output_dir = tmp_path / "out_complex"
    pub = DailyTradingAuditPublisher(result_dir, market_dir, output_dir, run_id=run_id)
    manifest = pub.publish()

    trade_df = pd.read_csv(output_dir / "2025-01-02" / AuditArtifactNames.TRADE_RECORDS_CSV)
    assert len(trade_df) == 4
    # Check reversal legs share decision_id
    assert trade_df.iloc[0]["decision_id"] == trade_df.iloc[1]["decision_id"]
    assert trade_df.iloc[0]["leg_index"] == 0
    assert trade_df.iloc[1]["leg_index"] == 1
    assert trade_df.iloc[0]["event_type"] == "reversal_close"
    assert trade_df.iloc[1]["event_type"] == "reversal_open"

    # Check partial fill
    assert trade_df.iloc[2]["execution_status"] == "partial_fill"
    assert trade_df.iloc[2]["executed_quantity"] == 0.5
    assert trade_df.iloc[2]["remaining_quantity"] == 0.5

    # Check rejection
    assert trade_df.iloc[3]["execution_status"] == "rejected"
    assert trade_df.iloc[3]["executed_quantity"] == 0.0
    assert trade_df.iloc[3]["remaining_quantity"] == 1.0


def test_multi_contract_and_trading_day_mapping(tmp_path: Path):
    # Contract 1: explicit column in market data with 2 trading days (including night session on previous calendar day)
    c1_dir = tmp_path / "c1"
    res1, mkt1 = _setup_mock_run(c1_dir, contract="fu2508", run_id="multi_run", n_steps=2, trading_day="2025-01-02")
    # Contract 2: NO trading_day column in market data, uses mapping file!
    c2_dir = tmp_path / "c2"
    res2, mkt2 = _setup_mock_run(c2_dir, contract="fu2510", run_id="multi_run", n_steps=2, include_trading_day_col=False)

    # Combine into a unified result_dir and market_dir
    unified_res = tmp_path / "unified_res"
    unified_mkt = tmp_path / "unified_mkt"
    (unified_res / "contracts" / "fu2508").parent.mkdir(parents=True, exist_ok=True)
    unified_mkt.mkdir(parents=True, exist_ok=True)

    shutil.copytree(res1 / "contracts" / "fu2508", unified_res / "contracts" / "fu2508")
    shutil.copytree(res2 / "contracts" / "fu2510", unified_res / "contracts" / "fu2510")
    shutil.copy(mkt1 / "fu2508.feather", unified_mkt / "fu2508.feather")
    shutil.copy(mkt2 / "fu2510.feather", unified_mkt / "fu2510.feather")

    # 1. Fail fast when contract lacks trading_day and no mapping provided
    output_dir = tmp_path / "out_multi"
    pub_fail = DailyTradingAuditPublisher(unified_res, unified_mkt, output_dir, run_id="multi_run")
    with pytest.raises(ValueError, match="Missing explicit TradingDay"):
        pub_fail.publish()
    assert not output_dir.exists()

    # 2. Provide explicit mapping file
    mapping_file = tmp_path / "mapping.json"
    mapping_data = {
        "fu2510": {
            "2025-01-02 09:30:00": "2025-01-02",
            "2025-01-02 09:40:00": "2025-01-02",
            "2025-01-02 09:50:00": "2025-01-02",
        }
    }
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(mapping_data, f)

    pub_success = DailyTradingAuditPublisher(
        unified_res,
        unified_mkt,
        output_dir,
        run_id="multi_run",
        trading_day_mapping=mapping_file,
    )
    manifest = pub_success.publish()
    assert manifest["contracts"] == ["fu2508", "fu2510"]
    assert "2025-01-02" in manifest["trading_days"]

    day_dir = output_dir / "2025-01-02"
    acc_df = pd.read_csv(day_dir / AuditArtifactNames.ACCOUNT_RECORDS_CSV)
    # Both contracts have independent accounts
    contracts_present = sorted(acc_df["contract"].unique())
    assert contracts_present == ["fu2508", "fu2510"]
    assert (acc_df["account_scope"] == "independent_contract_wallet").all()

    # 3. Deterministic repeated run produces identical outputs
    output_dir_2 = tmp_path / "out_multi_2"
    pub_repeat = DailyTradingAuditPublisher(
        unified_res,
        unified_mkt,
        output_dir_2,
        run_id="multi_run",
        trading_day_mapping=mapping_file,
    )
    manifest_2 = pub_repeat.publish()
    m1_clean = {k: v for k, v in manifest.items() if k != "created_at"}
    m2_clean = {k: v for k, v in manifest_2.items() if k != "created_at"}
    assert m1_clean == m2_clean

    # Compare CSV file bytes
    for fname in [AuditArtifactNames.PREDICTION_RECORDS_CSV, AuditArtifactNames.TRADE_RECORDS_CSV, AuditArtifactNames.ACCOUNT_RECORDS_CSV]:
        b1 = (output_dir / "2025-01-02" / fname).read_bytes()
        b2 = (output_dir_2 / "2025-01-02" / fname).read_bytes()
        assert b1 == b2


def test_strict_validation_failures_leave_no_official_output(tmp_path: Path):
    output_dir = tmp_path / "out_strict"

    # Test A: Missing audit artifact (e.g. audit_trades.json missing)
    res_a, mkt_a = _setup_mock_run(tmp_path / "sub_a")
    (res_a / "contracts" / "fu2508" / AuditArtifactNames.AUDIT_TRADES_JSON).unlink()
    pub_a = DailyTradingAuditPublisher(res_a, mkt_a, output_dir, run_id="run_a")
    with pytest.raises(ValueError, match="Missing audit-grade input"):
        pub_a.publish()
    assert not output_dir.exists()

    # Test B: Dangling foreign key in trade
    res_b, mkt_b = _setup_mock_run(tmp_path / "sub_b")
    trade_file = res_b / "contracts" / "fu2508" / AuditArtifactNames.AUDIT_TRADES_JSON
    with open(trade_file, encoding="utf-8") as f:
        tdata = json.load(f)
    tdata[0]["prediction_id"] = "pred_nonexistent_999999"
    with open(trade_file, "w", encoding="utf-8") as f:
        json.dump(tdata, f)
    pub_b = DailyTradingAuditPublisher(res_b, mkt_b, output_dir, run_id="run_b")
    with pytest.raises(ValueError, match="Foreign key violation"):
        pub_b.publish()
    assert not output_dir.exists()

    # Test C: Non-finite value (NaN in equity)
    res_c, mkt_c = _setup_mock_run(tmp_path / "sub_c")
    acc_file = res_c / "contracts" / "fu2508" / AuditArtifactNames.AUDIT_ACCOUNT_SNAPSHOTS_JSON
    with open(acc_file, encoding="utf-8") as f:
        adata = json.load(f)
    adata[1]["equity"] = float("nan")
    with open(acc_file, "w", encoding="utf-8") as f:
        json.dump(adata, f)
    pub_c = DailyTradingAuditPublisher(res_c, mkt_c, output_dir, run_id="run_c")
    with pytest.raises(ValueError, match="Non-finite value"):
        pub_c.publish()
    assert not output_dir.exists()

    # Test D: Equity identity violation (equity != wallet_balance + unrealized_pnl)
    res_d, mkt_d = _setup_mock_run(tmp_path / "sub_d")
    acc_file_d = res_d / "contracts" / "fu2508" / AuditArtifactNames.AUDIT_ACCOUNT_SNAPSHOTS_JSON
    with open(acc_file_d, encoding="utf-8") as f:
        adata_d = json.load(f)
    adata_d[1]["equity"] = adata_d[1]["wallet_balance"] + adata_d[1]["unrealized_pnl"] + 100.0
    with open(acc_file_d, "w", encoding="utf-8") as f:
        json.dump(adata_d, f)
    pub_d = DailyTradingAuditPublisher(res_d, mkt_d, output_dir, run_id="run_d")
    with pytest.raises(ValueError, match="Equity identity violation"):
        pub_d.publish()
    assert not output_dir.exists()

    # Test E: Reward reconciliation mismatch
    res_e, mkt_e = _setup_mock_run(tmp_path / "sub_e")
    acc_file_e = res_e / "contracts" / "fu2508" / AuditArtifactNames.AUDIT_ACCOUNT_SNAPSHOTS_JSON
    with open(acc_file_e, encoding="utf-8") as f:
        adata_e = json.load(f)
    adata_e[1]["reward"] = 999.0  # does not match equity delta
    with open(acc_file_e, "w", encoding="utf-8") as f:
        json.dump(adata_e, f)
    pub_e = DailyTradingAuditPublisher(res_e, mkt_e, output_dir, run_id="run_e")
    with pytest.raises(ValueError, match="Reward reconciliation mismatch"):
        pub_e.publish()
    assert not output_dir.exists()


def test_cli_invocation(tmp_path: Path):
    import subprocess
    result_dir, market_dir = _setup_mock_run(tmp_path)
    output_dir = tmp_path / "cli_out"

    cmd = [
        sys.executable,
        "-m",
        "analysis.diagnostics.daily_trading_audit",
        "--result-dir",
        str(result_dir),
        "--market-data-dir",
        str(market_dir),
        "--output-dir",
        str(output_dir),
        "--run-id",
        "cli_run_01",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[2]))
    assert proc.returncode == 0, f"CLI failed: {proc.stderr}"
    assert (output_dir / AuditArtifactNames.AUDIT_MANIFEST_JSON).is_file()
    assert (output_dir / "2025-01-02" / AuditArtifactNames.TRADE_RECORDS_CSV).is_file()


def test_cli_can_skip_missing_audit_input_for_optional_pipeline_integration(tmp_path: Path):
    import subprocess

    result_dir, market_dir = _setup_mock_run(tmp_path)
    metadata_path = result_dir / "contracts" / "fu2508" / AuditArtifactNames.AUDIT_METADATA_JSON
    metadata_path.unlink()
    output_dir = tmp_path / "optional_cli_out"

    cmd = [
        sys.executable,
        "-m",
        "analysis.diagnostics.daily_trading_audit",
        "--result-dir",
        str(result_dir),
        "--market-data-dir",
        str(market_dir),
        "--output-dir",
        str(output_dir),
        "--skip-missing-input",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[2]))

    assert proc.returncode == 0, proc.stderr
    assert "Skipping daily trading audit" in proc.stderr
    assert not output_dir.exists()
