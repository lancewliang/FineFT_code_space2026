from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.diagnostics.trading_diagnostics import TradingDiagnosticsCalculator
from common import ActionDecisionReasons, HistoryArtifactNames


def test_trading_diagnostics_calculator_e2e(tmp_path: Path):
    # Setup mock data directory with a feather file
    data_dir = tmp_path / "valid"
    data_dir.mkdir(parents=True, exist_ok=True)

    contract_name = "test_contract"
    n_steps = 25
    mark_prices = np.linspace(3000.0, 3100.0, n_steps)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n_steps, freq="10min"),
        "mark_price": mark_prices,
    })
    df.to_feather(data_dir / f"{contract_name}.feather")

    # Setup mock result directory with contracts/
    result_dir = tmp_path / "result"
    contract_result_dir = result_dir / "contracts" / contract_name
    contract_result_dir.mkdir(parents=True, exist_ok=True)

    # Mock trading artifacts: mostly Long (2)
    micro_actions = np.array([2] * n_steps, dtype=np.int32)
    micro_actions[0] = 1 # first step flat
    macro_actions = np.array([8] * n_steps, dtype=np.int32)
    reasons = np.array([ActionDecisionReasons.ACTION_PERSISTENCE.value] * n_steps, dtype=np.int32)
    reasons[1] = ActionDecisionReasons.POLICY_INFERENCE.value

    # Rewards: positive
    rewards = np.array([0.0] + [35.0] * (n_steps - 1), dtype=np.float64)
    total_asset = 6000.0 + np.cumsum(rewards)

    np.save(contract_result_dir / HistoryArtifactNames.REWARD_HISTORY_NPY, rewards)
    np.save(contract_result_dir / HistoryArtifactNames.TOTAL_ASSET_HISTORY_NPY, total_asset)
    np.save(contract_result_dir / HistoryArtifactNames.MICRO_ACTION_HISTORY_NPY, micro_actions)
    np.save(contract_result_dir / HistoryArtifactNames.MACRO_ACTION_HISTORY_NPY, macro_actions)
    np.save(contract_result_dir / HistoryArtifactNames.ACTION_DECISION_REASON_HISTORY_NPY, reasons)

    output_dir = tmp_path / "diagnostics_out"

    calculator = TradingDiagnosticsCalculator(
        result_dir=str(result_dir),
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        initial_wallet_balance=6000.0,
        commission_rate=0.0005,
        contract_unit=10.0,
        freq=12,
    )
    summary = calculator.run()

    # Assert outputs exist
    assert (output_dir / "diagnostics_summary.json").is_file()
    assert (output_dir / "contract_pnl_friction.csv").is_file()
    assert (output_dir / "contract_behavior_risk.csv").is_file()
    assert (output_dir / "macro_routing_distribution.csv").is_file()
    assert (output_dir / "diagnostics_summary.md").is_file()

    # Assert summary contents
    p = summary["portfolio"]
    assert p["contract_count"] == 1
    assert p["total_steps"] == n_steps
    assert p["win_rate_pct"] == 100.0
    assert p["total_net_pnl"] > 0
    assert p["total_gross_pnl"] > 0
    assert p["total_friction"] >= 0

    # Load JSON and verify
    with open(output_dir / "diagnostics_summary.json", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["portfolio"]["contract_count"] == 1
    assert len(loaded["contracts"]) == 1
    assert loaded["contracts"][0]["contract"] == contract_name
