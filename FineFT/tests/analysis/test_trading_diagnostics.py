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
    mark_prices = np.linspace(3000.0, 3100.0, n_steps + 1)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n_steps + 1, freq="10min"),
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
    rewards = np.array([0.0] + [3.5] * (n_steps - 1), dtype=np.float64)
    total_asset = np.r_[6000.0, 6000.0 + np.cumsum(rewards)]

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


def _diagnose(tmp_path, prices, actions, rewards, unit=1.0, executed=None, fees=None):
    data = tmp_path / "data"
    result = tmp_path / "result" / "contracts" / "c"
    data.mkdir(parents=True, exist_ok=True)
    result.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=len(prices), freq="10min"),
        "mark_price": prices,
    }).to_feather(data / "c.feather")
    np.save(result / HistoryArtifactNames.MICRO_ACTION_HISTORY_NPY, actions)
    np.save(result / HistoryArtifactNames.REWARD_HISTORY_NPY, rewards)
    np.save(result / HistoryArtifactNames.TOTAL_ASSET_HISTORY_NPY, np.r_[100, 100 + np.cumsum(rewards)])
    if executed is not None:
        np.save(result / HistoryArtifactNames.EXECUTED_POSITION_HISTORY_NPY, executed)
    if fees is not None:
        np.save(result / HistoryArtifactNames.COMMISSION_FEE_HISTORY_NPY, fees)
    calc = TradingDiagnosticsCalculator(str(result.parent.parent), str(data), contract_unit=unit, commission_rate=0.01)
    return calc, result


def test_terminal_interval_initial_open_and_execution_prices(tmp_path):
    calc, _ = _diagnose(tmp_path, [100, 110, 130], [2, 1], [9, -1])
    summary = calc.run()
    d = summary["contracts"][0]
    assert d["p_end"] == 130
    assert d["gross_pnl"] == 10
    assert d["net_pnl"] == 8
    assert d["trades"] == 2  # flat -> long at 100, long -> flat at 110
    assert d["turnover_volume"] == 210
    assert d["est_commission"] == pytest.approx(2.1)
    assert d["bh_tr"] == pytest.approx(0.3)
    assert summary["metadata"]["missing_macro_histories"] == ["c"]
    assert summary["portfolio"]["portfolio_mdd_pct"] is None


def test_initial_loss_and_unit_scaling_do_not_disappear(tmp_path):
    calc, _ = _diagnose(tmp_path, [100, 90, 90], [2, 1], [-10, 0], unit=10)
    d = calc.run()["contracts"][0]
    assert d["net_pnl"] == -100
    assert d["req_money"] == 1000
    assert d["tr"] == pytest.approx(-0.1)
    assert d["mdd"] == pytest.approx(0.1)
    assert d["friction_ratio"] is None


def test_actual_position_and_fee_override_requested_flat(tmp_path):
    calc, _ = _diagnose(tmp_path, [100, 90, 95], [1, 1], [-10, 5], executed=[1, 1], fees=[0, 0])
    d = calc.run()["contracts"][0]
    assert d["gross_pnl"] == -5
    assert d["est_commission"] == 0
    assert d["position_source"] == "executed_history"
    assert d["long_pct"] == 100


@pytest.mark.parametrize("corruption", ["reward_length", "nan", "equity_mismatch", "action"])
def test_invalid_histories_fail_fast(tmp_path, corruption):
    calc, result = _diagnose(tmp_path, [100, 110, 130], [2, 1], [9, -1])
    if corruption == "reward_length":
        np.save(result / HistoryArtifactNames.REWARD_HISTORY_NPY, [9])
    elif corruption == "nan":
        np.save(result / HistoryArtifactNames.REWARD_HISTORY_NPY, [np.nan, -1])
    elif corruption == "equity_mismatch":
        np.save(result / HistoryArtifactNames.TOTAL_ASSET_HISTORY_NPY, [100, 101, 102])
    else:
        np.save(result / HistoryArtifactNames.MICRO_ACTION_HISTORY_NPY, [2, 8])
    with pytest.raises(ValueError):
        calc.run()


def test_negative_friction_is_retained_and_exported(tmp_path):
    calc, _ = _diagnose(tmp_path, [100, 110], [2], [12])
    s = calc.run()
    assert s["contracts"][0]["friction"] == -2
    assert s["research_grades"]["friction_consumption_ratio_pct"] == "NeedsAudit"
    assert s["metadata"]["negative_friction_contracts"] == ["c"]
    audit = pd.read_csv(calc.output_dir / "c_step_pnl_audit.csv")
    assert audit["negative_friction"].all()


def test_short_run_can_be_loaded_without_inventing_daily_metrics(tmp_path):
    from analysis.diagnostics.batch_diagnostics_loader import BatchDiagnosticsLoader

    calc, _ = _diagnose(tmp_path, [100, 110], [2], [9])
    calc.run()
    digest = BatchDiagnosticsLoader(calc.output_dir).load()
    assert digest.portfolio.mean_annual_sr is None
    assert digest.verdict.status == "NEEDS_AUDIT"
    assert "Unavailable" in digest.format_text_report()
    json.dumps(digest.to_dict(), allow_nan=False)


def test_fee_residual_needs_audit_even_with_positive_gross_net_gap(tmp_path):
    calc, result = _diagnose(tmp_path, [100, 110], [2], [9], executed=[1], fees=[2])
    np.save(result / HistoryArtifactNames.MARKET_INDEX_HISTORY_NPY, [0, 1])
    s = calc.run()
    assert s["contracts"][0]["friction"] == 1
    assert s["metadata"]["negative_friction_steps"] == 0
    assert s["metadata"]["negative_execution_residual_steps"] == 1
    assert s["research_status"] == "NEEDS_AUDIT"


def test_no_market_advance_does_not_invent_interval_return(tmp_path):
    calc, result = _diagnose(tmp_path, [100, 110, 999], [2, 2], [9, -2], executed=[1, 1], fees=[1, 2])
    np.save(result / HistoryArtifactNames.MARKET_INDEX_HISTORY_NPY, [0, 1, 1])
    d = calc.run()["contracts"][0]
    assert d["gross_pnl"] == 10
    assert d["p_end"] == 110
    assert d["bh_tr"] == pytest.approx(0.1)


def test_frozen_methodology_does_not_pass_reported_validation_scores():
    from analysis.diagnostics.evaluation_standards import grade_research_metrics

    grades = grade_research_metrics({
        "portfolio_return_pct": 6.88,
        "portfolio_gross_return_pct": 7.53,
        "mean_annual_sr": 1.274,
        "win_rate_pct": 83.33,
    })
    assert grades["portfolio_return_pct"] == "Warning"
    assert grades["portfolio_gross_return_pct"] == "Fail"
    assert grades["mean_annual_sr"] == "Warning"
    assert grades["win_rate_pct"] == "Warning"
