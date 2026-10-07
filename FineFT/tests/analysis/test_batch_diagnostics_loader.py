from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.diagnostics.batch_diagnostics_loader import (
    BatchDiagnosticsLoader,
    BatchExperimentDigest,
)
from common import ArtifactNames, MetricColumns


def test_batch_diagnostics_loader_end_to_end(tmp_path: Path):
    analysis_dir = tmp_path / "analysis_result" / "DiHFT" / "high_level_heurstic" / "fu" / "10min_parallel"
    diag_dir = analysis_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # 1. Mock diagnostics_summary.json
    summary_data = {
        "portfolio": {
            "contract_count": 2,
            "total_steps": 1000,
            "total_capital": 10000.0,
            "total_gross_pnl": 5000.0,
            "total_net_pnl": 800.0,
            "portfolio_gross_return_pct": 50.0,
            "portfolio_return_pct": 8.0,
            "win_rate_pct": 100.0,
            "mean_annual_sr": 2.1,
            "mean_mdd_pct": 4.5,
            "mean_daily_cr": 1.8,
            "mean_daily_sor": 2.5,
            "total_turnover_volume": 200000.0,
            "portfolio_turnover_multiple": 20.0,
            "total_trades": 150,
            "total_commission": 2000.0,
            "total_slippage": 2200.0,
            "total_friction": 4200.0,
            "friction_consumption_ratio_pct": 84.0,
        },
        "contracts": [],
    }
    with open(diag_dir / "diagnostics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f)

    # 2. Mock contract_pnl_friction.csv
    df_pnl = pd.DataFrame([
        {
            "contract": "fu2501",
            "period": "2025-01 ~ 2025-02",
            "steps": 500,
            "p0": 3000.0,
            "p_end": 3100.0,
            "bh_tr_pct": 3.33,
            "bh_5x_tr_pct": 16.65,
            "gross_pnl": 3000.0,
            "net_pnl": 600.0,
            "friction": 2400.0,
            "friction_ratio_pct": 80.0,
            "trades": 80,
            "turnover_volume": 110000.0,
            "turnover_ratio": 22.0,
            "est_commission": 1100.0,
            "est_slippage": 1300.0,
            "dihft_tr_pct": 12.0,
            "annual_sr": 2.4,
            "mdd_pct": 4.0,
            "daily_cr": 2.0,
            "daily_sor": 2.8,
        },
        {
            "contract": "fu2505",
            "period": "2025-05 ~ 2025-06",
            "steps": 500,
            "p0": 2900.0,
            "p_end": 3050.0,
            "bh_tr_pct": 5.17,
            "bh_5x_tr_pct": 25.86,
            "gross_pnl": 2000.0,
            "net_pnl": 200.0,
            "friction": 1800.0,
            "friction_ratio_pct": 90.0,
            "trades": 70,
            "turnover_volume": 90000.0,
            "turnover_ratio": 18.0,
            "est_commission": 900.0,
            "est_slippage": 900.0,
            "dihft_tr_pct": 4.0,
            "annual_sr": 1.8,
            "mdd_pct": 5.0,
            "daily_cr": 1.6,
            "daily_sor": 2.2,
        },
    ])
    df_pnl.to_csv(diag_dir / "contract_pnl_friction.csv", index=False)

    # 3. Mock contract_behavior_risk.csv
    df_beh = pd.DataFrame([
        {
            "contract": "fu2501",
            "steps": 500,
            "long_pct": 40.0,
            "short_pct": 30.0,
            "flat_pct": 30.0,
            "mean_holding_bars": 3.8,
            "median_holding_bars": 3.0,
            "trades": 80,
            "policy_inference_steps": 200,
            "action_persistence_steps": 280,
            "defensive_preemptions": 10,
            "defensive_rule_closes": 5,
            "hard_stop_losses": 2,
            "stop_loss_cooldown_steps": 24,
            "circuit_breaker_suspension_steps": 0,
        },
        {
            "contract": "fu2505",
            "steps": 500,
            "long_pct": 35.0,
            "short_pct": 35.0,
            "flat_pct": 30.0,
            "mean_holding_bars": 4.2,
            "median_holding_bars": 3.0,
            "trades": 70,
            "policy_inference_steps": 180,
            "action_persistence_steps": 300,
            "defensive_preemptions": 12,
            "defensive_rule_closes": 4,
            "hard_stop_losses": 1,
            "stop_loss_cooldown_steps": 12,
            "circuit_breaker_suspension_steps": 0,
        },
    ])
    df_beh.to_csv(diag_dir / "contract_behavior_risk.csv", index=False)

    # 4. Mock macro_routing_distribution.csv
    df_mac = pd.DataFrame([
        {
            "contract": "fu2501",
            "steps": 500,
            "dynamic_0": 100,
            "dynamic_1": 150,
            "dynamic_2": 250,
        },
        {
            "contract": "fu2505",
            "steps": 500,
            "dynamic_0": 80,
            "dynamic_1": 120,
            "dynamic_2": 300,
        },
    ])
    df_mac.to_csv(diag_dir / "macro_routing_distribution.csv", index=False)

    # 5. Mock best_result.csv in analysis_dir
    df_best = pd.DataFrame([
        {
            MetricColumns.INDICATOR: MetricColumns.TR,
            MetricColumns.TR: 0.15,
            MetricColumns.ANNUAL_SR: 2.3,
            MetricColumns.MDD: 0.045,
            "ood_threshold": 0.005,
            "slope_margin_threshold": 0.12,
        }
    ])
    df_best.to_csv(analysis_dir / ArtifactNames.BEST_RESULT_CSV, index=False)

    loader = BatchDiagnosticsLoader(analysis_path=analysis_dir)
    digest = loader.load()

    assert digest.portfolio.contract_count == 2
    assert digest.portfolio.total_gross_pnl == 5000.0
    assert digest.portfolio.total_friction == 4200.0
    assert digest.portfolio.friction_consumption_ratio_pct == 84.0
    assert digest.macro_routing.dominant_slot == "dynamic_2"
    assert len(digest.top_winners) == 2
    assert digest.top_winners[0].contract == "fu2501"

    # Verdict flags check
    assert "EXCESSIVE_FRICTION_CONSUMPTION" in digest.verdict.flags
    assert "ULTRA_SHORT_HOLDING" in digest.verdict.flags

    # Serialization test
    d = digest.to_dict()
    assert d["portfolio"]["total_steps"] == 1000

    text_rep = digest.format_text_report()
    assert "DIHFT EXPERIMENT DIAGNOSTIC DIGEST" in text_rep
    assert "FRICTION OVERVIEW" in text_rep
    assert "ACTIONABLE ADJUSTMENT ADVICE" in text_rep
