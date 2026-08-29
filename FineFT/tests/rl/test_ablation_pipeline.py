import numpy as np
import polars as pl
import pytest

from FineFT.analysis.ablation_pipeline import (
    AblationGroupResult,
    RegimePerformanceRecord,
    compute_paired_bootstrap_ci,
    evaluate_regimes_from_detail_df,
)


def test_compute_paired_bootstrap_ci_returns_valid_bounds():
    sample_a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    sample_b = [1.5, 2.5, 3.8, 4.2, 5.9, 6.5, 7.8, 9.1]

    mean_diff, lower, upper = compute_paired_bootstrap_ci(sample_a, sample_b, num_bootstraps=200, seed=42)

    assert mean_diff > 0.0
    assert lower <= mean_diff <= upper


def test_evaluate_regimes_from_detail_df_conserves_16_bins():
    n_rows = 100
    steps = np.arange(n_rows)
    df = pl.DataFrame(
        {
            "timestamp": steps,
            "close": 100.0 * np.exp(0.001 * steps),
            "log_price_slope_48": np.linspace(-0.002, 0.002, n_rows),
            "step_reward": np.random.default_rng(42).normal(0.01, 0.05, n_rows),
            "trade_count_step": np.random.default_rng(42).choice([0, 1], size=n_rows),
            "commission_fee_step": np.full(n_rows, 0.0004),
            "slippage_step": np.zeros(n_rows),
            "margin_balance": 1e5 + np.cumsum(np.random.default_rng(42).normal(1.0, 10.0, n_rows)),
        }
    )

    quantiles = {
        "slope": [-0.001, 0.0, 0.001],
        "volatility": [0.0001, 0.0002, 0.0003],
    }

    result = evaluate_regimes_from_detail_df(df, quantiles, config_name="B", stage="valid")

    assert len(result.regime_records) == 16
    total_step_count = sum(r.step_count for r in result.regime_records)
    assert total_step_count == n_rows
    assert pytest.approx(sum(r.step_ratio for r in result.regime_records)) == 1.0
