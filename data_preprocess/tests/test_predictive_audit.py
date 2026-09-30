import numpy as np
import polars as pl
import pytest

from operator_futures.feature_selection.muti_contract.metrics import (
    calculate_future_return,
)
from operator_futures.feature_selection.muti_contract.predictive_audit import (
    execute_predictive_audit,
)
from operator_futures.feature_selection.muti_contract.types import (
    PredictiveAuditConfig,
)


def _create_synthetic_predictive_frames(n_rows: int = 300):
    rng = np.random.RandomState(42)
    prices_c1 = 10.0 + np.cumsum(rng.normal(0.01, 0.1, size=n_rows))
    prices_c2 = 20.0 + np.cumsum(rng.normal(0.01, 0.1, size=n_rows))
    df1 = pl.DataFrame({"mark_price": prices_c1})
    df2 = pl.DataFrame({"mark_price": prices_c2})
    ret1 = calculate_future_return(df1, 6)
    ret2 = calculate_future_return(df2, 6)

    # 1. Predictive feature (realistic IC ~ 0.15, within [0.02, 0.30])
    good_1 = np.pad(ret1 + rng.normal(scale=6.0 * np.std(ret1), size=len(ret1)), (0, 6))
    good_2 = np.pad(ret2 + rng.normal(scale=6.0 * np.std(ret2), size=len(ret2)), (0, 6))

    # 2. Pure noise feature (independent across contracts)
    noise_1 = rng.normal(size=n_rows)
    noise_2 = rng.normal(size=n_rows)

    # 3. Anti-causal coincident leak (|IC| >= 0.80 > 0.30)
    leak_1 = np.pad(ret1 + rng.normal(scale=0.1 * np.std(ret1), size=len(ret1)), (0, 6))
    leak_2 = np.pad(ret2 + rng.normal(scale=0.1 * np.std(ret2), size=len(ret2)), (0, 6))

    frames = {
        "c1": pl.DataFrame(
            {
                "mark_price": prices_c1,
                "good_feature": good_1,
                "noise_feature": noise_1,
                "leak_feature": leak_1,
            }
        ),
        "c2": pl.DataFrame(
            {
                "mark_price": prices_c2,
                "good_feature": good_2,
                "noise_feature": noise_2,
                "leak_feature": leak_2,
            }
        ),
    }
    return frames


def test_predictive_audit_filters_noise_and_detects_anti_causality():
    frames = _create_synthetic_predictive_frames(n_rows=300)
    features = ["good_feature", "noise_feature", "leak_feature"]

    config = PredictiveAuditConfig(
        min_abs_ic=0.02,
        min_sign_consistency=0.75,
        min_rank_ic_ir=0.40,
        ic_anomaly_ceiling=0.30,
        target_decision_window=6,
        windows_list=(6,),
        fdr_threshold=0.05,
    )

    agg_df, step_result = execute_predictive_audit(frames, features, config)

    # good_feature should survive
    assert "good_feature" in step_result.surviving_features
    # noise_feature should be dropped by Hard RankIC or Sign Consistency
    assert "noise_feature" in step_result.dropped_features
    # leak_feature should be caught by anti-causality ceiling
    assert "leak_feature" in step_result.dropped_features
    assert "leak_feature" in step_result.diagnostics["anti_causal_dropped"]


def test_predictive_audit_fdr_control():
    rng = np.random.RandomState(42)
    n = 300
    prices_c1 = 10.0 + np.cumsum(rng.normal(0.01, 0.1, size=n))
    prices_c2 = 20.0 + np.cumsum(rng.normal(0.01, 0.1, size=n))
    df1 = pl.DataFrame({"mark_price": prices_c1})
    df2 = pl.DataFrame({"mark_price": prices_c2})
    ret1 = calculate_future_return(df1, 6)
    ret2 = calculate_future_return(df2, 6)

    # 1 genuine signal with RankIC ~ 0.15 in both contracts
    sig_c1 = np.pad(ret1 + rng.normal(scale=6.0 * np.std(ret1), size=len(ret1)), (0, 6))
    sig_c2 = np.pad(ret2 + rng.normal(scale=6.0 * np.std(ret2), size=len(ret2)), (0, 6))

    # 10 pure noise features (independent across contracts)
    noise_dict_c1 = {f"noise_{i}": rng.normal(size=n) for i in range(10)}
    noise_dict_c2 = {f"noise_{i}": rng.normal(size=n) for i in range(10)}

    df_c1 = pl.DataFrame({"mark_price": prices_c1, "signal": sig_c1, **noise_dict_c1})
    df_c2 = pl.DataFrame({"mark_price": prices_c2, "signal": sig_c2, **noise_dict_c2})
    frames = {"c1": df_c1, "c2": df_c2}
    features = ["signal", *noise_dict_c1.keys()]

    config = PredictiveAuditConfig(
        min_abs_ic=0.02,
        min_sign_consistency=0.75,
        min_rank_ic_ir=0.40,
        ic_anomaly_ceiling=0.30,
        target_decision_window=6,
        windows_list=(6,),
        fdr_threshold=0.05,
    )

    agg_df, step_result = execute_predictive_audit(frames, features, config)
    assert "signal" in step_result.surviving_features
    for noise_name in noise_dict_c1.keys():
        assert noise_name in step_result.dropped_features
