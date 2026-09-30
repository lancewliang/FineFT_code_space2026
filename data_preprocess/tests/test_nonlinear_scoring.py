from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from operator_futures.feature_selection.muti_contract.metrics import (
    _catboost_importance,
)
from operator_futures.feature_selection.muti_contract.nonlinear_scoring import (
    execute_nonlinear_scoring,
)
from operator_futures.feature_selection.muti_contract.types import (
    NonlinearScoringConfig,
)


def test_purged_embargo_gap_in_catboost():
    # Verify that a purge gap of window_length separates train and eval pools
    n = 100
    df = pl.DataFrame({"feat_1": [float(i) for i in range(n)]})
    future_return = np.random.normal(size=n)
    window_length = 6

    captured_pools = []

    class MockPool:
        def __init__(self, data, label):
            self.data = data
            self.label = label
            captured_pools.append(self)

    class MockCatBoost:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, train_pool, eval_set=None, verbose=None):
            return self

        def get_feature_importance(self, fit_pool):
            return [1.0]

    with patch("catboost.Pool", MockPool), patch(
        "catboost.CatBoostRegressor", MockCatBoost
    ):
        imp = _catboost_importance(
            df, ["feat_1"], future_return, window_length=window_length
        )

    assert imp["feat_1"] == 1.0
    assert len(captured_pools) >= 2
    train_pool = captured_pools[0]
    eval_pool = captured_pools[1]

    # Split_idx is int(100 * 0.8) = 80
    # Purge window is 6, so train pool ends at 80 - 6 = 74
    assert len(train_pool.data) == 74
    # Eval pool starts at 80, length is 20
    assert len(eval_pool.data) == 20
    # Purged gap is exactly 6 bars between index 74 and 80


def test_nonlinear_scoring_composite_ranking_and_truncation():
    # 10 features
    features = [f"f_{i}" for i in range(10)]
    # Feature 0 has best RankIC and low PSI
    # Feature 9 has worst RankIC and high PSI
    agg_df = pl.DataFrame(
        {
            "feature": features,
            "RankIC_Mean": [0.20 - 0.02 * i for i in range(10)],
            "RankIC_Std": [0.05] * 10,
            "IC_Mean": [0.20 - 0.02 * i for i in range(10)],
            "IC_Std": [0.05] * 10,
        }
    )
    mean_psi = {f"f_{i}": 0.01 + 0.02 * i for i in range(10)}

    # Synthetic frames
    n = 50
    df = pl.DataFrame(
        {
            "mark_price": [10.0 + 0.1 * i for i in range(n)],
            **{f"f_{i}": [float(i + j) for j in range(n)] for i in range(10)},
        }
    )
    frames = {"c1": df}

    config = NonlinearScoringConfig(
        decision_window=6,
        composite_drop_ratio=0.10,  # drops 1 out of 10
        catboost_iterations=10,
        early_stopping_rounds=5,
    )

    class MockCatBoost:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, train_pool, eval_set=None, verbose=None):
            return self

        def get_feature_importance(self, fit_pool):
            # Monotonically decreasing importance from f_0 to f_9
            return [10.0 - i for i in range(10)]

    with patch("catboost.CatBoostRegressor", MockCatBoost):
        surviving, scored_df, step_result = execute_nonlinear_scoring(
            frames, features, agg_df, mean_psi, config
        )

    # 10 * 0.10 = 1 feature dropped (f_9 has lowest Priority)
    assert len(surviving) == 9
    assert surviving[0] == "f_0"
    assert "f_9" in step_result.dropped_features
    assert "Priority" in scored_df.columns
    assert step_result.diagnostics["decision_window"] == 6
