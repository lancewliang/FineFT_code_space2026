import numpy as np
import polars as pl
import pytest

from operator_futures.feature_selection.muti_contract.orthogonal_dedup import (
    compute_contract_normalized_spearman_correlation_matrix,
    execute_orthogonal_deduplication,
    prune_by_vif,
)
from operator_futures.feature_selection.muti_contract.types import (
    OrthogonalDedupConfig,
)


def test_spearman_rank_correlation_invariant_to_monotonic_nonlinear_shifts():
    rng = np.random.RandomState(42)
    x = rng.normal(size=200)
    # Monotonic cubic and exponential transforms
    x_cubic = x**3
    x_exp = np.exp(x)

    df = pl.DataFrame(
        {
            "x_base": x,
            "x_cubic": x_cubic,
            "x_exp": x_exp,
        }
    )
    frames = {"c1": df}

    corr_df = compute_contract_normalized_spearman_correlation_matrix(
        frames, ["x_base", "x_cubic", "x_exp"]
    )
    matrix = corr_df.select(["x_base", "x_cubic", "x_exp"]).to_numpy()

    # Ranks are identical, so Spearman correlation must be exactly 1.0
    assert np.allclose(matrix, 1.0, atol=1e-5)


def test_ward_clustering_collapses_collinear_clique():
    rng = np.random.RandomState(42)
    n = 200
    base_1 = rng.normal(size=n)
    base_2 = rng.normal(size=n)

    # Clique 1: 3 features based on base_1
    f1_a = base_1
    f1_b = base_1 + rng.normal(scale=0.01, size=n)
    f1_c = base_1**3

    # Clique 2: 2 features based on base_2
    f2_a = base_2
    f2_b = base_2 + rng.normal(scale=0.01, size=n)

    df = pl.DataFrame(
        {
            "f1_a": f1_a,
            "f1_b": f1_b,
            "f1_c": f1_c,
            "f2_a": f2_a,
            "f2_b": f2_b,
        }
    )
    frames = {"c1": df}
    features = ["f1_a", "f1_b", "f1_c", "f2_a", "f2_b"]
    priority_order = ["f1_a", "f2_a", "f1_b", "f1_c", "f2_b"]

    config = OrthogonalDedupConfig(
        dedup_method="cluster",
        cluster_distance_threshold=0.30,
        min_clusters=2,
        max_clusters=70,
    )

    result = execute_orthogonal_deduplication(frames, features, priority_order, config)

    # Should collapse clique 1 to f1_a and clique 2 to f2_a
    assert "f1_a" in result.surviving_features
    assert "f2_a" in result.surviving_features
    assert "f1_b" in result.dropped_features
    assert "f1_c" in result.dropped_features
    assert "f2_b" in result.dropped_features
    assert len(result.surviving_features) == 2


def test_backward_compatible_greedy_dedup_mode():
    df = pl.DataFrame(
        {
            "feat_1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feat_2": [1.01, 2.01, 3.01, 4.01, 5.01],  # Corr ~ 1.0 with feat_1
            "feat_3": [5.0, 2.0, 1.0, 4.0, 3.0],  # Low corr
        }
    )
    frames = {"c1": df}
    features = ["feat_1", "feat_2", "feat_3"]
    priority_order = ["feat_1", "feat_2", "feat_3"]

    config = OrthogonalDedupConfig(
        dedup_method="greedy",
        max_correlation=0.70,
    )

    result = execute_orthogonal_deduplication(frames, features, priority_order, config)

    assert result.diagnostics["dedup_method"] == "greedy"
    assert "feat_1" in result.surviving_features
    assert "feat_3" in result.surviving_features
    assert "feat_2" in result.dropped_features


def test_vif_pruning():
    # 2 features with correlation 0.999 -> VIF ~ 500 > 10.0
    features = ["f_a", "f_b"]
    corr = pl.DataFrame(
        {
            "f_a": [1.0, 0.999],
            "f_b": [0.999, 1.0],
            "feature": ["f_a", "f_b"],
        }
    )
    surviving, dropped = prune_by_vif(["f_a", "f_b"], corr, max_vif=10.0)
    assert len(surviving) == 1
    assert len(dropped) == 1
