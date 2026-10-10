import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from operator_futures.feature_selection.muti_contract.distribution_audit import (
    audit_distribution_drift,
    _compute_pairwise_psi_and_ks,
)
from operator_futures.feature_selection.muti_contract.pipeline import run_feature_selection

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_identical_distributions_yield_near_zero_psi():
    rng1 = np.random.RandomState(42)
    rng2 = np.random.RandomState(43)
    x1 = rng1.normal(0.0, 1.0, 500)
    x2 = rng2.normal(0.0, 1.0, 500)

    mean_psi, max_pair_psi, max_ks_d, min_ks_p = _compute_pairwise_psi_and_ks([x1, x2], num_bins=10)
    assert mean_psi < 0.05
    assert max_pair_psi < 0.05
    assert min_ks_p > 0.05


def test_shifted_mean_generates_high_psi_and_fails_gate():
    rng = np.random.RandomState(100)
    x1 = rng.normal(0.0, 1.0, 500)
    x2 = rng.normal(3.0, 1.0, 500)

    mean_psi, max_pair_psi, max_ks_d, min_ks_p = _compute_pairwise_psi_and_ks([x1, x2], num_bins=10)
    assert mean_psi > 1.0
    assert max_pair_psi > 1.0
    assert max_ks_d > 0.5


def test_variance_expansion_generates_high_psi_and_fails_gate():
    rng = np.random.RandomState(200)
    x1 = rng.normal(0.0, 1.0, 500)
    x2 = rng.normal(0.0, 5.0, 500)

    mean_psi, max_pair_psi, max_ks_d, min_ks_p = _compute_pairwise_psi_and_ks([x1, x2], num_bins=10)
    assert mean_psi > 0.25
    assert max_pair_psi > 0.25


def test_audit_distribution_drift_rejects_drifting_features():
    rng = np.random.RandomState(42)
    n = 200
    frames = {
        "c1": pl.DataFrame({
            "stationary_feature": rng.normal(0.0, 1.0, n),
            "drifting_feature": rng.normal(0.0, 1.0, n),
        }),
        "c2": pl.DataFrame({
            "stationary_feature": rng.normal(0.0, 1.0, n),
            "drifting_feature": rng.normal(4.0, 1.0, n),
        }),
    }

    result = audit_distribution_drift(
        frames=frames,
        feature_universe=["stationary_feature", "drifting_feature"],
        num_bins=10,
        max_mean_psi=0.10,
        max_pair_psi=0.25,
        min_drift_survivors=1,
    )

    assert "stationary_feature" in result.surviving_features
    assert "drifting_feature" in result.dropped_features
    assert not result.fallback_triggered
    assert result.metrics_df.height == 2
    row_stat = result.metrics_df.filter(pl.col("feature") == "stationary_feature").row(0, named=True)
    row_drift = result.metrics_df.filter(pl.col("feature") == "drifting_feature").row(0, named=True)
    assert row_stat["drift_passed"] is True
    assert row_drift["drift_passed"] is False
    assert row_drift["mean_psi"] > 1.0


def test_audit_distribution_drift_safety_fallback_relaxes_threshold():
    rng = np.random.RandomState(42)
    n = 300
    # Create features with mild drift: mean_psi ~ 0.12 (between 0.10 and 0.15)
    frames = {
        "c1": pl.DataFrame({
            f"mild_drift_{i}": rng.normal(0.0, 1.0, n) for i in range(5)
        }),
        "c2": pl.DataFrame({
            f"mild_drift_{i}": rng.normal(0.35, 1.0, n) for i in range(5)
        }),
    }
    features = [f"mild_drift_{i}" for i in range(5)]

    # With min_drift_survivors=3, if initial 0.10 threshold fails, it relaxes to 0.15
    result = audit_distribution_drift(
        frames=frames,
        feature_universe=features,
        num_bins=10,
        max_mean_psi=0.01,  # extremely strict initial threshold ensures 0 pass initially
        max_pair_psi=0.01,
        min_drift_survivors=3,
    )

    assert result.fallback_triggered is True
    assert len(result.surviving_features) >= 3


def test_constant_feature_and_single_contract_safety():
    frames = {
        "c1": pl.DataFrame({"const_feat": [1.0] * 50}),
        "c2": pl.DataFrame({"const_feat": [1.0] * 50}),
    }
    result = audit_distribution_drift(
        frames=frames,
        feature_universe=["const_feat"],
        num_bins=10,
        max_mean_psi=0.10,
        max_pair_psi=0.25,
    )
    assert result.surviving_features == ["const_feat"]
    assert result.metrics_df["mean_psi"][0] == 0.0

    # Single contract
    result_single = audit_distribution_drift(
        frames={"c1": frames["c1"]},
        feature_universe=["const_feat"],
        num_bins=10,
    )
    assert result_single.surviving_features == ["const_feat"]
    assert result_single.metrics_df["mean_psi"][0] == 0.0


def test_pipeline_persists_distribution_audit_metrics_csv(tmp_path, monkeypatch):
    from operator_futures.feature_selection.muti_contract.pipeline import run_feature_selection
    import test_commodity_multi_contract_feature_selection as base_test

    # Use existing helper to write test contracts
    base_test._write_long_split_contract(
        tmp_path,
        "train",
        "fu2601",
        [float(index) for index in range(15)],
        [float(14 - index) for index in range(15)],
    )
    base_test._write_long_split_contract(
        tmp_path,
        "train",
        "fu2605",
        [float(index + 1) for index in range(15)],
        [float(15 - index) for index in range(15)],
    )

    # Mock catboost to pass quickly
    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass
        def fit(self, *args, **kwargs):
            return self
        def get_feature_importance(self, *args, **kwargs):
            return np.ones(10)

    monkeypatch.setattr("catboost.CatBoostRegressor", FakeModel)

    from operator_futures.feature_selection.muti_contract.types import StreamFilterProfile
    toy_profile = StreamFilterProfile(
        name="toy",
        max_mean_psi=0.50,
        max_pair_psi=1.0,
        min_abs_ic=0.0,
        min_sign_consistency=0.0,
        min_rank_ic_ir=0.0,
        max_correlation=1.0,
        min_clusters=1,
        max_clusters=20,
        psi_weight=0.33,
        rank_ic_weight=0.33,
        catboost_weight=0.34,
        filter_micro_persistence=False,
    )
    result = run_feature_selection(
        root_path=tmp_path,
        split_path="PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST",
        save_path="PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION",
        symbol="fu",
        target_freq="5min",
        stage="train",
        orderbook_depth=5,
        min_abs_ic=0.0,
        max_correlation=1.0,
        composite_drop_ratio=0.0,
        max_mean_psi=0.10,
        max_pair_psi=0.25,
        vae_slope_profile=toy_profile,
        vae_volatility_profile=toy_profile,
        rl_profile=toy_profile,
    )

    dist_csv = result.output_dir / "distribution_audit_metrics.csv"
    assert dist_csv.exists()
    df = pl.read_csv(dist_csv)
    assert set(df.columns) == {"feature", "mean_psi", "max_pair_psi", "max_ks_d", "min_ks_p", "drift_passed"}
    assert df.height > 0
    assert result.manifest.distribution_audit_path == str(dist_csv)


def test_zero_isolated_adaptive_binning_retains_ten_bins():
    from operator_futures.feature_selection.muti_contract.distribution_audit import (
        _compute_bin_probabilities,
    )
    rng = np.random.RandomState(42)
    # 70% zeros, 30% positive values
    n = 1000
    zeros = np.zeros(700)
    positives_1 = rng.exponential(1.0, 300)
    positives_2 = rng.exponential(1.0, 300)

    arr1 = np.concatenate([zeros, positives_1])
    arr2 = np.concatenate([zeros, positives_2])

    probs, total_bins = _compute_bin_probabilities([arr1, arr2], num_bins=10)
    assert total_bins == 10
    assert len(probs[0]) == 10
    assert len(probs[1]) == 10
    # First bin is the zero bin, which should hold ~70% of probability
    assert abs(probs[0][0] - 0.70) < 0.05
    assert abs(probs[1][0] - 0.70) < 0.05
    assert abs(sum(probs[0]) - 1.0) < 1e-6
    assert abs(sum(probs[1]) - 1.0) < 1e-6


def test_forward_boundary_drift_outpost_rejects_future_drift():
    rng = np.random.RandomState(42)
    n = 300
    train_frames = {
        "c1": pl.DataFrame({
            "stable_feat": rng.normal(0.0, 1.0, n),
            "forward_drifting_feat": rng.normal(0.0, 1.0, n),
        }),
        "c2": pl.DataFrame({
            "stable_feat": rng.normal(0.0, 1.0, n),
            "forward_drifting_feat": rng.normal(0.0, 1.0, n),
        }),
    }
    # Outpost from earliest validation contract has forward_drifting_feat shifted
    outpost_frame = pl.DataFrame({
        "stable_feat": rng.normal(0.0, 1.0, n),
        "forward_drifting_feat": rng.normal(4.0, 1.0, n),
    })

    result = audit_distribution_drift(
        frames=train_frames,
        feature_universe=["stable_feat", "forward_drifting_feat"],
        num_bins=10,
        max_mean_psi=0.10,
        max_pair_psi=0.25,
        min_drift_survivors=1,
        forward_outpost_frame=outpost_frame,
        forward_outpost_max_psi=0.15,
    )

    assert "stable_feat" in result.surviving_features
    assert "forward_drifting_feat" in result.dropped_features
    assert result.forward_psi_by_feature["stable_feat"] <= 0.15
    assert result.forward_psi_by_feature["forward_drifting_feat"] > 1.0
    assert "forward_psi" in result.metrics_df.columns


def test_macro_continuous_features_pass_distribution_audit_with_macro_tier_policy():
    rng = np.random.RandomState(42)
    n = 300
    # Macro feature with realistic inter-contract macro cycle drift (mean_psi ~ 3.5)
    train_frames = {
        "c1": pl.DataFrame({
            "trend_beta_720": rng.normal(0.0, 1.0, n),
            "micro_drifting_feat": rng.normal(0.0, 1.0, n),
        }),
        "c2": pl.DataFrame({
            "trend_beta_720": rng.normal(3.5, 1.0, n),
            "micro_drifting_feat": rng.normal(3.5, 1.0, n),
        }),
    }
    outpost_frame = pl.DataFrame({
        "trend_beta_720": rng.normal(2.5, 1.0, n),
        "micro_drifting_feat": rng.normal(2.5, 1.0, n),
    })

    result = audit_distribution_drift(
        frames=train_frames,
        feature_universe=["trend_beta_720", "micro_drifting_feat"],
        num_bins=10,
        max_mean_psi=0.10,
        max_pair_psi=0.25,
        min_drift_survivors=1,
        forward_outpost_frame=outpost_frame,
        forward_outpost_max_psi=0.50,
    )

    # Macro feature should be preserved because cycle variation across contracts is expected
    assert "trend_beta_720" in result.surviving_features
    # Micro feature with identical drift must be rejected
    assert "micro_drifting_feat" in result.dropped_features
