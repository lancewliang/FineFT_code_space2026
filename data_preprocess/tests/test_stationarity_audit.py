import numpy as np
import polars as pl
import pytest

from operator_futures.feature_selection.muti_contract.stationarity_audit import (
    directional_half_life_bars,
    execute_stationarity_audit,
    lag1_autocorrelation,
    sign_alternation_rate,
)
from operator_futures.feature_selection.muti_contract.types import (
    StationarityAuditConfig,
)


def test_stationarity_audit_passes_stationary_and_rejects_random_walk():
    rng = np.random.RandomState(42)
    n = 300

    # 1. Stationary AR(1) process
    stationary = np.zeros(n)
    for t in range(1, n):
        stationary[t] = 0.2 * stationary[t - 1] + rng.normal()

    # 2. Cumulative random walk
    random_walk = np.cumsum(rng.normal(size=n))

    frames = {
        "c1": pl.DataFrame(
            {
                "stationary_feat": stationary,
                "random_walk_feat": random_walk,
            }
        ),
        "c2": pl.DataFrame(
            {
                "stationary_feat": stationary + rng.normal(scale=0.1, size=n),
                "random_walk_feat": random_walk + rng.normal(scale=0.1, size=n),
            }
        ),
    }

    # Set min_survivors_floor=1 so fallback doesn't force retention of both
    config = StationarityAuditConfig(
        adf_significance_level=0.05,
        min_passing_contract_ratio=0.70,
        min_survivors_floor=1,
        min_half_life_bars=0.0,
        max_sign_alternation_rate=1.0,
    )

    result = execute_stationarity_audit(
        frames, ["stationary_feat", "random_walk_feat"], config
    )

    assert "stationary_feat" in result.surviving_features
    assert "random_walk_feat" in result.dropped_features
    assert not result.diagnostics["fallback_triggered"]
    assert "random_walk_feat" in result.diagnostics["adf_dropped"]


def test_stationarity_audit_fallback_safeguard_triggers_when_survivors_low():
    rng = np.random.RandomState(42)
    n = 200

    # Create 5 random walks - none will pass p < 0.05
    frames = {
        "c1": pl.DataFrame(
            {f"rw_{i}": np.cumsum(rng.normal(size=n)) for i in range(5)}
        ),
        "c2": pl.DataFrame(
            {f"rw_{i}": np.cumsum(rng.normal(size=n)) for i in range(5)}
        ),
    }

    config = StationarityAuditConfig(
        adf_significance_level=0.05,
        min_passing_contract_ratio=0.70,
        min_survivors_floor=3,
        min_half_life_bars=0.0,
        max_sign_alternation_rate=1.0,
    )

    result = execute_stationarity_audit(frames, [f"rw_{i}" for i in range(5)], config)

    assert result.diagnostics["fallback_triggered"] is True
    assert len(result.surviving_features) >= 3


def test_stationarity_audit_persistence_half_life_and_sar_filtering():
    rng = np.random.RandomState(42)
    n = 300
    # Oscillating / noisy feature with high SAR and negative autocorrelation
    oscillating = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n)])
    # Stationary AR(1) with positive persistence
    persistent = np.zeros(n)
    for t in range(1, n):
        persistent[t] = 0.8 * persistent[t - 1] + rng.normal(scale=0.1)
    persistent += 5.0  # strictly positive so SAR = 0.0

    frames = {
        "c1": pl.DataFrame(
            {
                "oscillating_feat": oscillating,
                "persistent_feat": persistent,
            }
        )
    }

    config = StationarityAuditConfig(
        min_survivors_floor=1,
        min_half_life_bars=2.0,
        max_sign_alternation_rate=0.40,
    )

    result = execute_stationarity_audit(
        frames, ["oscillating_feat", "persistent_feat"], config
    )

    assert "persistent_feat" in result.surviving_features
    assert "oscillating_feat" in result.dropped_features
    assert (
        "oscillating_feat" in result.diagnostics["persistence_dropped"]
        or "oscillating_feat" in result.diagnostics["sar_dropped"]
    )
