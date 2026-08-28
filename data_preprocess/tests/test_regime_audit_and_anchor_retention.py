import json
from pathlib import Path
import numpy as np
import polars as pl
import pytest

from operator_futures.feature_selection.muti_contract.pipeline import run_feature_selection
from operator_futures.feature_selection.muti_contract.regime_audit import (
    MARKET_STATE_ANCHOR_COLUMNS,
    TARGET_REGIME_BINS,
    compute_regime_quantiles,
    audit_16_regimes,
)


def _create_mock_split_dataset(tmp_path: Path, num_contracts: int = 4, num_rows: int = 120):
    split_dir = (
        tmp_path
        / "PREPROCESS_DATASET"
        / "commodity-futures"
        / "SPLIT-TRAIN-VALID-TEST"
        / "30min"
        / "fu"
    )
    (split_dir / "train").mkdir(parents=True, exist_ok=True)
    (split_dir / "valid").mkdir(parents=True, exist_ok=True)

    steps = np.arange(num_rows, dtype=float)

    for i in range(num_contracts):
        contract_name = f"fu260{i+1}"
        slope_val = 0.001 * (i + 1)
        rng = np.random.default_rng(42 + i)
        noise = rng.normal(0.0, 0.002, num_rows)
        close = 100.0 * np.exp(slope_val * steps + noise)
        df = pl.DataFrame(
            {
                "timestamp": steps.astype(int),
                "contract": [contract_name] * num_rows,
                "close": close,
                "mark_price": close,
                # Add market state anchors
                "log_price_slope_48": np.full(num_rows, slope_val),
                "log_price_slope_96": np.full(num_rows, slope_val * 0.9),
                "trend_to_noise_48": np.full(num_rows, 0.5 * (i + 1)),
                "trend_to_noise_96": np.full(num_rows, 0.4 * (i + 1)),
                "signed_efficiency_48": np.full(num_rows, 0.8),
                "trend_r2_48": np.full(num_rows, 0.95),
                "log_return_vol_quantile_192": np.full(num_rows, 0.5),
                # Add ordinary candidate features
                "feature_a": np.sin(steps / 5.0) + noise,
                "feature_b": np.cos(steps / 5.0) - noise,
                # Add reward execution column
                "future_return_30m": rng.normal(0.0001 * (i + 1), 0.001, num_rows),
            }
        )
        df.write_ipc(split_dir / "train" / f"{contract_name}.feather")
        df.write_ipc(split_dir / "valid" / f"{contract_name}.feather")


def test_regime_audit_16_bins_conservation_and_valid_reuse(tmp_path):
    _create_mock_split_dataset(tmp_path, num_contracts=4, num_rows=100)

    res_train = run_feature_selection(
        root_path=tmp_path,
        symbol="fu",
        target_freq="30min",
        stage="train",
        enable_conditional_anchors=True,
    )

    train_manifest = res_train.manifest
    assert train_manifest.regime_quantiles is not None
    assert "slope" in train_manifest.regime_quantiles
    assert "volatility" in train_manifest.regime_quantiles
    assert len(train_manifest.regime_quantiles["slope"]) == 3
    assert len(train_manifest.regime_quantiles["volatility"]) == 3

    # Check 16-bin audit CSV
    audit_path = Path(train_manifest.regime_audit_path)
    assert audit_path.exists()

    audit_df = pl.read_csv(audit_path)
    # Check that all 16 (slope_bin, vol_bin) combinations exist
    bins_present = set(zip(audit_df["slope_bin"].to_list(), audit_df["vol_bin"].to_list()))
    expected_16_bins = {(s, v) for s in range(4) for v in range(4)}
    assert bins_present == expected_16_bins

    # Sum of step counts for unique (slope_bin, vol_bin) should equal total mature steps
    unique_bin_steps = (
        audit_df.select(["slope_bin", "vol_bin", "step_count"])
        .unique()
        ["step_count"]
        .sum()
    )
    expected_mature_steps = 4 * (100 - 47)  # 4 contracts x 53 mature steps
    assert unique_bin_steps == expected_mature_steps

    # Valid run: should reuse train quantiles
    res_valid = run_feature_selection(
        root_path=tmp_path,
        symbol="fu",
        target_freq="30min",
        stage="valid",
        enable_conditional_anchors=True,
    )
    valid_manifest = res_valid.manifest
    assert valid_manifest.regime_quantiles == train_manifest.regime_quantiles


def test_conditional_anchor_retention_and_regular_candidate_exclusion(tmp_path):
    _create_mock_split_dataset(tmp_path, num_contracts=4, num_rows=120)

    res_train = run_feature_selection(
        root_path=tmp_path,
        symbol="fu",
        target_freq="30min",
        stage="train",
        min_abs_ic=0.0001,  # low threshold for testing
        enable_conditional_anchors=True,
    )

    manifest = res_train.manifest
    selected = set(manifest.selected_features)

    # Any conditionally retained features in manifest should be market state anchors only
    if manifest.conditional_anchors_retained:
        retained_names = {item["feature"] for item in manifest.conditional_anchors_retained}
        assert retained_names.issubset(set(MARKET_STATE_ANCHOR_COLUMNS))
        assert not retained_names.intersection({"feature_a", "feature_b"})
