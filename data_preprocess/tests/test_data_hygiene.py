from pathlib import Path

import numpy as np
import polars as pl
import pytest

from operator_futures.feature_selection.muti_contract.data_hygiene import (
    execute_data_hygiene,
)
from operator_futures.feature_selection.muti_contract.types import (
    DataHygieneConfig,
)


def test_data_hygiene_near_zero_variance_rejection():
    # Variance < 1e-6 should be dropped
    df = pl.DataFrame(
        {
            "normal_feature": [1.0, 2.0, 3.0, 4.0, 5.0] * 20,
            "zero_var_feature": [1.0] * 100,
            "tiny_var_feature": [1.0 + 1e-4 * i for i in range(100)],  # var = 8.3e-5
            "micro_var_feature": [1.0 + 1e-5 * (i % 2) for i in range(100)],  # var = 2.5e-11
        }
    )
    frames = {"c1": df}
    config = DataHygieneConfig(min_variance=1e-6)
    cleaned_frames, step_result = execute_data_hygiene(
        frames, list(df.columns), config
    )

    assert "normal_feature" in step_result.surviving_features
    assert "tiny_var_feature" in step_result.surviving_features
    assert "zero_var_feature" in step_result.dropped_features
    assert "micro_var_feature" in step_result.dropped_features
    assert "zero_var_feature" in step_result.diagnostics["zero_variance_dropped"]
    assert "micro_var_feature" in step_result.diagnostics["zero_variance_dropped"]


def test_data_hygiene_quasi_constant_rejection():
    # Mode frequency >= 0.98 should be dropped
    # 99 out of 100 elements are 0.0 -> mode_freq = 0.99 >= 0.98
    df = pl.DataFrame(
        {
            "active_feature": [0.0] * 50 + [1.0] * 50,
            "pulse_feature": [0.0] * 99 + [100.0],
        }
    )
    frames = {"c1": df}
    config = DataHygieneConfig(max_mode_frequency=0.98)
    cleaned_frames, step_result = execute_data_hygiene(
        frames, list(df.columns), config
    )

    assert "active_feature" in step_result.surviving_features
    assert "pulse_feature" in step_result.dropped_features
    assert "pulse_feature" in step_result.diagnostics["quasi_constant_dropped"]


def test_data_hygiene_winsorization_and_disk_immutability(tmp_path: Path):
    # Test in-memory Winsorization clamps extreme spikes to 5 * IQR without mutating source files on disk
    source_feather = tmp_path / "c1.feather"
    raw_values = [1.0, 1.1, 0.9, 1.0, 1.2, 0.8, 1.05, 0.95, 1000.0, -1000.0] * 10
    df = pl.DataFrame({"spike_feature": raw_values})
    df.write_ipc(source_feather)

    # Read from disk
    loaded_df = pl.read_ipc(source_feather)
    frames = {"c1": loaded_df}

    config = DataHygieneConfig(
        enable_winsorization=True,
        winsorize_iqr_multiplier=5.0,
        min_variance=1e-6,
        max_mode_frequency=0.98,
    )
    cleaned_frames, step_result = execute_data_hygiene(
        frames, ["spike_feature"], config
    )

    assert "spike_feature" in step_result.surviving_features
    cleaned_vals = cleaned_frames["c1"]["spike_feature"].to_numpy()
    assert np.max(cleaned_vals) < 500.0
    assert np.min(cleaned_vals) > -500.0

    # Verify source file on disk is strictly unchanged
    disk_df = pl.read_ipc(source_feather)
    assert 1000.0 in disk_df["spike_feature"].to_list()
    assert -1000.0 in disk_df["spike_feature"].to_list()


def test_data_hygiene_blacklist_and_ablation():
    df = pl.DataFrame(
        {
            "keep_me": [1.0, 2.0, 3.0] * 10,
            "blacklisted": [1.0, 2.0, 3.0] * 10,
            "ablate_ratio_1": [1.0, 2.0, 3.0] * 10,
        }
    )
    frames = {"c1": df}
    config = DataHygieneConfig(
        feature_blacklist=("blacklisted",),
        feature_ablation_patterns=(r"_ratio_\d+$",),
    )
    cleaned_frames, step_result = execute_data_hygiene(
        frames, list(df.columns), config
    )

    assert step_result.surviving_features == ["keep_me"]
    assert "blacklisted" in step_result.diagnostics["blacklist_dropped"]
    assert "ablate_ratio_1" in step_result.diagnostics["ablation_dropped"]
