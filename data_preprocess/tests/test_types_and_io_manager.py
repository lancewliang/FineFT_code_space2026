from pathlib import Path

import numpy as np
import polars as pl
import pytest

from operator_futures.feature_selection.manifests import (
    FeatureSelectionContractRecord,
    FeatureSelectionManifest,
)
from operator_futures.feature_selection.muti_contract.io_manager import (
    PipelineIOManager,
    extract_state_features,
    load_feature_list,
)
from operator_futures.feature_selection.muti_contract.types import (
    DataHygieneConfig,
    DistributionAuditConfig,
    FeatureSelectionPipelineConfig,
    NonlinearScoringConfig,
    OrthogonalDedupConfig,
    PipelineStepResult,
    PredictiveAuditConfig,
    RegimeAuditConfig,
    StationarityAuditConfig,
)


def test_immutable_types_and_defaults():
    config = FeatureSelectionPipelineConfig(
        root_path=Path("/tmp"),
        symbol="fu",
        target_freq="5min",
        stage="train",
    )
    assert config.symbol == "fu"
    assert config.hygiene.min_variance == 1e-6
    assert config.drift.max_mean_psi == 0.10
    assert config.stationarity.adf_significance_level == 0.05
    assert config.predictive.min_sign_consistency == 0.75
    assert config.scoring.decision_window == 6
    assert config.dedup.cluster_distance_threshold == 0.50
    assert config.regime.regime_bins == 4

    with pytest.raises(Exception):
        config.symbol = "rb"  # frozen dataclass


def test_pipeline_step_result_contract():
    result = PipelineStepResult(
        step_name="data_hygiene",
        surviving_features=["alpha", "beta"],
        dropped_features=["constant_col"],
        diagnostics={"dropped_count": 1},
    )
    assert result.step_name == "data_hygiene"
    assert result.surviving_features == ["alpha", "beta"]
    assert result.dropped_features == ["constant_col"]
    assert result.diagnostics["dropped_count"] == 1


def test_io_manager_load_and_write(tmp_path: Path):
    split_dir = (
        tmp_path
        / "PREPROCESS_DATASET"
        / "commodity-futures"
        / "SPLIT-TRAIN-VALID-TEST"
        / "5min"
        / "fu"
        / "train"
    )
    split_dir.mkdir(parents=True, exist_ok=True)

    df_c1 = pl.DataFrame(
        {
            "timestamp": ["2026-01-01 09:05:00", "2026-01-01 09:00:00"],
            "trading_day": ["2026-01-01", "2026-01-01"],
            "contract": ["c1", "c1"],
            "mark_price": [10.0, 9.5],
            "bid1_price": [9.9, 9.4],
            "ask1_price": [10.1, 9.6],
            "alpha": [1.0, 2.0],
            "beta": [0.1, 0.2],
        }
    )
    df_c1.write_ipc(split_dir / "c1.feather")

    config = FeatureSelectionPipelineConfig(
        root_path=tmp_path,
        symbol="fu",
        target_freq="5min",
        stage="train",
    )
    io = PipelineIOManager(config)

    # 1. load_stage_frames
    frames = io.load_stage_frames()
    assert "c1" in frames
    # Verify sorting by timestamp
    assert frames["c1"]["timestamp"].to_list() == [
        "2026-01-01 09:00:00",
        "2026-01-01 09:05:00",
    ]

    # 2. resolve_initial_feature_universe
    raw_universe = io.resolve_initial_feature_universe(frames)
    assert sorted(raw_universe) == ["alpha", "beta"]

    # 3. write_filtered_outputs
    outputs = io.write_filtered_outputs(frames, ["alpha"])
    assert len(outputs) == 1
    assert outputs[0].contract == "c1"
    assert outputs[0].output_row_count == 2
    written_df = pl.read_ipc(outputs[0].output_path)
    assert "alpha" in written_df.columns
    assert "beta" not in written_df.columns
    assert "symbol" in written_df.columns

    # 4. save_selected_features
    saved_npy = io.save_selected_features(["alpha"])
    assert saved_npy.exists()
    assert np.load(saved_npy).tolist() == ["alpha"]

    # 5. save_manifest
    manifest = FeatureSelectionManifest(
        symbol="fu",
        target_freq="5min",
        stage="train",
        split_input_dir=str(io.input_dir),
        windows_list=[6],
        aggregate_metrics_path="dummy.csv",
    )
    saved_manifest = io.save_manifest(manifest)
    assert saved_manifest.exists()


def test_io_manager_validation_outpost(tmp_path: Path):
    valid_dir = (
        tmp_path
        / "PREPROCESS_DATASET"
        / "commodity-futures"
        / "SPLIT-TRAIN-VALID-TEST"
        / "5min"
        / "fu"
        / "valid"
    )
    valid_dir.mkdir(parents=True, exist_ok=True)

    df_v1 = pl.DataFrame(
        {
            "timestamp": ["2026-02-01 09:00:00"],
            "mark_price": [12.0],
            "alpha": [1.5],
            "beta": [0.3],
            "future_return": [0.01],
        }
    )
    df_v1.write_ipc(valid_dir / "c_valid_1.feather")

    config = FeatureSelectionPipelineConfig(
        root_path=tmp_path,
        symbol="fu",
        target_freq="5min",
        stage="train",
    )
    io = PipelineIOManager(config)
    outpost_df = io.load_validation_outpost_frame(["alpha", "beta"])
    assert outpost_df is not None
    assert set(outpost_df.columns) == {"alpha", "beta"}
    assert "mark_price" not in outpost_df.columns
    assert "future_return" not in outpost_df.columns
