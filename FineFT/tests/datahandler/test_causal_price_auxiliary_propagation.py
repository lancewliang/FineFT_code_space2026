import json

import numpy as np
import pandas as pd
import pytest

from datahandler.commodity_contract_dataset import (
    build_dataset_manifest,
    load_dataset_split_manifest,
    rebuild_train_slice_plan,
    write_stage_datasets,
    write_train_slices,
)


def _write_feature_lists(root):
    feature_dir = root / "features"
    feature_dir.mkdir()
    for name in ("rl_state_features", "vae_slope_state_features", "vae_volatility_state_features"):
        np.save(feature_dir / f"{name}.npy", np.array(["feature_a"]))
    return feature_dir


def test_scale_save_auxiliary_columns_survive_stage_copy_and_slices(tmp_path):
    feature_dir = _write_feature_lists(tmp_path)
    input_root = tmp_path / "scale_save" / "fu" / "10min"
    rows = 20
    source = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="10min"),
            "mark_price": np.linspace(100.0, 119.0, rows),
            "feature_a": np.arange(rows, dtype=float),
            "causal_mark_price_mean_abs_change_18": [0.0] * 18 + [1.0, 1.0],
            "causal_mark_price_scale_valid": [False] * 18 + [True, True],
            "source_row_index": np.arange(rows),
        }
    )
    for stage in ("train", "valid", "test"):
        target = input_root / stage
        target.mkdir(parents=True)
        source.to_feather(target / "fu2501.feather")
    (input_root / "causal_price_scale_manifest.json").write_text(
        json.dumps({"schema": "causal_price_scale.v1"}), encoding="utf-8"
    )
    split_manifest = tmp_path / "split.json"
    split_manifest.write_text(
        json.dumps(
            {
                "symbol": "fu",
                "target_freq": "10min",
                "sets": {
                    stage: {
                        "range": None,
                        "contracts": [{"contract": "fu2501", "output_row_count": rows}],
                        "skipped_contracts": [],
                    }
                    for stage in ("train", "valid", "test")
                },
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "dataset"
    manifest = build_dataset_manifest(
        load_dataset_split_manifest(split_manifest, "fu", "10min"),
        split_manifest,
        tmp_path / "scale_save",
        output_root,
        "fu",
        "10min",
        chunk_length=10,
        early_stop=1,
        rl_state_features_path=feature_dir / "rl_state_features.npy",
        vae_slope_state_features_path=feature_dir / "vae_slope_state_features.npy",
        vae_volatility_state_features_path=feature_dir / "vae_volatility_state_features.npy",
    )
    write_stage_datasets(manifest)
    rebuild_train_slice_plan(manifest, chunk_length=10, early_stop=1)
    write_train_slices(manifest)
    slice_frame = pd.read_feather(output_root / "fu" / "train" / "slice" / "df_1.feather")
    assert slice_frame["source_row_index"].tolist() == list(range(10, 20))
    assert slice_frame["causal_mark_price_scale_valid"].tolist() == [False] * 8 + [True, True]
    assert manifest.causal_price_scale_column == "causal_mark_price_mean_abs_change_18"


def test_new_manifest_rejects_missing_auxiliary_columns(tmp_path):
    input_root = tmp_path / "scale_save" / "fu" / "10min" / "train"
    input_root.mkdir(parents=True)
    pd.DataFrame({"timestamp": [1], "mark_price": [100.0], "feature_a": [1.0]}).to_feather(
        input_root / "fu2501.feather"
    )
    (input_root.parent / "causal_price_scale_manifest.json").write_text("{}", encoding="utf-8")
    feature_path = tmp_path / "rl.npy"
    np.save(feature_path, np.array(["feature_a"]))
    for stage in ("valid", "test"):
        target = input_root.parent / stage
        target.mkdir(parents=True)
        pd.DataFrame({"timestamp": [1], "mark_price": [100.0], "feature_a": [1.0]}).to_feather(
            target / "fu2501.feather"
        )
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"symbol": "fu", "target_freq": "10min", "sets": {
        stage: {"range": None, "contracts": [{"contract": "fu2501", "output_row_count": 1}], "skipped_contracts": []}
        for stage in ("train", "valid", "test")
    }}), encoding="utf-8")
    # The manifest discovers the source sidecar and consequently enables strict validation.
    dataset_manifest = build_dataset_manifest(
        load_dataset_split_manifest(split, "fu", "10min"), split, tmp_path / "scale_save",
        tmp_path / "dataset", "fu", "10min", 10, 0, feature_path,
    )
    with pytest.raises(ValueError, match="causal auxiliary columns"):
        write_stage_datasets(dataset_manifest)
