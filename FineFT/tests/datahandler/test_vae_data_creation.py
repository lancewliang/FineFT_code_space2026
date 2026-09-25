import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from datahandler.vae_data_creation import make_data


def test_make_data_skips_empty_label_directory(tmp_path, capsys):
    dataset_path = tmp_path / "dataset" / "demo"
    valid_path = dataset_path / "valid"
    (valid_path / "label_empty").mkdir(parents=True)
    (valid_path / "label_full").mkdir()

    np.save(dataset_path / "state_features.npy", np.array(["feature_a", "feature_b"]))
    pd.DataFrame({"feature_a": [1.0], "feature_b": [2.0]}).to_feather(
        valid_path / "label_full" / "df_0.feather"
    )
    pd.DataFrame({"feature_a": [3.0], "feature_b": [4.0]}).to_feather(
        dataset_path / "test.feather"
    )

    make_data(
        SimpleNamespace(
            base_path=str(tmp_path / "dataset"),
            dataset_name="demo",
            save_path=str(tmp_path / "dataset"),
            source_split="valid",
            labeling_method="slope",
        )
    )

    assert "skip empty label: label_empty" in capsys.readouterr().out
    assert not (dataset_path / "VAE_data" / "slope" / "label_empty.npy").exists()
    np.testing.assert_array_equal(
        np.load(dataset_path / "VAE_data" / "slope" / "label_full.npy"),
        np.array([[1.0, 2.0]]),
    )


def test_make_data_writes_contract_scoped_test_arrays(tmp_path):
    dataset_path = tmp_path / "dataset" / "fu"
    (dataset_path / "valid" / "label_0").mkdir(parents=True)
    (dataset_path / "test").mkdir()
    np.save(dataset_path / "state_features.npy", np.array(["feature_a", "feature_b"]))
    pd.DataFrame({"feature_a": [1.0], "feature_b": [2.0]}).to_feather(
        dataset_path / "valid" / "label_0" / "df_0.feather"
    )
    pd.DataFrame({"feature_a": [3.0], "feature_b": [4.0]}).to_feather(
        dataset_path / "test" / "df_fu2601.feather"
    )
    pd.DataFrame({"feature_a": [5.0], "feature_b": [6.0]}).to_feather(
        dataset_path / "test" / "df_fu2605.feather"
    )

    make_data(
        SimpleNamespace(
            base_path=str(tmp_path / "dataset"),
            dataset_name="fu",
            save_path=str(tmp_path / "dataset"),
            source_split="valid",
            labeling_method="slope",
        )
    )

    assert not (dataset_path / "VAE_data" / "test.npy").exists()
    np.testing.assert_array_equal(
        np.load(dataset_path / "VAE_data" / "test" / "test_fu2601.npy"),
        np.array([[3.0, 4.0]]),
    )
    np.testing.assert_array_equal(
        np.load(dataset_path / "VAE_data" / "test" / "test_fu2605.npy"),
        np.array([[5.0, 6.0]]),
    )


def test_make_data_writes_contract_scoped_valid_label_arrays(tmp_path):
    dataset_path = tmp_path / "dataset" / "fu"
    (dataset_path / "valid" / "fu2505" / "label_0").mkdir(parents=True)
    (dataset_path / "valid" / "fu2509" / "label_0").mkdir(parents=True)
    (dataset_path / "valid" / "fu2505" / "label_1").mkdir(parents=True)
    (dataset_path / "valid" / "processed").mkdir()
    (dataset_path / "test").mkdir()
    np.save(dataset_path / "state_features.npy", np.array(["feature_a", "feature_b"]))
    pd.DataFrame({"feature_a": [1.0], "feature_b": [2.0]}).to_feather(
        dataset_path / "valid" / "fu2505" / "label_0" / "df_0.feather"
    )
    pd.DataFrame({"feature_a": [3.0], "feature_b": [4.0]}).to_feather(
        dataset_path / "valid" / "fu2509" / "label_0" / "df_0.feather"
    )
    pd.DataFrame({"feature_a": [5.0], "feature_b": [6.0]}).to_feather(
        dataset_path / "valid" / "fu2505" / "label_1" / "df_0.feather"
    )
    pd.DataFrame({"feature_a": [7.0], "feature_b": [8.0]}).to_feather(
        dataset_path / "test" / "df_fu2505.feather"
    )

    make_data(
        SimpleNamespace(
            base_path=str(tmp_path / "dataset"),
            dataset_name="fu",
            save_path=str(tmp_path / "dataset"),
            source_split="valid",
            labeling_method="slope",
        )
    )

    assert not (dataset_path / "VAE_data" / "label_0.npy").exists()
    assert not (dataset_path / "VAE_data" / "fu2505" / "label_0.npy").exists()
    np.testing.assert_array_equal(
        np.load(dataset_path / "VAE_data" / "slope" / "fu2505" / "label_0.npy"),
        np.array([[1.0, 2.0]]),
    )
    np.testing.assert_array_equal(
        np.load(dataset_path / "VAE_data" / "slope" / "fu2509" / "label_0.npy"),
        np.array([[3.0, 4.0]]),
    )
    np.testing.assert_array_equal(
        np.load(dataset_path / "VAE_data" / "slope" / "fu2505" / "label_1.npy"),
        np.array([[5.0, 6.0]]),
    )


def test_make_data_reads_selected_labeling_method_directory(tmp_path):
    dataset_path = tmp_path / "dataset" / "fu"
    volatility_path = dataset_path / "valid" / "volatility"
    (volatility_path / "fu2505" / "label_0").mkdir(parents=True)
    (dataset_path / "test").mkdir()
    np.save(dataset_path / "state_features.npy", np.array(["feature_a", "feature_b"]))
    pd.DataFrame({"feature_a": [1.0], "feature_b": [2.0]}).to_feather(
        volatility_path / "fu2505" / "label_0" / "df_0.feather"
    )
    pd.DataFrame({"feature_a": [3.0], "feature_b": [4.0]}).to_feather(
        dataset_path / "test" / "df_fu2505.feather"
    )

    make_data(
        SimpleNamespace(
            base_path=str(tmp_path / "dataset"),
            dataset_name="fu",
            save_path=str(tmp_path / "dataset"),
            source_split="valid",
            labeling_method="volatility",
        )
    )

    np.testing.assert_array_equal(
        np.load(dataset_path / "VAE_data" / "volatility" / "fu2505" / "label_0.npy"),
        np.array([[1.0, 2.0]]),
    )


def test_make_data_reads_train_split_and_prunes_stale_contract_directories(tmp_path):
    dataset_path = tmp_path / "dataset" / "fu"
    train_path = dataset_path / "train" / "slope"
    (train_path / "fu2305" / "label_0").mkdir(parents=True)
    (dataset_path / "test").mkdir()

    # Pre-populate a stale contract directory from a previous valid run
    stale_contract_dir = dataset_path / "VAE_data" / "slope" / "fu2508"
    stale_contract_dir.mkdir(parents=True)
    np.save(stale_contract_dir / "label_0.npy", np.array([[99.0, 99.0]]))

    np.save(dataset_path / "state_features.npy", np.array(["feature_a", "feature_b"]))
    pd.DataFrame({"feature_a": [11.0], "feature_b": [12.0]}).to_feather(
        train_path / "fu2305" / "label_0" / "df_0.feather"
    )
    pd.DataFrame({"feature_a": [3.0], "feature_b": [4.0]}).to_feather(
        dataset_path / "test" / "df_fu2601.feather"
    )

    make_data(
        SimpleNamespace(
            base_path=str(tmp_path / "dataset"),
            dataset_name="fu",
            save_path=str(tmp_path / "dataset"),
            source_split="train",
            labeling_method="slope",
        )
    )

    # Stale valid directory fu2508 should be pruned
    assert not stale_contract_dir.exists()

    # New train contract fu2305 should be saved
    np.testing.assert_array_equal(
        np.load(dataset_path / "VAE_data" / "slope" / "fu2305" / "label_0.npy"),
        np.array([[11.0, 12.0]]),
    )
