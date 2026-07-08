from types import SimpleNamespace

import numpy as np
import pandas as pd

from FineFT.datahandler.vae_data_creation import make_data


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
        )
    )

    assert "skip empty label: label_empty" in capsys.readouterr().out
    assert not (dataset_path / "VAE_data" / "label_empty.npy").exists()
    np.testing.assert_array_equal(
        np.load(dataset_path / "VAE_data" / "label_full.npy"),
        np.array([[1.0, 2.0]]),
    )
