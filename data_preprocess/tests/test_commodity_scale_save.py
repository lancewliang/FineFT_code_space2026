import json
from pathlib import Path
import numpy as np
import polars as pl
import pandas as pd
import pytest

from operator_futures.scale_describe_save.muti_contract_scale_save import (
    main,
    parser,
    load_state_features,
)


def _create_mock_contract_df(contract: str, n_rows: int = 50) -> pl.DataFrame:
    np.random.seed(42)
    return pl.DataFrame({
        "timestamp": list(range(1, n_rows + 1)),
        "contract": [contract] * n_rows,
        "symbol": ["fu"] * n_rows,
        "ask1_price": [100.0 + i * 0.05 for i in range(n_rows)],
        "ask1_size": [10.0 + i for i in range(n_rows)],
        "bid1_price": [99.5 + i * 0.05 for i in range(n_rows)],
        "bid1_size": [12.0 + i for i in range(n_rows)],
        "LowerLimitPrice": [90.0] * n_rows,
        "UpperLimitPrice": [110.0] * n_rows,
        "funding_timestamp": list(range(1, n_rows + 1)),
        "funding_rate": [0.0] * n_rows,
        "index_price": [100.0] * n_rows,
        "mark_price": [100.0] * n_rows,
        # VAE stream specific features
        "base_time_day_progress": [i / float(n_rows) for i in range(n_rows)],
        "realized_volatility_6": [0.01 + 0.001 * (i % 5) for i in range(n_rows)],
        # RL stream specific features
        "level5_ofi_weighted_norm": [np.sin(i / 3.0) for i in range(n_rows)],
        "log_return_1": [0.0005 * ((i % 4) - 2) for i in range(n_rows)],
        "log_return_6": [0.001 * ((i % 6) - 3) for i in range(n_rows)],
        "depth_depletion": [0.1 * (i % 10) for i in range(n_rows)],
        # Shared features
        "wap_1_return_trend": [0.02 * (i - n_rows // 2) for i in range(n_rows)],
    })


def test_scale_save_union_features_and_dual_stream_slicing(tmp_path):
    vae_features = ["base_time_day_progress", "realized_volatility_6", "wap_1_return_trend"]
    rl_features = [
        "level5_ofi_weighted_norm",
        "log_return_1",
        "log_return_6",
        "depth_depletion",
        "wap_1_return_trend",
    ]
    union_features = sorted(list(set(vae_features) | set(rl_features)))

    fs_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/train"
    fs_dir.mkdir(parents=True, exist_ok=True)
    np.save(fs_dir / "vae_state_features.npy", np.array(vae_features))
    np.save(fs_dir / "rl_state_features.npy", np.array(rl_features))

    split_base = tmp_path / "PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/5min/fu"
    for stage, contract in [("train", "fu2601"), ("valid", "fu2602"), ("test", "fu2603")]:
        stage_dir = split_base / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        df = _create_mock_contract_df(contract, n_rows=50)
        df.write_ipc(stage_dir / f"{contract}.feather")

    save_path = "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE"
    args = parser.parse_args([
        "--root_path", str(tmp_path),
        "--symbols", "fu",
        "--target_freq", "5min",
        "--feature_selection_dir", str(fs_dir),
        "--save_path", save_path,
        "--scale_method", "rolling_zscore",
        "--rolling_window", "24",
        "--clip_mode", "tanh",
        "--soft_clip_m", "4.0",
        "--clip_min", "-5.0",
        "--clip_max", "5.0",
        "--passthrough_features", "base_time_day_progress",
    ])

    main(args)

    output_root = tmp_path / save_path / "fu" / "5min"
    assert (output_root / "scaler_manifest.json").exists()
    assert (output_root / "scale_diagnostics.csv").exists()
    assert not (output_root / "state_features.npy").exists()
    assert (output_root / "rl_state_features.npy").exists()
    assert (output_root / "vae_state_features.npy").exists()
    np.testing.assert_array_equal(
        np.load(output_root / "rl_state_features.npy"),
        np.array(rl_features),
    )
    np.testing.assert_array_equal(
        np.load(output_root / "vae_state_features.npy"),
        np.array(vae_features),
    )

    manifest_data = json.loads((output_root / "scaler_manifest.json").read_text(encoding="utf-8"))
    assert manifest_data["clip"]["mode"] == "tanh"
    assert manifest_data["clip"]["soft_clip_m"] == 4.0
    assert manifest_data["passthrough_state_features"] == ["base_time_day_progress"]

    for stage, contract in [("train", "fu2601"), ("valid", "fu2602"), ("test", "fu2603")]:
        scaled_feather = output_root / stage / f"{contract}.feather"
        assert scaled_feather.exists()
        scaled_df = pl.read_ipc(scaled_feather)

        # Scaled df must have zero null/NaN across all union features
        for f in union_features:
            assert scaled_df[f].null_count() == 0

        # Downstream zero-copy view slicing for RL state features
        rl_state_slice = scaled_df.select(rl_features).to_numpy()
        assert rl_state_slice.shape == (50, len(rl_features))
        assert not np.isnan(rl_state_slice).any()
        assert not np.isinf(rl_state_slice).any()

        # Downstream zero-copy view slicing for VAE regime features
        vae_state_slice = scaled_df.select(vae_features).to_numpy()
        assert vae_state_slice.shape == (50, len(vae_features))
        assert not np.isnan(vae_state_slice).any()
        assert not np.isinf(vae_state_slice).any()

        # Pandas interface compatibility (used by RL environments)
        pdf = pd.read_feather(scaled_feather)
        rl_arr = pdf[rl_features].values
        assert isinstance(rl_arr, np.ndarray)
        assert rl_arr.shape == (50, len(rl_features))
        assert not np.isnan(rl_arr).any()


def test_scale_save_preflight_validation_catches_missing_columns(tmp_path):
    union_features = ["feature_1", "feature_2", "missing_feature"]
    fs_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/train"
    fs_dir.mkdir(parents=True, exist_ok=True)
    np.save(fs_dir / "rl_state_features.npy", np.array(union_features))
    np.save(fs_dir / "vae_state_features.npy", np.array(union_features))

    split_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/5min/fu/train"
    split_dir.mkdir(parents=True, exist_ok=True)
    df = _create_mock_contract_df("fu2601", n_rows=20).with_columns([
        pl.lit(1.0).alias("feature_1"),
        pl.lit(2.0).alias("feature_2"),
    ])
    df.write_ipc(split_dir / "fu2601.feather")

    save_path = "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE"
    args = parser.parse_args([
        "--root_path", str(tmp_path),
        "--symbols", "fu",
        "--target_freq", "5min",
        "--feature_selection_dir", str(fs_dir),
        "--save_path", save_path,
    ])

    with pytest.raises(ValueError) as exc_info:
        main(args)
    assert "missing_feature" in str(exc_info.value)
