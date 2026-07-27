import json
from pathlib import Path
import numpy as np
import polars as pl
import pytest

from operator_futures.commodity.base_time_feature import BASE_TIME_FEATURE_COLUMNS
from operator_futures.scale_describe_save.muti_contract_scale_save import (
    main,
    parser,
    load_state_features,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_scale_save_passthrough_base_time_features(tmp_path):
    # Setup state_features.npy with normal feature + BASE_TIME_FEATURE_COLUMNS
    feature_list_file = tmp_path / "state_features.npy"
    all_features = ["normal_feature"] + list(BASE_TIME_FEATURE_COLUMNS)
    np.save(feature_list_file, np.array(all_features))

    split_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/5min/fu/train"
    split_dir.mkdir(parents=True, exist_ok=True)

    # Input dataframe with reward columns, normal_feature, and base_time_features
    n = 10
    df = pl.DataFrame({
        "timestamp": list(range(1, n + 1)),
        "contract": ["fu2601"] * n,
        "symbol": ["fu"] * n,
        "ask1_price": [100.0] * n,
        "ask1_size": [1.0] * n,
        "bid1_price": [99.0] * n,
        "bid1_size": [1.0] * n,
        "LowerLimitPrice": [90.0] * n,
        "UpperLimitPrice": [110.0] * n,
        "funding_timestamp": list(range(1, n + 1)),
        "funding_rate": [0.0] * n,
        "index_price": [100.0] * n,
        "mark_price": [100.0] * n,
        "normal_feature": [10.0 * i for i in range(n)],
        "trading_minute_progress": [0.1 * i for i in range(n)],
        "morning_session": [1.0] * n,
        "afternoon_session": [0.0] * n,
        "night_session": [0.0] * n,
        "is_opening_30m": [1.0] * n,
        "is_closing_30m": [0.0] * n,
        "contract_month_sin": [0.5] * n,
        "contract_month_cos": [0.5] * n,
        "contract_life_remaining_ratio": [0.8] * n,
    })
    df.write_ipc(split_dir / "fu2601.feather")

    save_path = "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE"
    args = parser.parse_args([
        "--root_path", str(tmp_path),
        "--symbols", "fu",
        "--target_freq", "5min",
        "--feature_list_path", str(feature_list_file),
        "--save_path", save_path,
        "--passthrough_features", *BASE_TIME_FEATURE_COLUMNS,
    ])

    main(args)

    output_root = tmp_path / save_path / "fu" / "5min"
    manifest_path = output_root / "scaler_manifest.json"
    assert manifest_path.exists()
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Manifest features list should only contain normal_feature
    feature_names_in_manifest = [f["feature"] for f in manifest_data["features"]]
    assert "normal_feature" in feature_names_in_manifest
    for col in BASE_TIME_FEATURE_COLUMNS:
        assert col not in feature_names_in_manifest

    # Manifest must record passthrough_state_features
    assert "passthrough_state_features" in manifest_data
    assert manifest_data["passthrough_state_features"] == list(BASE_TIME_FEATURE_COLUMNS)

    # Check scaled output feather file
    out_file = output_root / "train" / "fu2601.feather"
    assert out_file.exists()
    out_df = pl.read_ipc(out_file)

    # BASE_TIME_FEATURE values must remain unscaled/passthrough
    expected_progress = [0.1 * i for i in range(n)]
    assert np.allclose(out_df["trading_minute_progress"].to_list(), expected_progress)
    assert np.allclose(out_df["morning_session"].to_list(), [1.0] * n)
    assert np.allclose(out_df["contract_life_remaining_ratio"].to_list(), [0.8] * n)
