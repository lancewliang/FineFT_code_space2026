import json
from pathlib import Path
import numpy as np
import polars as pl
import pytest

from operator_futures.commodity.base_time_feature import BASE_TIME_FEATURE_COLUMNS
from operator_futures.feature_selection.muti_contract.pipeline import run_feature_selection

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_feature_selection_excludes_base_time_from_metrics_and_appends_to_final(tmp_path):
    split_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/5min/fu/train"
    split_dir.mkdir(parents=True, exist_ok=True)

    n = 20
    df = pl.DataFrame({
        "timestamp": list(range(1, n + 1)),
        "contract": ["fu2601"] * n,
        "symbol": ["fu"] * n,
        "mark_price": [10.0, 11.0, 13.0, 16.0, 20.0, 25.0, 31.0, 38.0, 46.0, 55.0] * 2,
        "bid1_price": [9.9, 10.9, 12.9, 15.9, 19.9, 24.9, 30.9, 37.9, 45.9, 54.9] * 2,
        "ask1_price": [10.1, 11.1, 13.1, 16.1, 20.1, 25.1, 30.1, 38.1, 46.1, 55.1] * 2,
        "ask1_size": [1.0] * n,
        "bid1_size": [1.0] * n,
        "LowerLimitPrice": [90.0] * n,
        "UpperLimitPrice": [110.0] * n,
        "funding_timestamp": list(range(1, n + 1)),
        "funding_rate": [0.0] * n,
        "index_price": [10.0, 11.0, 13.0, 16.0, 20.0, 25.0, 31.0, 38.0, 46.0, 55.0] * 2,
        "normal_feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 2,
        "normal_feature_2": [5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0] * 2,
        "trading_minute_progress": [i / n for i in range(n)],
        "morning_session": [1.0] * n,
        "afternoon_session": [0.0] * n,
        "night_session": [0.0] * n,
        "is_opening_30m": [1.0] * n,
        "is_closing_30m": [0.0] * n,
        "is_session_first_bar": [1.0, 1.0] + [0.0] * (n - 2),
        "is_session_last_bar": [0.0] * (n - 2) + [1.0, 1.0],
        "contract_month_sin": [0.5] * n,
        "contract_month_cos": [0.5] * n,
        "contract_life_remaining_ratio": [0.8] * n,
    })
    df.write_ipc(split_dir / "fu2601.feather")

    res = run_feature_selection(
        root_path=tmp_path,
        symbol="fu",
        target_freq="5min",
        stage="train",
        min_abs_ic=0.0,
        max_metric_std=100.0,
        max_correlation=1.0,
        composite_drop_ratio=0.0,
        mandatory_state_features=list(BASE_TIME_FEATURE_COLUMNS),
    )

    state_features_file = res.output_dir / "state_features.npy"
    assert state_features_file.exists()
    final_features = np.load(state_features_file, allow_pickle=True).tolist()

    # BASE_TIME_FEATURE_COLUMNS must appear at the end of final_features
    for col in BASE_TIME_FEATURE_COLUMNS:
        assert col in final_features

    assert final_features[-len(BASE_TIME_FEATURE_COLUMNS):] == BASE_TIME_FEATURE_COLUMNS

    manifest_path = res.output_dir / "feature_selection_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "mandatory_state_features" in manifest
    assert manifest["mandatory_state_features"] == BASE_TIME_FEATURE_COLUMNS


def test_feature_selection_rejects_blacklist_targeting_base_time(tmp_path):
    split_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/5min/fu/train"
    split_dir.mkdir(parents=True, exist_ok=True)

    n = 20
    df = pl.DataFrame({
        "timestamp": list(range(1, n + 1)),
        "contract": ["fu2601"] * n,
        "symbol": ["fu"] * n,
        "mark_price": [10.0, 11.0, 13.0, 16.0, 20.0, 25.0, 31.0, 38.0, 46.0, 55.0] * 2,
        "bid1_price": [9.9, 10.9, 12.9, 15.9, 19.9, 24.9, 30.9, 37.9, 45.9, 54.9] * 2,
        "ask1_price": [10.1, 11.1, 13.1, 16.1, 20.1, 25.1, 30.1, 38.1, 46.1, 55.1] * 2,
        "ask1_size": [1.0] * n,
        "bid1_size": [1.0] * n,
        "LowerLimitPrice": [90.0] * n,
        "UpperLimitPrice": [110.0] * n,
        "funding_timestamp": list(range(1, n + 1)),
        "funding_rate": [0.0] * n,
        "index_price": [10.0, 11.0, 13.0, 16.0, 20.0, 25.0, 31.0, 38.0, 46.0, 55.0] * 2,
        "normal_feature_1": [1.0 * i for i in range(n)],
        "trading_minute_progress": [i / n for i in range(n)],
        "morning_session": [1.0] * n,
        "afternoon_session": [0.0] * n,
        "night_session": [0.0] * n,
        "is_opening_30m": [1.0] * n,
        "is_closing_30m": [0.0] * n,
        "is_session_first_bar": [1.0, 1.0] + [0.0] * (n - 2),
        "is_session_last_bar": [0.0] * (n - 2) + [1.0, 1.0],
        "contract_month_sin": [0.5] * n,
        "contract_month_cos": [0.5] * n,
        "contract_life_remaining_ratio": [0.8] * n,
    })
    df.write_ipc(split_dir / "fu2601.feather")

    with pytest.raises(ValueError, match="blacklist"):
        run_feature_selection(
            root_path=tmp_path,
            symbol="fu",
            target_freq="5min",
            stage="train",
            feature_blacklist=["trading_minute_progress"],
            mandatory_state_features=list(BASE_TIME_FEATURE_COLUMNS),
        )
