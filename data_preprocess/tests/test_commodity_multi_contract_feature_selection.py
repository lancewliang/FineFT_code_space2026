import json
import sys
import types
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from operator_futures.feature_selection.muti_contract.metrics import (
    aggregate_metric_frames,
    calculate_metric_frame,
    calculate_sharpe,
)
from operator_futures.feature_selection.muti_contract import metrics
from operator_futures.feature_selection.muti_contract.pipeline import (
    _ordered_filter_features,
    run_feature_selection,
)


def _write_split_contract(root: Path, stage: str, contract: str, alpha, beta, gamma=None):
    stage_dir = (
        root
        / "PREPROCESS_DATASET"
        / "commodity-futures"
        / "SPLIT-TRAIN-VALID-TEST"
        / "5min"
        / "fu"
        / stage
    )
    stage_dir.mkdir(parents=True, exist_ok=True)
    values = {
        "timestamp": ["2026-01-01 09:00:00", "2026-01-01 09:05:00", "2026-01-01 09:10:00", "2026-01-01 09:15:00"],
        "trading_day": ["2026-01-01", "2026-01-01", "2026-01-01", "2026-01-01"],
        "contract": [contract, contract, contract, contract],
        "mark_price": [10.0, 11.0, 13.0, 16.0],
        "bid1_price": [9.9, 10.9, 12.9, 15.9],
        "ask1_price": [10.1, 11.1, 13.1, 16.1],
        "alpha": alpha,
        "beta": beta,
    }
    if gamma is not None:
        values["gamma"] = gamma
    frame = pl.DataFrame(values).with_columns(pl.col("timestamp").str.strptime(pl.Datetime))
    output = stage_dir / f"{contract}.feather"
    frame.write_ipc(output)
    return output


def _write_long_split_contract(
    root: Path, stage: str, contract: str, alpha, beta, gamma=None
):
    stage_dir = (
        root
        / "PREPROCESS_DATASET"
        / "commodity-futures"
        / "SPLIT-TRAIN-VALID-TEST"
        / "5min"
        / "fu"
        / stage
    )
    stage_dir.mkdir(parents=True, exist_ok=True)
    row_count = len(alpha)
    values = {
        "timestamp": [f"2026-01-01 09:{index:02d}:00" for index in range(row_count)],
        "trading_day": ["2026-01-01"] * row_count,
        "contract": [contract] * row_count,
        "mark_price": [10.0 + float(index * index) for index in range(row_count)],
        "bid1_price": [9.9 + float(index * index) for index in range(row_count)],
        "ask1_price": [10.1 + float(index * index) for index in range(row_count)],
        "alpha": alpha,
        "beta": beta,
    }
    if gamma is not None:
        values["gamma"] = gamma
    frame = pl.DataFrame(values).with_columns(pl.col("timestamp").str.strptime(pl.Datetime))
    output = stage_dir / f"{contract}.feather"
    frame.write_ipc(output)
    return output


@pytest.fixture
def fake_catboost(monkeypatch):
    calls = {}

    class FakePool:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    class FakeCatBoostRegressor:
        def __init__(self, **kwargs):
            calls["params"] = kwargs

        def fit(self, train_pool, eval_set=None, verbose=None):
            calls["fit"] = {
                "train_pool": train_pool,
                "eval_set": eval_set,
                "verbose": verbose,
            }

        def get_feature_importance(self, pool):
            calls["importance_pool"] = pool
            return np.linspace(0.25, 0.75, pool.x.shape[1])

    fake_module = types.SimpleNamespace(
        CatBoostRegressor=FakeCatBoostRegressor,
        Pool=FakePool,
    )
    monkeypatch.setitem(sys.modules, "catboost", fake_module)
    return calls


def test_calculate_sharpe_uses_single_feature_pseudo_returns():
    feature = np.array([1.0, 2.0, 3.0, 4.0])
    future_return = np.array([0.1, 0.2, -0.1, 0.3])

    result = calculate_sharpe(feature, future_return)

    z = (feature - feature.mean()) / feature.std(ddof=0)
    pseudo_returns = z * future_return
    expected = pseudo_returns.mean() / pseudo_returns.std(ddof=1)
    assert result == pytest.approx(expected)


def test_ic_matches_original_pairwise_nan_and_degenerate_handling():
    assert np.isnan(metrics.calculate_ic([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))

    result = metrics.calculate_ic(
        [1.0, np.nan, 3.0, 4.0],
        [2.0, 3.0, np.nan, 5.0],
    )

    assert result == pytest.approx(1.0)


def test_rank_ic_matches_original_argsort_rank_and_degenerate_handling():
    assert metrics.calculate_rank_ic([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0

    feature = np.array([10.0, 30.0, 20.0])
    target = np.array([3.0, 1.0, 2.0])
    expected = np.corrcoef(
        np.argsort(np.argsort(feature)),
        np.argsort(np.argsort(target)),
    )[0, 1]

    assert metrics.calculate_rank_ic(feature, target) == pytest.approx(expected)


def test_catboost_importance_matches_original_training_call(fake_catboost):
    frame = pl.DataFrame(
        {
            "mark_price": [10.0, 11.0, 13.0, 16.0],
            "alpha": [1.0, 2.0, 3.0, 4.0],
            "beta": [4.0, 3.0, 2.0, 1.0],
        }
    )

    result = calculate_metric_frame(frame, ["alpha", "beta"], windows_list=[1])

    assert fake_catboost["params"] == {
        "iterations": 1000,
        "learning_rate": 0.1,
        "depth": 6,
        "loss_function": "MAE",
        "task_type": "GPU",
        "random_seed": 42,
    }
    assert fake_catboost["fit"]["eval_set"] is not None
    assert fake_catboost["fit"]["verbose"] == 100
    assert fake_catboost["fit"]["train_pool"] is fake_catboost["importance_pool"]
    assert result["CatBoost Importance"].to_list() == [0.25, 0.75]


def test_metric_frame_uses_original_default_windows(fake_catboost):
    frame = pl.DataFrame(
        {
            "mark_price": [float(index) for index in range(15)],
            "alpha": [float(index) for index in range(15)],
            "beta": [float(14 - index) for index in range(15)],
        }
    )

    result = calculate_metric_frame(frame, ["alpha", "beta"])

    assert result["window"].to_list() == [1, 1, 6, 6, 12, 12]


def test_aggregate_metric_frames_writes_mean_std_median_columns():
    first = pl.DataFrame({"feature": ["alpha", "beta"], "IC": [0.5, 0.2], "Sharpe": [1.0, 0.5]})
    second = pl.DataFrame({"feature": ["alpha", "beta"], "IC": [0.7, 0.1], "Sharpe": [1.4, 0.4]})

    result = aggregate_metric_frames([first, second])

    alpha = result.filter(pl.col("feature") == "alpha").row(0, named=True)
    assert alpha["IC_Mean"] == pytest.approx(0.6)
    assert alpha["IC_Median"] == pytest.approx(0.6)
    assert alpha["IC_Std"] == pytest.approx(np.std([0.5, 0.7], ddof=1))
    assert "Sharpe_Mean" in result.columns
    assert "Sharpe_Std" in result.columns
    assert "Sharpe_Median" in result.columns


def test_composite_score_drops_bottom_ten_percent_with_rankic_priority():
    features = [f"feature_{index}" for index in range(10)]
    aggregate = pl.DataFrame(
        {
            "feature": features,
            "IC_Mean": [0.1] * 10,
            "IC_Std": [0.1] * 10,
            "RankIC_Mean": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01],
            "Sharpe_Mean": [0.0] * 9 + [100.0],
            "Permutation Importance_Mean": [0.0] * 9 + [100.0],
            "CatBoost Importance_Mean": [0.0] * 9 + [100.0],
        }
    )
    frames = {
        "fu2601": pl.DataFrame(
            {
                feature: [float(index), float(index + 1), float(index + 3)]
                for index, feature in enumerate(features)
            }
        )
    }

    selected, filter_results = _ordered_filter_features(
        frames,
        aggregate,
        features,
        min_abs_ic=0.01,
        max_metric_std=1.0,
        max_correlation=1.0,
    )

    assert "feature_9" not in selected
    assert filter_results["Composite Score"][:3] == [
        "feature_0",
        "feature_1",
        "feature_2",
    ]
    assert filter_results["Composite Score"] == features[:-1]
    assert filter_results["Composite Score Dropped"] == ["feature_9"]


def test_train_stage_writes_candidates_metrics_filtered_outputs_and_manifest(tmp_path, fake_catboost):
    _write_long_split_contract(
        tmp_path,
        "train",
        "fu2601",
        [float(index) for index in range(15)],
        [float(14 - index) for index in range(15)],
    )
    _write_long_split_contract(
        tmp_path,
        "train",
        "fu2605",
        [float(index + 1) for index in range(15)],
        [float(15 - index) for index in range(15)],
    )

    manifest = run_feature_selection(
        root_path=tmp_path,
        split_path="PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST",
        save_path="PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION",
        symbol="fu",
        target_freq="5min",
        stage="train",
        orderbook_depth=5,
        min_abs_ic=0.01,
        max_correlation=0.99,
    )

    stage_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/train"
    candidates = np.load(stage_dir / "state_features_candidate.npy", allow_pickle=True).tolist()
    assert candidates
    assert (stage_dir / "per_contract" / "fu2601_metrics.csv").exists()
    assert (stage_dir / "per_contract" / "fu2605_metrics.csv").exists()
    assert (stage_dir / "aggregate_metrics.csv").exists()
    assert (stage_dir / "feature_selection_manifest.json").exists()
    assert (stage_dir / "fu2601" / "df.feather").exists()
    assert manifest["stage"] == "train"
    assert manifest["selected_feature_file"].endswith("state_features_candidate.npy")
    assert manifest["selected_feature_count"] == len(candidates)
    metrics = pl.read_csv(stage_dir / "aggregate_metrics.csv")
    assert {"IC_Mean", "IC_Std", "IC_Median", "Sharpe_Mean", "Sharpe_Std", "Sharpe_Median"}.issubset(metrics.columns)
    per_contract_metrics = pl.read_csv(stage_dir / "per_contract" / "fu2601_metrics.csv")
    assert per_contract_metrics["window"].unique().sort().to_list() == [1, 6, 12]
    assert manifest["windows_list"] == [1, 6, 12]


def test_train_stage_rejects_illegal_feature_values_before_metrics(tmp_path, fake_catboost):
    _write_long_split_contract(
        tmp_path,
        "train",
        "fu2601",
        [0.0, 1.0, float("nan"), 3.0, 4.0, 5.0, float("inf"), 7.0, 8.0, 9.0, 10.0, 11.0, float("-inf"), 13.0, 14.0],
        [float(index) for index in range(15)],
    )

    with pytest.raises(ValueError) as exc_info:
        run_feature_selection(
            root_path=tmp_path,
            split_path="PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST",
            save_path="PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION",
            symbol="fu",
            target_freq="5min",
            stage="train",
            orderbook_depth=5,
            min_abs_ic=0.01,
            max_correlation=0.99,
        )

    message = str(exc_info.value)
    assert "Illegal data detected" in message
    assert "stage=train_feature_selection_input" in message
    assert "contract=fu2601" in message
    assert "alpha:nan=1" in message
    assert "alpha:infinite=2" in message
    assert "CatBoostRegressor" not in fake_catboost


def test_valid_stage_uses_train_candidates_and_writes_final_features(tmp_path, fake_catboost):
    _write_split_contract(tmp_path, "valid", "fu2601", [1.0, 2.0, 3.0, 4.0], [4.0, 4.0, 4.0, 4.0], gamma=[9.0, 8.0, 7.0, 6.0])
    train_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/train"
    train_dir.mkdir(parents=True)
    np.save(train_dir / "state_features_candidate.npy", np.array(["alpha"]))

    manifest = run_feature_selection(
        root_path=tmp_path,
        split_path="PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST",
        save_path="PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION",
        symbol="fu",
        target_freq="5min",
        stage="valid",
        orderbook_depth=5,
        min_abs_ic=0.01,
        max_correlation=0.99,
    )

    stage_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/valid"
    selected = np.load(stage_dir / "state_features.npy", allow_pickle=True).tolist()
    assert selected == ["alpha"]
    filtered = pl.read_ipc(stage_dir / "fu2601" / "df.feather")
    assert "alpha" in filtered.columns
    assert "gamma" not in filtered.columns
    assert filtered.get_column("symbol").unique().to_list() == ["fu"]
    assert manifest["candidate_feature_file"].endswith("state_features_candidate.npy")
    assert manifest["selected_feature_file"].endswith("state_features.npy")


def test_feature_selection_fails_for_missing_split_input(tmp_path):
    with pytest.raises(FileNotFoundError, match="split input directory"):
        run_feature_selection(
            root_path=tmp_path,
            split_path="PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST",
            save_path="PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION",
            symbol="fu",
            target_freq="5min",
            stage="train",
            orderbook_depth=5,
        )


def test_valid_stage_fails_when_candidate_file_is_empty(tmp_path):
    _write_split_contract(tmp_path, "valid", "fu2601", [1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0])
    train_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/train"
    train_dir.mkdir(parents=True)
    np.save(train_dir / "state_features_candidate.npy", np.array([]))

    with pytest.raises(ValueError, match="candidate feature list is empty"):
        run_feature_selection(
            root_path=tmp_path,
            split_path="PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST",
            save_path="PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION",
            symbol="fu",
            target_freq="5min",
            stage="valid",
            orderbook_depth=5,
        )


def test_valid_stage_fails_when_candidate_column_is_missing(tmp_path):
    _write_split_contract(tmp_path, "valid", "fu2601", [1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0])
    train_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/train"
    train_dir.mkdir(parents=True)
    np.save(train_dir / "state_features_candidate.npy", np.array(["missing_alpha"]))

    with pytest.raises(ValueError, match="missing_alpha"):
        run_feature_selection(
            root_path=tmp_path,
            split_path="PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST",
            save_path="PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION",
            symbol="fu",
            target_freq="5min",
            stage="valid",
            orderbook_depth=5,
        )
