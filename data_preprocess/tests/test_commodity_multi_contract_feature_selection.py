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
    build_parser,
    run_feature_selection,
)
from operator_futures.feature_selection.manifests import FeatureSelectionResult


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
        "trading_minute_progress": [0.0] * 4,
        "morning_session": [1.0] * 4,
        "afternoon_session": [0.0] * 4,
        "night_session": [0.0] * 4,
        "is_opening_30m": [1.0] * 4,
        "is_closing_30m": [0.0] * 4,
        "contract_month_sin": [0.5] * 4,
        "contract_month_cos": [0.5] * 4,
        "contract_life_remaining_ratio": [0.8] * 4,
    }
    if gamma is not None:
        values["gamma"] = gamma
    frame = pl.DataFrame(values).with_columns(pl.col("timestamp").str.strptime(pl.Datetime))
    output = stage_dir / f"{contract}.feather"
    frame.write_ipc(output)
    return output


def _write_long_split_contract(
    root: Path,
    stage: str,
    contract: str,
    alpha,
    beta,
    gamma=None,
    extra_features=None,
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
        "trading_minute_progress": [0.0] * row_count,
        "morning_session": [1.0] * row_count,
        "afternoon_session": [0.0] * row_count,
        "night_session": [0.0] * row_count,
        "is_opening_30m": [1.0] * row_count,
        "is_closing_30m": [0.0] * row_count,
        "contract_month_sin": [0.5] * row_count,
        "contract_month_cos": [0.5] * row_count,
        "contract_life_remaining_ratio": [0.8] * row_count,
    }
    if gamma is not None:
        values["gamma"] = gamma
    if extra_features is not None:
        values.update(extra_features)
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


def test_hard_filter_rejects_high_ic_feature_when_rankic_is_too_low():
    features = ["high_ic_low_rankic", "sufficient_rankic"]
    aggregate = pl.DataFrame(
        {
            "feature": features,
            "IC_Mean": [0.95, 0.02],
            "IC_Std": [0.1, 0.1],
            "RankIC_Mean": [0.0, 0.2],
            "Sharpe_Mean": [10.0, 0.0],
            "Permutation Importance_Mean": [10.0, 0.0],
            "CatBoost Importance_Mean": [10.0, 0.0],
        }
    )
    frames = {
        "fu2601": pl.DataFrame(
            {
                "high_ic_low_rankic": [1.0, 3.0, 2.0, 4.0],
                "sufficient_rankic": [1.0, 2.0, 3.0, 4.0],
            }
        )
    }

    selected, filter_results = _ordered_filter_features(
        frames,
        aggregate,
        features,
        min_abs_ic=0.1,
        max_metric_std=1.0,
        max_correlation=1.0,
    )

    assert "high_ic_low_rankic" not in filter_results["Hard Filter"]
    assert "sufficient_rankic" in filter_results["Hard Filter"]


def test_parser_accepts_runtime_feature_blacklist():
    args = build_parser().parse_args(
        [
            "--symbol",
            "fu",
            "--target_freq",
            "5min",
            "--stage",
            "train",
            "--feature_blacklist",
            "wap_1",
            "last_price",
        ]
    )

    assert args.feature_blacklist == ["wap_1", "last_price"]


def test_train_stage_writes_final_features_metrics_filtered_outputs_and_manifest(tmp_path, fake_catboost):
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
    selected_feature_path = stage_dir / "state_features.npy"
    assert selected_feature_path.exists()
    selected_features = np.load(selected_feature_path, allow_pickle=True).tolist()
    assert selected_features
    assert not (stage_dir / "state_features_candidate.npy").exists()
    assert (stage_dir / "per_contract" / "fu2601_metrics.csv").exists()
    assert (stage_dir / "per_contract" / "fu2605_metrics.csv").exists()
    assert (stage_dir / "aggregate_metrics.csv").exists()
    assert (stage_dir / "feature_selection_manifest.json").exists()
    assert (stage_dir / "fu2601" / "df.feather").exists()
    persisted_manifest = json.loads(
        (stage_dir / "feature_selection_manifest.json").read_text(encoding="utf-8")
    )
    assert isinstance(manifest, FeatureSelectionResult)
    assert manifest.output_dir == stage_dir
    assert manifest.manifest.stage == "train"
    assert manifest.manifest.selected_feature_file.endswith("train/state_features.npy")
    assert manifest.manifest.selected_feature_count == len(selected_features)
    assert persisted_manifest == manifest.manifest.to_dict()
    metrics = pl.read_csv(stage_dir / "aggregate_metrics.csv")
    assert {"IC_Mean", "IC_Std", "IC_Median", "Sharpe_Mean", "Sharpe_Std", "Sharpe_Median"}.issubset(metrics.columns)
    per_contract_metrics = pl.read_csv(stage_dir / "per_contract" / "fu2601_metrics.csv")
    assert per_contract_metrics["window"].unique().sort().to_list() == [1, 6, 12]
    assert manifest.manifest.windows_list == [1, 6, 12]


def test_train_stage_applies_feature_blacklist_only_to_final_outputs(tmp_path, fake_catboost):
    _write_long_split_contract(
        tmp_path,
        "train",
        "fu2601",
        [float(index) for index in range(15)],
        [float(14 - index) for index in range(15)],
        extra_features={
            "wap_1": [
                0.0,
                2.0,
                1.0,
                4.0,
                3.0,
                6.0,
                5.0,
                8.0,
                7.0,
                10.0,
                9.0,
                12.0,
                11.0,
                14.0,
                13.0,
            ]
        },
    )
    _write_long_split_contract(
        tmp_path,
        "train",
        "fu2605",
        [float(index + 1) for index in range(15)],
        [float(15 - index) for index in range(15)],
        extra_features={
            "wap_1": [
                1.0,
                3.0,
                2.0,
                5.0,
                4.0,
                7.0,
                6.0,
                9.0,
                8.0,
                11.0,
                10.0,
                13.0,
                12.0,
                15.0,
                14.0,
            ]
        },
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
        max_correlation=1.0,
        composite_drop_ratio=0.0,
        feature_blacklist=["wap_1", "mark_price", "ask1_price"],
    )

    stage_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/train"
    selected_features = np.load(
        stage_dir / "state_features.npy", allow_pickle=True
    ).tolist()
    filtered = pl.read_ipc(stage_dir / "fu2601" / "df.feather")
    aggregate = pl.read_csv(stage_dir / "aggregate_metrics.csv")
    persisted_manifest = json.loads(
        (stage_dir / "feature_selection_manifest.json").read_text(encoding="utf-8")
    )

    assert "wap_1" in aggregate["feature"].to_list()
    assert "wap_1" not in selected_features
    assert "wap_1" not in filtered.columns
    assert "mark_price" in filtered.columns
    assert "ask1_price" in filtered.columns
    assert manifest.manifest.feature_blacklist == ["wap_1", "mark_price", "ask1_price"]
    assert persisted_manifest["feature_blacklist"] == [
        "wap_1",
        "mark_price",
        "ask1_price",
    ]
    assert manifest.manifest.filter_results["Feature Blacklist Dropped"] == ["wap_1"]
    assert persisted_manifest["filter_results"]["Feature Blacklist Dropped"] == ["wap_1"]
    assert manifest.manifest.selected_feature_count == len(selected_features)


def test_train_stage_filters_fast_decay_micro_returns_by_persistence(
    tmp_path, fake_catboost
):
    row_count = 24
    fast_decay = [1.0 if index % 2 == 0 else -1.0 for index in range(row_count)]
    slow_signal = [float(index) for index in range(row_count)]
    _write_long_split_contract(
        tmp_path,
        "train",
        "fu2601",
        slow_signal,
        [float(row_count - index) for index in range(row_count)],
        extra_features={
            "wap_1_log_return_2": fast_decay,
            "mandatory_log_return_2": fast_decay,
            "trend_strength_norm": slow_signal,
        },
    )
    _write_long_split_contract(
        tmp_path,
        "train",
        "fu2605",
        [value + 1.0 for value in slow_signal],
        [float(row_count - index + 1) for index in range(row_count)],
        extra_features={
            "wap_1_log_return_2": fast_decay,
            "mandatory_log_return_2": fast_decay,
            "trend_strength_norm": [value + 1.0 for value in slow_signal],
        },
    )

    manifest = run_feature_selection(
        root_path=tmp_path,
        split_path="PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST",
        save_path="PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION",
        symbol="fu",
        target_freq="5min",
        stage="train",
        orderbook_depth=5,
        min_abs_ic=0.0,
        max_correlation=1.0,
        composite_drop_ratio=0.0,
        min_half_life_bars=1.0,
        mandatory_state_features=["mandatory_log_return_2"],
    )

    stage_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/train"
    selected_features = np.load(
        stage_dir / "state_features.npy", allow_pickle=True
    ).tolist()
    persisted_manifest = json.loads(
        (stage_dir / "feature_selection_manifest.json").read_text(encoding="utf-8")
    )

    assert "wap_1_log_return_2" not in selected_features
    assert "mandatory_log_return_2" in selected_features
    assert "trend_strength_norm" in selected_features
    assert manifest.manifest.filter_results["Persistence Filter Dropped"] == [
        "wap_1_log_return_2"
    ]
    assert persisted_manifest["persistence_filter"]["min_half_life_bars"] == 1.0
    assert persisted_manifest["persistence_filter"]["active_feature_pattern"] == (
        "_log_return_(1|2)$"
    )
    diagnostics = {
        row["feature"]: row
        for row in persisted_manifest["persistence_diagnostics"]
    }
    assert diagnostics["wap_1_log_return_2"]["active_filter"] is True
    assert diagnostics["wap_1_log_return_2"]["half_life_bars_median"] == 0.0
    assert diagnostics["trend_strength_norm"]["active_filter"] is False


def test_train_stage_semantically_deduplicates_equivalent_log_return_aliases(
    tmp_path, fake_catboost
):
    row_count = 24
    alias_values = [float(index) for index in range(row_count)]
    _write_long_split_contract(
        tmp_path,
        "train",
        "fu2601",
        alias_values,
        [float(row_count - index) for index in range(row_count)],
        extra_features={
            "wap_1_log_return_2": alias_values,
            "wap_1_log_return_6": alias_values,
        },
    )
    _write_long_split_contract(
        tmp_path,
        "train",
        "fu2605",
        [value + 1.0 for value in alias_values],
        [float(row_count - index + 1) for index in range(row_count)],
        extra_features={
            "wap_1_log_return_2": [value + 1.0 for value in alias_values],
            "wap_1_log_return_6": [value + 1.0 for value in alias_values],
        },
    )

    manifest = run_feature_selection(
        root_path=tmp_path,
        split_path="PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST",
        save_path="PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION",
        symbol="fu",
        target_freq="5min",
        stage="train",
        orderbook_depth=5,
        min_abs_ic=0.0,
        max_correlation=1.0,
        composite_drop_ratio=0.0,
    )

    stage_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/train"
    aggregate_features = pl.read_csv(stage_dir / "aggregate_metrics.csv")[
        "feature"
    ].to_list()
    selected_features = np.load(
        stage_dir / "state_features.npy", allow_pickle=True
    ).tolist()

    assert "wap_1_log_return_2" in aggregate_features
    assert "wap_1_log_return_6" not in aggregate_features
    assert "wap_1_log_return_6" not in selected_features
    assert manifest.manifest.filter_results[
        "Feature Semantic Deduplication Dropped"
    ] == ["wap_1_log_return_6"]


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


def test_valid_stage_evaluates_train_features_without_writing_downstream_features(tmp_path, fake_catboost):
    _write_split_contract(tmp_path, "valid", "fu2601", [1.0, 2.0, 3.0, 4.0], [4.0, 4.0, 4.0, 4.0], gamma=[9.0, 8.0, 7.0, 6.0])
    train_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/train"
    train_dir.mkdir(parents=True)
    train_feature_file = train_dir / "state_features.npy"
    np.save(train_feature_file, np.array(["alpha"]))

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
    manifest_path = stage_dir / "feature_selection_manifest.json"
    assert (stage_dir / "per_contract" / "fu2601_metrics.csv").exists()
    assert (stage_dir / "aggregate_metrics.csv").exists()
    assert manifest_path.exists()
    assert not (stage_dir / "state_features.npy").exists()
    assert not (stage_dir / "fu2601" / "df.feather").exists()
    persisted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(manifest, FeatureSelectionResult)
    assert manifest.output_dir == stage_dir
    assert manifest.manifest.evaluated_feature_file.endswith("train/state_features.npy")
    assert persisted_manifest == manifest.manifest.to_dict()
    assert persisted_manifest["evaluated_feature_file"] == manifest.manifest.evaluated_feature_file
    assert manifest.manifest.report_only is True
    assert manifest.manifest.evaluated_feature_count == 1
    assert manifest.manifest.evaluated_features == ["alpha"]
    assert persisted_manifest["report_only"] is True
    assert persisted_manifest["evaluated_feature_count"] == 1
    assert persisted_manifest["evaluated_features"] == ["alpha"]
    assert "filter_results" not in persisted_manifest
    assert "selected_feature_file" not in persisted_manifest
    assert "filtered_outputs" not in persisted_manifest


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


def test_valid_stage_fails_when_train_feature_file_is_empty(tmp_path):
    _write_split_contract(tmp_path, "valid", "fu2601", [1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0])
    train_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/train"
    train_dir.mkdir(parents=True)
    np.save(train_dir / "state_features.npy", np.array([]))

    with pytest.raises(ValueError, match="feature list is empty"):
        run_feature_selection(
            root_path=tmp_path,
            split_path="PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST",
            save_path="PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION",
            symbol="fu",
            target_freq="5min",
            stage="valid",
            orderbook_depth=5,
        )


def test_valid_stage_fails_when_train_feature_column_is_missing(tmp_path):
    _write_split_contract(tmp_path, "valid", "fu2601", [1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0])
    train_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/train"
    train_dir.mkdir(parents=True)
    np.save(train_dir / "state_features.npy", np.array(["missing_alpha"]))

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


def test_stability_filter_rejects_unstable_feature_with_low_rank_ic_ir():
    features = ["stable_feature", "unstable_feature"]
    aggregate = pl.DataFrame(
        {
            "feature": features,
            "IC_Mean": [0.05, 0.05],
            "IC_Std": [0.01, 0.20],
            "RankIC_Mean": [0.05, 0.05],
            "RankIC_Std": [0.02, 0.20],  # IR for stable = 2.5, for unstable = 0.25
            "Sharpe_Mean": [1.0, 1.0],
            "Permutation Importance_Mean": [0.1, 0.1],
            "CatBoost Importance_Mean": [0.1, 0.1],
        }
    )
    frames = {
        "fu2601": pl.DataFrame(
            {
                "stable_feature": [1.0, 2.0, 3.0],
                "unstable_feature": [1.0, 3.0, 2.0],
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
        min_rank_ic_ir=1.0,
    )

    assert "unstable_feature" not in filter_results["Stability Filter"]
    assert "stable_feature" in filter_results["Stability Filter"]


def test_calculate_future_return_sorts_unsorted_timestamps():
    from operator_futures.feature_selection.muti_contract.metrics import calculate_future_return
    frame = pl.DataFrame(
        {
            "timestamp": ["2026-01-01 09:10:00", "2026-01-01 09:00:00", "2026-01-01 09:05:00"],
            "mark_price": [13.0, 10.0, 11.0],
        }
    )
    # Sorted order: 09:00 (10.0), 09:05 (11.0), 09:10 (13.0)
    # Future return (window=1): at 09:00 -> (11-10)/10 = 0.1, at 09:05 -> (13-11)/11 = 2/11
    returns = calculate_future_return(frame, window=1)
    assert len(returns) == 2
    assert returns[0] == pytest.approx(0.1)
    assert returns[1] == pytest.approx(2.0 / 11.0)


def test_catboost_importance_uses_temporal_split_when_sample_size_large(fake_catboost):
    from operator_futures.feature_selection.muti_contract.metrics import calculate_metric_frame
    rows = 15
    frame = pl.DataFrame(
        {
            "mark_price": [float(i) for i in range(rows)],
            "alpha": [float(i) for i in range(rows)],
            "beta": [float(rows - i) for i in range(rows)],
        }
    )
    result = calculate_metric_frame(frame, ["alpha", "beta"], windows_list=[1])
    assert fake_catboost["fit"]["eval_set"] is not None
    assert fake_catboost["fit"]["eval_set"].x.shape[0] < fake_catboost["fit"]["train_pool"].x.shape[0] + fake_catboost["fit"]["eval_set"].x.shape[0]
