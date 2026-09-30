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
    _state_features,
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
        "early_stopping_rounds": 30,
    }
    assert fake_catboost["fit"]["eval_set"] is not None
    assert fake_catboost["fit"]["verbose"] == 100
    assert fake_catboost["fit"]["train_pool"] is fake_catboost["importance_pool"]
    assert result["CatBoost Importance"].to_list() == [0.25, 0.75]


def test_metric_frame_uses_intraday_default_windows(fake_catboost):
    frame = pl.DataFrame(
        {
            "mark_price": [float(index) for index in range(15)],
            "alpha": [float(index) for index in range(15)],
            "beta": [float(14 - index) for index in range(15)],
        }
    )

    result = calculate_metric_frame(frame, ["alpha", "beta"])

    assert result["window"].to_list() == [1, 1, 2, 2, 6, 6, 12, 12]


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


def test_state_features_exclude_raw_price_and_oi_levels_and_keep_derived_features():
    frame = pl.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "open_interest": [100.0, 200.0],
            "close": [1.0, 2.0],
            "mark_price": [1.0, 2.0],
            "wap_1": [1.0, 2.0],
            "vwap": [1.0, 2.0],
            "buy_wap": [1.0, 2.0],
            "buy_spread_oe_max": [1.0, 2.0],
            "open_interest_change_ratio_192": [0.0, 0.1],
            "prev_day_vwap_deviation_pct": [0.0, 0.1],
            "relative_spread": [1.0, 2.0],
            "close_log_return_1": [0.0, 0.1],
            "relative_strength": [0.0, 0.1],
        }
    )

    result = _state_features(frame, orderbook_depth=5)

    assert "open" not in result
    assert "high" not in result
    assert "low" not in result
    assert "open_interest" not in result
    assert "close" not in result
    assert "mark_price" not in result
    assert result == [
        "wap_1",
        "vwap",
        "buy_wap",
        "buy_spread_oe_max",
        "open_interest_change_ratio_192",
        "prev_day_vwap_deviation_pct",
        "relative_spread",
        "close_log_return_1",
        "relative_strength",
    ]


def test_state_features_excludes_close_and_treats_as_reward_column():
    frame = pl.DataFrame(
        {
            "close": [10.0, 11.0],
            "volume": [100.0, 200.0],
            "tradeval": [1000.0, 2000.0],
            "mark_price": [10.0, 11.0],
            "close_log_return_1": [0.0, 0.1],
            "feature_alpha": [1.0, 2.0],
        }
    )
    result = _state_features(frame, orderbook_depth=5)
    assert "close" not in result
    assert "volume" not in result
    assert "tradeval" not in result
    assert "mark_price" not in result
    assert "close_log_return_1" in result
    assert "feature_alpha" in result


def test_parser_restores_legacy_feature_selection_defaults():
    args = build_parser().parse_args(
        ["--symbol", "fu", "--target_freq", "5min", "--stage", "train"]
    )

    assert args.rank_ic_mode == "absolute"
    assert args.feature_ablation_patterns == []
    assert args.min_half_life_bars == 0.0


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
    assert not (stage_dir / "state_features.npy").exists()
    rl_feature_path = stage_dir / "rl_state_features.npy"
    assert rl_feature_path.exists()
    rl_features = np.load(rl_feature_path, allow_pickle=True).tolist()
    assert rl_features
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
    assert manifest.manifest.rl_feature_file.endswith("train/rl_state_features.npy")
    assert manifest.manifest.selected_feature_count == len(manifest.manifest.selected_features)
    assert persisted_manifest == manifest.manifest.to_dict()
    metrics = pl.read_csv(stage_dir / "aggregate_metrics.csv")
    assert {"IC_Mean", "IC_Std", "IC_Median", "Sharpe_Mean", "Sharpe_Std", "Sharpe_Median"}.issubset(metrics.columns)
    per_contract_metrics = pl.read_csv(stage_dir / "per_contract" / "fu2601_metrics.csv")
    assert per_contract_metrics["window"].unique().sort().to_list() == [1, 2, 6, 12]
    assert manifest.manifest.windows_list == [1, 2, 6, 12, 24, 48]


def test_train_stage_front_loads_feature_blacklist_preventing_metric_evaluation(tmp_path, fake_catboost):
    _write_long_split_contract(
        tmp_path,
        "train",
        "fu2601",
        [float(index) for index in range(15)],
        [float(14 - index) for index in range(15)],
        extra_features={
            "custom_signal": [
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
            "custom_signal": [
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
        feature_blacklist=["custom_signal", "mark_price", "ask1_price"],
        feature_ablation_patterns=[],
        rank_ic_mode="absolute",
    )

    stage_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/train"
    assert not (stage_dir / "state_features.npy").exists()
    selected_features = np.load(
        stage_dir / "rl_state_features.npy", allow_pickle=True
    ).tolist()
    filtered = pl.read_ipc(stage_dir / "fu2601" / "df.feather")
    aggregate = pl.read_csv(stage_dir / "aggregate_metrics.csv")
    persisted_manifest = json.loads(
        (stage_dir / "feature_selection_manifest.json").read_text(encoding="utf-8")
    )

    # Front-loaded blacklist purges before metric calculation
    assert "custom_signal" not in aggregate["feature"].to_list()
    assert "custom_signal" not in selected_features
    assert "custom_signal" not in filtered.columns
    assert "mark_price" in filtered.columns
    assert "ask1_price" in filtered.columns
    assert manifest.manifest.feature_blacklist == ["custom_signal", "mark_price", "ask1_price"]
    assert persisted_manifest["feature_blacklist"] == [
        "custom_signal",
        "mark_price",
        "ask1_price",
    ]
    assert manifest.manifest.filter_results["Feature Blacklist Dropped"] == ["custom_signal"]
    assert persisted_manifest["filter_results"]["Feature Blacklist Dropped"] == ["custom_signal"]
    assert manifest.manifest.selected_feature_count == len(selected_features)


def test_front_loaded_blacklist_eliminates_borrowed_knife_correlation_dropping(tmp_path, fake_catboost):
    # toxic_feature is correlated with valid_stationary_feature
    # toxic_feature is blacklisted.
    # Front-loaded blacklist must drop toxic_feature upfront, so valid_stationary_feature survives correlation filter.
    toxic = [float(i) for i in range(15)]
    valid = [float(i) for i in range(15)]
    # Use uncorrelated alpha/beta so only toxic and valid correlate
    independent_beta = [1.0, 0.0, -1.0, 0.5, -0.5, 0.8, -0.8, 0.2, -0.2, 0.1, -0.1, 0.3, -0.3, 0.4, -0.4]
    _write_long_split_contract(
        tmp_path,
        "train",
        "fu2601",
        independent_beta,
        independent_beta,
        extra_features={
            "toxic_feature": toxic,
            "valid_stationary_feature": valid,
        },
    )
    _write_long_split_contract(
        tmp_path,
        "train",
        "fu2605",
        independent_beta,
        independent_beta,
        extra_features={
            "toxic_feature": toxic,
            "valid_stationary_feature": valid,
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
        max_correlation=0.70,
        composite_drop_ratio=0.0,
        feature_blacklist=["toxic_feature"],
        feature_ablation_patterns=[],
        rank_ic_mode="absolute",
    )

    stage_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/train"
    assert not (stage_dir / "state_features.npy").exists()
    selected_features = np.load(
        stage_dir / "rl_state_features.npy", allow_pickle=True
    ).tolist()

    assert "toxic_feature" not in selected_features
    assert "valid_stationary_feature" in selected_features
    assert "toxic_feature" in manifest.manifest.filter_results["Feature Blacklist Dropped"]


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
        rank_ic_mode="absolute",
    )

    stage_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/train"
    assert not (stage_dir / "state_features.npy").exists()
    selected_features = np.load(
        stage_dir / "rl_state_features.npy", allow_pickle=True
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


def test_train_stage_does_not_filter_equivalent_log_return_aliases(
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
            "buy_volume_oe_trend_2": alias_values,
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
            "buy_volume_oe_trend_2": [value + 1.0 for value in alias_values],
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
    assert "wap_1_log_return_2" in aggregate_features
    assert "wap_1_log_return_6" in aggregate_features
    assert "buy_volume_oe_trend_2" in aggregate_features
    assert "Feature Semantic Deduplication Dropped" not in (
        manifest.manifest.filter_results
    )


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
    train_feature_file = train_dir / "rl_state_features.npy"
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
    assert manifest.manifest.evaluated_feature_file.endswith("train/rl_state_features.npy")
    assert persisted_manifest == manifest.manifest.to_dict()
    assert persisted_manifest["evaluated_feature_file"] == manifest.manifest.evaluated_feature_file
    assert manifest.manifest.report_only is True
    assert manifest.manifest.evaluated_feature_count == 1
    assert manifest.manifest.evaluated_features == ["alpha"]
    assert persisted_manifest["report_only"] is True
    assert persisted_manifest["evaluated_feature_count"] == 1
    assert persisted_manifest["evaluated_features"] == ["alpha"]
    assert "filter_results" not in persisted_manifest
    assert "rl_feature_file" not in persisted_manifest
    assert "vae_feature_file" not in persisted_manifest
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
    np.save(train_dir / "rl_state_features.npy", np.array([]))

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
    np.save(train_dir / "rl_state_features.npy", np.array(["missing_alpha"]))

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


def test_conditional_anchors_cannot_override_feature_blacklist(tmp_path, fake_catboost):
    _write_long_split_contract(
        tmp_path,
        "train",
        "fu2601",
        [float(index) for index in range(15)],
        [float(14 - index) for index in range(15)],
        extra_features={
            "log_price_slope_96": [float(i) for i in range(15)],
        },
    )
    _write_long_split_contract(
        tmp_path,
        "train",
        "fu2605",
        [float(index + 1) for index in range(15)],
        [float(15 - index) for index in range(15)],
        extra_features={
            "log_price_slope_96": [float(i + 1) for i in range(15)],
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
        min_abs_ic=0.001,
        max_correlation=1.0,
        composite_drop_ratio=0.0,
        feature_blacklist=["log_price_slope_96"],
        feature_ablation_patterns=[],
        rank_ic_mode="absolute",
        enable_conditional_anchors=True,
    )

    stage_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/train"
    assert not (stage_dir / "state_features.npy").exists()
    selected_features = np.load(stage_dir / "rl_state_features.npy", allow_pickle=True).tolist()
    assert "log_price_slope_96" not in selected_features


def test_contract_normalized_correlation_avoids_simpsons_paradox():
    from operator_futures.feature_selection.cor_util import compute_contract_normalized_correlation_matrix

    # Two contracts with vastly different baseline price levels (2,000 vs 4,000)
    # Inside each contract, feat_x and feat_y are orthogonal / uncorrelated (r = 0.0)
    # Naive vertical concatenation creates spurious correlation r > 0.999
    c1 = pl.DataFrame({
        "feat_x": [1999.0, 2001.0, 1999.0, 2001.0],
        "feat_y": [1999.0, 1999.0, 2001.0, 2001.0],
    })
    c2 = pl.DataFrame({
        "feat_x": [3999.0, 4001.0, 3999.0, 4001.0],
        "feat_y": [3999.0, 3999.0, 4001.0, 4001.0],
    })
    frames = {"fu2601": c1, "fu2605": c2}

    # Naive concatenation correlation
    naive_concat = pl.concat([c1, c2], how="vertical")
    naive_corr = naive_concat.select(pl.corr("feat_x", "feat_y")).item()
    assert naive_corr > 0.99  # Spurious Simpson's paradox correlation

    # Decentralized contract-normalized correlation matrix
    decentralized_df = compute_contract_normalized_correlation_matrix(frames, ["feat_x", "feat_y"])
    feat_y_corr = decentralized_df.filter(pl.col("feature") == "feat_x")["feat_y"].item()
    assert abs(feat_y_corr) < 0.01  # Truly uncorrelated

    # Under correlation filtering at max_correlation=0.70, both features survive
    from operator_futures.feature_selection.cor_util import select_feature
    surviving = select_feature(
        features=["feat_x", "feat_y"],
        corre_df=decentralized_df,
        theshold=0.70,
    )
    assert set(surviving) == {"feat_x", "feat_y"}


def test_stream_filter_profile_immutability_and_defaults():
    from operator_futures.feature_selection.muti_contract.types import (
        StreamFilterProfile,
        DEFAULT_VAE_PROFILE,
        DEFAULT_RL_PROFILE,
        FeatureSelectionPipelineConfig,
    )
    from pathlib import Path
    import pytest

    # Verify VAE Profile canonical values
    assert DEFAULT_VAE_PROFILE.name == "vae_regime"
    assert DEFAULT_VAE_PROFILE.max_mean_psi == 0.10
    assert DEFAULT_VAE_PROFILE.max_pair_psi == 0.20
    assert DEFAULT_VAE_PROFILE.min_abs_ic == 0.015
    assert DEFAULT_VAE_PROFILE.min_sign_consistency == 0.70
    assert DEFAULT_VAE_PROFILE.min_rank_ic_ir == 0.35
    assert DEFAULT_VAE_PROFILE.max_correlation == 0.65
    assert DEFAULT_VAE_PROFILE.min_clusters == 12
    assert DEFAULT_VAE_PROFILE.max_clusters == 18
    assert DEFAULT_VAE_PROFILE.psi_weight == 0.50
    assert DEFAULT_VAE_PROFILE.rank_ic_weight == 0.30
    assert DEFAULT_VAE_PROFILE.catboost_weight == 0.20
    assert DEFAULT_VAE_PROFILE.filter_micro_persistence is True
    assert DEFAULT_VAE_PROFILE.mandatory_feature_pattern == r"^(base_time_|time_|trading_minute_)"

    # Verify RL Profile canonical values
    assert DEFAULT_RL_PROFILE.name == "rl_decision"
    assert DEFAULT_RL_PROFILE.max_mean_psi == 0.25
    assert DEFAULT_RL_PROFILE.max_pair_psi == 0.35
    assert DEFAULT_RL_PROFILE.min_abs_ic == 0.020
    assert DEFAULT_RL_PROFILE.min_sign_consistency == 0.65
    assert DEFAULT_RL_PROFILE.min_rank_ic_ir == 0.30
    assert DEFAULT_RL_PROFILE.max_correlation == 0.80
    assert DEFAULT_RL_PROFILE.min_clusters == 50
    assert DEFAULT_RL_PROFILE.max_clusters == 65
    assert DEFAULT_RL_PROFILE.psi_weight == 0.15
    assert DEFAULT_RL_PROFILE.rank_ic_weight == 0.50
    assert DEFAULT_RL_PROFILE.catboost_weight == 0.35
    assert DEFAULT_RL_PROFILE.filter_micro_persistence is False
    assert DEFAULT_RL_PROFILE.mandatory_feature_pattern is None

    # Verify frozen immutability
    with pytest.raises(Exception):
        DEFAULT_VAE_PROFILE.min_clusters = 5  # type: ignore

    # Verify pipeline config defaults
    config = FeatureSelectionPipelineConfig(
        root_path=Path("/tmp"),
        symbol="fu",
        target_freq="5min",
        stage="train",
    )
    assert config.dual_stream is True
    assert config.vae_profile == DEFAULT_VAE_PROFILE
    assert config.rl_profile == DEFAULT_RL_PROFILE


def test_feature_selection_manifest_dual_stream_serialization(tmp_path: Path):
    from operator_futures.feature_selection.manifests import (
        FeatureSelectionManifest,
        StreamAuditRecord,
    )

    vae_record = StreamAuditRecord(
        profile_name="vae_regime",
        selected_features=["base_time_day_progress", "time_hour_sin"],
        selected_feature_count=2,
        filter_results={"psi_drop": ["feat_unstable"]},
        candidate_count=10,
        dropped_counts={"psi_drop": 1},
    )
    rl_record = StreamAuditRecord(
        profile_name="rl_decision",
        selected_features=["base_time_day_progress", "level5_ofi_weighted_norm"],
        selected_feature_count=2,
        filter_results={},
        candidate_count=10,
        dropped_counts={},
    )
    manifest = FeatureSelectionManifest(
        symbol="fu",
        target_freq="5min",
        stage="train",
        split_input_dir=str(tmp_path),
        windows_list=[6],
        aggregate_metrics_path=str(tmp_path / "metrics.csv"),
        stream_mode="dual",
        selected_features=["base_time_day_progress", "time_hour_sin", "level5_ofi_weighted_norm"],
        selected_feature_count=3,
        vae_stream=vae_record,
        rl_stream=rl_record,
    )

    manifest_path = tmp_path / "feature_selection_manifest.json"
    manifest.write_json(manifest_path)

    # Read back and verify deserialization roundtrip
    loaded = FeatureSelectionManifest.read_json(manifest_path)
    assert loaded.stream_mode == "dual"
    assert loaded.selected_feature_count == 3
    assert set(loaded.selected_features or []) == {"base_time_day_progress", "time_hour_sin", "level5_ofi_weighted_norm"}
    assert loaded.vae_stream is not None
    assert loaded.vae_stream.profile_name == "vae_regime"
    assert loaded.vae_stream.selected_feature_count == 2
    assert loaded.vae_stream.filter_results == {"psi_drop": ["feat_unstable"]}
    assert loaded.vae_stream.candidate_count == 10
    assert loaded.vae_stream.dropped_counts == {"psi_drop": 1}
    assert loaded.rl_stream is not None
    assert loaded.rl_stream.profile_name == "rl_decision"
    assert loaded.rl_stream.selected_feature_count == 2


def test_dual_stream_train_stage_writes_dual_artifacts_and_union(tmp_path, fake_catboost):
    from operator_futures.feature_selection.muti_contract.types import (
        StreamFilterProfile,
        DEFAULT_VAE_PROFILE,
        DEFAULT_RL_PROFILE,
    )

    row_count = 30
    rng = np.random.RandomState(42)

    # 15 time topology features + 5 mandatory contract features
    # 55 candidate features = 70 candidate features total
    time_features = {
        f"base_time_{i}": rng.normal(0, 1, row_count).tolist() for i in range(8)
    }
    time_features.update({
        f"time_{i}": rng.normal(0, 1, row_count).tolist() for i in range(4)
    })
    time_features.update({
        f"trading_minute_{i}": rng.normal(0, 1, row_count).tolist() for i in range(3)
    })

    mandatory_contract_features = {
        "cm_main_sub_spread": rng.normal(0, 1, row_count).tolist(),
        "cm_volume_ratio": rng.normal(0, 1, row_count).tolist(),
        "contract_month_sin": [0.5] * row_count,
        "contract_month_cos": [0.5] * row_count,
        "contract_life_remaining_ratio": [0.8] * row_count,
    }
    all_mandatory = list(time_features.keys()) + list(mandatory_contract_features.keys())

    # Candidate alpha features
    alpha_features = {}
    for i in range(50):
        # Independent features so they don't collapse during correlation clustering
        alpha_features[f"alpha_feat_{i}"] = (rng.normal(0, 1, row_count) + float(i)*0.01).tolist()
    alpha_features["level5_ofi_weighted_norm"] = rng.normal(0, 1, row_count).tolist()
    alpha_features["log_return_1"] = rng.normal(0, 1, row_count).tolist()
    alpha_features["log_return_2"] = rng.normal(0, 1, row_count).tolist()
    alpha_features["log_return_6"] = rng.normal(0, 1, row_count).tolist()

    all_extra = {}
    all_extra.update(time_features)
    all_extra.update(mandatory_contract_features)
    all_extra.update(alpha_features)

    _write_long_split_contract(
        tmp_path,
        "train",
        "fu2601",
        [float(i) for i in range(row_count)],
        [float(row_count - i) for i in range(row_count)],
        extra_features=all_extra,
    )
    _write_long_split_contract(
        tmp_path,
        "train",
        "fu2605",
        [float(i + 1) for i in range(row_count)],
        [float(row_count - i + 1) for i in range(row_count)],
        extra_features=all_extra,
    )

    test_vae_profile = StreamFilterProfile(
        name="vae_regime",
        max_mean_psi=0.10,
        max_pair_psi=0.20,
        min_abs_ic=0.0,
        min_sign_consistency=0.0,
        min_rank_ic_ir=0.0,
        max_correlation=0.65,
        min_clusters=5,
        max_clusters=18,
        psi_weight=0.50,
        rank_ic_weight=0.30,
        catboost_weight=0.20,
        filter_micro_persistence=True,
        mandatory_feature_pattern=r"^(base_time_|time_|trading_minute_)",
    )
    test_rl_profile = StreamFilterProfile(
        name="rl_decision",
        max_mean_psi=0.25,
        max_pair_psi=0.35,
        min_abs_ic=0.0,
        min_sign_consistency=0.0,
        min_rank_ic_ir=0.0,
        max_correlation=0.80,
        min_clusters=15,
        max_clusters=65,
        psi_weight=0.15,
        rank_ic_weight=0.50,
        catboost_weight=0.35,
        filter_micro_persistence=False,
        mandatory_feature_pattern=None,
    )

    res = run_feature_selection(
        root_path=tmp_path,
        split_path="PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST",
        save_path="PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION",
        symbol="fu",
        target_freq="5min",
        stage="train",
        orderbook_depth=5,
        min_abs_ic=0.0,
        max_correlation=0.99,
        composite_drop_ratio=0.0,
        mandatory_state_features=all_mandatory,
        dual_stream=True,
        vae_profile=test_vae_profile,
        rl_profile=test_rl_profile,
    )

    stage_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/train"
    vae_file = stage_dir / "vae_state_features.npy"
    rl_file = stage_dir / "rl_state_features.npy"
    manifest_file = stage_dir / "feature_selection_manifest.json"

    assert vae_file.exists()
    assert rl_file.exists()
    assert not (stage_dir / "state_features.npy").exists()
    assert manifest_file.exists()

    vae_feats = np.load(vae_file, allow_pickle=True).tolist()
    rl_feats = np.load(rl_file, allow_pickle=True).tolist()
    union_feats = list(dict.fromkeys(rl_feats + vae_feats))

    # Verify set union property
    assert set(union_feats) == set(vae_feats).union(set(rl_feats))

    # Verify manifest audit payload
    manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest_data["stream_mode"] == "dual"
    assert manifest_data["selected_feature_count"] == len(union_feats)
    assert manifest_data["selected_features"] == union_feats
    assert "vae_stream" in manifest_data
    assert "rl_stream" in manifest_data
    assert manifest_data["vae_stream"]["profile_name"] == "vae_regime"
    assert manifest_data["rl_stream"]["profile_name"] == "rl_decision"
    assert manifest_data["vae_stream"]["selected_feature_count"] == len(vae_feats)
    assert manifest_data["rl_stream"]["selected_feature_count"] == len(rl_feats)

    # Verify filtered contract feather outputs contain reward and union features
    df_fu2601 = pl.read_ipc(stage_dir / "fu2601" / "df.feather")
    for uf in union_feats:
        assert uf in df_fu2601.columns
    assert "symbol" in df_fu2601.columns


def test_dual_stream_micro_persistence_and_mandatory_isolation(tmp_path, fake_catboost):
    from operator_futures.feature_selection.muti_contract.types import (
        StreamFilterProfile,
        DEFAULT_VAE_PROFILE,
        DEFAULT_RL_PROFILE,
    )

    row_count = 24
    fast_decay = [1.0 if index % 2 == 0 else -1.0 for index in range(row_count)]
    slow_signal = [float(index) for index in range(row_count)]

    # Time topology mandatory vs contract-role mandatory
    mandatory_features = [
        "base_time_day_progress",
        "time_hour_sin",
        "contract_month_sin",
        "cm_volume_ratio",
    ]

    rng = np.random.RandomState(123)
    extra_features = {
        "base_time_day_progress": slow_signal,
        "time_hour_sin": slow_signal,
        "contract_month_sin": [0.5] * row_count,
        "cm_volume_ratio": slow_signal,
        "wap_1_log_return_2": fast_decay,
        "level5_ofi_weighted_norm": (rng.normal(0, 1, row_count)).tolist(),
        "stationary_alpha": (rng.normal(0, 1, row_count)).tolist(),
    }

    _write_long_split_contract(
        tmp_path, "train", "fu2601", slow_signal, slow_signal, extra_features=extra_features
    )
    _write_long_split_contract(
        tmp_path, "train", "fu2605", [v + 1 for v in slow_signal], slow_signal, extra_features=extra_features
    )

    # Use smaller min_clusters for this unit test
    custom_vae_profile = StreamFilterProfile(
        name="vae_regime",
        max_mean_psi=0.10,
        max_pair_psi=0.20,
        min_abs_ic=0.0,
        min_sign_consistency=0.0,
        min_rank_ic_ir=0.0,
        max_correlation=0.65,
        min_clusters=1,
        max_clusters=10,
        psi_weight=0.50,
        rank_ic_weight=0.30,
        catboost_weight=0.20,
        filter_micro_persistence=True,
        mandatory_feature_pattern=r"^(base_time_|time_|trading_minute_)",
    )
    custom_rl_profile = StreamFilterProfile(
        name="rl_decision",
        max_mean_psi=0.25,
        max_pair_psi=0.35,
        min_abs_ic=0.0,
        min_sign_consistency=0.0,
        min_rank_ic_ir=0.0,
        max_correlation=0.80,
        min_clusters=1,
        max_clusters=20,
        psi_weight=0.15,
        rank_ic_weight=0.50,
        catboost_weight=0.35,
        filter_micro_persistence=False,
        mandatory_feature_pattern=None,
    )

    res = run_feature_selection(
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
        mandatory_state_features=mandatory_features,
        dual_stream=True,
        vae_profile=custom_vae_profile,
        rl_profile=custom_rl_profile,
    )

    stage_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/train"
    assert not (stage_dir / "state_features.npy").exists()
    vae_feats = np.load(stage_dir / "vae_state_features.npy", allow_pickle=True).tolist()
    rl_feats = np.load(stage_dir / "rl_state_features.npy", allow_pickle=True).tolist()
    union_feats = list(dict.fromkeys(rl_feats + vae_feats))

    # VAE stream isolation:
    # 1. wap_1_log_return_2 is dropped by persistence filter
    assert "wap_1_log_return_2" not in vae_feats
    # 2. contract_month_sin, cm_volume_ratio are stripped (only time topology retained)
    assert "contract_month_sin" not in vae_feats
    assert "cm_volume_ratio" not in vae_feats
    assert "base_time_day_progress" in vae_feats
    assert "time_hour_sin" in vae_feats

    # RL stream preservation:
    # 1. wap_1_log_return_2 and level5_ofi_weighted_norm are unblocked and retained
    assert "wap_1_log_return_2" in rl_feats
    assert "level5_ofi_weighted_norm" in rl_feats
    # 2. All mandatory features preserved
    for mf in mandatory_features:
        assert mf in rl_feats

    # Union features contains everything from both streams
    assert "wap_1_log_return_2" in union_feats
    assert "level5_ofi_weighted_norm" in union_feats
    assert "contract_month_sin" in union_feats
    assert set(union_feats) == set(vae_feats).union(set(rl_feats))


def test_dual_stream_fail_fast_on_insufficient_clusters(tmp_path, fake_catboost):
    from operator_futures.feature_selection.muti_contract.types import StreamFilterProfile

    slow_signal = [float(i) for i in range(15)]
    _write_long_split_contract(
        tmp_path, "train", "fu2601", slow_signal, slow_signal,
        extra_features={"feat_1": slow_signal, "feat_2": slow_signal}
    )
    _write_long_split_contract(
        tmp_path, "train", "fu2605", [v + 1 for v in slow_signal], slow_signal,
        extra_features={"feat_1": slow_signal, "feat_2": slow_signal}
    )

    # Require min_clusters=10 when only 2 features are provided
    strict_profile = StreamFilterProfile(
        name="strict_test",
        max_mean_psi=0.10,
        max_pair_psi=0.20,
        min_abs_ic=0.0,
        min_sign_consistency=0.0,
        min_rank_ic_ir=0.0,
        max_correlation=0.65,
        min_clusters=10,
        max_clusters=20,
        psi_weight=0.50,
        rank_ic_weight=0.30,
        catboost_weight=0.20,
        filter_micro_persistence=False,
    )

    with pytest.raises(ValueError, match="configured minimum cluster count"):
        run_feature_selection(
            root_path=tmp_path,
            split_path="PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST",
            save_path="PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION",
            symbol="fu",
            target_freq="5min",
            stage="train",
            orderbook_depth=5,
            dual_stream=True,
            vae_profile=strict_profile,
            rl_profile=strict_profile,
        )


def test_end_to_end_pipeline_dual_stream_selection_scale_and_vae_data_creation(tmp_path, fake_catboost):
    from operator_futures.feature_selection.muti_contract.types import (
        StreamFilterProfile,
        DEFAULT_VAE_PROFILE,
        DEFAULT_RL_PROFILE,
    )
    from operator_futures.scale_describe_save.muti_contract_scale_save import (
        main as scale_save_main,
        parser as scale_save_parser,
    )
    import sys
    fineft_root = Path(__file__).resolve().parents[2] / "FineFT"
    if str(fineft_root) not in sys.path:
        sys.path.insert(0, str(fineft_root))
    from datahandler.commodity_contract_dataset import write_stage_datasets
    from datahandler.manifests import DatasetManifest, DatasetSetManifest, DatasetContractManifest
    from datahandler.vae_data_creation import make_data as vae_make_data
    from types import SimpleNamespace
    import pandas as pd

    row_count = 35
    rng = np.random.RandomState(42)

    # 15 time features + 5 mandatory contract features
    time_features = {
        f"base_time_{i}": rng.normal(0, 1, row_count).tolist() for i in range(8)
    }
    time_features.update({
        f"time_{i}": rng.normal(0, 1, row_count).tolist() for i in range(4)
    })
    time_features.update({
        f"trading_minute_{i}": rng.normal(0, 1, row_count).tolist() for i in range(3)
    })
    mandatory_contract_features = {
        "cm_main_sub_spread": rng.normal(0, 1, row_count).tolist(),
        "cm_volume_ratio": rng.normal(0, 1, row_count).tolist(),
        "contract_month_sin": [0.5] * row_count,
        "contract_month_cos": [0.5] * row_count,
        "contract_life_remaining_ratio": [0.8] * row_count,
    }
    all_mandatory = list(time_features.keys()) + list(mandatory_contract_features.keys())

    alpha_features = {}
    for i in range(50):
        alpha_features[f"alpha_feat_{i}"] = (rng.normal(0, 1, row_count) + float(i) * 0.01).tolist()
    alpha_features["level5_ofi_weighted_norm"] = rng.normal(0, 1, row_count).tolist()
    alpha_features["log_return_1"] = rng.normal(0, 1, row_count).tolist()
    alpha_features["log_return_2"] = rng.normal(0, 1, row_count).tolist()
    alpha_features["log_return_6"] = rng.normal(0, 1, row_count).tolist()

    all_extra = {}
    all_extra.update(time_features)
    all_extra.update(mandatory_contract_features)
    all_extra.update(alpha_features)

    # 1. Populate train, valid, test contracts
    _write_long_split_contract(
        tmp_path, "train", "fu2601",
        [float(i) for i in range(row_count)], [float(row_count - i) for i in range(row_count)],
        extra_features=all_extra,
    )
    _write_long_split_contract(
        tmp_path, "train", "fu2605",
        [float(i + 1) for i in range(row_count)], [float(row_count - i + 1) for i in range(row_count)],
        extra_features=all_extra,
    )
    _write_long_split_contract(
        tmp_path, "valid", "fu2609",
        [float(i + 2) for i in range(row_count)], [float(row_count - i + 2) for i in range(row_count)],
        extra_features=all_extra,
    )
    _write_long_split_contract(
        tmp_path, "test", "fu2611",
        [float(i + 3) for i in range(row_count)], [float(row_count - i + 3) for i in range(row_count)],
        extra_features=all_extra,
    )

    test_vae_profile = StreamFilterProfile(
        name="vae_regime",
        max_mean_psi=0.10,
        max_pair_psi=0.20,
        min_abs_ic=0.0,
        min_sign_consistency=0.0,
        min_rank_ic_ir=0.0,
        max_correlation=0.65,
        min_clusters=12,
        max_clusters=18,
        psi_weight=0.50,
        rank_ic_weight=0.30,
        catboost_weight=0.20,
        filter_micro_persistence=True,
        mandatory_feature_pattern=r"^(base_time_|time_|trading_minute_)",
    )
    test_rl_profile = StreamFilterProfile(
        name="rl_decision",
        max_mean_psi=0.25,
        max_pair_psi=0.35,
        min_abs_ic=0.0,
        min_sign_consistency=0.0,
        min_rank_ic_ir=0.0,
        max_correlation=0.80,
        min_clusters=50,
        max_clusters=65,
        psi_weight=0.15,
        rank_ic_weight=0.50,
        catboost_weight=0.35,
        filter_micro_persistence=False,
        mandatory_feature_pattern=None,
    )

    # 2. Stage 1 & 2: Run Dual-Stream Feature Selection
    res = run_feature_selection(
        root_path=tmp_path,
        split_path="PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST",
        save_path="PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION",
        symbol="fu",
        target_freq="5min",
        stage="train",
        orderbook_depth=5,
        min_abs_ic=0.0,
        max_correlation=0.99,
        composite_drop_ratio=0.0,
        mandatory_state_features=all_mandatory,
        dual_stream=True,
        vae_profile=test_vae_profile,
        rl_profile=test_rl_profile,
    )

    stage_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/train"
    vae_file = stage_dir / "vae_state_features.npy"
    rl_file = stage_dir / "rl_state_features.npy"
    manifest_file = stage_dir / "feature_selection_manifest.json"

    assert vae_file.exists()
    assert rl_file.exists()
    assert not (stage_dir / "state_features.npy").exists()

    vae_feats = np.load(vae_file, allow_pickle=True).tolist()
    rl_feats = np.load(rl_file, allow_pickle=True).tolist()
    union_feats = list(dict.fromkeys(rl_feats + vae_feats))

    # Manifest audit: VAE has 12-18 features, RL has 50-65 features
    assert 12 <= len(vae_feats) <= 18
    assert 50 <= len(rl_feats) <= 65
    assert set(union_feats) == set(vae_feats).union(set(rl_feats))

    # 3. Stage 3: Unified Scale-Save on Union State Features
    scale_save_args = scale_save_parser.parse_args([
        "--root_path", str(tmp_path),
        "--symbols", "fu",
        "--target_freq", "5min",
        "--feature_selection_dir", str(stage_dir),
        "--save_path", "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE",
        "--scale_method", "rolling_zscore",
        "--rolling_window", "24",
        "--clip_mode", "tanh",
        "--soft_clip_m", "4.0",
        "--clip_min", "-5.0",
        "--clip_max", "5.0",
        "--passthrough_features", "base_time_0",
    ])
    scale_save_main(scale_save_args)

    scale_root = tmp_path / "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/fu/5min"
    assert not (scale_root / "state_features.npy").exists()
    assert (scale_root / "rl_state_features.npy").exists()
    assert (scale_root / "vae_state_features.npy").exists()

    # 4. Step 2 Dataset Packaging: write_stage_datasets
    dataset_dest = tmp_path / "dataset" / "5min" / "fu"
    dataset_manifest = DatasetManifest(
        symbol="fu",
        target_freq="5min",
        dataset_split_manifest_path=str(tmp_path / "dataset_split_manifest.json"),
        state_features_source_path=str(scale_root / "rl_state_features.npy"),
        state_features_path=str(dataset_dest / "rl_state_features.npy"),
        sets={
            "train": DatasetSetManifest(
                range=None,
                contracts=[
                    DatasetContractManifest(
                        contract="fu2601",
                        input_path=str(scale_root / "train/fu2601.feather"),
                        output_path=str(dataset_dest / "train/fu2601.feather"),
                    )
                ],
                skipped_contracts=[],
            ),
            "test": DatasetSetManifest(
                range=None,
                contracts=[
                    DatasetContractManifest(
                        contract="fu2611",
                        input_path=str(scale_root / "test/fu2611.feather"),
                        output_path=str(dataset_dest / "test/df_fu2611.feather"),
                    )
                ],
                skipped_contracts=[],
            ),
        },
    )
    write_stage_datasets(dataset_manifest)

    assert not (dataset_dest / "state_features.npy").exists()
    assert (dataset_dest / "vae_state_features.npy").exists()
    assert (dataset_dest / "rl_state_features.npy").exists()

    # Create dummy dynamic slice feather under train/slope/fu2601/label_0/df_0.feather
    train_contract_df = pd.read_feather(dataset_dest / "train/fu2601.feather")
    slice_dir = dataset_dest / "train" / "slope" / "fu2601" / "label_0"
    slice_dir.mkdir(parents=True, exist_ok=True)
    train_contract_df.to_feather(slice_dir / "df_0.feather")

    # 5. Downstream VAE Ingestion: make_data
    vae_args = SimpleNamespace(
        base_path=str(tmp_path / "dataset" / "5min"),
        dataset_name="fu",
        save_path=str(tmp_path / "dataset" / "5min"),
        source_split="train",
        labeling_method="slope",
    )
    vae_make_data(vae_args)

    vae_data_dir = dataset_dest / "VAE_data"
    label_arr = np.load(vae_data_dir / "slope" / "fu2601" / "label_0.npy")
    test_arr = np.load(vae_data_dir / "test" / "test_fu2611.npy")

    # Both VAE arrays must match exact len(vae_feats) (12~18)
    assert label_arr.shape == (row_count, len(vae_feats))
    assert test_arr.shape == (row_count, len(vae_feats))
    assert not np.isnan(label_arr).any()
    assert not np.isnan(test_arr).any()

    # 6. Downstream Low-Level RL State Slicing
    rl_state_obs = train_contract_df[rl_feats].values
    assert rl_state_obs.shape == (row_count, len(rl_feats))
    assert not np.isnan(rl_state_obs).any()
