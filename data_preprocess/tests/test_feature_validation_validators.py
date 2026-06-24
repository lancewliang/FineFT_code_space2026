from pathlib import Path

import pandas as pd

from operator_futures.commodity.schema import get_reward_execution_columns
from operator_futures.feature_validation.models import ValidationConfig
from operator_futures.feature_validation.reference_adapters import (
    recompute_cross_section,
    recompute_ic_correlation,
    recompute_merge_concat,
    recompute_scale_save,
    recompute_time_feature,
)
from operator_futures.feature_validation.validators import (
    _compare_stage,
    validate_all_stages,
    validate_merge_concat,
    validate_scale_save,
)


def _write_feather(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_feather(path)


def test_validate_merge_concat_returns_partial_when_reference_is_not_wired(tmp_path):
    artifact = (
        tmp_path
        / "PREPROCESS_DATASET"
        / "commodity-futures"
        / "MERGE_CONCAT"
        / "CONCAT_FEATURE"
        / "fu"
        / "5min"
        / "2025-11-03-2025-11-08.feather"
    )
    _write_feather(
        artifact,
        [
            {"timestamp": "2025-11-03 09:00:00", "open": 100.0, "close": 100.5},
            {"timestamp": "2025-11-03 09:05:00", "open": 101.0, "close": 101.5},
        ],
    )
    config = ValidationConfig(
        root_path=tmp_path,
        symbol="fu",
        target_freq="5min",
        start_date="2025-11-03",
        end_date="2025-11-08",
        report_dir=tmp_path / "reports",
        sample_size=2,
    )

    result = validate_merge_concat(config)

    assert result.status in {"error", "partial", "fail"}
    assert result.message
    assert result.stage == "merge_concat"


def test_recompute_merge_concat_uses_commodity_merge_concat_intermediate_path(tmp_path):
    root = (
        tmp_path
        / "PREPROCESS_DATASET"
        / "commodity-futures"
        / "MERGE_CONCAT"
        / "MERGED_FEATURE"
        / "fu"
        / "5min"
    )
    for folder in ["CONCURRENT_FEATURE", "FUTURE_FEATURE"]:
        _write_feather(
            root / folder / "2025-11-03.feather",
            [
                {
                    "timestamp": "2025-11-03 09:00:00",
                    "ask1_price": 101.0,
                    "mark_price": 100.0,
                    "symbol": "fu2501",
                    "exchange": "SHFE",
                },
                {
                    "timestamp": "2025-11-03 09:05:00",
                    "ask1_price": 102.0,
                    "mark_price": 101.0,
                    "symbol": "fu2501",
                    "exchange": "SHFE",
                },
            ],
        )
    config = ValidationConfig(
        root_path=tmp_path,
        symbol="fu",
        target_freq="5min",
        start_date="2025-11-03",
        end_date="2025-11-04",
        report_dir=tmp_path / "reports",
    )

    result = recompute_merge_concat(config)

    assert result.shape[0] == 1
    assert result["timestamp"].tolist() == ["2025-11-03 09:05:00"]
    assert result.iloc[0, 1] == 102.0


def test_validate_scale_save_reports_missing_dependency_artifacts(tmp_path):
    artifact = (
        tmp_path
        / "PREPROCESS_DATASET"
        / "commodity-futures"
        / "SCALE_SAVE"
        / "fu"
        / "5min"
        / "2025-11-03-2025-11-08"
        / "df.feather"
    )
    _write_feather(artifact, [{"timestamp": "2025-11-03 09:00:00", "mark_price": 100.0}])

    config = ValidationConfig(
        root_path=tmp_path,
        symbol="fu",
        target_freq="5min",
        start_date="2025-11-03",
        end_date="2025-11-08",
        report_dir=tmp_path / "reports",
    )

    result = validate_scale_save(config)

    assert result.status == "error"
    assert "state_features.npy" in result.message


def test_compare_stage_uses_real_comparator():
    config = ValidationConfig(
        root_path=Path("."),
        symbol="fu",
        target_freq="5min",
        start_date="2025-11-03",
        end_date="2025-11-08",
        report_dir=Path("."),
        sample_size=1,
    )
    actual = pd.DataFrame([{"timestamp": "2025-11-03 09:00:00", "open": 101.0}])
    expected = pd.DataFrame([{"timestamp": "2025-11-03 09:00:00", "open": 100.0}])

    result = _compare_stage("merge_concat", actual, expected, config)

    assert result.status == "fail"
    assert result.checked_columns == 1
    assert result.mismatched_columns == ["open"]


def test_compare_stage_uses_reference_columns_for_variable_output_stages():
    config = ValidationConfig(
        root_path=Path("."),
        symbol="fu",
        target_freq="5min",
        start_date="2025-11-03",
        end_date="2025-11-08",
        report_dir=Path("."),
        sample_size=1,
    )
    actual = pd.DataFrame(
        [{"timestamp": "2025-11-03 09:00:00", "mark_price": 101.0}]
    )
    expected = pd.DataFrame(
        [{"timestamp": "2025-11-03 09:00:00", "mark_price": 100.0}]
    )

    result = _compare_stage("ic_correlation", actual, expected, config)

    assert result.status == "fail"
    assert result.checked_columns == 1
    assert result.mismatched_columns == ["mark_price"]


def test_recompute_time_feature_uses_reference_group_suffixes(tmp_path):
    artifact = (
        tmp_path
        / "PREPROCESS_DATASET"
        / "commodity-futures"
        / "MERGE_CONCAT"
        / "CONCAT_FEATURE"
        / "fu"
        / "5min"
        / "2025-11-03-2025-11-08.feather"
    )
    rows = []
    for index, timestamp in enumerate(
        pd.date_range("2025-11-03 09:00:00", periods=70, freq="5min")
    ):
        value = float(index + 100)
        rows.append(
            {
                "timestamp": timestamp,
                "open": value,
                "high": value + 2,
                "low": value - 2,
                "close": value + 1,
                "volume": value * 10,
                "bid1_price": value,
                "open_spread": value / 10,
                "high_spread": value / 10 + 2,
                "low_spread": value / 10 - 2,
                "close_spread": value / 10 + 1,
            }
        )
    _write_feather(artifact, rows)
    config = ValidationConfig(
        root_path=tmp_path,
        symbol="fu",
        target_freq="5min",
        start_date="2025-11-03",
        end_date="2025-11-08",
        report_dir=tmp_path / "reports",
    )

    result = recompute_time_feature(config, config.start_date, config.end_date)

    assert "bid1_price_log_return_2" in result.columns
    assert "log_volume_origin" in result.columns
    assert "roc_2_spread" in result.columns
    assert "roc_2" not in result.columns


def test_recompute_cross_section_accepts_five_level_snapshot(tmp_path):
    base_path = (
        tmp_path
        / "PREPROCESS_DATASET"
        / "commodity-futures"
        / "BASE_FEATURE"
        / "fu"
        / "5min"
        / "2025-11-03.feather"
    )
    snapshot_path = (
        tmp_path
        / "PREPROCESS_DATASET"
        / "commodity-futures"
        / "DOWNSCALE_ORDERBOOK_25"
        / "fu"
        / "5min"
        / "2025-11-03.feather"
    )
    _write_feather(
        base_path,
        [
            {
                "timestamp": "2025-11-03 09:00:00",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10.0,
                "tradeval": 1000.0,
                "vwap": 100.2,
                "awap": 100.1,
                "twap": 100.0,
                "ntrade_estimated": 10.0,
                "ntrade_up_estimated": 4.0,
                "ntrade_down_estimated": 3.0,
                "ntrade_flat_estimated": 3.0,
            }
        ],
    )
    snapshot_row = {"timestamp": "2025-11-03 09:00:00"}
    for level in range(1, 6):
        snapshot_row[f"ask{level}_price"] = 100.0 + level
        snapshot_row[f"ask{level}_size"] = float(level)
        snapshot_row[f"bid{level}_price"] = 100.0 - level
        snapshot_row[f"bid{level}_size"] = float(level + 1)
    _write_feather(snapshot_path, [snapshot_row])
    config = ValidationConfig(
        root_path=tmp_path,
        symbol="fu",
        target_freq="5min",
        start_date="2025-11-03",
        end_date="2025-11-08",
        report_dir=tmp_path / "reports",
    )

    result = recompute_cross_section(config, config.start_date)

    assert set(result) == {"kline", "quotes", "snapshot"}
    assert "ask5_size_n" in result["snapshot"].columns
    assert "ask6_size_n" not in result["snapshot"].columns


def test_ic_and_scale_reference_use_commodity_reward_schema(tmp_path):
    all_feature_path = (
        tmp_path
        / "PREPROCESS_DATASET"
        / "commodity-futures"
        / "ALL_FEATURE"
        / "fu"
        / "5min"
        / "2025-11-03-2025-11-08.feather"
    )
    rows = []
    for index, timestamp in enumerate(
        pd.date_range("2025-11-03 09:00:00", periods=20, freq="5min")
    ):
        row = {"timestamp": timestamp, "mark_price": float(100 + index)}
        for level in range(1, 6):
            row[f"ask{level}_price"] = float(100 + level + index)
            row[f"ask{level}_size"] = float(level)
            row[f"bid{level}_price"] = float(100 - level + index)
            row[f"bid{level}_size"] = float(level + 1)
        row["state_a"] = float(index)
        row["state_b"] = float(index * 2)
        rows.append(row)
    _write_feather(all_feature_path, rows)
    ic_dir = (
        tmp_path
        / "PREPROCESS_DATASET"
        / "commodity-futures"
        / "IC_RESULT"
        / "fu"
        / "5min"
        / "2025-11-03-2025-11-08"
    )
    _write_feather(ic_dir / "df.feather", rows)
    import numpy as np

    np.save(ic_dir / "state_features.npy", np.array(["state_a", "state_b"]))
    config = ValidationConfig(
        root_path=tmp_path,
        symbol="fu",
        target_freq="5min",
        start_date="2025-11-03",
        end_date="2025-11-08",
        report_dir=tmp_path / "reports",
    )

    ic_result = recompute_ic_correlation(config, config.start_date, config.end_date)
    scale_result = recompute_scale_save(config, config.start_date, config.end_date)

    assert "state_a" in scale_result.columns
    assert "state_b" in scale_result.columns
    assert "symbol" in scale_result.columns
    assert scale_result["symbol"].unique().tolist() == ["fu"]
    reward_columns = [
        column
        for column in get_reward_execution_columns(5)
        if column in pd.DataFrame(rows).columns
    ]
    assert ic_result.columns.tolist()[: len(reward_columns)] == reward_columns
    assert scale_result.columns.tolist()[: len(reward_columns)] == reward_columns


def test_validate_all_stages_includes_partial_and_error_states(tmp_path):
    results = validate_all_stages(
        ValidationConfig(
            root_path=tmp_path,
            symbol="fu",
            target_freq="5min",
            start_date="2025-11-03",
            end_date="2025-11-08",
            report_dir=tmp_path / "reports",
        )
    )

    assert {result.stage for result in results} == {
        "cross_section:kline",
        "cross_section:quotes",
        "cross_section:snapshot",
        "merge_concat",
        "time_feature",
        "merge_clean",
        "ic_correlation",
        "scale_save",
    }
    assert any(result.status in {"partial", "error"} for result in results)
