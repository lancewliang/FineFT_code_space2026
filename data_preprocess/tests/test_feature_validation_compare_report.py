from pathlib import Path
import json

import numpy as np
import pandas as pd

from operator_futures.feature_validation.compare import compare_frames
from operator_futures.feature_validation.expected_columns import EXPECTED_COLUMNS_BY_DOC
from operator_futures.feature_validation.models import StageResult, ValidationConfig, ValidationReport
from operator_futures.feature_validation.report import render_json_report, render_markdown_report


def test_expected_columns_are_fixed_docs_derived_lists():
    assert len(EXPECTED_COLUMNS_BY_DOC["base_feature"]) == 112
    assert len(EXPECTED_COLUMNS_BY_DOC["kline_feature"]) == 216
    assert len(EXPECTED_COLUMNS_BY_DOC["quotes_feature"]) == 69
    assert len(EXPECTED_COLUMNS_BY_DOC["snapshot_feature"]) == 82
    assert len(EXPECTED_COLUMNS_BY_DOC["reward_environment"]) == 106
    assert len(EXPECTED_COLUMNS_BY_DOC["time_feature"]) == 3375
    assert "ohlcv_feature_1" not in EXPECTED_COLUMNS_BY_DOC["time_feature"]
    assert "ohlc_feature_1" not in EXPECTED_COLUMNS_BY_DOC["time_feature"]
    assert "log_volume_origin" in EXPECTED_COLUMNS_BY_DOC["time_feature"]
    assert "vma_48_std_norm_sell" in EXPECTED_COLUMNS_BY_DOC["time_feature"]
    assert "roc_2_spread" in EXPECTED_COLUMNS_BY_DOC["time_feature"]
    assert "sumd_48_asksize" in EXPECTED_COLUMNS_BY_DOC["time_feature"]


def test_compare_frames_aligns_by_timestamp_and_applies_absolute_tolerance():
    actual = pd.DataFrame(
        {
            "timestamp": [3, 1, 2],
            "price": [30.0, 10.0, 20.0 + 5e-10],
            "volume": [300.0, 100.0, 200.25],
            "actual_only": [1, 1, 1],
        }
    )
    expected = pd.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "price": [10.0, 20.0, 30.0],
            "volume": [100.0, 200.0, 300.0],
            "expected_only": [2, 2, 2],
        }
    )

    result = compare_frames(
        stage="unit",
        actual=actual,
        expected=expected,
        expected_columns=["price", "volume", "missing_column"],
        tolerance=1e-9,
        sample_size=3,
    )

    assert result.status == "fail"
    assert result.checked_columns == 2
    assert result.missing_columns == ["missing_column"]
    assert result.extra_columns == ["actual_only"]
    assert result.max_abs_diff == 0.25
    assert len(result.sample_failures) == 1
    failure = result.sample_failures[0]
    assert failure.column == "volume"
    assert failure.timestamp == "2"
    assert failure.abs_diff == 0.25


def test_compare_frames_handles_duplicate_column_names_without_error():
    actual = pd.DataFrame([[1, 10.0, 99.0]], columns=["timestamp", "price", "price"])
    expected = pd.DataFrame([[1, 10.0]], columns=["timestamp", "price"])

    result = compare_frames(
        stage="unit",
        actual=actual,
        expected=expected,
        expected_columns=["price"],
        tolerance=1e-9,
        sample_size=1,
    )

    assert result.status == "pass"
    assert result.checked_columns == 1


def test_compare_frames_records_scalar_failure_values_for_duplicate_columns():
    actual = pd.DataFrame([[1, 11.0, 99.0]], columns=["timestamp", "price", "price"])
    expected = pd.DataFrame([[1, 10.0]], columns=["timestamp", "price"])

    result = compare_frames(
        stage="unit",
        actual=actual,
        expected=expected,
        expected_columns=["price"],
        tolerance=1e-9,
        sample_size=1,
    )

    assert result.status == "fail"
    assert result.sample_failures[0].actual == 11.0
    assert render_json_report(
        ValidationReport(
            config=ValidationConfig(
                root_path=Path("."),
                symbol="fu",
                target_freq="5min",
                start_date="2025-11-03",
                end_date="2025-11-08",
                report_dir=Path("."),
            ),
            stages=[result],
        )
    )


def test_compare_frames_records_json_serializable_numpy_scalars():
    actual = pd.DataFrame({"timestamp": [1], "price": np.array([11], dtype=np.int64)})
    expected = pd.DataFrame({"timestamp": [1], "price": np.array([10], dtype=np.int64)})

    result = compare_frames(
        stage="unit",
        actual=actual,
        expected=expected,
        expected_columns=["price"],
        tolerance=1e-9,
        sample_size=1,
    )
    payload = render_json_report(
        ValidationReport(
            config=ValidationConfig(
                root_path=Path("."),
                symbol="fu",
                target_freq="5min",
                start_date="2025-11-03",
                end_date="2025-11-08",
                report_dir=Path("."),
            ),
            stages=[result],
        )
    )

    assert payload["stages"][0]["sample_failures"][0]["actual"] == 11
    json.dumps(payload)


def test_reports_include_stage_status_counts_and_failures(tmp_path):
    config = ValidationConfig(
        root_path=tmp_path,
        symbol="fu",
        target_freq="5min",
        start_date="2025-11-03",
        end_date="2025-11-08",
        report_dir=tmp_path,
    )
    report = ValidationReport(
        config=config,
        stages=[
            StageResult(
                stage="base_feature",
                status="fail",
                checked_columns=1,
                missing_columns=["volume"],
                extra_columns=["actual_only"],
                mismatched_columns=["open"],
                max_abs_diff=2e-9,
            )
        ],
    )

    markdown = render_markdown_report(report)
    payload = render_json_report(report)

    assert "base_feature" in markdown
    assert "fail" in markdown
    assert "volume" in markdown
    assert payload["summary"]["status_counts"]["fail"] == 1
    assert payload["stages"][0]["max_abs_diff"] == 2e-9
