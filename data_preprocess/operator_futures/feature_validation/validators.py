from __future__ import annotations

from pathlib import Path

import pandas as pd

from operator_futures.feature_validation.compare import compare_frames
from operator_futures.feature_validation.expected_columns import EXPECTED_COLUMNS_BY_STAGE
from operator_futures.feature_validation.io import read_feather_frame
from operator_futures.feature_validation.reference_adapters import (
    recompute_cross_section,
    recompute_ic_correlation,
    recompute_merge_clean,
    recompute_merge_concat,
    recompute_scale_save,
    recompute_time_feature,
)
from operator_futures.feature_validation.models import StageResult, ValidationConfig


def _error_result(stage: str, exc: Exception) -> StageResult:
    return StageResult(stage=stage, status="error", message=f"{type(exc).__name__}: {exc}")


def _compare_stage(
    stage: str,
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    config: ValidationConfig,
) -> StageResult:
    expected_columns = [column for column in expected.columns if column != "timestamp"]
    return compare_frames(
        stage=stage,
        actual=actual,
        expected=expected,
        expected_columns=expected_columns,
        tolerance=config.tolerance,
        sample_size=config.sample_size,
    )


def validate_merge_concat(config: ValidationConfig) -> StageResult:
    stage = "merge_concat"
    try:
        actual = read_feather_frame(
            config.root_path
            / "PREPROCESS_DATASET"
            / "commodity-futures"
            / "MERGE_CONCAT"
            / "CONCAT_FEATURE"
            / config.symbol
            / config.target_freq
            / f"{config.start_date}-{config.end_date}.feather"
        )
        expected = recompute_merge_concat(config)
        return _compare_stage(stage, actual, expected, config)
    except Exception as exc:
        return _error_result(stage, exc)


def validate_time_feature(config: ValidationConfig) -> StageResult:
    stage = "time_feature"
    try:
        actual = read_feather_frame(
            config.root_path
            / "PREPROCESS_DATASET"
            / "commodity-futures"
            / "TIME_FEATURE"
            / config.symbol
            / config.target_freq
            / f"{config.start_date}-{config.end_date}.feather"
        )
        expected = recompute_time_feature(config, config.start_date, config.end_date)
        return _compare_stage(stage, actual, expected, config)
    except Exception as exc:
        return _error_result(stage, exc)


def validate_cross_section(config: ValidationConfig) -> list[StageResult]:
    expected = None
    try:
        expected = recompute_cross_section(config, config.start_date)
    except Exception as exc:
        return [
            _error_result("cross_section:kline", exc),
            _error_result("cross_section:quotes", exc),
            _error_result("cross_section:snapshot", exc),
        ]

    results = []
    for stage_name, folder in [
        ("cross_section:kline", "KLINE_FEATURE"),
        ("cross_section:quotes", "QUOTES_FEATURE"),
        ("cross_section:snapshot", "SNAPSHOT_FEATURE"),
    ]:
        try:
            actual = read_feather_frame(
                config.root_path
                / "PREPROCESS_DATASET"
                / "commodity-futures"
                / "CROSS_SECTION"
                / folder
                / config.symbol
                / config.target_freq
                / f"{config.start_date}.feather"
            )
            results.append(_compare_stage(stage_name, actual, expected[stage_name.split(":")[1]], config))
        except Exception as exc:
            results.append(_error_result(stage_name, exc))
    return results


def validate_merge_clean(config: ValidationConfig) -> StageResult:
    stage = "merge_clean"
    try:
        actual = read_feather_frame(
            config.root_path
            / "PREPROCESS_DATASET"
            / "commodity-futures"
            / "ALL_FEATURE"
            / config.symbol
            / config.target_freq
            / f"{config.start_date}-{config.end_date}.feather"
        )
        expected = recompute_merge_clean(config, config.start_date, config.end_date)
        return _compare_stage(stage, actual, expected, config)
    except Exception as exc:
        return _error_result(stage, exc)


def validate_ic_correlation(config: ValidationConfig) -> StageResult:
    stage = "ic_correlation"
    try:
        actual = read_feather_frame(
            config.root_path
            / "PREPROCESS_DATASET"
            / "commodity-futures"
            / "IC_RESULT"
            / config.symbol
            / config.target_freq
            / f"{config.start_date}-{config.end_date}"
            / "df.feather"
        )
        expected = recompute_ic_correlation(config, config.start_date, config.end_date)
        return _compare_stage(stage, actual, expected, config)
    except Exception as exc:
        return _error_result(stage, exc)


def validate_scale_save(config: ValidationConfig) -> StageResult:
    stage = "scale_save"
    try:
        state_path = (
            config.root_path
            / "PREPROCESS_DATASET"
            / "commodity-futures"
            / "SCALE_SAVE"
            / config.symbol
            / config.target_freq
            / f"{config.start_date}-{config.end_date}"
            / "state_features.npy"
        )
        if not state_path.exists():
            raise FileNotFoundError(str(state_path))
        actual = read_feather_frame(
            config.root_path
            / "PREPROCESS_DATASET"
            / "commodity-futures"
            / "SCALE_SAVE"
            / config.symbol
            / config.target_freq
            / f"{config.start_date}-{config.end_date}"
            / "df.feather"
        )
        expected = recompute_scale_save(config, config.start_date, config.end_date)
        return _compare_stage(stage, actual, expected, config)
    except Exception as exc:
        return _error_result(stage, exc)


def validate_all_stages(config: ValidationConfig) -> list[StageResult]:
    results = []
    results.extend(validate_cross_section(config))
    results.append(validate_merge_concat(config))
    results.append(validate_time_feature(config))
    results.append(validate_merge_clean(config))
    results.append(validate_ic_correlation(config))
    results.append(validate_scale_save(config))
    return results
