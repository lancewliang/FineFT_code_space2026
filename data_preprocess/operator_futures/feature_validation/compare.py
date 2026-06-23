from __future__ import annotations

import math
from dataclasses import asdict
from typing import Iterable

import numpy as np
import pandas as pd

from operator_futures.feature_validation.models import Mismatch, StageResult


def _normalize_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise ValueError("timestamp column is required for validation")
    result = df.copy()
    result["timestamp"] = result["timestamp"].astype(str)
    return result


def _sample_timestamps(timestamps: list[str], sample_size: int) -> list[str]:
    if sample_size <= 0 or len(timestamps) <= sample_size:
        return timestamps
    if sample_size == 1:
        return [timestamps[0]]
    step = (len(timestamps) - 1) / (sample_size - 1)
    indexes = sorted({round(i * step) for i in range(sample_size)})
    return [timestamps[index] for index in indexes]


def _to_float(value: object) -> float | None:
    value = _to_scalar(value)
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_scalar(value: object) -> object:
    if isinstance(value, pd.Series):
        if value.empty:
            return None
        value = value.iloc[0]
    if isinstance(value, np.generic):
        return value.item()
    return value


def compare_frames(
    stage: str,
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    expected_columns: Iterable[str],
    tolerance: float,
    sample_size: int,
) -> StageResult:
    actual = _normalize_timestamp_column(actual)
    expected = _normalize_timestamp_column(expected)
    expected_column_list = list(expected_columns)

    actual_columns = set(actual.columns) - {"timestamp"}
    reference_columns = set(expected.columns) - {"timestamp"}
    expected_set = set(expected_column_list)

    missing_columns = sorted(expected_set - actual_columns)
    extra_columns = sorted(actual_columns - expected_set)
    comparable_columns = sorted(expected_set & actual_columns & reference_columns)
    unverified_columns = sorted((expected_set & actual_columns) - reference_columns)

    actual_indexed = actual.set_index("timestamp")
    expected_indexed = expected.set_index("timestamp")
    common_timestamps = sorted(set(actual_indexed.index) & set(expected_indexed.index))
    sampled_timestamps = _sample_timestamps(common_timestamps, sample_size)

    failures: list[Mismatch] = []
    max_abs_diff = 0.0
    mismatched_columns: set[str] = set()

    for timestamp in sampled_timestamps:
        actual_row = actual_indexed.loc[timestamp]
        expected_row = expected_indexed.loc[timestamp]
        if isinstance(actual_row, pd.DataFrame):
            actual_row = actual_row.iloc[0]
        if isinstance(expected_row, pd.DataFrame):
            expected_row = expected_row.iloc[0]
        for column in comparable_columns:
            actual_value = actual_row[column]
            expected_value = expected_row[column]
            actual_scalar = _to_scalar(actual_value)
            expected_scalar = _to_scalar(expected_value)
            actual_float = _to_float(actual_value)
            expected_float = _to_float(expected_value)
            if actual_float is None and expected_float is None:
                continue
            if actual_float is None or expected_float is None:
                abs_diff = math.inf
            else:
                abs_diff = abs(actual_float - expected_float)
            max_abs_diff = max(max_abs_diff, abs_diff)
            if abs_diff > tolerance:
                mismatched_columns.add(column)
                if len(failures) < 50:
                    failures.append(
                        Mismatch(
                            stage=stage,
                            column=column,
                            timestamp=str(timestamp),
                            actual=actual_scalar,
                            expected=expected_scalar,
                            abs_diff=abs_diff,
                        )
                    )

    if not common_timestamps:
        status = "partial"
        message = "No overlapping timestamps between actual and reference outputs"
    elif missing_columns or mismatched_columns:
        status = "fail"
        message = ""
    elif unverified_columns:
        status = "partial"
        message = "Some expected columns were present but not produced by the reference adapter"
    else:
        status = "pass"
        message = ""

    return StageResult(
        stage=stage,
        status=status,
        checked_columns=len(comparable_columns),
        missing_columns=missing_columns,
        extra_columns=extra_columns,
        unverified_columns=unverified_columns,
        mismatched_columns=sorted(mismatched_columns),
        max_abs_diff=max_abs_diff,
        sample_failures=failures,
        message=message,
    )
