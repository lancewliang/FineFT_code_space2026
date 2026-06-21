from pathlib import Path

import numpy as np
import polars as pl


def assert_frame_contract(
    actual: pl.DataFrame,
    expected: pl.DataFrame,
    *,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> None:
    if actual.columns != expected.columns:
        raise AssertionError(
            f"column order mismatch: actual={actual.columns}, expected={expected.columns}"
        )

    if actual.height != expected.height:
        raise AssertionError(
            f"row count mismatch: actual={actual.height}, expected={expected.height}"
        )

    for column in expected.columns:
        actual_dtype = actual.schema[column]
        expected_dtype = expected.schema[column]
        if actual_dtype != expected_dtype:
            raise AssertionError(
                f"dtype mismatch for {column}: actual={actual_dtype}, expected={expected_dtype}"
            )

        actual_series = actual[column]
        expected_series = expected[column]
        if actual_dtype in (pl.Float32, pl.Float64):
            actual_values = actual_series.to_numpy()
            expected_values = expected_series.to_numpy()
            if not np.allclose(
                actual_values,
                expected_values,
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            ):
                raise AssertionError(f"float column {column} differs beyond tolerance")
            continue

        if actual_series.to_list() != expected_series.to_list():
            raise AssertionError(f"column {column} values differ")


def assert_no_pandas_engine(paths: list[Path]) -> None:
    offenders: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if "import pandas" in text or "from pandas" in text:
            offenders.append(str(path))

    if offenders:
        joined = ", ".join(offenders)
        raise AssertionError(f"pandas import remains in migrated engine files: {joined}")
