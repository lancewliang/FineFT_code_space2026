from datetime import datetime

import polars as pl
import pytest

from polars_compat import assert_frame_contract, assert_no_pandas_engine


def test_assert_frame_contract_accepts_matching_frames():
    expected = pl.DataFrame(
        {
            "timestamp": [
                datetime(2023, 1, 1, 9, 0, 0),
                datetime(2023, 1, 1, 9, 0, 10),
            ],
            "value": [1.0, 2.0],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
    actual = expected.clone()

    assert_frame_contract(actual, expected)


def test_assert_frame_contract_rejects_column_order_change():
    expected = pl.DataFrame({"timestamp": [1, 2], "value": [1.0, 2.0]})
    actual = pl.DataFrame({"value": [1.0, 2.0], "timestamp": [1, 2]})

    with pytest.raises(AssertionError, match="column order"):
        assert_frame_contract(actual, expected)


def test_assert_frame_contract_uses_strict_float_tolerance():
    expected = pl.DataFrame({"timestamp": [1, 2], "value": [1.0, 2.0]})
    actual = pl.DataFrame({"timestamp": [1, 2], "value": [1.0, 2.0 + 1e-9]})

    with pytest.raises(AssertionError, match="float column"):
        assert_frame_contract(actual, expected, rtol=1e-12, atol=1e-12)


def test_assert_no_pandas_engine_rejects_pandas_import(tmp_path):
    module = tmp_path / "module.py"
    module.write_text("import pandas as pd\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="pandas import"):
        assert_no_pandas_engine([module])
