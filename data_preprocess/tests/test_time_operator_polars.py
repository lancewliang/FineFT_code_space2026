from pathlib import Path
import os
import subprocess
import sys

import pandas as pd
import polars as pl

from operator_futures.feature_validation.pandas_reference.time_operator.multi_processing_util import (
    get_multi_feature_window_price as pandas_get_multi_feature_window_price,
    process_ohlc_single_window as pandas_process_ohlc_single_window,
    process_ohlcv_single_window as pandas_process_ohlcv_single_window,
    process_single_price_single_window as pandas_process_single_price_single_window,
)
from operator_futures.time_operator.multi_processing_util import (
    _process_ohlc_single_window_polars,
    _process_ohlcv_single_window_polars,
    get_multi_feature_window_price,
    get_multi_window_ohlcv,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_concat_feature_fixture(path: Path, depth: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(20):
        row = {
            "timestamp": idx,
            "open": 2600.0 + idx,
            "high": 2601.0 + idx,
            "low": 2599.0 + idx,
            "close": 2600.5 + idx,
            "volume": 100.0 + idx,
            "mark_price": 2600.25 + idx,
            "buy_spread_oe_max": 4.0,
            "sell_spread_oe_max": 4.0,
            "wap_1": 2600.2 + idx,
            "wap_2": 2600.3 + idx,
            "buy_wap": 2600.1 + idx,
            "sell_wap": 2600.4 + idx,
            "buy_volume_oe": 20.0 + idx,
            "sell_volume_oe": 21.0 + idx,
            "imblance_volume_oe": 1.0,
        }
        for level in range(1, depth + 1):
            row[f"bid{level}_price"] = 2600.0 + idx - level
            row[f"ask{level}_price"] = 2600.0 + idx + level
            row[f"bid{level}_size_n"] = 0.1 * level
            row[f"ask{level}_size_n"] = 0.2 * level
        rows.append(row)
    pl.DataFrame(rows).write_ipc(path)


def test_time_feature_multi_processing_targets_do_not_import_pandas():
    targets = [
        REPO_ROOT
        / "data_preprocess/operator_futures/time_operator/create_feature_multi_processing.py",
        REPO_ROOT / "data_preprocess/operator_futures/time_operator/multi_processing_util.py",
    ]
    for path in targets:
        text = path.read_text(encoding="utf-8")
        assert "import pandas" not in text
        assert "from pandas" not in text


def test_time_feature_cli_respects_orderbook_depth_and_output_contract(tmp_path):
    input_file = (
        tmp_path
        / "PREPROCESS_DATASET/commodity-futures/MERGE_CONCAT/CONCAT_FEATURE/fu/5min"
        / "2026-01-05-2026-01-06.feather"
    )
    _write_concat_feature_fixture(input_file, depth=5)

    subprocess.run(
        [
            sys.executable,
            "data_preprocess/operator_futures/time_operator/create_feature_multi_processing.py",
            "--root_path",
            str(tmp_path),
            "--data_path",
            "PREPROCESS_DATASET/commodity-futures/MERGE_CONCAT/CONCAT_FEATURE/",
            "--save_path",
            "PREPROCESS_DATASET/commodity-futures/TIME_FEATURE/",
            "--symbols",
            "fu",
            "--target_freq",
            "5min",
            "--start_date",
            "2026-01-05",
            "--end_date",
            "2026-01-06",
            "--windows",
            "2",
            "--orderbook_depth",
            "5",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        check=True,
    )

    output_file = (
        tmp_path
        / "PREPROCESS_DATASET/commodity-futures/TIME_FEATURE/fu/5min"
        / "2026-01-05-2026-01-06.feather"
    )
    out = pl.read_ipc(output_file)
    assert out.columns[0] == "timestamp"
    assert out.height > 0
    assert "bid5_price_log_return_2" in out.columns
    assert "bid6_price_log_return_2" not in out.columns


def test_get_multi_window_ohlcv_supports_multiple_windows_without_suffix_collision():
    frame = pl.DataFrame(
        {
            "timestamp": list(range(1, 21)),
            "open": [10.0 + idx for idx in range(20)],
            "high": [11.0 + idx for idx in range(20)],
            "low": [9.0 + idx for idx in range(20)],
            "close": [10.0 + idx for idx in range(20)],
            "volume": [100.0 + idx for idx in range(20)],
        }
    )

    out = get_multi_window_ohlcv(frame, [2, 3, 4])

    assert out.height > 0
    assert "log_volume" in out.columns


def test_ohlcv_single_window_matches_pandas_reference_formulas():
    timestamps = [1_700_000_000 + idx * 60 for idx in range(24)]
    pandas_frame = pd.DataFrame(
        {
            "open": [100.0, 101.0, 100.5, 102.0, 103.0, 102.5, 104.0, 103.5, 105.0, 104.0, 106.0, 105.5, 107.0, 106.5, 108.0, 107.0, 109.0, 108.5, 110.0, 109.5, 111.0, 110.5, 112.0, 111.5],
            "high": [101.0, 102.5, 101.5, 103.0, 104.5, 103.5, 105.0, 104.0, 106.5, 105.0, 107.0, 106.0, 108.5, 107.5, 109.0, 108.0, 110.5, 109.0, 111.0, 110.0, 112.5, 111.0, 113.0, 112.0],
            "low": [99.0, 100.0, 99.5, 101.0, 102.0, 101.5, 103.0, 102.5, 104.0, 103.0, 105.0, 104.5, 106.0, 105.5, 107.0, 106.0, 108.0, 107.5, 109.0, 108.0, 110.0, 109.5, 111.0, 110.0],
            "close": [100.5, 101.5, 100.8, 102.5, 103.2, 102.8, 104.4, 103.9, 105.5, 104.8, 106.2, 105.9, 107.4, 106.8, 108.3, 107.6, 109.2, 108.7, 110.1, 109.4, 111.2, 110.7, 112.3, 111.6],
            "volume": [1000.0, 980.0, 1030.0, 1010.0, 1080.0, 1060.0, 1110.0, 1090.0, 1150.0, 1120.0, 1180.0, 1160.0, 1210.0, 1190.0, 1250.0, 1220.0, 1290.0, 1260.0, 1320.0, 1300.0, 1360.0, 1330.0, 1400.0, 1370.0],
        },
        index=timestamps,
    )
    polars_frame = pl.from_pandas(pandas_frame.reset_index(names="timestamp"))

    expected = pandas_process_ohlcv_single_window(pandas_frame, 3).reset_index(
        names="timestamp"
    )
    actual = _process_ohlcv_single_window_polars(polars_frame, 3).to_pandas()

    pd.testing.assert_frame_equal(
        actual[expected.columns],
        expected,
        check_dtype=False,
        atol=1e-9,
        rtol=0,
    )


def test_ohlc_single_window_matches_pandas_reference_formulas():
    timestamps = [1_700_000_000 + idx * 60 for idx in range(12)]
    pandas_frame = pd.DataFrame(
        {
            "open": [10.0, 11.0, 10.5, 12.0, 13.0, 12.5, 14.0, 13.5, 15.0, 14.0, 16.0, 15.5],
            "high": [11.0, 12.5, 11.5, 13.0, 14.5, 13.5, 15.0, 14.0, 16.5, 15.0, 17.0, 16.0],
            "low": [9.0, 10.0, 9.5, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.0, 15.0, 14.5],
            "close": [10.5, 11.5, 10.8, 12.5, 13.2, 12.8, 14.4, 13.9, 15.5, 14.8, 16.2, 15.9],
        },
        index=timestamps,
    )
    polars_frame = pl.from_pandas(pandas_frame.reset_index(names="timestamp"))

    expected = pandas_process_ohlc_single_window(pandas_frame, 3).reset_index(
        names="timestamp"
    )
    actual = _process_ohlc_single_window_polars(polars_frame, 3).to_pandas()

    pd.testing.assert_frame_equal(
        actual[expected.columns],
        expected,
        check_dtype=False,
        atol=1e-9,
        rtol=0,
    )


def test_single_price_window_preserves_pandas_reference_nan_values():
    timestamps = [1_700_000_000 + idx * 60 for idx in range(8)]
    pandas_frame = pd.DataFrame(
        {"imblance_volume_oe": [1.0, -1.0, -2.0, 2.0, -3.0, -3.0, 4.0, -4.0]},
        index=timestamps,
    )
    polars_frame = pl.from_pandas(pandas_frame.reset_index(names="timestamp"))

    expected = pandas_process_single_price_single_window(
        pandas_frame["imblance_volume_oe"], 2
    ).reset_index(names="timestamp")
    actual = get_multi_feature_window_price(
        polars_frame, [2], ["imblance_volume_oe"]
    ).to_pandas()

    pd.testing.assert_frame_equal(
        actual[expected.columns],
        expected,
        check_dtype=False,
        atol=1e-9,
        rtol=0,
    )


def test_multi_feature_price_deduplicates_repeated_window_outputs_like_reference():
    pandas_frame = pd.DataFrame(
        {
            "feature_x": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
        },
        index=[1_700_000_000 + idx * 60 for idx in range(8)],
    )
    polars_frame = pl.from_pandas(pandas_frame.reset_index(names="timestamp"))

    expected = pandas_get_multi_feature_window_price(
        pandas_frame, [2, 6], ["feature_x"]
    ).reset_index(names="timestamp")
    actual = get_multi_feature_window_price(
        polars_frame, [2, 6], ["feature_x"]
    ).to_pandas()

    assert list(actual.columns) == list(expected.columns)
    pd.testing.assert_frame_equal(
        actual[expected.columns],
        expected,
        check_dtype=False,
        atol=1e-9,
        rtol=0,
    )


def test_ohlcv_window_two_matches_pandas_reference_degenerate_windows():
    timestamps = [1_700_000_000 + idx * 60 for idx in range(15)]
    close = [
        2766.0,
        2767.0,
        2768.0,
        2769.0,
        2770.0,
        2771.0,
        2772.0,
        2768.234875,
        2768.210526,
        2773.785714,
        2768.6,
        2774.0,
        2774.0,
        2773.0,
        2772.0,
    ]
    volume = [
        1000.0,
        1200.0,
        900.0,
        1300.0,
        1100.0,
        1500.0,
        1400.0,
        281.0,
        37788.0,
        8502.0,
        10328.0,
        12169.0,
        3525.0,
        2156.0,
        2522.0,
    ]
    pandas_frame = pd.DataFrame(
        {
            "open": close,
            "high": [value + 1.0 for value in close],
            "low": [value - 1.0 for value in close],
            "close": close,
            "volume": volume,
        },
        index=timestamps,
    )
    polars_frame = pl.from_pandas(pandas_frame.reset_index(names="timestamp"))

    expected = pandas_process_ohlcv_single_window(pandas_frame, 2).reset_index(
        names="timestamp"
    )
    actual = _process_ohlcv_single_window_polars(polars_frame, 2).to_pandas()

    pd.testing.assert_frame_equal(
        actual[expected.columns],
        expected,
        check_dtype=False,
        atol=1e-9,
        rtol=0,
    )
