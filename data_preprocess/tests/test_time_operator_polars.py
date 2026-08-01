from pathlib import Path
import os
import subprocess
import sys

import pytest
import pandas as pd
import polars as pl

from operator_futures.feature_validation.pandas_reference.time_operator.multi_processing_util import (
    get_multi_feature_window_price as pandas_get_multi_feature_window_price,
    process_ohlc_single_window as pandas_process_ohlc_single_window,
    process_ohlcv_single_window as pandas_process_ohlcv_single_window,
)
from operator_futures.time_operator.multi_processing_util import (
    _process_ohlc_single_window_polars,
    _process_ohlcv_single_window_polars,
    get_multi_feature_window_price,
    get_multi_window_ohlcv,
    get_risk_and_liquidity_state_features,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_concat_feature_fixture(
    path: Path,
    depth: int = 5,
    mark_price_nan_index: int | None = None,
    single_sided_ask_index: int | None = None,
) -> None:
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
            "tradeval": (2600.0 + idx) * (100.0 + idx),
            "open_interest": 1000.0 + idx,
            "mark_price": (
                float("nan") if idx == mark_price_nan_index else 2600.25 + idx
            ),
            "buy_spread_oe_max": 4.0,
            "sell_spread_oe_max": 4.0,
            "wap_1": 2600.2 + idx,
            "wap_2": 2600.3 + idx,
            "buy_wap": 2600.1 + idx,
            "sell_wap": 2600.4 + idx,
            "buy_volume_oe": 20.0 + idx,
            "sell_volume_oe": (
                0.0 if idx == single_sided_ask_index else 21.0 + idx
            ),
            "imblance_volume_oe": 1.0,
            "ask_side_empty": idx == single_sided_ask_index,
            "bid_side_empty": False,
            "LowerLimitPrice": 2400.0,
            "UpperLimitPrice": 3200.0,
        }
        for level in range(1, depth + 1):
            row[f"bid{level}_price"] = 2600.0 + idx - level
            row[f"ask{level}_price"] = 2600.0 + idx + level
            row[f"bid{level}_size_n"] = 0.1 * level
            row[f"ask{level}_size_n"] = (
                0.0 if idx == single_sided_ask_index else 0.2 * level
            )
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
    csv_output = output_file.with_suffix(".csv")
    assert csv_output.exists()
    assert pl.read_csv(csv_output).columns[0] == "timestamp"


def test_time_feature_cli_accepts_enhanced_single_sided_snapshot_input(tmp_path):
    input_file = (
        tmp_path
        / "PREPROCESS_DATASET/commodity-futures/MERGE_CONCAT/CONCAT_FEATURE/fu/5min"
        / "2026-01-05-2026-01-06.feather"
    )
    _write_concat_feature_fixture(input_file, depth=5, single_sided_ask_index=3)

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
    assert out.height > 0
    assert "ask_side_empty_log_return_2" not in out.columns
    assert "LowerLimitPrice_log_return_2" not in out.columns


def test_time_feature_cli_rejects_illegal_input_before_generating_features(tmp_path):
    input_file = (
        tmp_path
        / "PREPROCESS_DATASET/commodity-futures/MERGE_CONCAT/CONCAT_FEATURE/fu/5min"
        / "2026-01-05-2026-01-06.feather"
    )
    _write_concat_feature_fixture(input_file, depth=5, mark_price_nan_index=3)

    result = subprocess.run(
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
        capture_output=True,
        text=True,
    )

    output_file = (
        tmp_path
        / "PREPROCESS_DATASET/commodity-futures/TIME_FEATURE/fu/5min"
        / "2026-01-05-2026-01-06.feather"
    )
    message = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Illegal data detected" in message
    assert "stage=time_feature_input" in message
    assert "contract=fu" in message
    assert "mark_price:nan=1" in message
    assert not output_file.exists()


def test_multi_feature_price_preserves_equal_trends_for_distinct_price_levels():
    frame = pl.DataFrame(
        {
            "timestamp": list(range(8)),
            "ask4_price": [100.0 + idx for idx in range(8)],
            "ask5_price": [101.0 + idx for idx in range(8)],
        }
    )

    out = get_multi_feature_window_price(
        frame, [2], ["ask4_price", "ask5_price"]
    )

    assert "ask4_price_trend_2" in out.columns
    assert "ask5_price_trend_2" in out.columns


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


def test_single_price_window_cleans_signed_log_return_illegal_values():
    timestamps = [1_700_000_000 + idx * 60 for idx in range(8)]
    polars_frame = pl.DataFrame(
        {"imblance_volume_oe": [1.0, -1.0, -2.0, 2.0, -3.0, -3.0, 4.0, -4.0]},
    ).with_columns(pl.Series("timestamp", timestamps))
    actual = get_multi_feature_window_price(
        polars_frame, [2], ["imblance_volume_oe"]
    )

    assert "imblance_volume_oe_log_return_2" in actual.columns
    assert not actual.select(
        pl.any_horizontal(pl.selectors.float().is_nan()).any()
    ).item()
    assert not actual.select(
        pl.any_horizontal(pl.selectors.float().is_infinite()).any()
    ).item()
    assert actual["imblance_volume_oe_log_return_2"].to_list() == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]


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


def test_risk_and_liquidity_features_computes_expected_columns_for_windows():
    rows = []
    for idx in range(35):
        rows.append(
            {
                "timestamp": idx,
                "open": 2600.0 + (idx % 3),
                "high": 2605.0 + (idx % 3),
                "low": 2595.0 + (idx % 3),
                "close": 2602.0 + (idx % 3),
                "volume": 100.0 + idx * 10.0,
                "tradeval": 260000.0 + idx * 26000.0,
                "open_interest": 5000.0 + idx * 50.0,
            }
        )
    frame = pl.DataFrame(rows)
    res = get_risk_and_liquidity_state_features(frame, windows=[12, 20], symbol="fu", target_freq="5min")

    expected_prefixes = [
        "atr_pct",
        "historical_volatility",
        "rolling_volatility",
        "parkinson_volatility",
        "garman_klass_volatility",
        "realized_volatility",
        "relative_volume",
        "relative_amount",
        "relative_open_interest",
        "open_interest_change_ratio",
    ]
    for prefix in expected_prefixes:
        assert f"{prefix}_12" in res.columns
        assert f"{prefix}_20" in res.columns
        assert prefix not in res.columns

    for col in res.columns:
        if col != "timestamp":
            series = res.get_column(col)
            assert series.null_count() == 0
            assert not series.is_nan().any()
            assert not series.is_infinite().any()


def test_risk_and_liquidity_features_bars_per_day_uses_trading_sessions():
    rows = []
    for idx in range(30):
        # close increases deterministically
        price = 2600.0 * (1.001 ** idx)
        rows.append(
            {
                "timestamp": idx,
                "open": price,
                "high": price * 1.001,
                "low": price * 0.999,
                "close": price,
                "volume": 100.0,
                "tradeval": 260000.0,
                "open_interest": 5000.0,
            }
        )
    frame = pl.DataFrame(rows)
    res = get_risk_and_liquidity_state_features(frame, windows=[12], symbol="fu", target_freq="5min")
    hv_12 = res.item(0, "historical_volatility_12")
    # fu trading sessions total 345 minutes per day. 5min bar => 69 bars_per_day
    # log return r_t = ln(1.001) for all t
    # std of constant r_t is 0.0, so hv_12 should be 0.0
    assert hv_12 == pytest.approx(0.0, abs=1e-12)


def test_risk_and_liquidity_features_garman_klass_clips_negative_mean():
    rows = []
    for idx in range(25):
        rows.append(
            {
                "timestamp": idx,
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 100.0,
                "tradeval": 10000.0,
                "open_interest": 1000.0,
            }
        )
    frame = pl.DataFrame(rows)
    res = get_risk_and_liquidity_state_features(frame, windows=[12], symbol="fu", target_freq="5min")
    gk_12 = res.item(0, "garman_klass_volatility_12")
    assert gk_12 == 0.0


def test_risk_and_liquidity_features_open_interest_change_ratio_zeroes_non_positive_prev_oi():
    rows = []
    for idx in range(25):
        oi = 0.0 if idx < 12 else 500.0
        rows.append(
            {
                "timestamp": idx,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 100.0,
                "tradeval": 10000.0,
                "open_interest": oi,
            }
        )
    frame = pl.DataFrame(rows)
    res = get_risk_and_liquidity_state_features(frame, windows=[12], symbol="fu", target_freq="5min")
    # shifted 12 rows back is 0.0, so ratio output must be 0.0
    ratio_12 = res.item(0, "open_interest_change_ratio_12")
    assert ratio_12 == 0.0


def test_risk_and_liquidity_features_rejects_missing_open_interest():
    rows = [{"timestamp": 0, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 100.0, "tradeval": 10000.0}]
    frame = pl.DataFrame(rows)
    with pytest.raises(ValueError, match="open_interest"):
        get_risk_and_liquidity_state_features(frame, windows=[12], symbol="fu", target_freq="5min")


def test_risk_and_liquidity_features_rejects_non_positive_prices():
    rows = [{"timestamp": 0, "open": 0.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 100.0, "tradeval": 10000.0, "open_interest": 1000.0}]
    frame = pl.DataFrame(rows)
    with pytest.raises(ValueError, match="Invalid price"):
        get_risk_and_liquidity_state_features(frame, windows=[12], symbol="fu", target_freq="5min")


from operator_futures.time_operator.time_operator_util import process_enhanced_state_features

def test_process_enhanced_state_features():
    rows = []
    for i in range(50):
        rows.append({
            "timestamp": i,
            "close": 100.0 + i * 0.1,
            "volume": 100.0 + (i % 5) * 10,
            "open_interest": 1000.0 + i * 5,
            "ntrade_up_estimated": 10 + i,
            "ntrade_down_estimated": 5 + (i % 3),
            "relative_bid_ask_spread": 0.001 + (i % 2) * 0.0005,
            "garman_klass_volatility_12": 0.01 + (i % 4) * 0.002,
            "parkinson_volatility_12": 0.01 + (i % 3) * 0.001,
            "cm_main_sub_log_price_ratio": 0.02 + i * 0.001,
            "cm_main_sub_open_interest_share_sub": 0.3 + i * 0.002,
        })
    df = pl.DataFrame(rows)
    res = process_enhanced_state_features(df)
    
    expected_cols = [
        "trade_direction_net_ratio_5m",
        "trade_direction_persistence_20m",
        "spread_widening_zscore_48",
        "price_velocity_10m",
        "price_acceleration_10m_norm",
        "garman_klass_vol_quantile_192",
        "parkinson_vol_zscore_192",
        "price_oi_vol_interaction_10m",
        "oi_change_rate_norm_10m",
        "cm_main_sub_log_price_spread_velocity_10m",
        "cm_open_interest_shift_speed_10m",
    ]
    for col in expected_cols:
        assert col in res.columns
        assert res[col].null_count() == 0
        assert not res[col].is_nan().any()
