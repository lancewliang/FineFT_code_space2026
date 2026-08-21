from datetime import datetime, timedelta
import numpy as np
import polars as pl
import pytest

from operator_futures.commodity.downscale import (
    downscale_quote_ofi_features,
    downscale_quote_microstructure_features,
)
from operator_futures.time_operator.time_operator_util import (
    process_enhanced_state_features,
)
from operator_futures.data_quality import DataQualityValidator
from operator_futures.feature_validation.expected_columns import ENHANCED_FEATURE_COLUMNS


def _sample_quote_snapshots(n: int = 50) -> pl.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "timestamp": datetime(2026, 2, 2, 9, 0, 0) + timedelta(minutes=i),
                "BidPrice1": 100.0 + i * 0.05,
                "AskPrice1": 100.2 + i * 0.05,
                "BidPrice2": 99.9,
                "AskPrice2": 100.3,
                "BidPrice3": 99.8,
                "AskPrice3": 100.4,
                "BidPrice4": 99.7,
                "AskPrice4": 100.5,
                "BidPrice5": 99.6,
                "AskPrice5": 100.6,
                "BidVolume1": 10.0 + (i % 5),
                "AskVolume1": 15.0 + (i % 3),
                "BidVolume2": 8.0,
                "AskVolume2": 12.0,
                "BidVolume3": 6.0,
                "AskVolume3": 10.0,
                "BidVolume4": 4.0,
                "AskVolume4": 8.0,
                "BidVolume5": 2.0,
                "AskVolume5": 6.0,
                "LastPrice": 100.1 + i * 0.05,
                "LowPrice": 99.5,
                "HighPrice": 101.5,
                "LowerLimitPrice": 90.0,
                "UpperLimitPrice": 110.0,
            }
        )
    return pl.DataFrame(rows)


def test_level5_ofi_and_spread_features():
    df = _sample_quote_snapshots(40)
    ofi_df = downscale_quote_ofi_features(df, window_rows=5, depth=5)
    assert "level5_ofi_weighted_norm" in ofi_df.columns
    assert "ofi_5m_norm" in ofi_df.columns
    assert ofi_df["level5_ofi_weighted_norm"].null_count() == 0
    assert not ofi_df["level5_ofi_weighted_norm"].is_nan().any()

    micro_df = downscale_quote_microstructure_features(df, window_rows=5)
    assert "relative_bid_ask_spread" in micro_df.columns
    assert micro_df["relative_bid_ask_spread"].null_count() == 0


def test_depth_depletion_and_replenishment_features():
    df = _sample_quote_snapshots(40)
    micro_df = downscale_quote_microstructure_features(df, window_rows=5)
    assert "ask_depth_depletion_5m" in micro_df.columns
    assert "bid_depth_depletion_5m" in micro_df.columns
    assert "depth_replenishment_ratio_20m" in micro_df.columns
    assert micro_df["ask_depth_depletion_5m"].null_count() == 0
    assert micro_df["bid_depth_depletion_5m"].null_count() == 0
    assert micro_df["depth_replenishment_ratio_20m"].null_count() == 0


def test_enhanced_time_operator_features():
    rows = []
    for i in range(220):
        close = 100.5 + i * 0.1
        volume = 100.0 + (i % 7) * 10
        rows.append(
            {
                "timestamp": datetime(2026, 2, 2, 9, 0, 0) + timedelta(minutes=i),
                "open": 100.0 + i * 0.1,
                "high": 101.0 + i * 0.1,
                "low": 99.0 + i * 0.1,
                "close": close,
                "volume": volume,
                "tradeval": close * volume,
                "open_interest": 5000.0 + i * 10,
                "ntrade_up_estimated": 30,
                "ntrade_down_estimated": 10,
                "relative_bid_ask_spread": 0.001 + (i % 3) * 0.0002,
                "garman_klass_volatility_12": 0.01 + (i % 5) * 0.001,
                "parkinson_volatility_12": 0.01 + (i % 4) * 0.001,
                "cm_main_sub_log_price_ratio": 0.015 + i * 0.0005,
                "cm_main_sub_open_interest_share_sub": 0.25 + i * 0.001,
            }
        )
    df = pl.DataFrame(rows)
    res = process_enhanced_state_features(df)

    expected = [
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
        "vwap_slope_96",
        "vwap_slope_192",
        "ema_slope_96",
        "ema_slope_192",
        "plus_di_14",
        "minus_di_14",
        "adx_14",
        "cvd_slope_96",
        "cvd_slope_192",
    ]
    for col in expected:
        assert col in res.columns
        assert res[col].null_count() == 0
        assert not res[col].is_nan().any()
        assert not res[col].is_infinite().any()

    assert "raw_vwap" not in res.columns
    assert "raw_ema" not in res.columns
    assert "macd" not in res.columns
    assert res["vwap_slope_96"][96] == pytest.approx(9.6 / 100.5 / 96)
    assert res["cvd_slope_96"][95] == pytest.approx(0.5)


def test_data_quality_validator_passes_enhanced_features():
    df = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 2, 2, 9, 0, 0)],
            "level5_ofi_weighted_norm": [0.15],
            "relative_bid_ask_spread": [0.0008],
            "ask_depth_depletion_5m": [0.05],
            "bid_depth_depletion_5m": [0.03],
            "depth_replenishment_ratio_20m": [1.02],
            "trade_direction_net_ratio_5m": [0.33],
            "price_velocity_10m": [0.12],
            "price_acceleration_10m_norm": [0.04],
        }
    )
    cols = [c for c in df.columns if c != "timestamp"]
    DataQualityValidator.validate_no_illegal_values(
        df,
        stage="test_enhanced",
        contract="fu2605",
        trading_day="2026-02-02",
        columns=cols,
    )
