from datetime import datetime, timedelta
import json
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
from operator_futures.feature_selection.ic_correlation import (
    select_reward_state_features,
)
from operator_futures.scale_describe_save.muti_contract_scale_save import (
    main as scale_save_main,
    parser as scale_save_parser,
)


MARKET_STATE_ANCHOR_COLUMNS = [
    "log_price_slope_48",
    "log_price_slope_96",
    "trend_to_noise_48",
    "trend_to_noise_96",
    "signed_efficiency_48",
    "trend_r2_48",
    "log_return_vol_quantile_192",
]


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


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_market_state_anchors_preserve_smooth_trend_direction(direction):
    expected_log_slope = direction * 0.002
    steps = np.arange(260, dtype=float)
    frame = pl.DataFrame(
        {
            "timestamp": steps.astype(int),
            "close": 100.0 * np.exp(expected_log_slope * steps),
        }
    )

    result = process_enhanced_state_features(frame)
    last = result.row(-1, named=True)

    assert set(MARKET_STATE_ANCHOR_COLUMNS).issubset(result.columns)
    assert last["log_price_slope_48"] == pytest.approx(expected_log_slope)
    assert last["log_price_slope_96"] == pytest.approx(expected_log_slope)
    assert np.sign(last["trend_to_noise_48"]) == direction
    assert np.sign(last["trend_to_noise_96"]) == direction
    assert last["signed_efficiency_48"] == pytest.approx(direction)
    assert last["trend_r2_48"] == pytest.approx(1.0)
    assert 0.0 <= last["log_return_vol_quantile_192"] <= 1.0
    assert np.isfinite(
        result.select(MARKET_STATE_ANCHOR_COLUMNS).to_numpy()
    ).all()


def test_market_state_anchor_r2_preserves_small_smooth_trend():
    steps = np.arange(48, dtype=float)
    frame = pl.DataFrame(
        {
            "timestamp": steps.astype(int),
            "close": 100.0 * np.exp(1e-7 * steps),
        }
    )

    result = process_enhanced_state_features(frame)

    assert result.row(-1, named=True)["trend_r2_48"] == pytest.approx(1.0)


@pytest.mark.parametrize("invalid_close", [0.0, -1.0, np.nan, np.inf])
def test_market_state_anchors_reject_invalid_close(invalid_close):
    close = np.full(48, 100.0)
    close[-1] = invalid_close
    frame = pl.DataFrame({"timestamp": np.arange(48), "close": close})

    with pytest.raises(ValueError, match="close"):
        process_enhanced_state_features(frame)


def test_market_state_anchor_windows_reset_at_contract_boundary():
    first_steps = np.arange(96, dtype=float)
    second_steps = np.arange(48, dtype=float)
    frame = pl.DataFrame(
        {
            "timestamp": np.arange(144),
            "contract": ["fu2601"] * 96 + ["fu2605"] * 48,
            "close": np.concatenate(
                [100.0 * np.exp(0.002 * first_steps), np.full(48, 100.0)]
            ),
        }
    )

    result = process_enhanced_state_features(frame)
    second_contract = result.filter(pl.col("contract") == "fu2605")

    assert second_contract.row(0, named=True)["log_price_slope_48"] == 0.0
    assert second_contract.row(46, named=True)["log_price_slope_48"] == 0.0
    assert second_contract.row(47, named=True)["log_price_slope_48"] == 0.0
    assert second_contract.row(47, named=True)["trend_r2_48"] == 0.0


def test_market_state_anchors_are_neutral_for_constant_prices_and_warmup():
    frame = pl.DataFrame(
        {"timestamp": np.arange(260), "close": np.full(260, 100.0)}
    )

    result = process_enhanced_state_features(frame)
    directional_columns = [
        "log_price_slope_48",
        "log_price_slope_96",
        "trend_to_noise_48",
        "trend_to_noise_96",
        "signed_efficiency_48",
        "trend_r2_48",
    ]

    assert (result.select(directional_columns).to_numpy() == 0.0).all()
    assert (
        result.head(47).select(MARKET_STATE_ANCHOR_COLUMNS).to_numpy() == 0.0
    ).all()
    assert np.isfinite(
        result.select(MARKET_STATE_ANCHOR_COLUMNS).to_numpy()
    ).all()


def test_market_state_anchors_are_registered_enhanced_feature_candidates():
    steps = np.arange(260, dtype=float)
    result = process_enhanced_state_features(
        pl.DataFrame(
            {
                "timestamp": steps.astype(int),
                "close": 100.0 * np.exp(0.001 * steps),
            }
        )
    )
    _, candidate_state_features = select_reward_state_features(
        result,
        market_type="commodity_futures",
        orderbook_depth=5,
    )

    assert set(MARKET_STATE_ANCHOR_COLUMNS).issubset(ENHANCED_FEATURE_COLUMNS)
    assert set(MARKET_STATE_ANCHOR_COLUMNS).issubset(candidate_state_features)


def test_market_state_anchors_are_causal_and_scale_invariant_for_noisy_trend():
    steps = np.arange(320, dtype=float)
    close = 100.0 * np.exp(0.001 * steps + 0.01 * np.sin(steps / 7.0))
    prefix = pl.DataFrame({"timestamp": np.arange(260), "close": close[:260]})
    full = pl.DataFrame({"timestamp": np.arange(320), "close": close})
    scaled = full.with_columns((pl.col("close") * 37.5).alias("close"))

    prefix_result = process_enhanced_state_features(prefix)
    full_result = process_enhanced_state_features(full)
    scaled_result = process_enhanced_state_features(scaled)

    np.testing.assert_allclose(
        prefix_result.select(MARKET_STATE_ANCHOR_COLUMNS).to_numpy(),
        full_result.head(260).select(MARKET_STATE_ANCHOR_COLUMNS).to_numpy(),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        full_result.select(MARKET_STATE_ANCHOR_COLUMNS).to_numpy(),
        scaled_result.select(MARKET_STATE_ANCHOR_COLUMNS).to_numpy(),
        rtol=1e-9,
        atol=1e-9,
    )
    last = full_result.row(-1, named=True)
    assert last["log_price_slope_48"] > 0.0
    assert last["trend_to_noise_48"] > 0.0
    assert -1.0 <= last["signed_efficiency_48"] <= 1.0
    assert 0.0 <= last["trend_r2_48"] <= 1.0
    assert 0.0 <= last["log_return_vol_quantile_192"] <= 1.0


def test_market_state_anchors_pass_nan_validation_and_scale_save(tmp_path):
    steps = np.arange(260, dtype=float)
    generated = process_enhanced_state_features(
        pl.DataFrame(
            {
                "timestamp": steps.astype(int),
                "contract": ["fu2601"] * len(steps),
                "close": 100.0
                * np.exp(0.001 * steps + 0.01 * np.sin(steps / 7.0)),
            }
        )
    )
    DataQualityValidator.validate_no_illegal_values(
        generated,
        stage="enhanced_feature",
        contract="fu2601",
        trading_day="fixture",
        columns=MARKET_STATE_ANCHOR_COLUMNS,
    )

    feature_list_path = tmp_path / "state_features.npy"
    np.save(feature_list_path, np.array(MARKET_STATE_ANCHOR_COLUMNS))
    input_dir = (
        tmp_path
        / "PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST"
        / "30min"
        / "fu"
        / "train"
    )
    input_dir.mkdir(parents=True)
    generated.select(
        ["timestamp", "contract", *MARKET_STATE_ANCHOR_COLUMNS]
    ).write_ipc(input_dir / "fu2601.feather")

    args = scale_save_parser.parse_args(
        [
            "--root_path",
            str(tmp_path),
            "--symbols",
            "fu",
            "--target_freq",
            "30min",
            "--feature_list_path",
            str(feature_list_path),
        ]
    )
    scale_save_main(args)

    output_root = (
        tmp_path / "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/fu/30min"
    )
    manifest = json.loads(
        (output_root / "scaler_manifest.json").read_text(encoding="utf-8")
    )
    scaled = pl.read_ipc(output_root / "train/fu2601.feather")

    assert [item["feature"] for item in manifest["features"]] == (
        MARKET_STATE_ANCHOR_COLUMNS
    )
    assert manifest["passthrough_state_features"] == []
    assert np.isfinite(scaled.select(MARKET_STATE_ANCHOR_COLUMNS).to_numpy()).all()


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
