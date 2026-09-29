import json
from pathlib import Path
import numpy as np
import polars as pl
import pytest

from operator_futures.commodity.base_time_feature import BASE_TIME_FEATURE_COLUMNS
from operator_futures.scale_describe_save.muti_contract_scale_save import (
    main,
    parser,
    load_state_features,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_scale_save_passthrough_base_time_features(tmp_path):
    # Setup state_features.npy with normal feature + BASE_TIME_FEATURE_COLUMNS
    feature_list_file = tmp_path / "state_features.npy"
    all_features = ["normal_feature"] + list(BASE_TIME_FEATURE_COLUMNS)
    np.save(feature_list_file, np.array(all_features))

    split_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/5min/fu/train"
    split_dir.mkdir(parents=True, exist_ok=True)

    # Input dataframe with reward columns, normal_feature, and base_time_features
    n = 10
    df = pl.DataFrame({
        "timestamp": list(range(1, n + 1)),
        "contract": ["fu2601"] * n,
        "symbol": ["fu"] * n,
        "ask1_price": [100.0] * n,
        "ask1_size": [1.0] * n,
        "bid1_price": [99.0] * n,
        "bid1_size": [1.0] * n,
        "LowerLimitPrice": [90.0] * n,
        "UpperLimitPrice": [110.0] * n,
        "funding_timestamp": list(range(1, n + 1)),
        "funding_rate": [0.0] * n,
        "index_price": [100.0] * n,
        "mark_price": [100.0] * n,
        "normal_feature": [10.0 * i for i in range(n)],
        "trading_minute_progress": [0.1 * i for i in range(n)],
        "morning_session": [1.0] * n,
        "afternoon_session": [0.0] * n,
        "night_session": [0.0] * n,
        "is_opening_30m": [1.0] * n,
        "is_closing_30m": [0.0] * n,
        "is_session_first_bar": [1.0, 1.0] + [0.0] * (n - 2),
        "is_session_last_bar": [0.0] * (n - 2) + [1.0, 1.0],
        "contract_month_sin": [0.5] * n,
        "contract_month_cos": [0.5] * n,
        "contract_life_remaining_ratio": [0.8] * n,
        "prev_day_contract_role_tier": [1.0] * n,
    })
    df.write_ipc(split_dir / "fu2601.feather")

    save_path = "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE"
    args = parser.parse_args([
        "--root_path", str(tmp_path),
        "--symbols", "fu",
        "--target_freq", "5min",
        "--feature_list_path", str(feature_list_file),
        "--save_path", save_path,
        "--passthrough_features", *BASE_TIME_FEATURE_COLUMNS,
    ])

    main(args)

    output_root = tmp_path / save_path / "fu" / "5min"
    manifest_path = output_root / "scaler_manifest.json"
    assert manifest_path.exists()
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Manifest features list should only contain normal_feature
    feature_names_in_manifest = [f["feature"] for f in manifest_data["features"]]
    assert "normal_feature" in feature_names_in_manifest
    for col in BASE_TIME_FEATURE_COLUMNS:
        assert col not in feature_names_in_manifest

    # Manifest must record passthrough_state_features
    assert "passthrough_state_features" in manifest_data
    assert manifest_data["passthrough_state_features"] == list(BASE_TIME_FEATURE_COLUMNS)

    # Check scaled output feather file
    out_file = output_root / "train" / "fu2601.feather"
    assert out_file.exists()
    out_df = pl.read_ipc(out_file)

    # BASE_TIME_FEATURE values must remain unscaled/passthrough
    expected_progress = [0.1 * i for i in range(n)]
    assert np.allclose(out_df["trading_minute_progress"].to_list(), expected_progress)
    assert np.allclose(out_df["morning_session"].to_list(), [1.0] * n)
    assert out_df["is_session_first_bar"].to_list() == [
        1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]
    assert out_df["is_session_last_bar"].to_list() == [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
    ]
    assert np.allclose(out_df["contract_life_remaining_ratio"].to_list(), [0.8] * n)
    assert np.allclose(out_df["prev_day_contract_role_tier"].to_list(), [1.0] * n)


def test_resolve_feature_clip_bounds_fat_tailed():
    from operator_futures.scale_describe_save.muti_contract_scale_save import (
        resolve_feature_clip_bounds,
    )
    # General features should retain global clip bounds [-5.0, 5.0]
    assert resolve_feature_clip_bounds("normal_feature", -5.0, 5.0) == (-5.0, 5.0)
    assert resolve_feature_clip_bounds("rsi_12", -5.0, 5.0) == (-5.0, 5.0)

    # Fat-tailed features should clamp to [-4.0, 4.0]
    assert resolve_feature_clip_bounds("vstd_24_origin", -5.0, 5.0) == (-4.0, 4.0)
    assert resolve_feature_clip_bounds("cm_m1_m2_log_price_spread_velocity_10m", -5.0, 5.0) == (-4.0, 4.0)
    assert resolve_feature_clip_bounds("cm_open_interest_shift_speed_10m", -5.0, 5.0) == (-4.0, 4.0)
    assert resolve_feature_clip_bounds("ask_size_topk_size_5_increments", -5.0, 5.0) == (-4.0, 4.0)
    assert resolve_feature_clip_bounds("bid_size_topk_size_5_increments", -5.0, 5.0) == (-4.0, 4.0)
    assert resolve_feature_clip_bounds("sell_spread_oe_max_trend_192", -5.0, 5.0) == (-4.0, 4.0)

    # Tighter custom bounds should be preserved
    assert resolve_feature_clip_bounds("vstd_24_origin", -3.0, 3.0) == (-3.0, 3.0)


def test_vstd_operator_bounds():
    from operator_futures.time_operator.multi_processing_util import _process_ohlcv_single_window_polars

    # Near-zero and zero volume should not cause division explosion in vstd
    df = pl.DataFrame({
        "timestamp": list(range(10)),
        "open": [100.0] * 10,
        "high": [101.0] * 10,
        "low": [99.0] * 10,
        "close": [100.0] * 10,
        "volume": [0.0, 0.0, 1e-12, 0.5, 100.0, 500.0, 0.0, 0.0, 10.0, 10.0],
    })
    result = _process_ohlcv_single_window_polars(df, window=3)
    assert "vstd_3" in result.columns
    vstd_vals = result["vstd_3"].to_numpy()
    assert np.all(np.isfinite(vstd_vals))
    assert np.all(vstd_vals >= 0.0)
    assert np.all(vstd_vals <= 10.0)


def test_is_volatility_feature_helper():
    from operator_futures.scale_describe_save.muti_contract_scale_save import (
        is_volatility_feature,
        is_log_transformed_feature,
    )
    # Volatility patterns
    assert is_volatility_feature("realized_volatility_192") is True
    assert is_volatility_feature("garman_klass_volatility_16") is True
    assert is_volatility_feature("rolling_volatility_48") is True
    assert is_volatility_feature("parkinson_volatility_96") is True
    assert is_volatility_feature("historical_volatility_24") is True
    assert is_volatility_feature("bollinger_bandwidth_96_origin") is True
    assert is_volatility_feature("bollinger_bandwidth_48") is True
    assert is_volatility_feature("rsi_12") is False
    assert is_volatility_feature("normal_feature") is False
    assert is_volatility_feature("log_price_slope_48") is False

    # Volume activity patterns (ADR-0033)
    assert is_log_transformed_feature("vma_24_std_norm_origin") is True
    assert is_log_transformed_feature("relative_volume_10") is True
    assert is_log_transformed_feature("relative_amount_48") is True
    assert is_log_transformed_feature("wvma_48_origin") is True

    # Signed volume/trend features must NOT be log-transformed
    assert is_log_transformed_feature("buy_volume_oe_trend_6") is False
    assert is_log_transformed_feature("imblance_volume_oe_trend_16") is False
    assert is_log_transformed_feature("sell_volume_oe_log_return_2") is False
    assert is_log_transformed_feature("price_oi_vol_interaction_10m") is False


def test_log_volatility_transformation_in_scale_save(tmp_path):
    from operator_futures.scale_describe_save.muti_contract_scale_save import (
        VOLATILITY_LOG_EPSILON,
    )

    feature_list_file = tmp_path / "state_features.npy"
    features = ["realized_volatility_192", "normal_feature"]
    np.save(feature_list_file, np.array(features))

    split_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/5min/fu/train"
    split_dir.mkdir(parents=True, exist_ok=True)

    n = 100
    np.random.seed(42)
    raw_vol = np.exp(np.random.normal(-4.0, 0.5, size=n))
    normal_vals = np.random.normal(50.0, 10.0, size=n)

    df = pl.DataFrame({
        "timestamp": list(range(1, n + 1)),
        "contract": ["fu2601"] * n,
        "symbol": ["fu"] * n,
        "ask1_price": [100.0] * n,
        "ask1_size": [1.0] * n,
        "bid1_price": [99.0] * n,
        "bid1_size": [1.0] * n,
        "LowerLimitPrice": [90.0] * n,
        "UpperLimitPrice": [110.0] * n,
        "funding_timestamp": list(range(1, n + 1)),
        "funding_rate": [0.0] * n,
        "index_price": [100.0] * n,
        "mark_price": [100.0] * n,
        "realized_volatility_192": raw_vol,
        "normal_feature": normal_vals,
    })
    df.write_ipc(split_dir / "fu2601.feather")

    save_path = "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE"
    args = parser.parse_args([
        "--root_path", str(tmp_path),
        "--symbols", "fu",
        "--target_freq", "5min",
        "--feature_list_path", str(feature_list_file),
        "--save_path", save_path,
    ])

    main(args)

    output_root = tmp_path / save_path / "fu" / "5min"
    manifest_path = output_root / "scaler_manifest.json"
    assert manifest_path.exists()
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

    stats_by_feat = {f["feature"]: f for f in manifest_data["features"]}
    assert stats_by_feat["realized_volatility_192"]["is_log_transformed"] is True
    assert stats_by_feat["normal_feature"]["is_log_transformed"] is False

    # Volatility center must be in log space (around -4.0), not raw space (around 0.02)
    vol_center = stats_by_feat["realized_volatility_192"]["center"]
    assert -5.0 < vol_center < -3.0

    # Verify scaled dataframe matches exact log-transformation formula
    out_file = output_root / "train" / "fu2601.feather"
    out_df = pl.read_ipc(out_file)
    vol_scale = stats_by_feat["realized_volatility_192"]["scale"]
    expected_vol_scaled = np.clip(
        (np.log(np.maximum(raw_vol, 0.0) + VOLATILITY_LOG_EPSILON) - vol_center) / vol_scale,
        -5.0,
        5.0,
    )
    assert np.allclose(out_df["realized_volatility_192"].to_numpy(), expected_vol_scaled, atol=1e-6)


def test_orderbook_spread_physical_depth_bounding():
    import pandas as pd
    from operator_futures.cross_section.base_feature_util import process_snapshot_features

    # Create dummy 5-depth orderbook dataframe with flat/identical prices (simulating auction or zero depth)
    df = pd.DataFrame({
        "bid1_price": [100.0, 100.0],
        "bid2_price": [100.0, 99.0],
        "bid3_price": [100.0, 98.0],
        "bid4_price": [100.0, 97.0],
        "bid5_price": [100.0, 96.0],
        "ask1_price": [101.0, 101.0],
        "ask2_price": [101.0, 102.0],
        "ask3_price": [101.0, 103.0],
        "ask4_price": [101.0, 104.0],
        "ask5_price": [101.0, 105.0],
        "bid1_size": [10.0, 10.0],
        "bid2_size": [10.0, 10.0],
        "bid3_size": [10.0, 10.0],
        "bid4_size": [10.0, 10.0],
        "bid5_size": [10.0, 10.0],
        "ask1_size": [10.0, 10.0],
        "ask2_size": [10.0, 10.0],
        "ask3_size": [10.0, 10.0],
        "ask4_size": [10.0, 10.0],
        "ask5_size": [10.0, 10.0],
    })

    # Orderbook depth features are returned as Polars DataFrame
    price_df = process_snapshot_features(df, topk=5, depth=5)
    # Row 0 has bid1 == bid5 == 100.0 (diff 0.0). With ADR-0030 bounding, minimum is 4.0
    assert price_df["buy_spread_oe_max"][0] == 4.0
    assert price_df["sell_spread_oe_max"][0] == 4.0
    assert price_df["buy_spread_oe_max"][1] == 4.0
    assert price_df["sell_spread_oe_max"][1] == 4.0

    # Polars version
    pl_df = pl.from_pandas(df)
    pl_res = process_snapshot_features(pl_df, topk=5, depth=5)
    assert pl_res["buy_spread_oe_max"][0] == 4.0
    assert pl_res["sell_spread_oe_max"][0] == 4.0
    assert pl_res["buy_spread_oe_max"][1] == 4.0
    assert pl_res["sell_spread_oe_max"][1] == 4.0


def test_adr0030_blacklist_coverage():
    script_path = REPO_ROOT / "data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh"
    text = script_path.read_text(encoding="utf-8")

    expected_features = [
        "min_96_origin",
        "max_96_origin",
        "pivot_s2_48_origin",
        "pivot_s1_24_origin",
        "pivot_s1_6_origin",
        "bollinger_lower_12_origin",
        "max_192_std_norm_origin",
        "cm_current_main_spread_rolling_zscore_192",
        "cm_main_sub_spread_rolling_zscore_192",
        "cm_main_sub_volume_share_sub",
        "cm_current_main_volume_share_current",
        "cm_current_sub_volume_share_current",
        "sell_spread_oe_max_trend_6",
        "buy_spread_oe_max_trend_6",
        "buy_spread_oe_max",
        "sell_spread_oe_max",
    ]
    for feat in expected_features:
        assert feat in text, f"Feature {feat} missing from fu_full_process.sh blacklist"


def test_volume_activity_log_transformation_in_scale_save(tmp_path):
    feature_list_file = tmp_path / "state_features.npy"
    features = ["vma_24_std_norm_origin", "buy_volume_oe_trend_6"]
    np.save(feature_list_file, np.array(features))

    split_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/5min/fu/train"
    split_dir.mkdir(parents=True, exist_ok=True)

    n = 50
    np.random.seed(42)
    raw_vma = np.exp(np.random.normal(1.0, 0.3, size=n))
    raw_trend = np.random.normal(0.0, 2.0, size=n)

    df = pl.DataFrame({
        "timestamp": list(range(1, n + 1)),
        "contract": ["fu2601"] * n,
        "symbol": ["fu"] * n,
        "ask1_price": [100.0] * n,
        "ask1_size": [1.0] * n,
        "bid1_price": [99.0] * n,
        "bid1_size": [1.0] * n,
        "LowerLimitPrice": [90.0] * n,
        "UpperLimitPrice": [110.0] * n,
        "funding_timestamp": list(range(1, n + 1)),
        "funding_rate": [0.0] * n,
        "index_price": [100.0] * n,
        "mark_price": [100.0] * n,
        "vma_24_std_norm_origin": raw_vma,
        "buy_volume_oe_trend_6": raw_trend,
    })
    df.write_ipc(split_dir / "fu2601.feather")

    save_path = "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE"
    args = parser.parse_args([
        "--root_path", str(tmp_path),
        "--symbols", "fu",
        "--target_freq", "5min",
        "--feature_list_path", str(feature_list_file),
        "--save_path", save_path,
    ])

    main(args)

    output_root = tmp_path / save_path / "fu" / "5min"
    manifest_path = output_root / "scaler_manifest.json"
    assert manifest_path.exists()
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

    stats_by_feat = {f["feature"]: f for f in manifest_data["features"]}
    assert stats_by_feat["vma_24_std_norm_origin"]["is_log_transformed"] is True
    assert stats_by_feat["buy_volume_oe_trend_6"]["is_log_transformed"] is False
