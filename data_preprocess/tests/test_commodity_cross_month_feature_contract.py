import pytest
import numpy as np
import polars as pl
import json

from operator_futures.commodity.cross_month_feature import (
    CROSS_MONTH_FEATURE_COLUMNS,
    CROSS_MONTH_PAIRING_MODES,
    generate_delivery_month_sequence_features,
    generate_main_sub_cross_month_features,
    resolve_cross_month_feature_input,
    resolve_previous_main_sub_role,
    validate_cross_month_feature_columns,
    write_cross_month_feature_for_day,
)


def test_cross_month_feature_contract_exposes_stable_columns_and_pairing_modes():
    assert CROSS_MONTH_PAIRING_MODES == ["main_sub", "delivery_month_sequence"]
    assert CROSS_MONTH_FEATURE_COLUMNS == [
        "cm_contract_role_main",
        "cm_contract_role_sub",
        "cm_contract_role_other",
        "cm_current_main_log_price_ratio",
        "cm_current_main_relative_price_spread",
        "cm_current_main_volume_share_current",
        "cm_current_main_open_interest_share_current",
        "cm_current_sub_log_price_ratio",
        "cm_current_sub_relative_price_spread",
        "cm_current_sub_volume_share_current",
        "cm_current_sub_open_interest_share_current",
        "cm_main_sub_log_price_ratio",
        "cm_main_sub_relative_price_spread",
        "cm_main_sub_volume_share_sub",
        "cm_main_sub_open_interest_share_sub",
        "cm_m1_m2_log_price_ratio",
        "cm_m2_m3_log_price_ratio",
        "cm_m1_m2_relative_price_spread",
        "cm_m2_m3_relative_price_spread",
        "cm_m1_m2_m3_butterfly_ratio",
        "cm_m1_m2_open_interest_share_m2",
        "cm_m2_m3_open_interest_share_m3",
        "cm_main_sub_log_price_spread_velocity_10m",
        "cm_open_interest_shift_speed_10m",
    ]
    assert validate_cross_month_feature_columns(CROSS_MONTH_FEATURE_COLUMNS) == list(
        CROSS_MONTH_FEATURE_COLUMNS
    )


def test_cross_month_feature_contract_rejects_absolute_price_columns():
    with pytest.raises(ValueError, match="No Absolute Price Rule"):
        validate_cross_month_feature_columns(
            [
                "cm_main_price",
                "cm_main_sub_price_spread",
                "cm_m1_m2_log_price_ratio",
            ]
        )


def test_cross_month_feature_contract_distinguishes_missing_file_from_gap_fill(
    tmp_path,
):
    missing = tmp_path / "CROSS_MONTH_FEATURE/fu/fu2601/5min/2026-01-05.feather"

    assert resolve_cross_month_feature_input(missing, required=False) is None
    with pytest.raises(ValueError, match="missing required CROSS_MONTH_FEATURE"):
        resolve_cross_month_feature_input(missing, required=True)

    existing = tmp_path / "CROSS_MONTH_FEATURE/fu/fu2601/5min/2026-01-06.feather"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"placeholder")

    assert resolve_cross_month_feature_input(existing, required=True) == existing


def test_generate_main_sub_cross_month_features_for_other_contract():
    current = pl.DataFrame(
        {
            "timestamp": [1, 2],
            "close": [90.0, 91.0],
            "volume": [10.0, 20.0],
            "open_interest": [100.0, 110.0],
        }
    )
    main = pl.DataFrame(
        {
            "timestamp": [1],
            "close": [100.0],
            "volume": [30.0],
            "open_interest": [300.0],
        }
    )
    sub = pl.DataFrame(
        {
            "timestamp": [1, 2],
            "close": [80.0, 81.0],
            "volume": [60.0, 80.0],
            "open_interest": [600.0, 890.0],
        }
    )

    features = generate_main_sub_cross_month_features(
        current_contract="fu2603",
        current_bars=current,
        main_contract="fu2601",
        main_bars=main,
        sub_contract="fu2602",
        sub_bars=sub,
        current_role="other",
    )

    assert features.columns == ["timestamp"] + CROSS_MONTH_FEATURE_COLUMNS
    assert features["cm_contract_role_main"].to_list() == [0.0, 0.0]
    assert features["cm_contract_role_sub"].to_list() == [0.0, 0.0]
    assert features["cm_contract_role_other"].to_list() == [1.0, 1.0]

    assert np.isclose(features["cm_current_main_log_price_ratio"][0], np.log(0.9))
    assert np.isclose(
        features["cm_current_main_relative_price_spread"][0],
        (90.0 - 100.0) / 90.0,
    )
    assert np.isclose(
        features["cm_current_main_volume_share_current"][0],
        10.0 / (10.0 + 30.0),
    )
    assert np.isclose(
        features["cm_current_main_open_interest_share_current"][0],
        100.0 / (100.0 + 300.0),
    )

    assert np.isclose(features["cm_current_sub_log_price_ratio"][1], np.log(91.0 / 81.0))
    assert np.isclose(
        features["cm_current_sub_open_interest_share_current"][1],
        110.0 / (110.0 + 890.0),
    )
    assert np.isclose(features["cm_main_sub_log_price_ratio"][0], np.log(100.0 / 80.0))
    assert np.isclose(
        features["cm_main_sub_open_interest_share_sub"][0],
        600.0 / (300.0 + 600.0),
    )

    assert features["cm_current_main_log_price_ratio"][1] == 0.0
    assert features["cm_main_sub_log_price_ratio"][1] == 0.0


def test_resolve_previous_main_sub_role_avoids_same_day_role_leakage():
    roles = {
        "20260105": {
            "fu2601": "main",
            "fu2602": "sub",
            "fu2603": "other",
        },
        "20260106": {
            "fu2603": "main",
            "fu2601": "sub",
            "fu2602": "other",
        },
    }

    resolved = resolve_previous_main_sub_role(
        main_sub_roles=roles,
        trading_day="20260106",
        current_contract="fu2603",
    )

    assert resolved.role_trading_day == "20260105"
    assert resolved.current_role == "other"
    assert resolved.main_contract == "fu2601"
    assert resolved.sub_contract == "fu2602"


def test_resolve_previous_main_sub_role_returns_none_without_prior_roles():
    res = resolve_previous_main_sub_role(
        main_sub_roles={
            "20260105": {
                "fu2601": "main",
                "fu2602": "sub",
            }
        },
        trading_day="20260105",
        current_contract="fu2601",
    )
    assert res is None


def test_generate_delivery_month_sequence_features_sorts_by_delivery_month():
    current = pl.DataFrame(
        {
            "timestamp": [1, 2],
            "close": [100.0, 101.0],
            "volume": [10.0, 10.0],
            "open_interest": [100.0, 100.0],
        }
    )
    contract_bars = {
        "fu2612": pl.DataFrame(
            {
                "timestamp": [1, 2],
                "close": [130.0, 131.0],
                "volume": [30.0, 30.0],
                "open_interest": [300.0, 300.0],
            }
        ),
        "fu2605": pl.DataFrame(
            {
                "timestamp": [1, 2],
                "close": [110.0, 111.0],
                "volume": [20.0, 20.0],
                "open_interest": [200.0, 200.0],
            }
        ),
        "fu2601": pl.DataFrame(
            {
                "timestamp": [1, 2],
                "close": [100.0, 101.0],
                "volume": [10.0, 10.0],
                "open_interest": [100.0, 100.0],
            }
        ),
    }

    features = generate_delivery_month_sequence_features(
        current_bars=current,
        contract_bars=contract_bars,
    )

    assert features.columns == ["timestamp"] + CROSS_MONTH_FEATURE_COLUMNS
    assert np.isclose(features["cm_m1_m2_log_price_ratio"][0], np.log(100.0 / 110.0))
    assert np.isclose(features["cm_m2_m3_log_price_ratio"][0], np.log(110.0 / 130.0))
    assert np.isclose(
        features["cm_m1_m2_relative_price_spread"][0],
        (100.0 - 110.0) / 100.0,
    )
    assert np.isclose(
        features["cm_m2_m3_relative_price_spread"][0],
        (110.0 - 130.0) / 110.0,
    )
    assert np.isclose(
        features["cm_m1_m2_m3_butterfly_ratio"][0],
        (2.0 * 110.0 - 100.0 - 130.0) / 110.0,
    )
    assert np.isclose(
        features["cm_m1_m2_open_interest_share_m2"][0],
        200.0 / (100.0 + 200.0),
    )
    assert np.isclose(
        features["cm_m2_m3_open_interest_share_m3"][0],
        300.0 / (200.0 + 300.0),
    )


def test_generate_delivery_month_sequence_features_fills_aligned_gaps():
    current = pl.DataFrame(
        {
            "timestamp": [1, 2],
            "close": [100.0, 101.0],
            "volume": [10.0, 10.0],
            "open_interest": [100.0, 100.0],
        }
    )
    contract_bars = {
        "fu2601": current,
        "fu2605": pl.DataFrame(
            {
                "timestamp": [1],
                "close": [110.0],
                "volume": [20.0],
                "open_interest": [200.0],
            }
        ),
        "fu2612": pl.DataFrame(
            {
                "timestamp": [1, 2],
                "close": [130.0, 131.0],
                "volume": [30.0, 30.0],
                "open_interest": [300.0, 300.0],
            }
        ),
    }

    features = generate_delivery_month_sequence_features(
        current_bars=current,
        contract_bars=contract_bars,
    )

    assert features["cm_m1_m2_log_price_ratio"][1] == 0.0
    assert features["cm_m2_m3_log_price_ratio"][1] == 0.0
    assert features["cm_m1_m2_open_interest_share_m2"][1] == 0.0
    assert features["cm_m2_m3_open_interest_share_m3"][1] == 0.0


def test_generate_delivery_month_sequence_features_rejects_invalid_contract_set():
    current = pl.DataFrame(
        {
            "timestamp": [1],
            "close": [100.0],
            "volume": [10.0],
            "open_interest": [100.0],
        }
    )

    with pytest.raises(ValueError, match="at least 3"):
        generate_delivery_month_sequence_features(
            current_bars=current,
            contract_bars={"fu2601": current, "fu2605": current},
        )

    with pytest.raises(ValueError, match="delivery month"):
        generate_delivery_month_sequence_features(
            current_bars=current,
            contract_bars={"fu2601": current, "fu2605": current, "fuXX": current},
        )


def _write_base_feature(root, symbol, contract, target_freq, date, close, volume, oi):
    path = root / "BASE_FEATURE" / symbol / contract / target_freq
    path.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "timestamp": [1, 2],
            "close": [close, close + 1.0],
            "volume": [volume, volume],
            "open_interest": [oi, oi],
        }
    ).write_ipc(path / f"{date}.feather")


def test_write_cross_month_feature_for_day_uses_previous_roles_and_writes_file(tmp_path):
    data_root = tmp_path / "PREPROCESS_DATASET/commodity-futures"
    summary_path = data_root / "CONTINUOUS_RAW/fu/main_contract_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "symbol": "fu",
                "commodity_name": "燃料油",
                "start_date": "2026-01-05",
                "end_date": "2026-01-07",
                "selection_rule": "test",
                "main_sub_roles": {
                    "20260105": {
                        "fu2601": "main",
                        "fu2605": "sub",
                        "fu2612": "other",
                    },
                    "20260106": {
                        "fu2612": "main",
                        "fu2601": "sub",
                        "fu2605": "other",
                    },
                },
                "contracts": [
                    {
                        "contract": "fu2601",
                        "last_trading_day": "20260120",
                        "total_trading_day_count": 20,
                        "selected_months": ["2026-01"],
                        "trading_day_count": 1,
                        "trading_days": [
                            {
                                "trading_day": "20260105",
                                "date": "2026-01-05",
                                "source_file": "fu2601.csv",
                                "daily_volume": 100.0,
                            }
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_base_feature(data_root, "fu", "fu2601", "5min", "2026-01-06", 100.0, 10.0, 100.0)
    _write_base_feature(data_root, "fu", "fu2605", "5min", "2026-01-06", 110.0, 20.0, 200.0)
    _write_base_feature(data_root, "fu", "fu2612", "5min", "2026-01-06", 130.0, 30.0, 300.0)

    output_path = write_cross_month_feature_for_day(
        root_path=tmp_path,
        summary_path=summary_path,
        symbol="fu",
        contract="fu2612",
        target_freq="5min",
        date="2026-01-06",
    )

    assert output_path == (
        data_root / "CROSS_MONTH_FEATURE/fu/fu2612/5min/2026-01-06.feather"
    )
    out = pl.read_ipc(output_path)
    assert out.columns == ["timestamp"] + CROSS_MONTH_FEATURE_COLUMNS
    assert out["cm_contract_role_other"].to_list() == [1.0, 1.0]
    assert np.isclose(out["cm_current_main_log_price_ratio"][0], np.log(130.0 / 100.0))
    assert np.isclose(out["cm_current_sub_log_price_ratio"][0], np.log(130.0 / 110.0))
    assert np.isclose(out["cm_m1_m2_log_price_ratio"][0], np.log(100.0 / 110.0))


def test_write_cross_month_feature_for_day_fallback_when_no_prior_roles(tmp_path):
    data_root = tmp_path / "PREPROCESS_DATASET/commodity-futures"
    summary_path = data_root / "CONTINUOUS_RAW/fu/main_contract_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "symbol": "fu",
                "commodity_name": "燃料油",
                "start_date": "2026-01-05",
                "end_date": "2026-01-07",
                "selection_rule": "test",
                "main_sub_roles": {
                    "20260105": {
                        "fu2601": "main",
                        "fu2605": "sub",
                    },
                },
                "contracts": [
                    {
                        "contract": "fu2601",
                        "last_trading_day": "20260120",
                        "total_trading_day_count": 20,
                        "selected_months": ["2026-01"],
                        "trading_day_count": 1,
                        "trading_days": [
                            {
                                "trading_day": "20260105",
                                "date": "2026-01-05",
                                "source_file": "fu2601.csv",
                                "daily_volume": 100.0,
                            }
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_base_feature(data_root, "fu", "fu2601", "5min", "2026-01-05", 100.0, 10.0, 100.0)

    output_path = write_cross_month_feature_for_day(
        root_path=tmp_path,
        summary_path=summary_path,
        symbol="fu",
        contract="fu2601",
        target_freq="5min",
        date="2026-01-05",
    )

    out = pl.read_ipc(output_path)
    assert out.columns == ["timestamp"] + CROSS_MONTH_FEATURE_COLUMNS
    assert out["cm_contract_role_main"].to_list() == [0.0, 0.0]
    assert out["cm_contract_role_sub"].to_list() == [0.0, 0.0]
    assert out["cm_contract_role_other"].to_list() == [0.0, 0.0]
    assert out["cm_current_main_log_price_ratio"].to_list() == [0.0, 0.0]
    assert out["cm_current_sub_log_price_ratio"].to_list() == [0.0, 0.0]
