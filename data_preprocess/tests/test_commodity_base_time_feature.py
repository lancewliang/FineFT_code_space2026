import numpy as np
import polars as pl
import pytest
from datetime import datetime

from operator_futures.commodity.base_time_feature import (
    BASE_TIME_FEATURE_COLUMNS,
    generate_base_time_features,
)


def test_base_time_feature_columns_constant():
    expected = [
        "trading_minute_progress",
        "morning_session",
        "afternoon_session",
        "night_session",
        "is_opening_30m",
        "is_closing_30m",
        "is_session_first_bar",
        "is_session_last_bar",
        "contract_month_sin",
        "contract_month_cos",
        "contract_life_remaining_ratio",
        "prev_day_contract_role_tier",
    ]
    assert BASE_TIME_FEATURE_COLUMNS == expected


def test_generate_base_time_features_schema_and_values():
    timestamps = [
        datetime(2026, 1, 5, 9, 0, 0),    # Morning start
        datetime(2026, 1, 5, 9, 30, 0),   # Morning inside 30m
        datetime(2026, 1, 5, 10, 15, 0),  # Morning end (session 9:00-10:15)
        datetime(2026, 1, 5, 13, 30, 0),  # Afternoon start (session 13:30-15:00)
        datetime(2026, 1, 5, 14, 45, 0),  # Afternoon closing 30m
        datetime(2026, 1, 5, 21, 0, 0),   # Night start (session 21:00-23:00)
    ]
    base_df = pl.DataFrame({"timestamp": timestamps})

    res = generate_base_time_features(
        base_df=base_df,
        symbol="fu",
        contract="fu2605",
        trading_day="20260105",
        last_trading_day="20260116",
        total_trading_day_count=10,
    )

    assert res.columns == ["timestamp"] + BASE_TIME_FEATURE_COLUMNS

    # Month 05: sin(2*pi*5/12), cos(2*pi*5/12)
    expected_sin = np.sin(2 * np.pi * 5 / 12)
    expected_cos = np.cos(2 * np.pi * 5 / 12)
    assert np.isclose(res["contract_month_sin"][0], expected_sin)
    assert np.isclose(res["contract_month_cos"][0], expected_cos)

    # Trading minute progress
    # Session 9:00 - 10:15 (75 min)
    assert res["trading_minute_progress"][0] == 0.0
    assert np.isclose(res["trading_minute_progress"][1], 30.0 / 75.0)
    assert res["trading_minute_progress"][2] == 1.0

    # Morning, afternoon, night one-hot
    assert res["morning_session"][0] == 1.0
    assert res["afternoon_session"][0] == 0.0
    assert res["night_session"][0] == 0.0

    assert res["morning_session"][3] == 0.0
    assert res["afternoon_session"][3] == 1.0
    assert res["night_session"][3] == 0.0

    assert res["morning_session"][5] == 0.0
    assert res["afternoon_session"][5] == 0.0
    assert res["night_session"][5] == 1.0

    # Opening 30m / Closing 30m
    assert res["is_opening_30m"][0] == 1.0
    assert res["is_closing_30m"][0] == 0.0

    assert res["is_opening_30m"][1] == 1.0
    assert res["is_closing_30m"][1] == 0.0

    assert res["is_opening_30m"][2] == 0.0
    assert res["is_closing_30m"][2] == 1.0

    assert res["is_opening_30m"][4] == 0.0
    assert res["is_closing_30m"][4] == 1.0

    # Contract life remaining ratio
    # 2026-01-05 to 2026-01-16 (10 business days: 5,6,7,8,9, 12,13,14,15,16) -> remaining = 10
    # ratio = 10 / 10 = 1.0
    assert np.isclose(res["contract_life_remaining_ratio"][0], 1.0)


def test_generate_base_time_features_last_trading_day_ratio():
    timestamps = [datetime(2026, 1, 16, 9, 0, 0)]
    base_df = pl.DataFrame({"timestamp": timestamps})

    res = generate_base_time_features(
        base_df=base_df,
        symbol="fu",
        contract="fu2605",
        trading_day="20260116",
        last_trading_day="20260116",
        total_trading_day_count=10,
    )

    # On last trading day, remaining = 1, ratio = 1 / 10 = 0.1
    assert np.isclose(res["contract_life_remaining_ratio"][0], 0.1)


def test_session_boundary_features_mark_first_and_last_two_observed_bars():
    timestamps = [
        datetime(2026, 1, 5, 9, 0),
        datetime(2026, 1, 5, 9, 30),
        datetime(2026, 1, 5, 10, 0),
        datetime(2026, 1, 5, 10, 15),
        datetime(2026, 1, 5, 10, 30),
        datetime(2026, 1, 5, 11, 0),
        datetime(2026, 1, 5, 11, 30),
        datetime(2026, 1, 5, 13, 30),
        datetime(2026, 1, 5, 21, 0),
        datetime(2026, 1, 5, 21, 30),
        datetime(2026, 1, 5, 22, 0),
        datetime(2026, 1, 5, 22, 30),
        datetime(2026, 1, 5, 23, 0),
    ]
    base_df = pl.DataFrame({"timestamp": timestamps})

    result = generate_base_time_features(
        base_df=base_df,
        symbol="fu",
        contract="fu2605",
        trading_day="20260105",
        last_trading_day="20260116",
        total_trading_day_count=10,
    )

    assert result["is_session_first_bar"].to_list() == [
        1.0, 1.0, 0.0, 0.0,
        1.0, 1.0, 0.0,
        1.0,
        1.0, 1.0, 0.0, 0.0, 0.0,
    ]
    assert result["is_session_last_bar"].to_list() == [
        0.0, 0.0, 1.0, 1.0,
        0.0, 1.0, 1.0,
        1.0,
        0.0, 0.0, 0.0, 1.0, 1.0,
    ]


def test_generate_base_time_features_previous_day_role_tier():
    timestamps = [
        datetime(2026, 1, 5, 9, 0, 0),
        datetime(2026, 1, 5, 9, 30, 0),
    ]
    base_df = pl.DataFrame({"timestamp": timestamps})

    # 1. Main role -> 1.0
    roles_main = {
        "20260102": {"fu2605": "main", "fu2609": "sub"},
    }
    res_main = generate_base_time_features(
        base_df=base_df,
        symbol="fu",
        contract="fu2605",
        trading_day="20260105",
        last_trading_day="20260116",
        total_trading_day_count=10,
        main_sub_roles=roles_main,
    )
    assert res_main["prev_day_contract_role_tier"].to_list() == [1.0, 1.0]

    # 2. Sub role -> 0.5
    roles_sub = {
        "20260102": {"fu2601": "main", "fu2605": "sub"},
    }
    res_sub = generate_base_time_features(
        base_df=base_df,
        symbol="fu",
        contract="fu2605",
        trading_day="20260105",
        last_trading_day="20260116",
        total_trading_day_count=10,
        main_sub_roles=roles_sub,
    )
    assert res_sub["prev_day_contract_role_tier"].to_list() == [0.5, 0.5]

    # 3. Other role -> 0.0
    roles_other = {
        "20260102": {"fu2601": "main", "fu2609": "sub", "fu2605": "other"},
    }
    res_other = generate_base_time_features(
        base_df=base_df,
        symbol="fu",
        contract="fu2605",
        trading_day="20260105",
        last_trading_day="20260116",
        total_trading_day_count=10,
        main_sub_roles=roles_other,
    )
    assert res_other["prev_day_contract_role_tier"].to_list() == [0.0, 0.0]

    # 4. Unrecorded contract in roles -> 0.0
    roles_unrecorded = {
        "20260102": {"fu2601": "main", "fu2609": "sub"},
    }
    res_unrecorded = generate_base_time_features(
        base_df=base_df,
        symbol="fu",
        contract="fu2605",
        trading_day="20260105",
        last_trading_day="20260116",
        total_trading_day_count=10,
        main_sub_roles=roles_unrecorded,
    )
    assert res_unrecorded["prev_day_contract_role_tier"].to_list() == [0.0, 0.0]

    # 5. Cold start / no prior days -> 0.0
    res_cold_start = generate_base_time_features(
        base_df=base_df,
        symbol="fu",
        contract="fu2605",
        trading_day="20260105",
        last_trading_day="20260116",
        total_trading_day_count=10,
        main_sub_roles=None,
    )
    assert res_cold_start["prev_day_contract_role_tier"].to_list() == [0.0, 0.0]


def test_generate_and_write_base_time_feature_with_summary_json(tmp_path):
    import json
    from operator_futures.commodity.base_time_feature import generate_and_write_base_time_feature

    # Setup base feature feather
    timestamps = [datetime(2026, 1, 5, 9, 0, 0)]
    base_df = pl.DataFrame({"timestamp": timestamps})
    base_path = tmp_path / "base.feather"
    base_df.write_ipc(base_path)

    # Setup dummy summary JSON with main_sub_roles
    summary_data = {
        "symbol": "fu",
        "commodity_name": "燃料油",
        "start_date": "2026-01-01",
        "end_date": "2026-01-31",
        "contracts": [
            {
                "contract": "fu2605",
                "trading_day_count": 1,
                "selected_months": ["2026-01"],
                "last_trading_day": "20260116",
                "total_trading_day_count": 10,
                "trading_days": [
                    {
                        "trading_day": "20260105",
                        "date": "2026-01-05",
                        "source_file": str(tmp_path / "dummy.csv"),
                        "daily_volume": 1000.0,
                    }
                ],
            }
        ],
        "main_sub_roles": {
            "20260102": {
                "fu2605": "main",
                "fu2609": "sub",
            }
        },
    }
    summary_file = tmp_path / "main_contract_summary.json"
    summary_file.write_text(json.dumps(summary_data), encoding="utf-8")

    out_path = generate_and_write_base_time_feature(
        base_feature_path=base_path,
        output_root=tmp_path / "output",
        symbol="fu",
        contract="fu2605",
        target_freq="5min",
        date="2026-01-05",
        last_trading_day="20260116",
        total_trading_day_count=10,
        summary_path=summary_file,
    )

    written_df = pl.read_ipc(out_path)
    assert "prev_day_contract_role_tier" in written_df.columns
    assert written_df["prev_day_contract_role_tier"].to_list() == [1.0]
