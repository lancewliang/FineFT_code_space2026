from datetime import datetime

import numpy as np
import polars as pl
import pytest

from operator_futures.commodity.mixed_frequency_feature import (
    MIXED_FREQUENCY_FEATURE_COLUMNS,
    combine_daily_weekly_mixed_frequency_features,
    write_mixed_frequency_feature_for_day,
)
from operator_futures.commodity.daily_base_feature import (
    generate_daily_base_features,
    write_daily_base_feature_for_range,
)
from operator_futures.commodity.daily_mixed_frequency_feature import (
    generate_daily_mixed_frequency_features_from_base,
    write_daily_mixed_frequency_feature_for_day,
)
from operator_futures.commodity.weekly_base_feature import (
    generate_weekly_base_features,
    write_weekly_base_feature_for_range,
)
from operator_futures.commodity.weekly_mixed_frequency_feature import (
    generate_weekly_mixed_frequency_features_from_base,
    write_weekly_mixed_frequency_feature_for_day,
)


def _bars_for_day(trading_day: str, first_ts: datetime, open_base: float) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [
                first_ts,
                first_ts.replace(hour=first_ts.hour + 1),
            ],
            "trading_day": [trading_day, trading_day],
            "open": [open_base, open_base + 3.0],
            "high": [open_base + 8.0, open_base + 10.0],
            "low": [open_base - 5.0, open_base + 1.0],
            "close": [open_base + 3.0, open_base + 5.0],
            "volume": [10.0, 20.0],
            "tradeval": [1000.0, 2200.0],
            "open_interest": [100.0, 110.0],
        }
    )


def test_generate_mixed_frequency_features_uses_previous_trading_day():
    frame = pl.concat(
        [
            _bars_for_day("2026-01-05", datetime(2026, 1, 5, 9), 100.0),
            _bars_for_day("2026-01-06", datetime(2026, 1, 6, 9), 200.0),
        ]
    )

    daily_feature = generate_daily_mixed_frequency_features_from_base(
        target_bars=frame.select("timestamp", "trading_day"),
        daily_base=generate_daily_base_features(frame),
    )
    weekly_feature = generate_weekly_mixed_frequency_features_from_base(
        target_bars=frame.select("timestamp", "trading_day"),
        weekly_base=generate_weekly_base_features(frame),
    )
    result = combine_daily_weekly_mixed_frequency_features(
        daily_feature=daily_feature,
        weekly_feature=weekly_feature,
    )

    assert result.columns == ["timestamp", *MIXED_FREQUENCY_FEATURE_COLUMNS]
    first_day_rows = result.head(2)
    assert first_day_rows.select(MIXED_FREQUENCY_FEATURE_COLUMNS).to_numpy().sum() == 0.0

    row = result.row(2, named=True)
    assert np.isclose(row["prev_day_return"], 0.05)
    assert np.isclose(row["prev_day_range_pct"], 15.0 / 105.0)
    assert np.isclose(row["prev_day_body_pct"], 0.05)
    assert np.isclose(row["prev_day_upper_shadow_pct"], 0.05)
    assert np.isclose(row["prev_day_lower_shadow_pct"], 0.05)
    assert row["prev_day_volume"] == 30.0
    assert row["prev_day_tradeval"] == 3200.0
    assert np.isclose(row["prev_day_open_interest_change"], 0.1)
    assert np.isclose(row["prev_day_turnover_rate"], 30.0 / 110.0)


def test_generate_mixed_frequency_features_uses_previous_calendar_week():
    previous_week = pl.concat(
        [
            _bars_for_day("2026-01-05", datetime(2026, 1, 5, 9), 100.0),
            _bars_for_day("2026-01-06", datetime(2026, 1, 6, 9), 120.0),
        ]
    )
    current_week = pl.concat(
        [
            _bars_for_day("2026-01-12", datetime(2026, 1, 12, 9), 200.0),
            _bars_for_day("2026-01-14", datetime(2026, 1, 14, 9), 300.0),
        ]
    )

    frame = pl.concat([previous_week, current_week])
    daily_feature = generate_daily_mixed_frequency_features_from_base(
        target_bars=frame.select("timestamp", "trading_day"),
        daily_base=generate_daily_base_features(frame),
    )
    weekly_feature = generate_weekly_mixed_frequency_features_from_base(
        target_bars=frame.select("timestamp", "trading_day"),
        weekly_base=generate_weekly_base_features(frame),
    )
    result = combine_daily_weekly_mixed_frequency_features(
        daily_feature=daily_feature,
        weekly_feature=weekly_feature,
    )

    monday = result.row(4, named=True)
    wednesday = result.row(6, named=True)
    for row in (monday, wednesday):
        assert np.isclose(row["prev_week_return"], 25.0 / 100.0)
        assert np.isclose(row["prev_week_range_pct"], 35.0 / 125.0)
        assert row["prev_week_volume"] == 60.0
        assert row["prev_week_tradeval"] == 6400.0
        assert np.isclose(row["prev_week_open_interest_change"], 0.1)
        assert np.isclose(row["prev_week_turnover_rate"], 60.0 / 110.0)


def test_daily_and_weekly_base_outputs_one_row_per_day_and_week():
    frame = pl.concat(
        [
            _bars_for_day("2026-01-05", datetime(2026, 1, 5, 9), 100.0),
            _bars_for_day("2026-01-06", datetime(2026, 1, 6, 9), 120.0),
            _bars_for_day("2026-01-12", datetime(2026, 1, 12, 9), 200.0),
        ]
    )

    daily_base = generate_daily_base_features(frame)
    weekly_base = generate_weekly_base_features(frame)

    assert daily_base.height == 3
    assert daily_base["trading_day"].to_list() == [
        "2026-01-05",
        "2026-01-06",
        "2026-01-12",
    ]
    assert weekly_base.height == 2
    assert weekly_base["calendar_week"].to_list() == ["2026-W02", "2026-W03"]
    assert weekly_base["week_start"].to_list() == ["2026-01-05", "2026-01-12"]


def test_mixed_frequency_features_rejects_missing_base_columns():
    target_bars = pl.DataFrame({"timestamp": [1], "trading_day": ["2026-01-05"]})
    with pytest.raises(ValueError, match="daily Mixed-frequency Base Data"):
        generate_daily_mixed_frequency_features_from_base(
            target_bars=target_bars,
            daily_base=pl.DataFrame({"trading_day": ["2026-01-04"]}),
        )


def test_write_mixed_frequency_feature_for_day_uses_base_feature_history(tmp_path):
    base_dir = (
        tmp_path
        / "PREPROCESS_DATASET"
        / "commodity-futures"
        / "BASE_FEATURE"
        / "fu"
        / "fu2601"
        / "5min"
    )
    base_dir.mkdir(parents=True)
    _bars_for_day("2026-01-05", datetime(2026, 1, 5, 9), 100.0).drop(
        "trading_day"
    ).write_ipc(base_dir / "2026-01-05.feather")
    _bars_for_day("2026-01-06", datetime(2026, 1, 6, 9), 200.0).drop(
        "trading_day"
    ).write_ipc(base_dir / "2026-01-06.feather")

    daily_path = write_daily_base_feature_for_range(
        root_path=tmp_path,
        symbol="fu",
        contract="fu2601",
        target_freq="5min",
        start_date="2026-01-05",
        end_date="2026-01-07",
    )
    weekly_path = write_weekly_base_feature_for_range(
        root_path=tmp_path,
        symbol="fu",
        contract="fu2601",
        target_freq="5min",
        start_date="2026-01-05",
        end_date="2026-01-07",
    )
    assert pl.read_ipc(daily_path).height == 2
    assert pl.read_ipc(weekly_path).height == 1

    daily_feature_path = write_daily_mixed_frequency_feature_for_day(
        root_path=tmp_path,
        symbol="fu",
        contract="fu2601",
        target_freq="5min",
        date="2026-01-06",
        start_date="2026-01-05",
        end_date="2026-01-07",
    )
    weekly_feature_path = write_weekly_mixed_frequency_feature_for_day(
        root_path=tmp_path,
        symbol="fu",
        contract="fu2601",
        target_freq="5min",
        date="2026-01-06",
        start_date="2026-01-05",
        end_date="2026-01-07",
    )
    assert pl.read_ipc(daily_feature_path)["prev_day_volume"].to_list() == [30.0, 30.0]
    assert pl.read_ipc(weekly_feature_path)["prev_week_return"].to_list() == [0.0, 0.0]

    out_path = write_mixed_frequency_feature_for_day(
        root_path=tmp_path,
        symbol="fu",
        contract="fu2601",
        target_freq="5min",
        date="2026-01-06",
    )

    output = pl.read_ipc(out_path)
    assert output["prev_day_volume"].to_list() == [30.0, 30.0]
    assert output["prev_week_return"].to_list() == [0.0, 0.0]
