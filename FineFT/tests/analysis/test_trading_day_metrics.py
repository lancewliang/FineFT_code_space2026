import numpy as np
import pandas as pd
import pytest

from analysis.calculate_metric.calculate_metric import (
    calculate_metric,
    assign_trading_days_from_timestamps,
)


def test_assign_trading_days_from_timestamps():
    """Verify night sessions (>= 20:00) map to the next trading day, day sessions map to calendar date."""
    timestamps = [
        "2024-07-23 09:00:00",  # Day session -> 2024-07-23
        "2024-07-23 15:00:00",  # Day session -> 2024-07-23
        "2024-07-23 21:00:00",  # Night session -> 2024-07-24
        "2024-07-23 23:00:00",  # Night session -> 2024-07-24
        "2024-07-24 09:00:00",  # Day session -> 2024-07-24
        "2024-07-26 15:00:00",  # Friday day session -> 2024-07-26
        "2024-07-26 21:00:00",  # Friday night session -> Monday 2024-07-29
        "2024-07-29 09:00:00",  # Monday day session -> 2024-07-29
        "2025-06-27 21:00:00",  # Final Friday night session at dataset end -> Monday 2025-06-30
    ]
    s = pd.Series(timestamps)
    tds = assign_trading_days_from_timestamps(s)

    assert tds.iloc[0] == "2024-07-23"
    assert tds.iloc[1] == "2024-07-23"
    assert tds.iloc[2] == "2024-07-24"
    assert tds.iloc[3] == "2024-07-24"
    assert tds.iloc[4] == "2024-07-24"
    assert tds.iloc[5] == "2024-07-26"
    assert tds.iloc[6] == "2024-07-29"
    assert tds.iloc[7] == "2024-07-29"
    assert tds.iloc[8] == "2025-06-30"


def test_calculate_metric_with_trading_days():
    """Verify calculate_metric aggregates by trading_day and annualizes using 242 trading days."""
    capital = 12000.0
    # Create 3 trading days of steps
    td1 = ["2024-07-23"] * 10
    td2 = ["2024-07-24"] * 10
    td3 = ["2024-07-25"] * 10
    trading_days = td1 + td2 + td3

    # Daily PnL: Day 1 = +120, Day 2 = -60, Day 3 = +240
    step_rewards = [12.0] * 10 + [-6.0] * 10 + [24.0] * 10
    rewards_with_init = np.r_[0.0, step_rewards]

    tr, daily_vol, mdd, downside_dev, annual_sr, daily_cr, daily_sor = calculate_metric(
        capital,
        rewards_with_init,
        trading_days=trading_days,
        annual_trading_days=242,
    )

    total_net = sum(step_rewards)
    assert tr == pytest.approx(total_net / capital)
    assert tr == pytest.approx(300.0 / 12000.0)

    # Equities at end of days:
    # Day 0: 12000
    # Day 1: 12120 -> return = 120 / 12000 = 0.0100
    # Day 2: 12060 -> return = -60 / 12120 = -0.004950495
    # Day 3: 12300 -> return = 240 / 12060 = 0.0199004975
    daily_rets = np.array([120.0 / 12000.0, -60.0 / 12120.0, 240.0 / 12060.0])
    expected_vol = np.std(daily_rets)
    assert daily_vol == pytest.approx(expected_vol)

    expected_sr = (np.mean(daily_rets) / expected_vol) * np.sqrt(242)
    assert annual_sr == pytest.approx(expected_sr)


def test_calculate_metric_backward_compatibility():
    """Verify calculate_metric without trading_days continues to work using freq."""
    capital = 12000.0
    rewards = np.zeros(25)
    tr, daily_vol, mdd, downside_dev, annual_sr, daily_cr, daily_sor = calculate_metric(
        capital, rewards, freq=12
    )
    assert tr == 0.0
    assert mdd == 0.0
    assert annual_sr == 0.0
