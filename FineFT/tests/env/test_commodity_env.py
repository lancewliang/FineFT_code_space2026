import pandas as pd
import pytest

from env.env_class.futures_util import change_of_wallet
from env.env_initiate.commodity_initiate import initiate_commodity_env


def _df():
    rows = []
    for t, price in enumerate([2600.0, 2601.0]):
        row = {
            "timestamp": pd.Timestamp("2023-01-03 21:00:00")
            + pd.Timedelta(minutes=5 * t),
            "mark_price": price,
            "feature_a": float(t),
        }
        for level in range(1, 6):
            row[f"ask{level}_price"] = price + level
            row[f"ask{level}_size"] = 10
            row[f"bid{level}_price"] = price - level
            row[f"bid{level}_size"] = 10
        rows.append(row)
    return pd.DataFrame(rows)


def test_commodity_env_reset_has_no_funding_countdown():
    env = initiate_commodity_env(
        _df(), ["feature_a"], max_holding_number=1, position_choices=3
    )
    state, info = env.reset()

    assert state.shape == (1,)
    assert "funding_count_down_hour" not in info
    assert "funding_count_down_minute" not in info
    assert len(info["ask_qyts"]) == 5
    assert len(info["bid_qyts"]) == 5


def test_commodity_env_step_uses_configured_fees_and_no_funding():
    env = initiate_commodity_env(
        _df(),
        ["feature_a"],
        max_holding_number=1,
        position_choices=3,
        buy_fee_rate=0.0001,
        sell_fee_rate=0.0003,
    )
    env.reset()
    _, _, _, info = env.step(
        env.env_map_position_leverage_to_action(1, env.leverage_choices[0])
    )

    assert env.buy_fee_rate == 0.0001
    assert env.sell_fee_rate == 0.0003
    assert "funding_count_down_hour" not in info


def test_wallet_change_can_use_buy_and_sell_fee_rates():
    opened = change_of_wallet(
        markprice=100.0,
        ask_prices=[101.0],
        ask_qtys=[5.0],
        bid_prices=[99.0],
        bid_qtys=[5.0],
        long_estimated_rate=0.0,
        short_estimated_rate=0.0,
        commission_rate=0.0,
        buy_fee_rate=0.1,
        sell_fee_rate=0.3,
        previous_leverage=5,
        previous_position=0,
        previous_initial_margine=0,
        previous_unrealized_pnL=0,
        previous_wallet_balance=1000.0,
        current_leverage=5,
        current_position=1,
    )

    assert opened[4] == 1000.0 - 101.0 * 0.1

    closed = change_of_wallet(
        markprice=100.0,
        ask_prices=[101.0],
        ask_qtys=[5.0],
        bid_prices=[99.0],
        bid_qtys=[5.0],
        long_estimated_rate=0.0,
        short_estimated_rate=0.0,
        commission_rate=0.0,
        buy_fee_rate=0.1,
        sell_fee_rate=0.3,
        previous_leverage=5,
        previous_position=1,
        previous_initial_margine=20.0,
        previous_unrealized_pnL=-1.0,
        previous_wallet_balance=opened[4],
        current_leverage=5,
        current_position=0,
    )

    expected_wallet = opened[4] - 1.0 + 99.0 - 100.0 - 99.0 * 0.3
    assert closed[4] == pytest.approx(expected_wallet)
