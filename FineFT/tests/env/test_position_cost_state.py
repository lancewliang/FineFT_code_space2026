import numpy as np
import pytest

from env.env_class.base_env import Base_Env
from env.env_class.futures_util import change_of_wallet


def _make_env(
    *,
    initial_position=0.0,
    max_holding_number=4,
    position_choices=5,
    ask_prices=None,
    ask_qtys=None,
    bid_prices=None,
    bid_qtys=None,
    markprices=None,
    buy_fee_rate=0.01,
    sell_fee_rate=0.01,
    allow_reverse_position=True,
):
    row_count = 7
    markprices = np.asarray(
        markprices if markprices is not None else [100.0] * row_count,
        dtype=float,
    )
    row_count = len(markprices)

    def _orderbook(values, default):
        if values is None:
            return np.tile(default, (row_count, 1))
        return np.asarray(values, dtype=float)

    ask_prices = _orderbook(ask_prices, [101.0, 102.0])
    ask_qtys = _orderbook(ask_qtys, [10.0, 10.0])
    bid_prices = _orderbook(bid_prices, [99.0, 98.0])
    bid_qtys = _orderbook(bid_qtys, [10.0, 10.0])
    timestamps = np.arange(row_count).astype("timedelta64[m]") + np.datetime64(
        "2025-01-01T00:00:00"
    )
    initial_margin = abs(markprices[0] * initial_position)

    return Base_Env(
        state_array=np.zeros((row_count, 3)),
        ask_prices_array=ask_prices,
        bid_prices_array=bid_prices,
        ask_qtys_array=ask_qtys,
        bid_qtys_array=bid_qtys,
        markprice_array=markprices,
        timestamp_array=timestamps,
        funding_rate_array=np.zeros(row_count),
        funding_timestamp_array=timestamps,
        max_holding_number=max_holding_number,
        position_choices=position_choices,
        leverage_choice=[1],
        long_estimated_rate=0.0,
        short_estimated_rate=0.0,
        commission_rate=0.0,
        buy_fee_rate=buy_fee_rate,
        sell_fee_rate=sell_fee_rate,
        initial_state=(100_000.0, initial_margin, 0.0, initial_position, 1),
        allow_reverse_position=allow_reverse_position,
    )


def test_reset_exposes_zero_cost_for_flat_position_without_changing_trading_info():
    env = _make_env()

    _, info = env.reset()

    assert env.current_holding_opening_price == 0.0
    assert env.current_holding_average_price == 0.0
    assert info["current_holding_opening_price"] == 0.0
    assert info["current_holding_average_price"] == 0.0
    assert info["trading_info"].shape == (4,)


def test_reset_uses_first_mark_price_for_nonzero_initial_position():
    env = _make_env(initial_position=-2.0, markprices=[123.0] * 7)

    _, info = env.reset()

    assert env.current_holding_opening_price == 123.0
    assert env.current_holding_average_price == 123.0
    assert info["current_holding_opening_price"] == 123.0
    assert info["current_holding_average_price"] == 123.0


def test_open_long_uses_actual_ask_fills_and_opening_fee():
    ask_prices = np.tile([101.0, 103.0], (7, 1))
    ask_qtys = np.tile([1.0, 10.0], (7, 1))
    env = _make_env(ask_prices=ask_prices, ask_qtys=ask_qtys)
    env.reset()

    _, _, _, info = env.step(env.env_map_position_leverage_to_action(2.0, 1))

    expected_price = (204.0 + 2.04) / 2.0
    assert env.current_holding_opening_price == pytest.approx(expected_price)
    assert env.current_holding_average_price == pytest.approx(expected_price)
    assert info["current_holding_opening_price"] == pytest.approx(expected_price)
    assert info["current_holding_average_price"] == pytest.approx(expected_price)


def test_open_short_uses_actual_bid_fills_and_opening_fee():
    bid_prices = np.tile([99.0, 97.0], (7, 1))
    bid_qtys = np.tile([1.0, 10.0], (7, 1))
    env = _make_env(bid_prices=bid_prices, bid_qtys=bid_qtys)
    env.reset()

    _, _, _, info = env.step(env.env_map_position_leverage_to_action(-2.0, 1))

    expected_price = (196.0 - 1.96) / 2.0
    assert env.current_holding_opening_price == pytest.approx(expected_price)
    assert env.current_holding_average_price == pytest.approx(expected_price)
    assert info["current_holding_opening_price"] == pytest.approx(expected_price)
    assert info["current_holding_average_price"] == pytest.approx(expected_price)


def test_add_reduce_and_close_follow_current_holding_cost_lifecycle():
    ask_prices = np.array(
        [[101.0, 103.0], [104.0, 106.0], *([[101.0, 103.0]] * 5)]
    )
    ask_qtys = np.tile([1.0, 10.0], (7, 1))
    env = _make_env(ask_prices=ask_prices, ask_qtys=ask_qtys)
    env.reset()

    env.step(env.env_map_position_leverage_to_action(2.0, 1))
    first_opening_price = (204.0 + 2.04) / 2.0
    env.step(env.env_map_position_leverage_to_action(4.0, 1))
    added_price = (210.0 + 2.10) / 2.0

    assert env.current_holding_opening_price == pytest.approx(first_opening_price)
    assert env.current_holding_average_price == pytest.approx(
        (2.0 * first_opening_price + 2.0 * added_price) / 4.0
    )

    opening_price_before_reduce = env.current_holding_opening_price
    average_price_before_reduce = env.current_holding_average_price
    env.step(env.env_map_position_leverage_to_action(2.0, 1))

    assert env.current_holding_opening_price == opening_price_before_reduce
    assert env.current_holding_average_price == average_price_before_reduce

    _, _, _, info = env.step(env.env_map_position_leverage_to_action(0.0, 1))

    assert env.current_holding_opening_price == 0.0
    assert env.current_holding_average_price == 0.0
    assert info["current_holding_opening_price"] == 0.0
    assert info["current_holding_average_price"] == 0.0


def test_partial_open_uses_actual_filled_quantity_instead_of_target_quantity():
    ask_prices = np.tile([101.0, 103.0], (7, 1))
    ask_qtys = np.tile([1.0, 1.0], (7, 1))
    env = _make_env(ask_prices=ask_prices, ask_qtys=ask_qtys)
    env.reset()

    _, _, _, info = env.step(env.env_map_position_leverage_to_action(4.0, 1))

    assert env.position == 2.0
    expected_price = (204.0 + 2.04) / 2.0
    assert env.current_holding_opening_price == pytest.approx(expected_price)
    assert env.current_holding_average_price == pytest.approx(expected_price)
    assert info["current_holding_average_price"] == pytest.approx(expected_price)


def test_reverse_resets_cost_from_only_the_new_direction_opening_leg():
    bid_prices = np.array(
        [[99.0, 97.0], [98.0, 96.0], *([[99.0, 97.0]] * 5)]
    )
    bid_qtys = np.tile([1.0, 10.0], (7, 1))
    env = _make_env(bid_prices=bid_prices, bid_qtys=bid_qtys)
    env.reset()
    env.step(env.env_map_position_leverage_to_action(2.0, 1))

    _, _, _, info = env.step(env.env_map_position_leverage_to_action(-2.0, 1))

    assert env.position == -2.0
    expected_new_opening_price = (194.0 - 1.94) / 2.0
    assert env.current_holding_opening_price == pytest.approx(
        expected_new_opening_price
    )
    assert env.current_holding_average_price == pytest.approx(
        expected_new_opening_price
    )
    assert info["current_holding_opening_price"] == pytest.approx(
        expected_new_opening_price
    )


def test_reverse_that_only_closes_the_old_position_clears_cost_state():
    env = _make_env(initial_position=2.0)
    env.initial_state = (10.0, 200.0, 0.0, 2.0, 1)
    env.reset()

    _, _, _, info = env.step(env.env_map_position_leverage_to_action(-4.0, 1))

    assert env.position == 0.0
    assert env.current_holding_opening_price == 0.0
    assert env.current_holding_average_price == 0.0
    assert info["current_holding_opening_price"] == 0.0
    assert info["current_holding_average_price"] == 0.0


def test_terminal_step_info_keeps_exposing_current_holding_cost():
    env = _make_env(markprices=[100.0, 100.0])
    env.reset()
    long_action = env.env_map_position_leverage_to_action(2.0, 1)
    env.step(long_action)

    _, _, terminal, info = env.step(long_action)

    assert terminal is True
    assert info["current_holding_opening_price"] == pytest.approx(102.01)
    assert info["current_holding_average_price"] == pytest.approx(102.01)


def test_wallet_change_keeps_legacy_six_values_and_names_opening_leg_metadata():
    result = change_of_wallet(
        markprice=100.0,
        ask_prices=np.array([101.0, 103.0]),
        ask_qtys=np.array([1.0, 10.0]),
        bid_prices=np.array([99.0, 97.0]),
        bid_qtys=np.array([1.0, 10.0]),
        long_estimated_rate=0.0,
        short_estimated_rate=0.0,
        commission_rate=0.0,
        buy_fee_rate=0.01,
        sell_fee_rate=0.01,
        previous_leverage=1,
        previous_position=0.0,
        previous_initial_margine=0.0,
        previous_unrealized_pnL=0.0,
        previous_wallet_balance=100_000.0,
        current_leverage=1,
        current_position=2.0,
    )

    legacy_values = tuple(result)
    assert len(result) == 6
    assert result[:6] == legacy_values
    assert result.opened_quantity == 2.0
    assert result.opened_value == 204.0
    assert result.opening_fee == pytest.approx(2.04)
