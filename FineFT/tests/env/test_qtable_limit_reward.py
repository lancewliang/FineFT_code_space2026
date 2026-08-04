import numpy as np
import pandas as pd
import pytest

from env.env_class.futures_util import create_optimal_q_table, create_optimal_q_table_from_df


def _base_arrays(n=3):
    # constant markprice + symmetric book -> PnL from price is zero, so the
    # with-limit vs without-limit difference isolates the limit reward term.
    ask_prices = np.tile([100.0, 101.0], (n, 1))
    bid_prices = np.tile([99.0, 98.0], (n, 1))
    ask_qtys = np.tile([10.0, 10.0], (n, 1))
    bid_qtys = np.tile([10.0, 10.0], (n, 1))
    markprices = np.full(n, 100.0)
    timestamps = np.arange(1, n + 1)
    funding_rates = np.zeros(n)
    funding_timestamps = np.zeros(n)
    return ask_prices, bid_prices, ask_qtys, bid_qtys, markprices, timestamps, funding_rates, funding_timestamps


def _common_kwargs():
    return dict(
        max_holding_number=2,
        position_choices=3,
        leverage_choice=[1],
        commission_rate=0.0,
        long_estimated_rate=0.0,
        short_estimated_rate=0.0,
        gamma=1.0,
    )


def test_create_optimal_q_table_limit_reward_incorporated():
    n = 3
    arrays = _base_arrays(n)
    # action map for position_choices=3, leverage=[1], max_holding=2:
    #   0 -> short(-2), 1 -> flat(0), 2 -> long(2)
    # limit up active only at index 1 (current_timestamp_index = -2, the first
    # computed row; its future row -1 stays zero, so no gamma propagation)
    is_limit_up = np.array([False, True, False])
    is_limit_down = np.zeros(n, dtype=bool)
    depth_up = np.zeros(n)
    depth_down = np.zeros(n)
    upper = np.full(n, 110.0)
    lower = np.full(n, 90.0)

    q_no_limit = create_optimal_q_table(*arrays, **_common_kwargs())

    q_limit = create_optimal_q_table(
        *arrays,
        **_common_kwargs(),
        is_limit_up_array=is_limit_up,
        is_limit_down_array=is_limit_down,
        limit_up_ask_depth_ratio_5_array=depth_up,
        limit_down_bid_depth_ratio_5_array=depth_down,
        upper_limit_prices_array=upper,
        lower_limit_prices_array=lower,
        enable_limit_reward=True,
        limit_hold_bonus=1.0,
        limit_stay_bonus=0.5,
        limit_reverse_penalty=1.5,
        near_limit_threshold=0.003,
    )

    # current_action=1 (flat, pos 0) at index -2 where limit up is active
    diff_open_long = q_limit[-2, 1, 2] - q_no_limit[-2, 1, 2]
    diff_open_short = q_limit[-2, 1, 0] - q_no_limit[-2, 1, 0]
    assert diff_open_long == pytest.approx(1.0)   # 0 -> long : +hold_bonus
    assert diff_open_short == pytest.approx(-1.5)  # 0 -> short : -reverse_penalty


def test_create_optimal_q_table_limit_disabled_by_default():
    # No limit params -> behaves identically to a table built without shaping
    arrays = _base_arrays(3)
    q_default = create_optimal_q_table(*arrays, **_common_kwargs())
    q_explicit_off = create_optimal_q_table(
        *arrays, **_common_kwargs(), enable_limit_reward=False
    )
    np.testing.assert_array_equal(q_default, q_explicit_off)


def _make_limit_df(n=3, limit_up_idx=1):
    cols = {
        "mark_price": np.full(n, 100.0),
        "timestamp": np.arange(1, n + 1),
        "funding_rate": np.zeros(n),
        "funding_timestamp": np.zeros(n),
    }
    for i in (1, 2):  # order_book_depth=2 is enough
        cols[f"ask{i}_price"] = 100.0
        cols[f"ask{i}_size"] = 10.0
        cols[f"bid{i}_price"] = 100.0
        cols[f"bid{i}_size"] = 10.0
    is_up = np.zeros(n, dtype=bool)
    is_up[limit_up_idx] = True
    cols["limit_up_single_sided_ratio"] = is_up.astype(float)
    cols["limit_down_single_sided_ratio"] = np.zeros(n)
    cols["limit_up_ask_depth_ratio_5"] = np.zeros(n)
    cols["limit_down_bid_depth_ratio_5"] = np.zeros(n)
    cols["UpperLimitPrice"] = np.full(n, 110.0)
    cols["LowerLimitPrice"] = np.full(n, 90.0)
    return pd.DataFrame(cols)


def test_create_optimal_q_table_from_df_passes_limit_reward_through():
    df = _make_limit_df(n=3, limit_up_idx=1)
    common = dict(
        max_holding_number=2,
        position_choices=3,
        leverage_choice=[1],
        commission_rate=0.0,
        long_estimated_rate=0.0,
        short_estimated_rate=0.0,
        gamma=1.0,
        order_book_depth=2,
    )
    q_no_limit = create_optimal_q_table_from_df(df, **common)
    q_limit = create_optimal_q_table_from_df(
        df, **common, enable_limit_reward=True, limit_hold_bonus=1.0,
        limit_reverse_penalty=1.5,
    )
    # action 1 = flat(0); at index -2 (=1) limit up is active
    assert (q_limit[-2, 1, 2] - q_no_limit[-2, 1, 2]) == pytest.approx(1.0)
    assert (q_limit[-2, 1, 0] - q_no_limit[-2, 1, 0]) == pytest.approx(-1.5)
