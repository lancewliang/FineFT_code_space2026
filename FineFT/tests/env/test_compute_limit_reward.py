import math

import pytest

from env.env_class.futures_util import compute_limit_reward


# Default limit config used across cases (matches Base_Env defaults)
HOLD = 1.0
STAY = 0.5
REV = 1.5
THRESH = 0.003


def _call(
    old_position,
    new_position,
    *,
    is_limit_up=False,
    is_limit_down=False,
    limit_up_ask_depth_ratio_5=0.0,
    limit_down_bid_depth_ratio_5=0.0,
    upper_limit_price=None,
    lower_limit_price=None,
    markprice=100.0,
    enable_limit_reward=True,
    **overrides,
):
    kwargs = dict(
        limit_hold_bonus=HOLD,
        limit_stay_bonus=STAY,
        limit_reverse_penalty=REV,
        near_limit_threshold=THRESH,
    )
    kwargs.update(overrides)
    return compute_limit_reward(
        old_position,
        new_position,
        is_limit_up,
        is_limit_down,
        limit_up_ask_depth_ratio_5,
        limit_down_bid_depth_ratio_5,
        upper_limit_price,
        lower_limit_price,
        markprice,
        enable_limit_reward,
        **kwargs,
    )


def test_disabled_returns_zero():
    # Even with limit up + long, disabled flag forces zero
    assert _call(0, 4, is_limit_up=True, enable_limit_reward=False) == 0.0


def test_no_limit_active_returns_zero():
    assert _call(0, 4) == 0.0


def test_limit_up_open_long_positive():
    # Open long from flat in limit up -> +hold_bonus * intensity(1.0)
    assert _call(0, 4, is_limit_up=True) == pytest.approx(HOLD)


def test_limit_up_reverse_to_short_negative():
    # Long -> short in limit up -> -reverse_penalty * intensity(1.0)
    assert _call(4, -4, is_limit_up=True) == pytest.approx(-REV)


def test_limit_down_open_short_positive():
    # Open short from flat in limit down -> +hold_bonus * intensity(1.0)
    assert _call(0, -4, is_limit_down=True) == pytest.approx(HOLD)


def test_limit_down_reverse_to_long_negative():
    # Short -> long in limit down -> -reverse_penalty * intensity(1.0)
    assert _call(-4, 4, is_limit_down=True) == pytest.approx(-REV)


def test_depth_ratio_scaling_open_long():
    # Same action (0->4) under limit-up depth ratio: 0.4 vs 1.0 -> 0.4 < 1.0
    r_low = _call(0, 4, limit_up_ask_depth_ratio_5=0.4, upper_limit_price=110.0)
    r_high = _call(0, 4, limit_up_ask_depth_ratio_5=1.0, upper_limit_price=110.0)
    assert r_low == pytest.approx(HOLD * 0.4)
    assert r_high == pytest.approx(HOLD * 1.0)
    assert r_high > r_low


def test_near_limit_up_scales_by_distance():
    # price 99.5, upper 100, threshold 0.01 -> halfway into the near-limit band
    # distance = (100-99.5)/100 = 0.005; intensity = 1 - 0.005/0.01 = 0.5
    r = _call(
        0,
        4,
        upper_limit_price=100.0,
        markprice=99.5,
        near_limit_threshold=0.01,
    )
    assert r == pytest.approx(HOLD * 0.5)


def test_near_limit_down_scales_by_distance():
    # price 100.5, lower 100, threshold 0.01 -> intensity 0.5; open short rewarded
    r = _call(
        0,
        -4,
        lower_limit_price=100.0,
        markprice=100.5,
        near_limit_threshold=0.01,
    )
    assert r == pytest.approx(HOLD * 0.5)


def test_exact_limit_takes_precedence_over_near_limit():
    # is_limit_up True -> intensity 1.0 regardless of being inside near-limit band
    r = _call(
        0,
        4,
        is_limit_up=True,
        upper_limit_price=100.0,
        markprice=99.5,
        near_limit_threshold=0.01,
    )
    assert r == pytest.approx(HOLD * 1.0)


def test_hold_same_direction_unchanged_gets_stay_bonus():
    # Long 4 -> 4 (no move) in limit up: hold_bonus + stay_bonus
    r = _call(4, 4, is_limit_up=True)
    assert r == pytest.approx(HOLD + STAY)


def test_reduce_same_direction_penalized():
    # Long 4 -> 2 in limit up: small negative (-stay_bonus)
    r = _call(4, 2, is_limit_up=True)
    assert r == pytest.approx(-STAY)


def test_close_same_direction_penalized():
    # Long 4 -> 0 in limit up: small negative (-stay_bonus)
    r = _call(4, 0, is_limit_up=True)
    assert r == pytest.approx(-STAY)


def test_flat_unchanged_no_bonus():
    # 0 -> 0 in limit up: no same-direction position to encourage
    r = _call(0, 0, is_limit_up=True)
    assert r == 0.0
