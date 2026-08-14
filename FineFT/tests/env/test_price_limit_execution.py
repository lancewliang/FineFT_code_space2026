import numpy as np
import pandas as pd
import pytest

from env.env_initiate.base_initiate import initiate_base_env


def _make_env(
    *,
    initial_position=0.0,
    is_limit_up=False,
    is_limit_down=False,
    limit_up_ratio=0.0,
    limit_down_ratio=0.0,
    upper_limit_price=110.0,
    lower_limit_price=90.0,
):
    row_count = 6
    data = {
        "mark_price": np.full(row_count, 100.0),
        "timestamp": pd.date_range("2026-01-01", periods=row_count, freq="1min"),
        "funding_rate": np.zeros(row_count),
        "funding_timestamp": pd.date_range(
            "2026-01-01", periods=row_count, freq="1min"
        ),
        "feature": np.zeros(row_count),
        "limit_up_single_sided_ratio": np.full(
            row_count, max(float(is_limit_up), limit_up_ratio)
        ),
        "limit_down_single_sided_ratio": np.full(
            row_count, max(float(is_limit_down), limit_down_ratio)
        ),
        "limit_up_ask_depth_ratio_5": np.zeros(row_count),
        "limit_down_bid_depth_ratio_5": np.zeros(row_count),
        "UpperLimitPrice": np.full(row_count, upper_limit_price),
        "LowerLimitPrice": np.full(row_count, lower_limit_price),
    }
    for level in range(1, 26):
        data[f"ask{level}_price"] = np.full(row_count, 100.0 + level)
        data[f"ask{level}_size"] = np.full(row_count, 10.0)
        data[f"bid{level}_price"] = np.full(row_count, 100.0 - level)
        data[f"bid{level}_size"] = np.full(row_count, 10.0)

    initial_margin = abs(initial_position * 100.0)
    return initiate_base_env(
        pd.DataFrame(data),
        feature_list=["feature"],
        max_holding_number=4,
        position_choices=5,
        leverage_choice=[1],
        long_estimated_rate=0.0,
        short_estimated_rate=0.0,
        commission_rate=0.01,
        initial_state=(100_000.0, initial_margin, 0.0, initial_position, 1),
        allow_reverse_position=True,
    )


def _action(env, position):
    return env.env_map_position_leverage_to_action(position, 1)


def _assert_no_execution(info):
    assert info["commission_fee_step"] == 0.0
    assert info["realized_pnl_step"] == 0.0
    assert info["slippage_step"] == 0.0


def test_limit_down_blocks_long_reduction_in_mask_and_direct_step():
    env = _make_env(initial_position=2.0, is_limit_down=True)

    _, reset_info = env.reset()

    assert reset_info["avaliable_action"][_action(env, 2.0)] == 1
    assert reset_info["avaliable_action"][_action(env, 4.0)] == 1
    assert reset_info["avaliable_action"][_action(env, 0.0)] == 0
    assert reset_info["avaliable_action"][_action(env, -2.0)] == 0

    _, _, _, step_info = env.step(_action(env, 0.0))

    assert env.position == 2.0
    _assert_no_execution(step_info)
    assert step_info["avaliable_action"][_action(env, 0.0)] == 0


def test_limit_up_blocks_short_reduction_in_mask_and_direct_step():
    env = _make_env(initial_position=-2.0, is_limit_up=True)

    _, reset_info = env.reset()

    assert reset_info["avaliable_action"][_action(env, -2.0)] == 1
    assert reset_info["avaliable_action"][_action(env, -4.0)] == 1
    assert reset_info["avaliable_action"][_action(env, 0.0)] == 0
    assert reset_info["avaliable_action"][_action(env, 2.0)] == 0

    _, _, _, step_info = env.step(_action(env, 0.0))

    assert env.position == -2.0
    _assert_no_execution(step_info)
    assert step_info["avaliable_action"][_action(env, 0.0)] == 0


def test_limit_down_allows_buying_to_reduce_a_short_position():
    env = _make_env(initial_position=-2.0, is_limit_down=True)

    _, reset_info = env.reset()

    assert reset_info["avaliable_action"][_action(env, -4.0)] == 0
    assert reset_info["avaliable_action"][_action(env, 0.0)] == 1

    _, _, _, step_info = env.step(_action(env, 0.0))

    assert env.position == 0.0
    assert step_info["commission_fee_step"] > 0.0


def test_limit_up_allows_selling_to_reduce_a_long_position():
    env = _make_env(initial_position=2.0, is_limit_up=True)

    _, reset_info = env.reset()

    assert reset_info["avaliable_action"][_action(env, 4.0)] == 0
    assert reset_info["avaliable_action"][_action(env, 0.0)] == 1

    _, _, _, step_info = env.step(_action(env, 0.0))

    assert env.position == 0.0
    assert step_info["commission_fee_step"] > 0.0


@pytest.mark.parametrize(
    ("limit_kwargs", "blocked_position", "allowed_position"),
    [
        ({"is_limit_down": True}, -2.0, 2.0),
        ({"is_limit_up": True}, 2.0, -2.0),
    ],
)
def test_flat_position_blocks_only_the_forbidden_opening_direction(
    limit_kwargs, blocked_position, allowed_position
):
    env = _make_env(**limit_kwargs)

    _, reset_info = env.reset()

    assert reset_info["avaliable_action"][_action(env, blocked_position)] == 0
    assert reset_info["avaliable_action"][_action(env, allowed_position)] == 1

    _, _, _, blocked_info = env.step(_action(env, blocked_position))

    assert env.position == 0.0
    _assert_no_execution(blocked_info)

    allowed_env = _make_env(**limit_kwargs)
    allowed_env.reset()
    allowed_env.step(_action(allowed_env, allowed_position))

    assert allowed_env.position == allowed_position


@pytest.mark.parametrize(
    ("limit_kwargs", "initial_position", "blocked_add_position"),
    [
        ({"is_limit_down": True}, -2.0, -4.0),
        ({"is_limit_up": True}, 2.0, 4.0),
    ],
)
def test_price_limit_blocks_direct_same_direction_add(
    limit_kwargs, initial_position, blocked_add_position
):
    env = _make_env(initial_position=initial_position, **limit_kwargs)
    env.reset()

    _, _, _, step_info = env.step(_action(env, blocked_add_position))

    assert env.position == initial_position
    _assert_no_execution(step_info)


@pytest.mark.parametrize(
    ("limit_kwargs", "initial_position", "reverse_position"),
    [
        ({"is_limit_down": True}, 2.0, -2.0),
        ({"is_limit_up": True}, -2.0, 2.0),
    ],
)
def test_price_limit_rejects_the_entire_reverse_position(
    limit_kwargs, initial_position, reverse_position
):
    env = _make_env(initial_position=initial_position, **limit_kwargs)

    _, reset_info = env.reset()
    reverse_action = _action(env, reverse_position)

    assert reset_info["avaliable_action"][reverse_action] == 0

    _, _, _, step_info = env.step(reverse_action)

    assert env.position == initial_position
    _assert_no_execution(step_info)


@pytest.mark.parametrize(
    ("ratio_kwargs", "blocked_position"),
    [
        ({"limit_up_ratio": 0.1}, 2.0),
        ({"limit_down_ratio": 0.1}, -2.0),
    ],
)
def test_single_sided_ratio_derives_hard_limit_state(
    ratio_kwargs, blocked_position
):
    env = _make_env(**ratio_kwargs)

    _, reset_info = env.reset()

    blocked_action = _action(env, blocked_position)
    assert reset_info["avaliable_action"][blocked_action] == 0

    _, _, _, step_info = env.step(blocked_action)

    assert env.position == 0.0
    _assert_no_execution(step_info)


def test_near_limit_price_without_explicit_state_does_not_block_execution():
    env = _make_env(upper_limit_price=100.1, lower_limit_price=99.9)

    _, reset_info = env.reset()

    assert reset_info["avaliable_action"][_action(env, -2.0)] == 1
    assert reset_info["avaliable_action"][_action(env, 2.0)] == 1
