import numpy as np
import pandas as pd
import pytest

from env.env_class.base_env import INFO_DIAGNOSTIC_FIELDS
from env.env_class.futures_util import rule_based_close
from env.env_initiate.base_initiate import initiate_base_env


def _df(depth):
    rows = []
    for t, price in enumerate([100.0, 101.0]):
        timestamp = pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=t)
        row = {
            "timestamp": timestamp,
            "mark_price": price,
            "funding_rate": 0.0,
            "funding_timestamp": timestamp + pd.Timedelta(hours=8),
            "feature_a": float(t),
        }
        for level in range(1, depth + 1):
            row[f"ask{level}_price"] = price + level
            row[f"ask{level}_size"] = 10.0
            row[f"bid{level}_price"] = price - level
            row[f"bid{level}_size"] = 10.0
        rows.append(row)
    return pd.DataFrame(rows)


def test_base_env_uses_configured_order_book_depth():
    env = initiate_base_env(
        _df(depth=5),
        ["feature_a"],
        max_holding_number=1,
        position_choices=3,
        order_book_depth=5,
    )

    _, info = env.reset()

    assert len(info["ask_qyts"]) == 5
    assert len(info["bid_qyts"]) == 5


def test_base_env_single_row_dataset():
    df_single = _df(depth=5).iloc[:1]
    env = initiate_base_env(
        df_single,
        ["feature_a"],
        max_holding_number=1,
        position_choices=3,
        order_book_depth=5,
    )
    s, info = env.reset()
    assert env.terminal is True
    s_, r, done, info = env.step(0)
    assert done is True
    assert r == 0.0


def test_get_info_field_returns_removed_diagnostic_values():
    env = initiate_base_env(
        _df(depth=5),
        ["feature_a"],
        max_holding_number=1,
        position_choices=3,
        order_book_depth=5,
    )
    _, info = env.reset()
    _, _, _, step_info = env.step(0)

    assert env.day == 1
    assert env.get_info_field("current_timestamp") == env.timestamp_array[1]
    assert env.get_info_field("previous_timestamp") == env.timestamp_array[0]
    assert env.get_info_field("current_markprice") == env.current_markprice
    assert (
        env.get_info_field("funding_count_down")
        == env.funding_timestamp_array[env.day] - env.timestamp_array[env.day]
    )
    assert 0.0 <= env.get_info_field("funding_count_down_second") < 60.0
    assert (
        env.get_info_field("single_holding_return_rate")
        == env.single_holding_return_rate
    )
    assert env.get_info_field("limit_reward") == env.last_limit_reward
    # 已移除字段不得再出现在 reset / step 默认返回中
    for field in INFO_DIAGNOSTIC_FIELDS:
        assert field not in info
        assert field not in step_info


def test_get_info_field_rejects_unknown_field():
    env = initiate_base_env(
        _df(depth=5),
        ["feature_a"],
        max_holding_number=1,
        position_choices=3,
        order_book_depth=5,
    )
    env.reset()
    with pytest.raises(ValueError, match="unknown info field"):
        env.get_info_field("not_a_field")


def test_get_info_field_previous_timestamp_before_step_raises():
    env = initiate_base_env(
        _df(depth=5),
        ["feature_a"],
        max_holding_number=1,
        position_choices=3,
        order_book_depth=5,
    )
    env.reset()
    assert env.day == 0
    with pytest.raises(ValueError, match="previous_timestamp"):
        env.get_info_field("previous_timestamp")


def test_immediate_liquidation_info_has_no_personal_state():
    """动作后立即爆仓分支的 info 不得包含 personal_state。

    历史缺陷：该分支曾返回 set 字面量导致 rule_based_close 索引崩溃；
    现 personal_state 已彻底从 info 移除，仓位/杠杆改为显式传参。
    """
    env = initiate_base_env(
        _df(depth=5),
        ["feature_a"],
        max_holding_number=4,
        position_choices=5,
        order_book_depth=5,
        initial_state=(90.0, 80.0, 0.0, 4.0, 5),
        maintenance_margin_ratio_dict={"0": [0.9, 0]},
    )
    _, reset_info = env.reset()
    act = env.env_map_position_leverage_to_action(4, 5)
    _, _, done, info = env.step(act)

    assert done is True
    assert "personal_state" not in reset_info
    assert "personal_state" not in info

    # rule_based_close 仅依赖 info 的订单簿/动作列表 + 显式 position/leverage
    action = rule_based_close(
        info, 12, [5], [-2.0, -1.0, 0.0, 1.0, 2.0], env.position, env.leverage
    )
    assert action == 12  # 爆仓后仓位归零 -> 直接返回 zero_position_action
