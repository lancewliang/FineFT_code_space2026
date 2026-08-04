import pytest
import numpy as np
import pandas as pd
from env.env_initiate.base_initiate import initiate_base_env
from env.env_initiate.commodity_initiate import initiate_commodity_env


def _make_dummy_df(num_rows=10, include_limit_cols=True):
    data = {
        "mark_price": np.linspace(100, 105, num_rows),
        "timestamp": pd.date_range("2026-01-01", periods=num_rows, freq="1s"),
        "funding_rate": np.zeros(num_rows),
        "funding_timestamp": pd.date_range("2026-01-01", periods=num_rows, freq="1s"),
        "feat1": np.random.randn(num_rows),
    }
    for i in range(1, 26):
        data[f"bid{i}_price"] = np.full(num_rows, 99.0)
        data[f"ask{i}_price"] = np.full(num_rows, 101.0)
        data[f"bid{i}_size"] = np.full(num_rows, 10.0)
        data[f"ask{i}_size"] = np.full(num_rows, 10.0)

    if include_limit_cols:
        data["limit_up_single_sided_ratio"] = np.zeros(num_rows)
        data["limit_down_single_sided_ratio"] = np.zeros(num_rows)
        data["limit_up_ask_depth_ratio_5"] = np.zeros(num_rows)
        data["limit_down_bid_depth_ratio_5"] = np.zeros(num_rows)
        data["UpperLimitPrice"] = np.full(num_rows, 110.0)
        data["LowerLimitPrice"] = np.full(num_rows, 90.0)

    return pd.DataFrame(data)


def test_missing_limit_columns_raises_error():
    df = _make_dummy_df(include_limit_cols=False)
    with pytest.raises(ValueError, match="DataFrame 缺少必须的涨跌停列"):
        initiate_base_env(df, feature_list=["feat1"], enable_limit_reward=True)

    with pytest.raises(ValueError, match="DataFrame 缺少必须的涨跌停列"):
        initiate_commodity_env(df, feature_list=["feat1"], depth=5, enable_limit_reward=True)


def test_disabled_by_default():
    df = _make_dummy_df(include_limit_cols=True)
    env = initiate_base_env(df, feature_list=["feat1"], enable_limit_reward=False)
    env.reset()
    _, reward, _, info = env.step(0)
    assert info["limit_reward"] == 0.0


def test_limit_up_reward_and_penalty():
    df = _make_dummy_df(num_rows=5, include_limit_cols=True)
    df["limit_up_single_sided_ratio"] = 1.0

    env = initiate_base_env(
        df,
        feature_list=["feat1"],
        enable_limit_reward=True,
        limit_hold_bonus=1.0,
        limit_stay_bonus=0.5,
        limit_reverse_penalty=1.5,
        position_choices=5,
        max_holding_number=4,
        allow_reverse_position=True,
    )
    env.reset()

    # Action to take positive position (long)
    action_long = env.env_map_position_leverage_to_action(4, 5)
    _, _, _, info = env.step(action_long)
    assert info["limit_reward"] > 0.0

    # Step to short position in limit up
    action_short = env.env_map_position_leverage_to_action(-4, 5)
    _, _, _, info2 = env.step(action_short)
    assert info2["limit_reward"] < 0.0


def test_limit_down_reward_and_penalty():
    df = _make_dummy_df(num_rows=5, include_limit_cols=True)
    df["limit_down_single_sided_ratio"] = 1.0

    env = initiate_base_env(
        df,
        feature_list=["feat1"],
        enable_limit_reward=True,
        limit_hold_bonus=1.0,
        limit_stay_bonus=0.5,
        limit_reverse_penalty=1.5,
        position_choices=5,
        max_holding_number=4,
        allow_reverse_position=True,
    )
    env.reset()

    # Action to take negative position (short)
    action_short = env.env_map_position_leverage_to_action(-4, 5)
    _, _, _, info = env.step(action_short)
    assert info["limit_reward"] > 0.0

    # Action to take long position in limit down
    action_long = env.env_map_position_leverage_to_action(4, 5)
    _, _, _, info2 = env.step(action_long)
    assert info2["limit_reward"] < 0.0


def test_depth_ratio_scaling():
    df1 = _make_dummy_df(num_rows=5, include_limit_cols=True)
    df1["limit_up_ask_depth_ratio_5"] = 0.4

    df2 = _make_dummy_df(num_rows=5, include_limit_cols=True)
    df2["limit_up_ask_depth_ratio_5"] = 1.0

    env1 = initiate_base_env(df1, feature_list=["feat1"], enable_limit_reward=True, position_choices=5, max_holding_number=4)
    env2 = initiate_base_env(df2, feature_list=["feat1"], enable_limit_reward=True, position_choices=5, max_holding_number=4)

    env1.reset()
    env2.reset()

    action_long = env1.env_map_position_leverage_to_action(4, 5)
    _, _, _, info1 = env1.step(action_long)
    _, _, _, info2 = env2.step(action_long)

    assert info2["limit_reward"] > info1["limit_reward"]
