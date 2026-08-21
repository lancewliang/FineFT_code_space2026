import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

FINEFT_ROOT = Path(__file__).resolve().parents[2]
if str(FINEFT_ROOT) not in sys.path:
    sys.path.insert(0, str(FINEFT_ROOT))

from env.env_initiate.base_initiate import initiate_base_env
from env.env_class.base_env import TRADING_INFO_KEYS


def _sample_data(rows=250):
    timestamps = pd.date_range("2026-01-01", periods=rows, freq="min")
    data = {
        "timestamp": timestamps,
        "funding_timestamp": timestamps + pd.Timedelta(hours=8),
        "funding_rate": np.zeros(rows),
        "mark_price": np.linspace(100.0, 105.0, rows),
        "feature_a": np.linspace(0.0, 1.0, rows),
    }
    for level in range(1, 26):
        data[f"ask{level}_price"] = data["mark_price"] + level * 0.01
        data[f"ask{level}_size"] = np.full(rows, 10.0)
        data[f"bid{level}_price"] = data["mark_price"] - level * 0.01
        data[f"bid{level}_size"] = np.full(rows, 10.0)
    return pd.DataFrame(data), ["feature_a"]


def test_trading_info_keys_constant():
    assert TRADING_INFO_KEYS == (
        "position_exposure",
        "single_holding_return_rate",
        "single_holding_max_drawdown",
        "current_holding_duration_norm",
    )


def test_invalid_holding_duration_norm_steps_raises():
    df, features = _sample_data()
    with pytest.raises(ValueError):
        initiate_base_env(df, features, holding_duration_norm_steps=0)
    with pytest.raises(ValueError):
        initiate_base_env(df, features, holding_duration_norm_steps=-10)


def test_reset_returns_trading_info_zeros():
    df, features = _sample_data()
    env = initiate_base_env(df, features, allow_reverse_position=True)
    state, info = env.reset()
    assert "trading_info" in info
    assert isinstance(info["trading_info"], np.ndarray)
    assert info["trading_info"].shape == (4,)
    np.testing.assert_array_almost_equal(info["trading_info"], np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))


def test_nonzero_reset_starts_duration_at_one_step():
    df, features = _sample_data()
    initial_state = (10000.0, 160.0, 0.0, 8.0, 5)
    env = initiate_base_env(df, features, initial_state=initial_state, holding_duration_norm_steps=180)
    state, info = env.reset()
    assert info["trading_info"].shape == (4,)
    assert abs(info["trading_info"][3] - (1.0 / 180.0)) < 1e-6


def test_holding_duration_lifecycle():
    df, features = _sample_data()
    norm_steps = 10
    env = initiate_base_env(df, features, allow_reverse_position=True, holding_duration_norm_steps=norm_steps)
    state, info = env.reset()
    assert info["trading_info"][3] == 0.0

    # 1. Open long position (action for +4.0 position)
    pos4_action = env.env_map_position_leverage_to_action(4, env.leverage_choices[0])
    state, reward, done, info = env.step(pos4_action)
    assert abs(info["trading_info"][3] - (1.0 / norm_steps)) < 1e-6

    # 2. Same-direction hold (+4.0 -> +4.0)
    state, reward, done, info = env.step(pos4_action)
    assert abs(info["trading_info"][3] - (2.0 / norm_steps)) < 1e-6

    # 3. Same-direction add (+4.0 -> +8.0)
    pos8_action = env.env_map_position_leverage_to_action(8, env.leverage_choices[0])
    state, reward, done, info = env.step(pos8_action)
    assert abs(info["trading_info"][3] - (3.0 / norm_steps)) < 1e-6

    # 4. Same-direction reduce (+8.0 -> +4.0)
    state, reward, done, info = env.step(pos4_action)
    assert abs(info["trading_info"][3] - (4.0 / norm_steps)) < 1e-6

    # 5. Reverse position (+4.0 -> -8.0)
    neg8_action = env.env_map_position_leverage_to_action(-8, env.leverage_choices[0])
    state, reward, done, info = env.step(neg8_action)
    assert abs(info["trading_info"][3] - (1.0 / norm_steps)) < 1e-6

    # 6. Close to flat (-8.0 -> 0)
    flat_action = env.env_map_position_leverage_to_action(0, env.leverage_choices[0])
    state, reward, done, info = env.step(flat_action)
    assert info["trading_info"][3] == 0.0


def test_holding_duration_clipping():
    df, features = _sample_data(rows=30)
    norm_steps = 5
    env = initiate_base_env(df, features, holding_duration_norm_steps=norm_steps)
    state, info = env.reset()

    pos8_action = env.env_map_position_leverage_to_action(8, env.leverage_choices[0])
    # Step 1 -> duration 1/5 = 0.2
    state, reward, done, info = env.step(pos8_action)
    assert abs(info["trading_info"][3] - 0.2) < 1e-6

    # Step 2..6 -> duration increases past norm_steps (5)
    for _ in range(6):
        state, reward, done, info = env.step(pos8_action)

    # Must clip to 1.0
    assert info["trading_info"][3] == 1.0


def test_single_holding_return_accumulates_across_same_direction_holds():
    df, features = _sample_data(rows=20)
    env = initiate_base_env(df, features, allow_reverse_position=True)
    _, info = env.reset()
    long_action = env.env_map_position_leverage_to_action(4, env.leverage_choices[0])

    _, _, _, _ = env.step(long_action)
    first_return = env.single_holding_return
    _, _, _, _ = env.step(long_action)
    second_return = env.single_holding_return
    expected_increment = env.position * (df["mark_price"].iloc[2] - df["mark_price"].iloc[1])

    assert second_return == pytest.approx(first_return + expected_increment)


def test_reset_restarts_holding_duration_for_a_new_episode():
    df, features = _sample_data(rows=20)
    env = initiate_base_env(
        df,
        features,
        allow_reverse_position=True,
        holding_duration_norm_steps=10,
        initial_state=(100000.0, 80.0, 0.0, 4.0, 5),
    )
    _, info = env.reset()
    long_action = env.env_map_position_leverage_to_action(4, env.leverage_choices[0])

    _, _, _, info = env.step(long_action)
    _, _, _, info = env.step(long_action)
    assert info["trading_info"][3] == pytest.approx(3.0 / 10.0)

    _, reset_info = env.reset()

    assert reset_info["trading_info"][3] == pytest.approx(1.0 / 10.0)
