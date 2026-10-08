import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

FINEFT_ROOT = Path(__file__).resolve().parents[2]
if str(FINEFT_ROOT) not in sys.path:
    sys.path.insert(0, str(FINEFT_ROOT))

from env.env_initiate.base_initiate import initiate_base_env
from env.env_initiate.simple_initiate import initiate_simple_env
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
        "peak_return_rate",
        "instant_profit_retracement",
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
    assert info["trading_info"].shape == (5,)
    np.testing.assert_array_almost_equal(info["trading_info"], np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))


def test_nonzero_reset_starts_duration_at_one_step():
    df, features = _sample_data()
    initial_state = (10000.0, 160.0, 0.0, 8.0, 5)
    env = initiate_base_env(df, features, initial_state=initial_state, holding_duration_norm_steps=180)
    state, info = env.reset()
    assert info["trading_info"].shape == (5,)
    assert abs(info["trading_info"][4] - (1.0 / 180.0)) < 1e-6


def test_holding_duration_lifecycle():
    df, features = _sample_data()
    norm_steps = 10
    env = initiate_base_env(df, features, allow_reverse_position=True, holding_duration_norm_steps=norm_steps)
    state, info = env.reset()
    assert info["trading_info"][4] == 0.0

    # 1. Open long position (action for +4.0 position)
    pos4_action = env.env_map_position_leverage_to_action(4, env.leverage_choices[0])
    state, reward, done, info = env.step(pos4_action)
    assert abs(info["trading_info"][4] - (1.0 / norm_steps)) < 1e-6

    # 2. Same-direction hold (+4.0 -> +4.0)
    state, reward, done, info = env.step(pos4_action)
    assert abs(info["trading_info"][4] - (2.0 / norm_steps)) < 1e-6

    # 3. Same-direction add (+4.0 -> +8.0)
    pos8_action = env.env_map_position_leverage_to_action(8, env.leverage_choices[0])
    state, reward, done, info = env.step(pos8_action)
    assert abs(info["trading_info"][4] - (3.0 / norm_steps)) < 1e-6

    # 4. Same-direction reduce (+8.0 -> +4.0)
    state, reward, done, info = env.step(pos4_action)
    assert abs(info["trading_info"][4] - (4.0 / norm_steps)) < 1e-6

    # 5. Reverse position (+4.0 -> -8.0)
    neg8_action = env.env_map_position_leverage_to_action(-8, env.leverage_choices[0])
    state, reward, done, info = env.step(neg8_action)
    assert abs(info["trading_info"][4] - (1.0 / norm_steps)) < 1e-6

    # 6. Close to flat (-8.0 -> 0)
    flat_action = env.env_map_position_leverage_to_action(0, env.leverage_choices[0])
    state, reward, done, info = env.step(flat_action)
    assert info["trading_info"][4] == 0.0


def test_holding_duration_clipping():
    df, features = _sample_data(rows=30)
    norm_steps = 5
    env = initiate_base_env(df, features, holding_duration_norm_steps=norm_steps)
    state, info = env.reset()

    pos8_action = env.env_map_position_leverage_to_action(8, env.leverage_choices[0])
    # Step 1 -> duration 1/5 = 0.2
    state, reward, done, info = env.step(pos8_action)
    assert abs(info["trading_info"][4] - 0.2) < 1e-6

    # Step 2..6 -> duration increases past norm_steps (5)
    for _ in range(6):
        state, reward, done, info = env.step(pos8_action)

    # Must clip to 1.0
    assert info["trading_info"][4] == 1.0


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
    assert info["trading_info"][4] == pytest.approx(3.0 / 10.0)

    _, reset_info = env.reset()

    assert reset_info["trading_info"][4] == pytest.approx(1.0 / 10.0)


def test_simple_env_single_holding_return_accumulates_across_holds():
    df, features = _sample_data(rows=20)
    env = initiate_simple_env(
        df,
        features,
        leverage_choice=[1],
        initial_state=(100000.0, 0.0, 0.0, 0.0, 1),
    )
    env.reset()
    long_action = env.env_map_position_leverage_to_action(4, 1)

    env.step(long_action)
    first_return = env.single_holding_return
    env.step(long_action)
    second_return = env.single_holding_return
    expected_increment = env.position * (df["mark_price"].iloc[2] - df["mark_price"].iloc[1])

    assert second_return == pytest.approx(first_return + expected_increment)


def test_augmented_markov_state_peak_and_retracement_lifecycle():
    prices = [100.0, 100.0, 104.0, 110.0, 107.0, 98.0, 98.0]
    timestamps = pd.date_range("2026-01-01", periods=len(prices), freq="min")
    data = {
        "timestamp": timestamps,
        "funding_timestamp": timestamps + pd.Timedelta(hours=8),
        "funding_rate": np.zeros(len(prices)),
        "mark_price": np.array(prices),
        "feat1": np.zeros(len(prices)),
    }
    for level in range(1, 26):
        data[f"ask{level}_price"] = data["mark_price"] + level * 0.01
        data[f"ask{level}_size"] = np.full(len(prices), 10.0)
        data[f"bid{level}_price"] = data["mark_price"] - level * 0.01
        data[f"bid{level}_size"] = np.full(len(prices), 10.0)
    df = pd.DataFrame(data)

    env = initiate_base_env(
        df,
        ["feat1"],
        max_holding_number=1,
        position_choices=3,
        leverage_choice=[1],
        initial_state=(1e5, 0.0, 0.0, 0.0, 1),
        commission_rate=0.0,
        holding_duration_norm_steps=100,
    )

    # 0. Reset -> pos 0 -> all 5 fields zero
    _, info = env.reset()
    assert info["trading_info"].shape == (5,)
    np.testing.assert_array_equal(info["trading_info"], np.zeros(5, dtype=np.float32))

    # 1. Open Long at 100.0
    _, _, _, info = env.step(2)
    assert info["trading_info"][0] == 1.0
    assert info["trading_info"][1] == pytest.approx(0.0, abs=1e-3)
    assert info["trading_info"][2] == pytest.approx(0.0, abs=1e-3)
    assert info["trading_info"][3] == 0.0
    assert info["trading_info"][4] == pytest.approx(1.0 / 100.0)

    # 2. Hold Long at 104.0: return +4%, peak +4%, deadband (<8%) -> retracement 0.0
    _, _, _, info = env.step(2)
    assert info["trading_info"][0] == 1.0
    assert info["trading_info"][1] == pytest.approx(0.04, abs=1e-3)
    assert info["trading_info"][2] == pytest.approx(0.04, abs=1e-3)
    assert info["trading_info"][3] == 0.0
    assert info["trading_info"][4] == pytest.approx(2.0 / 100.0)

    # 3. Hold Long at 110.0: return +10%, peak +10%, retracement 0.0
    _, _, _, info = env.step(2)
    assert info["trading_info"][0] == 1.0
    assert info["trading_info"][1] == pytest.approx(0.10, abs=1e-3)
    assert info["trading_info"][2] == pytest.approx(0.10, abs=1e-3)
    assert info["trading_info"][3] == 0.0
    assert info["trading_info"][4] == pytest.approx(3.0 / 100.0)

    # 4. Hold Long at 107.0: return +7%, peak +10%, retracement (0.10 - 0.07)/0.10 = 0.30
    _, _, _, info = env.step(2)
    assert info["trading_info"][0] == 1.0
    assert info["trading_info"][1] == pytest.approx(0.07, abs=1e-3)
    assert info["trading_info"][2] == pytest.approx(0.10, abs=1e-3)
    assert info["trading_info"][3] == pytest.approx(0.30, abs=1e-3)
    assert info["trading_info"][4] == pytest.approx(4.0 / 100.0)

    # 5. Drop below entry at 98.0: return -2%, peak +10%, retracement clamped to 1.0
    _, _, _, info = env.step(2)
    assert info["trading_info"][0] == 1.0
    assert info["trading_info"][1] == pytest.approx(-0.02, abs=1e-3)
    assert info["trading_info"][2] == pytest.approx(0.10, abs=1e-3)
    assert info["trading_info"][3] == pytest.approx(1.0, abs=1e-3)
    assert info["trading_info"][4] == pytest.approx(5.0 / 100.0)

    # 6. Exit to flat: action 1 -> trading_info reset to all zeros
    _, _, _, info = env.step(1)
    np.testing.assert_array_equal(info["trading_info"], np.zeros(5, dtype=np.float32))


def test_augmented_markov_state_position_reversal_resets_accumulators():
    prices = [100.0, 100.0, 110.0, 110.0, 99.0]
    timestamps = pd.date_range("2026-01-01", periods=len(prices), freq="min")
    data = {
        "timestamp": timestamps,
        "funding_timestamp": timestamps + pd.Timedelta(hours=8),
        "funding_rate": np.zeros(len(prices)),
        "mark_price": np.array(prices),
        "feat1": np.zeros(len(prices)),
    }
    for level in range(1, 26):
        data[f"ask{level}_price"] = data["mark_price"] + level * 0.01
        data[f"ask{level}_size"] = np.full(len(prices), 10.0)
        data[f"bid{level}_price"] = data["mark_price"] - level * 0.01
        data[f"bid{level}_size"] = np.full(len(prices), 10.0)
    df = pd.DataFrame(data)

    env = initiate_base_env(
        df,
        ["feat1"],
        max_holding_number=1,
        position_choices=3,
        leverage_choice=[1],
        initial_state=(1e5, 0.0, 0.0, 0.0, 1),
        commission_rate=0.0,
        holding_duration_norm_steps=100,
        allow_reverse_position=True,
    )

    env.reset()
    # 1. Open Long at 100.0 (action 2)
    env.step(2)
    # 2. Hold Long at 110.0: return +10%, peak +10%
    _, _, _, info = env.step(2)
    assert info["trading_info"][1] == pytest.approx(0.10, abs=1e-3)
    assert info["trading_info"][2] == pytest.approx(0.10, abs=1e-3)
    assert env.get_info_field("peak_return_rate") == pytest.approx(0.10, abs=1e-3)

    # 3. Direct position reversal from Long (+1) to Short (-1) at 110.0 (action 0)
    _, _, _, rev_info = env.step(0)
    assert rev_info["trading_info"][0] == -1.0
    assert rev_info["trading_info"][1] == 0.0
    assert rev_info["trading_info"][2] == 0.0
    assert rev_info["trading_info"][3] == 0.0
    assert rev_info["trading_info"][4] == pytest.approx(1.0 / 100.0)
    assert env.episode_peak_return_rate == 0.0
    assert env.instant_profit_retracement == 0.0
    assert env.get_info_field("peak_return_rate") == 0.0

    # 4. Hold Short as price moves from 110.0 to 99.0 (+10% gain for short)
    _, _, _, hold_info = env.step(0)
    assert hold_info["trading_info"][0] == -1.0
    assert hold_info["trading_info"][1] == pytest.approx(0.10, abs=1e-3)
    assert hold_info["trading_info"][2] == pytest.approx(0.10, abs=1e-3)
    assert hold_info["trading_info"][3] == 0.0
    assert hold_info["trading_info"][4] == pytest.approx(2.0 / 100.0)
    assert env.get_info_field("peak_return_rate") == pytest.approx(0.10, abs=1e-3)
