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


def _sample_data(rows=50):
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
    )


def test_reset_returns_trading_info_zeros():
    df, features = _sample_data()
    env = initiate_base_env(df, features, allow_reverse_position=True)
    state, info = env.reset()
    assert "trading_info" in info
    assert isinstance(info["trading_info"], np.ndarray)
    assert info["trading_info"].shape == (3,)
    np.testing.assert_array_almost_equal(info["trading_info"], np.array([0.0, 0.0, 0.0], dtype=np.float32))


def test_active_position_calculates_exposure_and_trading_info():
    df, features = _sample_data()
    env = initiate_base_env(df, features, allow_reverse_position=True)
    state, info = env.reset()

    # action 8 -> long position (+8.0)
    state, reward, done, info = env.step(8)
    assert "trading_info" in info
    trading_info = info["trading_info"]
    max_abs_pos = max(abs(p) for p in env.position_list)
    expected_exposure = env.position / max_abs_pos
    assert expected_exposure > 0.0
    assert abs(trading_info[0] - expected_exposure) < 1e-6
    assert len(trading_info) == 3


def test_close_position_resets_trading_info_to_zeros():
    df, features = _sample_data()
    env = initiate_base_env(df, features, allow_reverse_position=True)
    state, info = env.reset()

    # Open long position
    state, reward, done, info = env.step(8)
    assert info["trading_info"][0] != 0.0

    # Close to 0 position
    flat_action = env.env_map_position_leverage_to_action(0, env.leverage_choices[0])
    state, reward, done, info = env.step(flat_action)
    np.testing.assert_array_almost_equal(info["trading_info"], np.array([0.0, 0.0, 0.0], dtype=np.float32))


def test_direction_change_resets_holding_metrics():
    df, features = _sample_data()
    env = initiate_base_env(df, features, allow_reverse_position=True)
    state, info = env.reset()

    # Open long position (action 8)
    state, reward, done, info = env.step(8)
    # Next step: keep long position to accumulate return
    state, reward, done, info = env.step(8)

    # Reverse direction from long to short (action 0 is short -8.0)
    state, reward, done, info = env.step(0)
    trading_info = info["trading_info"]
    max_abs_pos = max(abs(p) for p in env.position_list)
    expected_short_exposure = env.position / max_abs_pos
    assert expected_short_exposure < 0.0
    assert abs(trading_info[0] - expected_short_exposure) < 1e-6
    # Return rate and drawdown for the brand-new short holding should be 0.0
    assert trading_info[1] == 0.0
    assert trading_info[2] == 0.0
