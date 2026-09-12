import numpy as np
import pandas as pd
import pytest

from env.env_initiate.demo_initiate import initiate_demo_env


def _build_dummy_df(num_rows: int = 20, with_regime: bool = True):
    data = {
        "mark_price": np.linspace(100.0, 105.0, num_rows),
        "timestamp": pd.date_range("2026-01-01", periods=num_rows, freq="30min"),
        "funding_rate": np.zeros(num_rows),
        "funding_timestamp": pd.date_range("2026-01-01", periods=num_rows, freq="30min"),
        "feature1": np.random.randn(num_rows),
        "feature2": np.random.randn(num_rows),
    }
    for i in range(1, 26):
        data[f"bid{i}_price"] = np.linspace(99.0, 104.0, num_rows)
        data[f"ask{i}_price"] = np.linspace(101.0, 106.0, num_rows)
        data[f"bid{i}_size"] = np.full(num_rows, 10.0)
        data[f"ask{i}_size"] = np.full(num_rows, 10.0)

    if with_regime:
        data["regime_grid_id"] = np.array([i % 9 for i in range(num_rows)], dtype=np.int64)

    return pd.DataFrame(data)


def test_env_exposes_regime_grid_id_when_present():
    df = _build_dummy_df(num_rows=20, with_regime=True)
    env = initiate_demo_env(
        df=df,
        feature_list=["feature1", "feature2"],
        order_book_depth=5,
    )
    state, info = env.reset()
    assert "regime_grid_id" in info
    assert info["regime_grid_id"] == 0

    state, reward, done, info = env.step(0)
    assert "regime_grid_id" in info
    assert info["regime_grid_id"] == 1


def test_env_defaults_to_negative_one_when_regime_absent():
    df = _build_dummy_df(num_rows=20, with_regime=False)
    env = initiate_demo_env(
        df=df,
        feature_list=["feature1", "feature2"],
        order_book_depth=5,
    )
    state, info = env.reset()
    assert "regime_grid_id" in info
    assert info["regime_grid_id"] == -1

    state, reward, done, info = env.step(0)
    assert "regime_grid_id" in info
    assert info["regime_grid_id"] == -1
