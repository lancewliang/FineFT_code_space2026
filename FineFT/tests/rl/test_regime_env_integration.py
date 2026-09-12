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


def test_env_and_regime_stratified_buffer_end_to_end(tmp_path):
    from RL.util.regime_stratified_replay_buffer import RegimeStratifiedReplayBuffer
    import torch

    rows = 60
    # Assign alternating grid IDs 0, 1, 2 (all under low vol)
    grid_pattern = np.array([0] * 20 + [1] * 20 + [2] * 20, dtype=np.int64)

    data = {
        "mark_price": np.linspace(100.0, 110.0, rows),
        "timestamp": pd.date_range("2026-01-01", periods=rows, freq="30min"),
        "funding_rate": np.zeros(rows),
        "funding_timestamp": pd.date_range("2026-01-01", periods=rows, freq="30min"),
        "feature1": np.arange(rows, dtype=float),
        "feature2": np.arange(rows, dtype=float) * 2.0,
        "regime_grid_id": grid_pattern,
    }
    for i in range(1, 26):
        data[f"bid{i}_price"] = np.linspace(99.0, 109.0, rows)
        data[f"ask{i}_price"] = np.linspace(101.0, 111.0, rows)
        data[f"bid{i}_size"] = np.full(rows, 10.0)
        data[f"ask{i}_size"] = np.full(rows, 10.0)

    df = pd.DataFrame(data)
    env = initiate_demo_env(
        df=df,
        feature_list=["feature1", "feature2"],
        order_book_depth=5,
    )

    buffer = RegimeStratifiedReplayBuffer(
        total_buffer_size=900,
        batch_size=9,
        device="cpu",
    )

    state, info = env.reset()
    for step_i in range(15):
        next_state, reward, done, next_info = env.step(0)
        transition = (state, info, 0, reward, next_state, next_info, done)
        buffer.add_transition(transition)
        state = next_state
        info = next_info

    # All first 15 steps had grid_id = 0
    grid_lengths = buffer.get_grid_lengths()
    assert grid_lengths[0] == 15
    for q_idx in range(1, 9):
        assert grid_lengths[q_idx] == 0

    # Draw batch activating queue 0
    sampler = buffer.create_sampler(epoch_index=0, active_grid_ids=[0])
    assert sampler is not None
    batch = sampler.sample()
    assert batch[0].shape[0] == 9
