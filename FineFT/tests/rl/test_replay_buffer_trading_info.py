import sys
from pathlib import Path
import numpy as np
import torch
import pytest

FINEFT_ROOT = Path(__file__).resolve().parents[2]
if str(FINEFT_ROOT) not in sys.path:
    sys.path.insert(0, str(FINEFT_ROOT))

from RL.util.replay_buffer_DQN import (
    NETWORK_INFO_KEYS,
    Multi_step_ReplayBuffer_multi_info,
)


def test_network_info_keys_includes_trading_info():
    assert "trading_info" in NETWORK_INFO_KEYS


def test_replay_buffer_multi_info_samples_trading_info():
    device = "cpu"
    buffer = Multi_step_ReplayBuffer_multi_info(
        buffer_size=10,
        batch_size=2,
        device=device,
        seed=42,
        gamma=0.99,
        n_step=1,
    )
    state = np.ones(10, dtype=np.float32)
    next_state = np.ones(10, dtype=np.float32)
    
    info = {
        "avaliable_action": np.array([1, 1, 0], dtype=np.float32),
        "previous_action": 0,
        "q_value": np.zeros(3, dtype=np.float32),
        "high_level_state": np.zeros(5, dtype=np.float32),
        "funding_count_down_hour": 1.0,
        "funding_count_down_minute": 30.0,
        "trading_info": np.array([0.5, 0.02, -0.01], dtype=np.float32),
    }
    next_info = {
        "avaliable_action": np.array([1, 1, 0], dtype=np.float32),
        "previous_action": 1,
        "q_value": np.zeros(3, dtype=np.float32),
        "high_level_state": np.zeros(5, dtype=np.float32),
        "funding_count_down_hour": 1.0,
        "funding_count_down_minute": 29.0,
        "trading_info": np.array([0.5, 0.03, -0.01], dtype=np.float32),
    }

    for _ in range(5):
        buffer.add(state, info, 1, 0.5, next_state, next_info, False)

    states, infos, actions, rewards, next_states, next_infos, dones = buffer.sample()

    assert "trading_info" in infos
    assert "trading_info" in next_infos
    assert isinstance(infos["trading_info"], torch.Tensor)
    assert infos["trading_info"].shape == (2, 3)
    assert next_infos["trading_info"].shape == (2, 3)
    assert infos["trading_info"].dtype == torch.float32


def test_replay_buffer_multi_info_sample_evaluate_trading_info():
    device = "cpu"
    buffer = Multi_step_ReplayBuffer_multi_info(
        buffer_size=10,
        batch_size=2,
        device=device,
        seed=42,
        gamma=0.99,
        n_step=1,
    )
    state = np.ones(10, dtype=np.float32)
    next_state = np.ones(10, dtype=np.float32)
    
    info = {
        "avaliable_action": np.array([1, 1, 0], dtype=np.float32),
        "previous_action": 0,
        "q_value": np.zeros(3, dtype=np.float32),
        "high_level_state": np.zeros(5, dtype=np.float32),
        "funding_count_down_hour": 1.0,
        "funding_count_down_minute": 30.0,
        "trading_info": np.array([0.5, 0.02, -0.01], dtype=np.float32),
    }
    next_info = {
        "avaliable_action": np.array([1, 1, 0], dtype=np.float32),
        "previous_action": 1,
        "q_value": np.zeros(3, dtype=np.float32),
        "high_level_state": np.zeros(5, dtype=np.float32),
        "funding_count_down_hour": 1.0,
        "funding_count_down_minute": 29.0,
        "trading_info": np.array([0.5, 0.03, -0.01], dtype=np.float32),
    }

    for _ in range(3):
        buffer.add(state, info, 1, 0.5, next_state, next_info, False)

    states, infos, actions, rewards, next_states, next_infos, dones = buffer.sample_evaluate()

    assert "trading_info" in infos
    assert "trading_info" in next_infos
    assert infos["trading_info"].shape == (3, 3)
    assert next_infos["trading_info"].shape == (3, 3)
