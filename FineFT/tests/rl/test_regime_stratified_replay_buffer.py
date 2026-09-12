import numpy as np
import pytest
import torch

from RL.util.regime_stratified_replay_buffer import (
    accumulate_trajectory_n_step,
    RegimeStratifiedReplayBuffer,
    StratifiedStackedSampler,
    get_active_grid_ids_for_epoch,
    DIRECTIONAL_REGIME_PHASES,
)


def _make_dummy_transition(step_idx: int, grid_id: int, reward: float = 1.0, done: bool = False):
    state = np.array([float(step_idx), float(grid_id)], dtype=np.float32)
    next_state = np.array([float(step_idx + 1), float(grid_id)], dtype=np.float32)
    info = {
        "previous_action": 0,
        "regime_grid_id": grid_id,
        "trading_info": np.array([1.0, 0.0, 0.0, 0.1], dtype=np.float32),
        "avaliable_action": np.array([1, 1, 1], dtype=np.int64),
        "funding_count_down_hour": 0.0,
        "funding_count_down_minute": 0.0,
        "q_value": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    }
    next_info = dict(info)
    action = 1
    return (state, info, action, reward, next_state, next_info, done)


def test_accumulate_trajectory_n_step():
    gamma = 0.9
    n_step = 3
    # 5 步单合约 trajectory: rewards = [1, 2, 3, 4, 5]
    trajectory = [
        _make_dummy_transition(i, grid_id=0, reward=float(i + 1), done=(i == 4))
        for i in range(5)
    ]
    accumulated = accumulate_trajectory_n_step(trajectory, gamma=gamma, n_step=n_step)
    assert len(accumulated) == 5

    # step 0: r = 1 + 0.9*2 + 0.81*3 = 1 + 1.8 + 2.43 = 5.23, next_state = state_3, done=False
    t0 = accumulated[0]
    assert np.isclose(t0[3], 1.0 + 0.9 * 2.0 + 0.81 * 3.0)
    assert np.allclose(t0[4], trajectory[3][0])
    assert t0[6] is False

    # step 1: r = 2 + 0.9*3 + 0.81*4 = 2 + 2.7 + 3.24 = 7.94, next_state = state_4, done=False
    t1 = accumulated[1]
    assert np.isclose(t1[3], 2.0 + 0.9 * 3.0 + 0.81 * 4.0)
    assert np.allclose(t1[4], trajectory[4][0])
    assert t1[6] is False

    # step 2: 跨入终点 step 4, r = 3 + 0.9*4 + 0.81*5 = 3 + 3.6 + 4.05 = 10.65, done=True
    t2 = accumulated[2]
    assert np.isclose(t2[3], 3.0 + 0.9 * 4.0 + 0.81 * 5.0)
    assert t2[6] is True

    # step 3: 尾部截断（仅剩 2 步：3 与 4），r = 4 + 0.9*5 = 8.5, done=True
    t3 = accumulated[3]
    assert np.isclose(t3[3], 4.0 + 0.9 * 5.0)
    assert t3[6] is True

    # step 4: 尾部截断（仅剩 1 步：4），r = 5, done=True
    t4 = accumulated[4]
    assert np.isclose(t4[3], 5.0)
    assert t4[6] is True


def test_regime_stratified_buffer_routing_and_isolation():
    buffer = RegimeStratifiedReplayBuffer(
        total_buffer_size=900,
        batch_size=32,
        device="cpu",
        seed=42,
        num_grids=9,
    )
    assert buffer.grid_capacity == 100

    buffer.add_transition(_make_dummy_transition(0, grid_id=0))
    buffer.add_transition(_make_dummy_transition(1, grid_id=0))
    buffer.add_transition(_make_dummy_transition(2, grid_id=0))

    buffer.add_transition(_make_dummy_transition(0, grid_id=5))
    buffer.add_transition(_make_dummy_transition(1, grid_id=5))

    buffer.add_transition(_make_dummy_transition(0, grid_id=8))

    lengths = buffer.get_grid_lengths()
    assert lengths[0] == 3
    assert lengths[5] == 2
    assert lengths[8] == 1
    assert lengths[1] == 0
    assert buffer.total_len() == 6


def test_regime_stratified_buffer_priority_replacement():
    buffer = RegimeStratifiedReplayBuffer(
        total_buffer_size=900,
        batch_size=32,
        device="cpu",
        seed=42,
        num_grids=9,
    )
    t1 = _make_dummy_transition(0, grid_id=0, reward=1.0)
    buffer.add_transition(t1, td_error=1.5)
    assert buffer.get_grid_lengths()[0] == 1

    t2 = _make_dummy_transition(0, grid_id=0, reward=2.0)
    buffer.add_transition(t2, td_error=1.0)
    assert buffer.get_grid_lengths()[0] == 1
    assert buffer.buffers[0].memory[0].reward == 1.0

    t3 = _make_dummy_transition(0, grid_id=0, reward=9.0)
    buffer.add_transition(t3, td_error=3.0)
    assert buffer.get_grid_lengths()[0] == 1
    assert buffer.buffers[0].memory[0].reward == 9.0


def test_stratified_stacked_sampler_balanced_sampling():
    buffer = RegimeStratifiedReplayBuffer(
        total_buffer_size=9000,
        batch_size=60,
        device="cpu",
        seed=42,
        num_grids=9,
    )
    for g in [0, 3, 6]:
        for i in range(100):
            buffer.add_transition(_make_dummy_transition(i, grid_id=g, reward=float(g)))

    tensor_dicts = buffer.extract_stacked_tensor_dicts()
    sampler = StratifiedStackedSampler(
        tensor_dicts=tensor_dicts,
        active_grid_ids=[0, 3, 6],
        batch_size=60,
        device="cpu",
    )

    states, infos, actions, rewards, next_states, next_infos, dones = sampler.sample()
    assert states.shape[0] == 60
    assert actions.shape[0] == 60
    assert rewards.shape[0] == 60

    rew_list = rewards.squeeze().numpy().tolist()
    assert rew_list.count(0.0) == 20
    assert rew_list.count(3.0) == 20
    assert rew_list.count(6.0) == 20


def test_stratified_stacked_sampler_replacement_fallback():
    buffer = RegimeStratifiedReplayBuffer(
        total_buffer_size=9000,
        batch_size=30,
        device="cpu",
        seed=42,
        num_grids=9,
    )
    for i in range(50):
        buffer.add_transition(_make_dummy_transition(i, grid_id=0, reward=0.0))
    for i in range(3):
        buffer.add_transition(_make_dummy_transition(i, grid_id=1, reward=1.0))

    tensor_dicts = buffer.extract_stacked_tensor_dicts()
    sampler = StratifiedStackedSampler(
        tensor_dicts=tensor_dicts,
        active_grid_ids=[0, 1],
        batch_size=30,
        device="cpu",
    )

    states, infos, actions, rewards, next_states, next_infos, dones = sampler.sample()
    assert states.shape[0] == 30
    rew_list = rewards.squeeze().numpy().tolist()
    assert rew_list.count(0.0) == 15
    assert rew_list.count(1.0) == 15


def test_directional_regime_curriculum_rotation():
    assert get_active_grid_ids_for_epoch(0, block_epochs=3) == [0, 3, 6]
    assert get_active_grid_ids_for_epoch(1, block_epochs=3) == [0, 3, 6]
    assert get_active_grid_ids_for_epoch(2, block_epochs=3) == [0, 3, 6]

    assert get_active_grid_ids_for_epoch(3, block_epochs=3) == [1, 4, 7]
    assert get_active_grid_ids_for_epoch(4, block_epochs=3) == [1, 4, 7]
    assert get_active_grid_ids_for_epoch(5, block_epochs=3) == [1, 4, 7]

    assert get_active_grid_ids_for_epoch(6, block_epochs=3) == [2, 5, 8]
    assert get_active_grid_ids_for_epoch(7, block_epochs=3) == [2, 5, 8]
    assert get_active_grid_ids_for_epoch(8, block_epochs=3) == [2, 5, 8]

    assert get_active_grid_ids_for_epoch(9, block_epochs=3) == [0, 3, 6]


def test_regime_stratified_buffer_create_sampler():
    buffer = RegimeStratifiedReplayBuffer(
        total_buffer_size=9000,
        batch_size=30,
        device="cpu",
        seed=42,
        num_grids=9,
    )
    for g in range(9):
        for i in range(10):
            buffer.add_transition(_make_dummy_transition(i, grid_id=g, reward=float(g)))

    sampler_p0 = buffer.create_sampler(epoch_index=0, block_epochs=3)
    assert sampler_p0.active_grid_ids == [0, 3, 6]

    sampler_p1 = buffer.create_sampler(epoch_index=4, block_epochs=3)
    assert sampler_p1.active_grid_ids == [1, 4, 7]

    sampler_p2 = buffer.create_sampler(epoch_index=8, block_epochs=3)
    assert sampler_p2.active_grid_ids == [2, 5, 8]
