import numpy as np
import pytest
import torch
from unittest.mock import MagicMock
from RL.util.regime_stratified_replay_buffer import (
    RegimeStratifiedReplayBuffer,
    get_active_grid_ids_for_epoch,
)
from RL.DiHFT.low_level import parallel_diverse_train as pdt


def _make_dummy_transition(step_idx: int, grid_id: int):
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
    return (state, info, 1, float(grid_id), next_state, dict(info), False)


def test_end_to_end_curriculum_phase_rotation_and_balanced_updates(monkeypatch):
    monkeypatch.setattr(pdt, "UPDATE_WINDOWS_PER_EPOCH", 1)

    # 1. 填充 9 个格子的数据
    buffer = RegimeStratifiedReplayBuffer(
        total_buffer_size=9000,
        batch_size=60,
        device="cpu",
        seed=42,
        num_grids=9,
    )
    for g in range(9):
        for i in range(50):
            buffer.add_transition(_make_dummy_transition(i, grid_id=g))

    assert buffer.total_len() == 450

    trainer = MagicMock()
    trainer.batch_size = 60
    trainer.device = "cpu"
    trainer.curriculum_block_epochs = 3
    trainer.update_counter = 0
    trainer.writer = MagicMock()

    observed_active_grids = []

    def fake_update(tr, states, infos, actions, rewards, next_states, next_infos, dones):
        tr.update_counter += 1
        # rewards 中包含的是 grid_id
        active_ids = sorted(list(set(rewards.squeeze().long().numpy().tolist())))
        observed_active_grids.append(active_ids)
        return (1.0, 0.2, 0.8)

    monkeypatch.setattr(pdt, "update", fake_update)

    # 运行 9 个 epoch，block_epochs=3
    # Epoch 0..2: Phase 0 [0, 3, 6] (Downtrend)
    # Epoch 3..5: Phase 1 [1, 4, 7] (Range/Flat)
    # Epoch 6..8: Phase 2 [2, 5, 8] (Uptrend)
    for epoch in range(9):
        pdt.run_diverse_training_phase(
            trainer=trainer,
            buffer_diverse=buffer,
            update_count=1,
            epoch_index=epoch,
        )

    assert len(observed_active_grids) == 9
    for epoch in range(3):
        assert observed_active_grids[epoch] == [0, 3, 6]

    for epoch in range(3, 6):
        assert observed_active_grids[epoch] == [1, 4, 7]

    for epoch in range(6, 9):
        assert observed_active_grids[epoch] == [2, 5, 8]
