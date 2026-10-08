from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from RL.DiHFT.low_level.parallel_diverse_train import (
    WorkerRoundResult,
    WorkerTransitionRecord,
    write_round_transitions_to_buffer,
)
from RL.util.regime_stratified_replay_buffer import (
    GPURegimeStratifiedReplayBuffer,
    RegimeStratifiedReplayBuffer,
)
from RL.util.update import soft_copy_params


def _make_sample_transition(step_idx: int, grid_id: int, reward: float = 1.0) -> tuple:
    state = np.array([float(step_idx), float(grid_id)], dtype=np.float32)
    next_state = np.array([float(step_idx + 1), float(grid_id)], dtype=np.float32)
    info = {
        "previous_action": 0,
        "regime_grid_id": grid_id,
        "trading_info": np.array([1.0, 0.0, 0.0, 0.0, 0.1], dtype=np.float32),
        "avaliable_action": np.array([1, 1, 1], dtype=np.int64),
        "funding_count_down_hour": 0.0,
        "funding_count_down_minute": 0.0,
        "q_value": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    }
    next_info = dict(info)
    action = 1
    done = False
    return (state, info, action, reward, next_state, next_info, done)


def test_gpu_regime_stratified_replay_buffer_device_assignment():
    target_device = "cuda" if torch.cuda.is_available() else "cpu"
    buf = GPURegimeStratifiedReplayBuffer(
        total_buffer_size=900,
        batch_size=18,
        device=target_device,
        seed=42,
        num_grids=9,
    )
    t = _make_sample_transition(0, grid_id=8)
    buf.add_transition(t)

    expected_device_type = torch.device(target_device).type
    assert buf.states[8].device.type == expected_device_type
    assert buf.actions[8].device.type == expected_device_type
    assert buf.rewards[8].device.type == expected_device_type
    assert buf.q_values[8].device.type == expected_device_type

    sampler = buf.create_sampler(epoch_index=0, block_epochs=3)
    states, infos, actions, rewards, next_states, next_infos, dones = sampler.sample()
    assert states.device.type == expected_device_type
    assert actions.device.type == expected_device_type
    assert rewards.device.type == expected_device_type
    assert infos["avaliable_action"].device.type == expected_device_type


def test_add_round_records_bulk_dma_transfer():
    target_device = "cuda" if torch.cuda.is_available() else "cpu"
    buf = GPURegimeStratifiedReplayBuffer(
        total_buffer_size=900,
        batch_size=18,
        device=target_device,
        seed=42,
        num_grids=9,
    )

    t1 = _make_sample_transition(1, grid_id=1, reward=10.0)
    t2 = _make_sample_transition(2, grid_id=1, reward=20.0)
    t3 = _make_sample_transition(1, grid_id=1, reward=10.0)  # duplicate

    round_res = WorkerRoundResult(
        df_index=0,
        epoch_index=0,
        context_index=0,
        initial_action=0,
        round_counter=0,
        worker_steps=3,
        transitions=[
            WorkerTransitionRecord(step_index=0, transition=t1, td_error=0.5),
            WorkerTransitionRecord(step_index=1, transition=t2, td_error=0.8),
            WorkerTransitionRecord(step_index=2, transition=t3, td_error=0.2),
        ],
        rollout_metrics=[],
        done=True,
    )

    duplicates = write_round_transitions_to_buffer(buf, [round_res])
    assert duplicates == 1
    assert buf.get_grid_lengths()[1] == 2
    assert buf.rewards[1][0, 0].item() == 10.0
    assert buf.rewards[1][1, 0].item() == 20.0


def test_soft_copy_params_vectorized_lerp():
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 8)
            self.fc2 = nn.Linear(8, 2)

    torch.manual_seed(42)
    online_net = SimpleNet()
    target_net = SimpleNet()

    for p in target_net.parameters():
        p.data.zero_()
    for p in online_net.parameters():
        p.data.fill_(1.0)

    tau = 0.05
    soft_copy_params(online_net, target_net, tau)

    for p in target_net.parameters():
        assert torch.allclose(p.data, torch.full_like(p.data, 0.05))

    # Second soft copy step: 0.05 + 0.05 * (1.0 - 0.05) = 0.0975
    soft_copy_params(online_net, target_net, tau)
    for p in target_net.parameters():
        assert torch.allclose(p.data, torch.full_like(p.data, 0.0975))


def test_end_to_end_parallel_diverse_pipeline_integration(tmp_path, monkeypatch):
    """验证端到端流水线集成：GPU 经验池批量写入、异步非阻塞评测、课程轮转与梯度迭代。"""
    import queue
    import types
    from RL.DiHFT.low_level import parallel_diverse_train as pdt

    events = []
    episode_counter = {"count": 0}

    class DummyInputQueue:
        def __init__(self, df_index, result_queue):
            self.df_index = df_index
            self.result_queue = result_queue

        def put(self, message):
            if type(message).__name__ == "ResetWorkerTask":
                return
            episode_counter["count"] += 1
            idx = episode_counter["count"]
            info = {
                "previous_action": 0,
                "regime_grid_id": 8,  # Phase 0 active grid
                "trading_info": np.zeros(5, dtype=np.float32),
                "avaliable_action": np.array([1, 1, 1], dtype=np.int64),
                "funding_count_down_hour": 0.0,
                "funding_count_down_minute": 0.0,
                "q_value": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            }
            transition = (
                np.array([float(idx), 0.0], dtype=np.float32),
                info,
                1,
                1.0,
                np.array([float(idx) + 1.0, 0.0], dtype=np.float32),
                dict(info),
                True,
            )
            self.result_queue.put(
                pdt.WorkerRoundResult(
                    df_index=self.df_index,
                    epoch_index=message.epoch_index,
                    context_index=message.context_index,
                    initial_action=message.initial_action,
                    round_counter=message.round_counter,
                    worker_steps=1,
                    transitions=[
                        pdt.WorkerTransitionRecord(
                            step_index=0,
                            transition=transition,
                            td_error=0.5,
                        )
                    ],
                    rollout_metrics=[
                        pdt.RolloutMetrics(
                            epoch_index=message.epoch_index,
                            context_index=message.context_index,
                            initial_action=message.initial_action,
                            df_index=self.df_index,
                            transition_count=1,
                            reward_sum=1.0,
                            final_balance=101.0,
                            return_rate=0.01,
                        )
                    ],
                    done=True,
                )
            )

    trainer = MagicMock()
    trainer.total_df_index_length = 2
    trainer.num_epoch = 2
    trainer.N = 1
    trainer.position_choices = 2
    trainer.epsilon_init = 1.0
    trainer.epsilon_min = 0.1
    trainer.ada_init = 256.0
    trainer.ada_min = 0.0
    trainer.lr_init = 0.005
    trainer.lr_min = 0.001
    trainer.batch_size = 2
    trainer.buffer_size = 900
    trainer.curriculum_block_epochs = 3
    trainer.update_times = 2
    trainer.n_step = 1
    trainer.gamma = 0.99
    trainer.update_counter = 0
    trainer.optimizer = types.SimpleNamespace(param_groups=[{"lr": 0.0}])
    trainer.writer = MagicMock()
    trainer.tech_indicator_list = ["f1", "f2"]
    trainer.N_ACTIONS = 3
    trainer.hidden_nodes = 16
    trainer.time_info_dim = 2
    trainer.device = "cpu"
    trainer.eval_interval = 1
    trainer.model_path = str(tmp_path / "model")
    trainer.eval_net = pdt.ensemble_Qnet(
        N_STATES=2,
        N_ACTIONS=3,
        hidden_nodes=16,
        TIME_INFO_DIM=2,
        ensemble_number=1,
    )

    def mock_start_workers(trainer, train_df_cache, env_kwargs, shared_model=None, shared_market_data=None):
        events.append("start_workers")
        trainer.worker_result_queue = queue.Queue()
        trainer.worker_task_queue = queue.Queue()
        trainer.worker_processes = [MagicMock()]
        trainer.worker_input_queues = {
            0: DummyInputQueue(0, trainer.worker_result_queue),
            1: DummyInputQueue(1, trainer.worker_result_queue),
        }

    monkeypatch.setattr(pdt, "start_parallel_workers", mock_start_workers)
    monkeypatch.setattr(pdt, "shutdown_exploration_workers", lambda tr: events.append("shutdown_workers"))
    monkeypatch.setattr(pdt, "save_parallel_epoch_model", lambda tr, ep: events.append(("save_model", ep)))
    monkeypatch.setattr(pdt, "save_diverse_buffer", lambda buf, p: events.append("save_buffer"))

    def fake_update(tr, *args, **kwargs):
        events.append("update")
        tr.update_counter += 1
        return (1.0, 0.5, 0.5)

    monkeypatch.setattr(pdt, "update", fake_update)

    eval_calls = []
    def fake_eval(*args, **kwargs):
        eval_calls.append(kwargs.get("epoch_index"))
        return {"mean_return_rate": 0.05, "profit_ratio": 1.0}

    monkeypatch.setattr(pdt, "run_periodic_greedy_evaluation", fake_eval)

    buffer_diverse = GPURegimeStratifiedReplayBuffer(
        total_buffer_size=900,
        batch_size=2,
        device="cpu",
        seed=42,
        num_grids=9,
    )

    pdt.run_parallel_diverse_training(
        trainer=trainer,
        train_df_cache={},
        env_kwargs={},
        buffer_diverse=buffer_diverse,
        step_counter_diverse=0,
        diverse_rollout_latest_metrics_by_df={},
    )

    assert trainer.update_counter == 4  # 2 epochs * 2 updates
    assert buffer_diverse.get_grid_lengths()[8] > 0
    assert len(eval_calls) == 2  # evaluated on both epochs
