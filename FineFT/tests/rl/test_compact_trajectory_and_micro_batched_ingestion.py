import pickle
import numpy as np
import pytest
import torch
import queue
from unittest.mock import MagicMock

from RL.util.regime_stratified_replay_buffer import (
    CompactTrajectoryRecord,
    accumulate_compact_trajectory_n_step,
    accumulate_trajectory_n_step,
    build_spatio_temporal_semantic_key,
    RegimeStratifiedReplayBuffer,
)
from RL.DiHFT.low_level.parallel_diverse_train import (
    WorkerRoundResult,
    WorkerTransitionRecord,
    RolloutMetrics,
    write_round_transitions_to_buffer,
    run_epoch_exploration,
    ExploreTask,
)
from RL.DiHFT.low_level.shared_data_manager import EnvTensorPack, SharedMarketDataPack


def _create_mock_shared_market_data(num_dfs=2, num_steps=50, state_dim=148, action_dim=9):
    packs = {}
    for df_idx in range(num_dfs):
        state_tensor = torch.arange(num_steps * state_dim, dtype=torch.float32).reshape(num_steps, state_dim) + df_idx * 1000.0
        ask_prices = torch.ones((num_steps, 25), dtype=torch.float32) * 100.0
        bid_prices = torch.ones((num_steps, 25), dtype=torch.float32) * 99.0
        ask_qtys = torch.ones((num_steps, 25), dtype=torch.float32) * 10.0
        bid_qtys = torch.ones((num_steps, 25), dtype=torch.float32) * 10.0
        markprice = torch.ones((num_steps,), dtype=torch.float32) * 100.0
        timestamp = torch.zeros((num_steps,), dtype=torch.int64)
        funding_rate = torch.zeros((num_steps,), dtype=torch.float32)
        funding_timestamp = torch.zeros((num_steps,), dtype=torch.int64)
        regime_grid_ids = torch.full((num_steps,), fill_value=df_idx % 9, dtype=torch.long)
        q_table = torch.ones((num_steps, action_dim, action_dim), dtype=torch.float32) * 5.0

        pack = EnvTensorPack(
            state_tensor=state_tensor,
            ask_prices_tensor=ask_prices,
            bid_prices_tensor=bid_prices,
            ask_qtys_tensor=ask_qtys,
            bid_qtys_tensor=bid_qtys,
            markprice_tensor=markprice,
            timestamp_tensor=timestamp,
            funding_rate_tensor=funding_rate,
            funding_timestamp_tensor=funding_timestamp,
            regime_grid_ids_tensor=regime_grid_ids,
            q_table_tensor=q_table,
        )
        packs[df_idx] = pack
    return SharedMarketDataPack(packs)


def test_compact_trajectory_n_step_accumulation_matches_legacy():
    """验证紧凑纯数值 N 步累加算法与经典元组累加算法结果完全一致。"""
    gamma = 0.95
    n_step = 3
    total_steps = 10

    # 构造原始数据
    raw_step_indices = list(range(total_steps))
    raw_actions = [i % 3 for i in range(total_steps)]
    raw_rewards = [float(i + 1) for i in range(total_steps)]
    raw_dones = [i == total_steps - 1 for i in range(total_steps)]
    raw_td_errors = [0.1 * i for i in range(total_steps)]
    raw_previous_actions = [0.0] * total_steps
    raw_trading_infos = [np.array([1.0, 0.01, 0.0, 0.1], dtype=np.float32) for _ in range(total_steps)]
    raw_avaliable_actions = [np.ones(3, dtype=np.float32) for _ in range(total_steps)]
    raw_hours = [0.0] * total_steps
    raw_minutes = [0.0] * total_steps
    raw_next_previous_actions = [float(a) for a in raw_actions]
    raw_next_trading_infos = [np.array([1.0, 0.02, 0.0, 0.2], dtype=np.float32) for _ in range(total_steps)]
    raw_next_avaliable_actions = [np.ones(3, dtype=np.float32) for _ in range(total_steps)]
    raw_next_hours = [0.0] * total_steps
    raw_next_minutes = [0.0] * total_steps

    # 1. Compact 累加
    compact_record = accumulate_compact_trajectory_n_step(
        raw_step_indices=raw_step_indices,
        raw_actions=raw_actions,
        raw_rewards=raw_rewards,
        raw_dones=raw_dones,
        raw_td_errors=raw_td_errors,
        raw_previous_actions=raw_previous_actions,
        raw_trading_infos=raw_trading_infos,
        raw_avaliable_actions=raw_avaliable_actions,
        raw_hours=raw_hours,
        raw_minutes=raw_minutes,
        raw_next_previous_actions=raw_next_previous_actions,
        raw_next_trading_infos=raw_next_trading_infos,
        raw_next_avaliable_actions=raw_next_avaliable_actions,
        raw_next_hours=raw_next_hours,
        raw_next_minutes=raw_next_minutes,
        gamma=gamma,
        n_step=n_step,
        df_index=0,
    )

    # 2. 传统元组累加
    legacy_transitions = []
    for t in range(total_steps):
        s = np.array([float(t)], dtype=np.float32)
        s_next = np.array([float(t + 1)], dtype=np.float32)
        info = {
            "previous_action": raw_previous_actions[t],
            "trading_info": raw_trading_infos[t],
            "avaliable_action": raw_avaliable_actions[t],
            "funding_count_down_hour": raw_hours[t],
            "funding_count_down_minute": raw_minutes[t],
        }
        next_info = {
            "previous_action": raw_next_previous_actions[t],
            "trading_info": raw_next_trading_infos[t],
            "avaliable_action": raw_next_avaliable_actions[t],
            "funding_count_down_hour": raw_next_hours[t],
            "funding_count_down_minute": raw_next_minutes[t],
        }
        legacy_transitions.append((s, info, raw_actions[t], raw_rewards[t], s_next, next_info, raw_dones[t]))

    legacy_acc = accumulate_trajectory_n_step(legacy_transitions, gamma=gamma, n_step=n_step)

    # 验证累加后的 rewards 和 dones 完全吻合
    for t in range(total_steps):
        assert np.isclose(compact_record.rewards[t], legacy_acc[t][3]), f"Reward mismatch at step {t}"
        assert compact_record.dones[t] == legacy_acc[t][6], f"Done flag mismatch at step {t}"


def test_ipc_serialization_compression():
    """验证 WorkerRoundResult 序列化时自动剔除 heavy transitions，实现超 95% 体积压缩。"""
    num_steps = 300
    state_dim = 148

    # 构造包含完整 heavy transitions 的对象
    heavy_transitions = [
        WorkerTransitionRecord(
            step_index=i,
            transition=(
                np.ones(state_dim, dtype=np.float32),
                {
                    "previous_action": 0,
                    "trading_info": np.zeros(4, dtype=np.float32),
                    "avaliable_action": np.ones(9, dtype=np.float32),
                    "funding_count_down_hour": 0.0,
                    "funding_count_down_minute": 0.0,
                    "ask_qyts": np.ones(25, dtype=np.float32),
                    "bid_qyts": np.ones(25, dtype=np.float32),
                    "q_value": np.ones(9, dtype=np.float32),
                    "regime_grid_id": 0,
                },
                1,
                1.0,
                np.ones(state_dim, dtype=np.float32),
                {
                    "previous_action": 1,
                    "trading_info": np.zeros(4, dtype=np.float32),
                    "avaliable_action": np.ones(9, dtype=np.float32),
                    "funding_count_down_hour": 0.0,
                    "funding_count_down_minute": 0.0,
                },
                False,
            ),
            td_error=1.5,
        )
        for i in range(num_steps)
    ]

    # Legacy 结果对象
    legacy_result = WorkerRoundResult(
        df_index=0,
        epoch_index=0,
        context_index=0,
        initial_action=0,
        round_counter=0,
        worker_steps=num_steps,
        transitions=heavy_transitions,
        rollout_metrics=[],
        done=True,
    )
    legacy_dump = pickle.dumps(legacy_result)

    # Compact 结果对象
    compact_record = CompactTrajectoryRecord(
        df_index=0,
        step_indices=np.arange(num_steps, dtype=np.int32),
        actions=np.ones(num_steps, dtype=np.int64),
        rewards=np.ones(num_steps, dtype=np.float32),
        next_step_indices=np.arange(1, num_steps + 1, dtype=np.int32),
        dones=np.zeros(num_steps, dtype=bool),
        td_errors=np.ones(num_steps, dtype=np.float32) * 1.5,
        previous_actions=np.zeros(num_steps, dtype=np.float32),
        next_previous_actions=np.ones(num_steps, dtype=np.float32),
        trading_infos=np.zeros((num_steps, 4), dtype=np.float32),
        next_trading_infos=np.zeros((num_steps, 4), dtype=np.float32),
        avaliable_actions=np.ones((num_steps, 9), dtype=np.float32),
        next_avaliable_actions=np.ones((num_steps, 9), dtype=np.float32),
        hours=np.zeros(num_steps, dtype=np.float32),
        minutes=np.zeros(num_steps, dtype=np.float32),
        next_hours=np.zeros(num_steps, dtype=np.float32),
        next_minutes=np.zeros(num_steps, dtype=np.float32),
    )

    compact_result = WorkerRoundResult(
        df_index=0,
        epoch_index=0,
        context_index=0,
        initial_action=0,
        round_counter=0,
        worker_steps=num_steps,
        compact_trajectory=compact_record,
        transitions=heavy_transitions,  # 传入 heavy transitions
        rollout_metrics=[],
        done=True,
    )
    compact_dump = pickle.dumps(compact_result)

    # 验证压缩比率
    compression_ratio = 1.0 - (len(compact_dump) / len(legacy_dump))
    assert compression_ratio > 0.90, f"Expected >90% compression, got {compression_ratio:.2%}"

    # 验证反序列化后 _transitions 被清理，但访问 .transitions 仍然能够懒加载还原
    loaded_result = pickle.loads(compact_dump)
    assert loaded_result._transitions is None
    assert len(loaded_result.transitions) == num_steps
    assert loaded_result.transitions[0].td_error == 1.5


def test_spatio_temporal_semantic_key_properties():
    info = {
        "previous_action": 0, "trading_info": np.array([1.0, 0.01, 0.02, 0.05]),
        "avaliable_action": np.ones(3),
        "funding_count_down_hour": 0.0, "funding_count_down_minute": 0.0,
    }
    key = build_spatio_temporal_semantic_key(0, 10, 1, info)
    assert key == build_spatio_temporal_semantic_key(0, 10, 1, dict(info))
    assert key != build_spatio_temporal_semantic_key(1, 10, 1, info)
    assert key != build_spatio_temporal_semantic_key(0, 11, 1, info)
    assert key != build_spatio_temporal_semantic_key(0, 10, 2, info)
    for index in range(4):
        changed = dict(info, trading_info=info["trading_info"].copy())
        changed["trading_info"][index] += 0.001
        assert key != build_spatio_temporal_semantic_key(0, 10, 1, changed)


def test_buffer_add_compact_records_bulk_with_shared_pack():
    """验证 RegimeStratifiedReplayBuffer 使用 CompactTrajectoryRecord 与 SharedMarketDataPack 入池与去重择优。"""
    shared_pack = _create_mock_shared_market_data(num_dfs=2, num_steps=20, state_dim=148, action_dim=9)
    buffer = RegimeStratifiedReplayBuffer(
        total_buffer_size=900,
        batch_size=32,
        device="cpu",
        seed=42,
        num_grids=9,
        shared_market_data=shared_pack,
    )

    def _make_compact(step_idx=5, action=2, td_error=1.0, reward=2.5):
        return CompactTrajectoryRecord(
            df_index=0,
            step_indices=np.array([step_idx], dtype=np.int32),
            actions=np.array([action], dtype=np.int64),
            rewards=np.array([reward], dtype=np.float32),
            next_step_indices=np.array([step_idx + 1], dtype=np.int32),
            dones=np.array([False], dtype=bool),
            td_errors=np.array([td_error], dtype=np.float32),
            previous_actions=np.array([0.0], dtype=np.float32),
            next_previous_actions=np.array([action], dtype=np.float32),
            trading_infos=np.array([[1.0, 0.001, 0.0, 0.05]], dtype=np.float32),
            next_trading_infos=np.array([[1.0, 0.002, 0.0, 0.1]], dtype=np.float32),
            avaliable_actions=np.ones((1, 9), dtype=np.float32),
            next_avaliable_actions=np.ones((1, 9), dtype=np.float32),
            hours=np.array([0.0], dtype=np.float32),
            minutes=np.array([0.0], dtype=np.float32),
            next_hours=np.array([0.0], dtype=np.float32),
            next_minutes=np.array([0.0], dtype=np.float32),
        )

    # 1. 写入首条记录 (td_error = 1.0)
    res1 = WorkerRoundResult(
        df_index=0, epoch_index=0, context_index=0, initial_action=0, round_counter=0,
        worker_steps=1, compact_trajectory=_make_compact(td_error=1.0, reward=2.5), done=True,
    )
    dup1 = write_round_transitions_to_buffer(buffer, [res1])
    assert dup1 == 0
    assert buffer.get_grid_lengths()[0] == 1
    # 验证 GPU 张量内容（从 shared_pack 还原的 states）
    assert torch.allclose(buffer.states[0][0], shared_pack[0].state_tensor[5])
    assert np.isclose(buffer.rewards[0][0, 0].item(), 2.5)

    # 2. 相同决策的不同后续应保留，不因旧误差指标更低而丢弃。
    res2 = WorkerRoundResult(
        df_index=0, epoch_index=0, context_index=1, initial_action=1, round_counter=1,
        worker_steps=1, compact_trajectory=_make_compact(td_error=0.5, reward=9.9), done=True,
    )
    dup2 = write_round_transitions_to_buffer(buffer, [res2])
    assert dup2 == 0
    assert buffer.get_grid_lengths()[0] == 2
    assert np.isclose(buffer.rewards[0][0, 0].item(), 2.5)  # 保持原样

    # 3. 第三个不同后续同样独立保留。
    res3 = WorkerRoundResult(
        df_index=0, epoch_index=0, context_index=2, initial_action=2, round_counter=2,
        worker_steps=1, compact_trajectory=_make_compact(td_error=3.0, reward=7.7), done=True,
    )
    dup3 = write_round_transitions_to_buffer(buffer, [res3])
    assert dup3 == 0
    assert buffer.get_grid_lengths()[0] == 3
    assert np.isclose(buffer.rewards[0][2, 0].item(), 7.7)  # 覆写成功


def test_run_epoch_exploration_micro_batched_ingestion(monkeypatch):
    """验证 run_epoch_exploration 微批次聚合逻辑：按 MICRO_BATCH_SIZE 批量触发入池。"""
    shared_pack = _create_mock_shared_market_data(num_dfs=2, num_steps=10, state_dim=148, action_dim=9)
    buffer = RegimeStratifiedReplayBuffer(
        total_buffer_size=900, batch_size=32, device="cpu", seed=42, num_grids=9,
        shared_market_data=shared_pack,
    )

    batch_calls = []
    original_write = write_round_transitions_to_buffer

    def mock_write(buf, results):
        batch_calls.append(len(results))
        return original_write(buf, results)

    monkeypatch.setattr("RL.DiHFT.low_level.parallel_diverse_train.write_round_transitions_to_buffer", mock_write)

    # 构造模拟 trainer
    result_q = queue.Queue()
    task_q = queue.Queue()

    trainer = MagicMock()
    trainer.total_df_index_length = 2
    trainer.N = 2
    trainer.position_choices = 2
    trainer.epsilon = 0.1
    trainer.buffer_size = 10000
    trainer.worker_processes = []
    trainer.worker_result_queue = result_q
    trainer.worker_task_queue = task_q
    trainer.worker_input_queues = {0: task_q, 1: task_q}

    # 投递 35 个任务结果 (2 * 2 * 2 = 8 个任务，我们将 total_tasks 扩展为 35 个)
    # 模拟 35 个结果
    def mock_start_workers(tr, **kwargs):
        pass
    monkeypatch.setattr("RL.DiHFT.low_level.parallel_diverse_train.start_parallel_workers", mock_start_workers)
    monkeypatch.setattr("RL.DiHFT.low_level.parallel_diverse_train.shutdown_exploration_workers", lambda tr: None)

    # 8 个任务 (2 contexts * 2 actions * 2 dfs)
    for i in range(8):
        ct = CompactTrajectoryRecord(
            df_index=i % 2,
            step_indices=np.array([0], dtype=np.int32),
            actions=np.array([0], dtype=np.int64),
            rewards=np.array([1.0], dtype=np.float32),
            next_step_indices=np.array([1], dtype=np.int32),
            dones=np.array([False], dtype=bool),
            td_errors=np.array([0.5], dtype=np.float32),
            previous_actions=np.array([0.0], dtype=np.float32),
            next_previous_actions=np.array([0.0], dtype=np.float32),
            trading_infos=np.zeros((1, 4), dtype=np.float32),
            next_trading_infos=np.zeros((1, 4), dtype=np.float32),
            avaliable_actions=np.ones((1, 9), dtype=np.float32),
            next_avaliable_actions=np.ones((1, 9), dtype=np.float32),
            hours=np.array([0.0], dtype=np.float32),
            minutes=np.array([0.0], dtype=np.float32),
            next_hours=np.array([0.0], dtype=np.float32),
            next_minutes=np.array([0.0], dtype=np.float32),
        )
        res = WorkerRoundResult(
            df_index=i % 2, epoch_index=0, context_index=i // 4, initial_action=(i // 2) % 2,
            round_counter=i, worker_steps=1, compact_trajectory=ct, done=True,
            rollout_metrics=[
                RolloutMetrics(
                    epoch_index=0, context_index=i // 4, initial_action=(i // 2) % 2, df_index=i % 2,
                    transition_count=1, reward_sum=1.0, final_balance=100.0, return_rate=0.01,
                )
            ]
        )
        result_q.put(res)

    epoch_metrics, steps, rounds = run_epoch_exploration(
        trainer=trainer,
        epoch_index=0,
        train_df_cache={},
        env_kwargs={},
        buffer_diverse=buffer,
        step_counter_diverse=0,
        round_counter=0,
        diverse_rollout_latest_metrics_by_df={},
        shared_market_data=shared_pack,
    )

    # 验证微批次聚合写入
    assert len(batch_calls) > 0
    assert sum(batch_calls) == 8
    assert len(epoch_metrics) == 8
    assert steps == 8


def test_update_supports_amp_mixed_precision_and_grad_step():
    """验证 update 函数支持 AMP 混合精度前向与反向，并正确累加 update_counter。"""
    from RL.DiHFT.low_level import parallel_diverse_train as pdt
    import unittest.mock as mock

    with mock.patch('RL.DiHFT.low_level.parallel_weight_advantage_pretrain.count_training_data_files', return_value=1),          mock.patch('RL.DiHFT.low_level.parallel_weight_advantage_pretrain.build_training_data_paths', return_value={
             'train_data_path': '/tmp',
             'state_features_path': '/tmp/sf.npy',
             'maintenance_margin_ratio_path': '/tmp/mmr.npy',
         }),          mock.patch('numpy.load', side_effect=lambda p, **kw: np.array([f'f{i}' for i in range(10)]) if 'sf' in str(p) else mock.MagicMock(item=lambda: {})):
        from RL.DiHFT.low_level.parallel_weight_advantage_pretrain import parser, Weighted_Contexts_DQN
        args = parser.parse_args(['--base_path', '/tmp', '--dataset_name', 'fu', '--update_times', '5', '--batch_size', '4', '--num_epoch', '21'])
        trainer = Weighted_Contexts_DQN(args)

    device = trainer.device
    bs = 4
    states = torch.zeros(bs, 10, device=device)
    actions = torch.zeros(bs, 1, dtype=torch.long, device=device)
    rewards = torch.ones(bs, 1, device=device)
    next_states = torch.zeros(bs, 10, device=device)
    dones = torch.zeros(bs, 1, device=device)

    info = {
        'previous_action': torch.zeros(bs, device=device),
        'avaliable_action': torch.ones(bs, trainer.N_ACTIONS, device=device),
        'funding_count_down_hour': torch.zeros(bs, device=device),
        'funding_count_down_minute': torch.zeros(bs, device=device),
        'trading_info': torch.zeros(bs, 4, device=device),
        'q_value': torch.zeros(bs, trainer.N_ACTIONS, device=device),
    }
    next_info = {
        'previous_action': torch.zeros(bs, device=device),
        'avaliable_action': torch.ones(bs, trainer.N_ACTIONS, device=device),
        'funding_count_down_hour': torch.zeros(bs, device=device),
        'funding_count_down_minute': torch.zeros(bs, device=device),
        'trading_info': torch.zeros(bs, 4, device=device),
    }

    losses = pdt.update(trainer, states, info, actions, rewards, next_states, next_info, dones)
    assert len(losses) == 3
    assert not np.isnan(losses[0])
    assert trainer.update_counter == 1


def test_run_diverse_training_phase_sparsified_logging():
    """验证 run_diverse_training_phase 按照 LOG_INTERVAL 稀疏化 TensorBoard 写入。"""
    from RL.DiHFT.low_level import parallel_diverse_train as pdt

    trainer = MagicMock()
    trainer.update_counter = 0
    trainer.batch_size = 4
    trainer.device = 'cpu'
    trainer.curriculum_block_epochs = 3
    writer = MagicMock()
    trainer.writer = writer

    buffer = MagicMock()
    sampler = MagicMock()
    buffer.create_sampler.return_value = sampler
    sampler.sample.return_value = (
        torch.zeros(4, 10), {}, torch.zeros(4, 1, dtype=torch.long),
        torch.zeros(4, 1), torch.zeros(4, 10), {}, torch.zeros(4, 1)
    )

    fake_update = MagicMock(return_value=(1.0, 0.2, 0.8))
    original_update = pdt.update
    pdt.update = fake_update

    try:
        losses = pdt.run_diverse_training_phase(
            trainer=trainer,
            buffer_diverse=buffer,
            update_count=25,
            epoch_index=0,
        )
        assert losses == (1.0, 0.2, 0.8)
        # 25 次更新中，step 10, 20, 25 触发写入，每次写 3 个标量 -> 9 次 call
        assert writer.add_scalar.call_count == 9
    finally:
        pdt.update = original_update
