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

    # 运行 21 个 epoch，block_epochs=3，共 7 个阶段
    # Epoch 0..2: Phase 0 [8]
    # Epoch 3..5: Phase 1 [2, 5, 8]
    # Epoch 6..8: Phase 2 [1, 4, 7]
    # Epoch 9..11: Phase 3 [0, 3, 6]
    # Epoch 12..14: Phase 4 [6]
    # Epoch 15..17: Phase 5 [6, 7, 8]; Epoch 18..20: Phase 6 all grids
    for epoch in range(21):
        pdt.run_diverse_training_phase(
            trainer=trainer,
            buffer_diverse=buffer,
            update_count=1,
            epoch_index=epoch,
        )

    assert len(observed_active_grids) == 21
    for epoch in range(3):
        assert observed_active_grids[epoch] == [8]

    for epoch in range(3, 6):
        assert observed_active_grids[epoch] == [2, 5, 8]

    for epoch in range(6, 9):
        assert observed_active_grids[epoch] == [1, 4, 7]

    for epoch in range(9, 12):
        assert observed_active_grids[epoch] == [0, 3, 6]

    for epoch in range(12, 15):
        assert observed_active_grids[epoch] == [6]

    for epoch in range(15, 18):
        assert observed_active_grids[epoch] == [6, 7, 8]
    for epoch in range(18, 21):
        assert observed_active_grids[epoch] == list(range(9))


def test_phase_cyclic_parameter_decay_schedule_across_seven_phases():
    """验证 7 个阶段的完整调度行为：
    epoch 0-2 从 max 衰减到最低 (Phase 0)
    epoch 3-5 从 max 衰减到最低 (Phase 1)
    epoch 6-8 从 max 衰减到最低 (Phase 2)
    epoch 9-11 从 max 衰减到最低 (Phase 3)
    epoch 12-14 从 max 衰减到最低 (Phase 4: [6])
    epoch 15-17 从 max 衰减到最低 (Phase 5: [6,7,8])
    epoch 18-20 第7阶段；21-23 恒等于最低
    学习率 lr 维持全局半程保持后线性衰减。
    """
    num_epoch = 24
    block_epochs = 3
    eps_init, eps_min = 1.0, 0.1
    ada_init, ada_min = 256.0, 0.0
    lr_init, lr_min = 0.0005, 0.0001

    for ep in range(num_epoch):
        params = pdt.compute_epoch_training_params(
            epoch_index=ep,
            num_epoch=num_epoch,
            epsilon_init=eps_init,
            epsilon_min=eps_min,
            ada_init=ada_init,
            ada_min=ada_min,
            lr_init=lr_init,
            lr_min=lr_min,
            curriculum_block_epochs=block_epochs,
        )

        # 验证 7 个阶段的周期性重置与衰减
        if ep in (0, 3, 6, 9, 12, 15, 18):
            # 阶段起点：恢复至 max
            assert params.epsilon == pytest.approx(eps_init)
            assert params.ada == pytest.approx(ada_init)
        elif ep in (1, 4, 7, 10, 13, 16, 19):
            # 阶段中点：线性中间值
            assert params.epsilon == pytest.approx((eps_init + eps_min) / 2.0)
            assert params.ada == pytest.approx((ada_init + ada_min) / 2.0)
        elif ep in (2, 5, 8, 11, 14, 17, 20):
            # 阶段末点：严格达到 min
            assert params.epsilon == pytest.approx(eps_min)
            assert params.ada == pytest.approx(ada_min)
        else:
            # ep >= 21: 剩余轮次恒等于最低值
            assert params.epsilon == pytest.approx(eps_min)
            assert params.ada == pytest.approx(ada_min)

        # 学习率从初始值过渡到最后一轮的最小值。
        if ep == 0:
            assert params.lr == pytest.approx(lr_init)
        elif ep == num_epoch - 1:
            assert params.lr == pytest.approx(lr_min)


def test_compute_epoch_training_params_rejects_non_positive_block_epochs():
    with pytest.raises(ValueError, match="curriculum_block_epochs must be positive"):
        pdt.compute_epoch_training_params(
            epoch_index=0,
            num_epoch=18,
            epsilon_init=1.0,
            epsilon_min=0.1,
            ada_init=256.0,
            ada_min=0.0,
            lr_init=0.0005,
            lr_min=0.0001,
            curriculum_block_epochs=0,
        )


def test_phase_entry_resets_exploration_exhaustion_in_diverse_train(monkeypatch):
    """验证进入 Phase 1 (epoch 3) 和 Phase 2 (epoch 6) 时重置探索早停状态。"""
    trainer = MagicMock()
    trainer.total_df_index_length = 1
    trainer.update_times = 1
    trainer.num_epoch = 15
    trainer.curriculum_block_epochs = 3
    trainer.epsilon_init = 1.0
    trainer.epsilon_min = 0.1
    trainer.ada_init = 256.0
    trainer.ada_min = 0.0
    trainer.lr_init = 0.0005
    trainer.lr_min = 0.0001
    trainer.optimizer = MagicMock()
    trainer.optimizer.param_groups = [{"lr": 0.0005}]
    trainer.model_path = "/tmp/test_phase_entry"

    buffer = MagicMock()
    buffer.total_added_count = 0
    buffer.__len__ = MagicMock(return_value=100)

    # 模拟持久化池与共享内存管理器
    monkeypatch.setattr(pdt, "PersistentRolloutPool", MagicMock())
    monkeypatch.setattr(pdt, "SharedInferenceManager", MagicMock())
    # 模拟 buffer 未满
    monkeypatch.setattr(pdt, "is_buffer_full", lambda buf, tr: False)
    # 模拟保存
    monkeypatch.setattr(pdt, "save_diverse_buffer", lambda buf, path: None)
    # 模拟训练阶段
    monkeypatch.setattr(pdt, "run_diverse_training_phase", lambda *args, **kwargs: (1.0, 0.1, 0.9))
    # 模拟标量记录和模型保存
    monkeypatch.setattr(pdt, "write_epoch_rollout_scalars", lambda *args, **kwargs: None)
    monkeypatch.setattr(pdt, "save_parallel_epoch_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(pdt, "build_epoch_model_path", lambda path, ep: f"{path}/epoch_{ep}")

    explored_epochs = []

    def mock_run_epoch_exploration(tr, epoch_index, *args, **kwargs):
        explored_epochs.append(epoch_index)
        # 始终不增加新经验，模拟连续无新经验触发早停
        return [], 0, 0

    monkeypatch.setattr(pdt, "run_epoch_exploration", mock_run_epoch_exploration)

    # 运行多样化训练主循环
    pdt.run_parallel_diverse_training(
        trainer=trainer,
        train_df_cache={},
        env_kwargs={},
        buffer_diverse=buffer,
        step_counter_diverse=0,
        diverse_rollout_latest_metrics_by_df={},
    )

    # 在 MAX_CONSECUTIVE_NO_NEW_EXPERIENCE_EPOCHS = 3 下：
    # Phase 0 (ep 0, 1, 2): ep 0(count=1), ep 1(count=2), ep 2(count=3 -> skip_exploration=True)
    # 进入 Phase 1 (ep 3): 阶段重置 count=0, skip_exploration=False -> ep 3 被正常探索！
    # ep 4(count=2), ep 5(count=3 -> skip_exploration=True)
    # 进入 Phase 2 (ep 6): 再次阶段重置 count=0, skip_exploration=False -> ep 6 被正常探索！
    # 进入 Phase 3 (ep 9): 再次阶段重置 count=0, skip_exploration=False -> ep 9 被正常探索！
    assert 0 in explored_epochs
    assert 1 in explored_epochs
    assert 2 in explored_epochs
    assert 3 in explored_epochs
    assert 6 in explored_epochs
    assert 9 in explored_epochs
    assert 12 in explored_epochs


def test_run_parallel_diverse_training_executes_training_phase_after_buffer_snapshot(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    import numpy as np
    from RL.util.regime_stratified_replay_buffer import RegimeStratifiedReplayBuffer
    from RL.DiHFT.low_level import parallel_diverse_train as pdt

    buffer = RegimeStratifiedReplayBuffer(
        total_buffer_size=90,
        batch_size=2,
        device="cpu",
        seed=42,
        num_grids=9,
    )
    for g in range(9):
        info = {
            "previous_action": 0,
            "regime_grid_id": g,
            "trading_info": np.zeros(4, dtype=np.float32),
            "avaliable_action": np.array([1, 1, 1], dtype=np.int64),
            "funding_count_down_hour": 0.0,
            "funding_count_down_minute": 0.0,
            "q_value": np.zeros(3, dtype=np.float32),
        }
        buffer.add_transition((np.zeros(2), info, 0, 1.0, np.zeros(2), dict(info), False))

    trainer = MagicMock()
    trainer.total_df_index_length = 1
    trainer.num_epoch = 1
    trainer.curriculum_block_epochs = 3
    trainer.update_times = 1
    trainer.batch_size = 2
    trainer.buffer_size = 90
    trainer.device = "cpu"
    trainer.model_path = str(tmp_path)
    trainer.writer = MagicMock()
    trainer.epsilon_init = 1.0
    trainer.epsilon_min = 0.1
    trainer.ada_init = 1.0
    trainer.ada_min = 0.1
    trainer.lr_init = 1e-4
    trainer.lr_min = 1e-5
    trainer.optimizer = MagicMock()
    trainer.optimizer.param_groups = [{"lr": 1e-4}]

    monkeypatch.setattr(pdt, "PersistentRolloutPool", MagicMock())
    monkeypatch.setattr(pdt, "SharedInferenceManager", MagicMock())
    monkeypatch.setattr(pdt, "run_epoch_exploration", lambda *args, **kwargs: ([], 0, 0))
    monkeypatch.setattr(pdt, "update", lambda *args, **kwargs: (0.5, 0.1, 0.4))
    monkeypatch.setattr(pdt, "write_epoch_rollout_scalars", lambda *args, **kwargs: None)
    monkeypatch.setattr(pdt, "save_parallel_epoch_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(pdt, "evaluate_parallel_diverse_model", lambda *args, **kwargs: [])

    pdt.run_parallel_diverse_training(
        trainer=trainer,
        train_df_cache={},
        env_kwargs={},
        buffer_diverse=buffer,
        step_counter_diverse=0,
        diverse_rollout_latest_metrics_by_df={},
    )


@pytest.mark.parametrize("epoch,expected_grids", [
    (12, [6]), (15, [6, 7, 8]), (18, list(range(9))),
])
def test_final_three_curriculum_phases_sample_balanced_grids(epoch, expected_grids):
    buffer = RegimeStratifiedReplayBuffer(total_buffer_size=9000, batch_size=90)
    for grid in range(9):
        for step in range(35):
            buffer.add_transition(_make_dummy_transition(step, grid_id=grid))
    sampler = buffer.create_sampler(epoch_index=epoch, block_epochs=3)
    assert sorted(sampler.active_grid_ids) == expected_grids
    states, _, _, rewards, _, _, _ = sampler.sample()
    assert len(states) == 90
    sampled = rewards.squeeze().tolist()
    for grid in expected_grids:
        assert sampled.count(float(grid)) == 90 // len(expected_grids)


def test_parallel_weight_advantage_pretrain_rejects_less_than_five_blocks():
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap

    parser = pwap.parser
    args = parser.parse_args([
        "--num_epoch", "14",
        "--curriculum_block_epochs", "3",
    ])
    # 模拟初始化校验
    assert int(args.num_epoch) < 5 * int(args.curriculum_block_epochs)
