import numpy as np
import pytest
from RL.DiHFT.low_level.parallel_diverse_train import (
    build_semantic_transition_key,
    WorkerTransitionRecord,
    WorkerRoundResult,
    write_round_transitions_to_buffer,
)
from RL.util.regime_stratified_replay_buffer import RegimeStratifiedReplayBuffer


def test_build_semantic_transition_key_ignores_qvalue_and_float_noise():
    """语义键应对 q_value 差异与连续浮点数噪声（收益率、回撤）保持不变。"""
    info_base = {
        "previous_action": 0,
        "q_value": np.array([100.0, 50.0, 10.0]),
        "trading_info": np.array([1.0, 0.001234, 0.00045, 0.05]),  # 1.0=多头, 收益0.0012, 回撤0.0004, 时间0.05
    }

    state = np.array([1.0, 2.0])
    key_base = build_semantic_transition_key(
        state=state,
        action=2,
        info=info_base,
    )

    # 1. 改变 q_value（例如不同的前序监督值或不同模型输出）
    info_different_q = {
        "previous_action": 0,
        "q_value": np.array([999.0, -10.0, 0.0]),
        "trading_info": np.array([1.0, 0.001234, 0.00045, 0.05]),
    }
    key_q = build_semantic_transition_key(
        state=state,
        action=2,
        info=info_different_q,
    )
    assert key_base == key_q

    # 2. 改变 trading_info 中的连续浮点收益率与回撤（路径历史噪声）
    info_different_floats = {
        "previous_action": 0,
        "q_value": np.array([100.0, 50.0, 10.0]),
        "trading_info": np.array([1.0, 0.008888, 0.00199, 0.05]),
    }
    key_floats = build_semantic_transition_key(
        state=state,
        action=2,
        info=info_different_floats,
    )
    assert key_base == key_floats


def test_build_semantic_transition_key_distinguishes_discrete_decisions():
    """语义键必须区分不同的市场状态、动作、前序动作与持仓方向。"""
    state = np.array([1.0, 2.0])
    info = {
        "previous_action": 0,
        "trading_info": np.array([1.0, 0.001, 0.0, 0.05]),
    }
    base = build_semantic_transition_key(state=state, action=2, info=info)

    # 不同的 state
    assert base != build_semantic_transition_key(state=np.array([1.0, 2.1]), action=2, info=info)
    # 不同的 action
    assert base != build_semantic_transition_key(state=state, action=0, info=info)
    # 不同的 previous_action
    info_prev_act = {"previous_action": 1, "trading_info": np.array([1.0, 0.001, 0.0, 0.05])}
    assert base != build_semantic_transition_key(state=state, action=2, info=info_prev_act)
    # 不同的持仓方向 (空头 -1.0 vs 多头 1.0)
    info_short = {"previous_action": 0, "trading_info": np.array([-1.0, 0.001, 0.0, 0.05])}
    assert base != build_semantic_transition_key(state=state, action=2, info=info_short)


def test_worker_transition_record_supports_td_error():
    """WorkerTransitionRecord 必须支持可选的 td_error 字段，且默认为 0.0。"""
    t = (np.array([1.0]), {"previous_action": 0}, 1, 0.5, np.array([2.0]), {}, False)
    rec_default = WorkerTransitionRecord(step_index=0, transition=t)
    assert rec_default.td_error == 0.0

    rec_custom = WorkerTransitionRecord(step_index=1, transition=t, td_error=3.14)
    assert rec_custom.td_error == 3.14


def test_write_round_transitions_semantic_dedup_and_td_error_replacement():
    """验证 write_round_transitions_to_buffer 基于语义键去重，并在新样本 TD-Error 更高时执行择优替换。"""
    buffer = RegimeStratifiedReplayBuffer(
        total_buffer_size=900,
        batch_size=32,
        device="cpu",
        seed=42,
        num_grids=9,
    )

    def make_transition(tag: str, q_val: float = 0.0, pnl_float: float = 0.0):
        # 相同语义：df_index=0, step_index=10, action=2, previous_action=0, pos_dir=1
        info = {
            "previous_action": 0,
            "regime_grid_id": 0,
            "q_value": np.array([q_val, 0.0, 0.0]),
            "trading_info": np.array([1.0, pnl_float, 0.0, 0.05]),
            "tag": tag,
        }
        return (np.array([1.0, 2.0]), info, 2, 0.5, np.array([1.5, 2.5]), dict(info), False)

    # 1. 插入第一条经验，td_error = 1.0
    t1 = make_transition("first", q_val=10.0, pnl_float=0.001)
    round_1 = WorkerRoundResult(
        df_index=0,
        epoch_index=0,
        context_index=0,
        initial_action=0,
        round_counter=0,
        worker_steps=1,
        transitions=[WorkerTransitionRecord(step_index=10, transition=t1, td_error=1.0)],
        rollout_metrics=[],
        done=True,
    )
    duplicates_1 = write_round_transitions_to_buffer(buffer, [round_1])
    assert duplicates_1 == 0
    assert buffer.get_grid_lengths()[0] == 1
    assert buffer.buffers[0].memory[0].info["tag"] == "first"

    # 2. 遇到相同语义的经验（不同 q_val 与 float），但 td_error = 0.4（更低），应丢弃
    t2 = make_transition("second_lower_td", q_val=999.0, pnl_float=0.009)
    round_2 = WorkerRoundResult(
        df_index=0,
        epoch_index=0,
        context_index=1,
        initial_action=1,
        round_counter=0,
        worker_steps=1,
        transitions=[WorkerTransitionRecord(step_index=10, transition=t2, td_error=0.4)],
        rollout_metrics=[],
        done=True,
    )
    duplicates_2 = write_round_transitions_to_buffer(buffer, [round_2])
    assert duplicates_2 == 1
    assert buffer.get_grid_lengths()[0] == 1
    assert buffer.buffers[0].memory[0].info["tag"] == "first"  # 保持原样

    # 3. 遇到相同语义的经验，但 td_error = 2.8（更高），应就地替换旧样本
    t3 = make_transition("third_higher_td", q_val=-50.0, pnl_float=-0.002)
    round_3 = WorkerRoundResult(
        df_index=0,
        epoch_index=0,
        context_index=2,
        initial_action=2,
        round_counter=0,
        worker_steps=1,
        transitions=[WorkerTransitionRecord(step_index=10, transition=t3, td_error=2.8)],
        rollout_metrics=[],
        done=True,
    )
    duplicates_3 = write_round_transitions_to_buffer(buffer, [round_3])
    assert duplicates_3 == 0  # 替换不计为普通丢弃重复
    assert buffer.get_grid_lengths()[0] == 1  # 长度不变
    assert buffer.buffers[0].memory[0].info["tag"] == "third_higher_td"  # 成功替换为更高 TD-Error 的样本


def test_multi_step_replay_buffer_multi_info_replace_in_place():
    """测试真实的 Multi_step_ReplayBuffer_multi_info 就地替换。"""
    from RL.util.replay_buffer_DQN import Multi_step_ReplayBuffer_multi_info

    buf = Multi_step_ReplayBuffer_multi_info(
        buffer_size=10,
        batch_size=1,
        device="cpu",
        seed=42,
        gamma=0.99,
        n_step=1,
    )

    t1 = (np.array([1.0]), {"tag": "old"}, 0, 1.0, np.array([2.0]), {"tag": "old"}, False)
    t2 = (np.array([1.0]), {"tag": "new"}, 0, 2.0, np.array([2.0]), {"tag": "new"}, False)

    buf.add(*t1)
    assert len(buf) == 1
    assert buf.memory[0].info["tag"] == "old"

    buf.replace(0, *t2)
    assert len(buf) == 1
    assert buf.memory[0].info["tag"] == "new"
    assert buf.memory[0].reward == 2.0


def test_explore_round_computes_td_error_on_transitions(monkeypatch):
    """测试 DfRolloutWorkerRunner.explore_round 探索过程中会计算单步 td_error 并存入 record。"""
    import torch
    from RL.DiHFT.low_level import parallel_diverse_train as pdt

    class FakeEnv:
        def __init__(self):
            self.step_calls = 0
            self.unrealized_pnl = 0.0
            self.wallet_balance = 100.0

        def reset(self):
            return (
                np.zeros(2),
                {
                    "previous_action": 0,
                    "avaliable_action": [1, 1, 1],
                    "avaiable_action_list": [0, 1, 2],
                    "funding_count_down_hour": 0,
                    "funding_count_down_minute": 0,
                    "trading_info": np.zeros(4),
                    "q_value": np.array([10.0, 5.0, 0.0]),
                },
            )

        def step(self, action):
            self.step_calls += 1
            done = self.step_calls >= 2
            return (
                np.ones(2),
                1.5,
                done,
                {
                    "previous_action": action,
                    "avaliable_action": [1, 1, 1],
                    "avaiable_action_list": [0, 1, 2],
                    "funding_count_down_hour": 0,
                    "funding_count_down_minute": 0,
                    "trading_info": np.zeros(4),
                    "q_value": np.array([8.0, 4.0, 1.0]),
                },
            )

    class FakeModel(torch.nn.Module):
        def __call__(self, **kwargs):
            # 返回 Q 值: batch_size=1, N=1, N_ACTIONS=3: Q=[2.0, 1.0, 0.0]
            return torch.tensor([[[2.0, 1.0, 0.0]]], dtype=torch.float32)

    monkeypatch.setattr(pdt, "build_initial_state", lambda *args, **kwargs: (None, None, None, "init"))
    monkeypatch.setattr(pdt, "create_demo_env", lambda *args, **kwargs: FakeEnv())
    monkeypatch.setattr(pdt, "create_parallel_worker_model", lambda config: FakeModel())

    worker_config = {
        "df_indices": [0],
        "train_df_by_df": {0: "df0"},
        "env_kwargs": {},
        "device": "cpu",
        "leverage_choices": [1],
        "position_list": [0.0],
        "initial_wallet_balance": 100.0,
        "initial_unrealized_pnL": 0.0,
    }
    runner = pdt.DfRolloutWorkerRunner(worker_config)
    runner.reset_task(pdt.ResetWorkerTask(df_index=0, epoch_index=0, context_index=0, initial_action=0))

    result = runner.explore_round(
        pdt.ExploreWorkerRound(
            df_index=0,
            epoch_index=0,
            context_index=0,
            initial_action=0,
            round_counter=0,
            state_dict={},
            epsilon=0.0,  # 贪婪选择 action=0 (Q=2.0)
        )
    )

    assert len(result.transitions) == 2
    # action=0: Agent Q=2.0, Teacher Q=10.0 -> td_error = |10.0 - 2.0| = 8.0 > 0
    for record in result.transitions:
        assert record.td_error > 0.0
    assert result.transitions[0].td_error == pytest.approx(8.0)
