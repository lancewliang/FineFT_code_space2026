from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from RL.DiHFT.low_level import parallel_diverse_train as diverse
from RL.DiHFT.low_level import parallel_pretrain as pretrain
from RL.util.regime_stratified_replay_buffer import (
    CompactTrajectoryRecord,
    RegimeStratifiedReplayBuffer,
    build_semantic_transition_key,
)


class ConstantQ(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(float(value)))

    def forward(self, state, **kwargs):
        return self.value.expand(len(state), 1, 3)

    def get_best_q(self, state, **kwargs):
        return self.value.expand(len(state), 1)


@pytest.mark.parametrize("stage", ["pretrain", "diverse"])
@pytest.mark.parametrize("n_step", [1, 18])
@pytest.mark.parametrize("next_value", [-10.0, 10.0])
def test_real_update_bootstraps_at_n_step_horizon(monkeypatch, stage, n_step, next_value):
    captured = {}
    net = ConstantQ(0.0)
    trainer = SimpleNamespace(
        device="cpu", batch_size=2, N=1, N_ACTIONS=3, gamma=0.992,
        n_step=n_step, eval_net=net, target_net=ConstantQ(next_value),
        if_use_hubber_loss=False, neighbor_size=1, ada=0.0,
        grad_clip=None, tau=0.0, update_counter=0,
        optimizer=torch.optim.SGD(net.parameters(), lr=0.01),
    )

    def capture_pretrain(prediction, target):
        captured["target"] = target.detach().clone()
        return (prediction - target).square()

    def capture_diverse(errors, neighbor_size):
        # Current Q is zero, so the residual is the actual Bellman target.
        captured["target"] = errors.detach().clone().squeeze(-1)
        return torch.ones(2, 1), errors.square().mean()

    trainer.loss_func_pretrain = capture_pretrain
    module = pretrain if stage == "pretrain" else diverse
    monkeypatch.setattr(module, "calculate_paper_supervisor_kl_loss", lambda values, *args: values.sum() * 0)
    if stage == "diverse":
        monkeypatch.setattr(module, "calculate_paper_partial_loss", capture_diverse)
    info = {
        "previous_action": torch.zeros(2),
        "avaliable_action": torch.ones(2, 3),
        "funding_count_down_hour": torch.zeros(2),
        "funding_count_down_minute": torch.zeros(2),
        "trading_info": torch.zeros(2, 4),
        "q_value": torch.zeros(2, 3),
    }
    update = pretrain.update_pretrain if stage == "pretrain" else diverse.update
    update(
        trainer, torch.zeros(2, 2), info, torch.zeros(2, 1, dtype=torch.long),
        torch.tensor([[2.0], [3.0]]), torch.ones(2, 2), info,
        torch.tensor([[0.0], [1.0]]),
    )
    expected = torch.tensor([[2.0 + 0.992 ** n_step * next_value], [3.0]])
    torch.testing.assert_close(captured["target"], expected)


def transition(step=0, reward=1.0, grid=0):
    info = {
        "previous_action": 2, "regime_grid_id": grid,
        "trading_info": np.array([1.0, 0.01, 0.02, 9 / 180], dtype=np.float32),
        "avaliable_action": np.ones(3, dtype=np.float32),
        "funding_count_down_hour": 1.0, "funding_count_down_minute": 2.0,
        "q_value": np.zeros(3, dtype=np.float32),
    }
    return (np.array([step, 2.0], dtype=np.float32), info, 2, reward,
            np.array([step + 1, 2.0], dtype=np.float32), deepcopy(info), False)


def buffer(capacity=100, seed=42):
    return RegimeStratifiedReplayBuffer(capacity, batch_size=2, num_grids=1, seed=seed)


@pytest.mark.parametrize("field,index,value", [
    ("trading_info", 0, 0.5), ("trading_info", 1, -0.01),
    ("trading_info", 2, 0.03), ("trading_info", 3, 10 / 180),
    ("avaliable_action", 0, 0.0), ("previous_action", None, 1),
    ("funding_count_down_hour", None, 2.0),
    ("funding_count_down_minute", None, 3.0),
])
def test_model_visible_holding_context_is_not_merged(field, index, value):
    first = transition()
    second = deepcopy(first)
    if index is None:
        second[1][field] = value
    else:
        second[1][field][index] = value
    replay = buffer()
    replay.add_transition(first)
    replay.add_transition(second)
    assert len(replay) == 2
    assert len(replay.decision_groups[0]) == 2


def test_identity_uses_float32_model_precision_and_ignores_teacher_opinion():
    first = transition()
    second = deepcopy(first)
    second[1]["q_value"][:] = 999
    second[1]["trading_info"] = second[1]["trading_info"].astype(np.float64) + 1e-12
    assert build_semantic_transition_key(first[0], first[2], first[1]) == build_semantic_transition_key(second[0], second[2], second[1])


@pytest.mark.parametrize("component", ["reward", "next_state", "next_holding", "next_mask", "next_time", "done"])
def test_different_learning_outcomes_can_coexist(component):
    first = transition()
    second = list(deepcopy(first))
    if component == "reward":
        second[3] = -9.0
    elif component == "next_state":
        second[4][0] = 99.0
    elif component == "next_holding":
        second[5]["trading_info"][1] = -0.2
    elif component == "next_mask":
        second[5]["avaliable_action"][0] = 0
    elif component == "next_time":
        second[5]["funding_count_down_hour"] = 8
    else:
        second[6] = True
    replay = buffer()
    replay.add_transition(first, td_error=999.0)
    replay.add_transition(tuple(second), td_error=0.0)
    assert len(replay) == 2
    assert len(replay.decision_groups[0]) == 1


def test_group_cap_random_rotation_is_seeded_and_does_not_follow_reward_rank():
    left, right = buffer(), buffer()
    for reward in range(20):
        left.add_transition(transition(reward=reward), td_error=1000 - reward)
        right.add_transition(transition(reward=reward), td_error=1000 - reward)
    assert len(left) == 4
    assert max(map(len, left.decision_groups[0].values())) == 4
    torch.testing.assert_close(left.rewards[0], right.rewards[0])
    assert set(left.rewards[0][:4, 0].tolist()) != {16, 17, 18, 19}
    assert left.retention_counts["group_replacements"] == 16
    assert len(left.seen_fingerprints[0]) == len(left.keys_by_idx[0]) == 4
    # Repeating a retained outcome cannot consume a fifth slot.
    s, info, a, r, next_s, done, next_info = left.slots[0][0]
    left.add_transition((s, info, a, r, next_s, next_info, done))
    assert len(left) == 4
    assert left.retention_counts["duplicates_skipped"] == 1


def test_fifo_eviction_cleans_up_groups_and_retains_new_contexts_at_capacity():
    replay = buffer(capacity=3)
    for step in range(100):
        replay.add_transition(transition(step=step))
    assert len(replay) == 3
    assert len(replay.decision_groups[0]) == 3
    assert len(replay.seen_fingerprints[0]) == len(replay.keys_by_idx[0]) == 3
    assert set(replay.states[0][:3, 0].tolist()) == {97, 98, 99}
    assert replay.retention_counts["fifo_evictions"] == 97
    assert replay.extract_stacked_tensor_dicts()[0]["replay_semantics_version"] == 2


def _compact_record(reward, holding_return=0.01):
    return CompactTrajectoryRecord(
        df_index=0, step_indices=np.array([0]), actions=np.array([2]),
        rewards=np.array([reward], dtype=np.float32), next_step_indices=np.array([1]),
        dones=np.array([False]), td_errors=np.array([0.0]),
        previous_actions=np.array([2.0]), next_previous_actions=np.array([2.0]),
        trading_infos=np.array([[1.0, holding_return, 0.02, 9 / 180]], dtype=np.float32),
        next_trading_infos=np.array([[1.0, 0.03, 0.02, 10 / 180]], dtype=np.float32),
        avaliable_actions=np.ones((1, 3)), next_avaliable_actions=np.ones((1, 3)),
        hours=np.ones(1), minutes=np.ones(1) * 2,
        next_hours=np.ones(1), next_minutes=np.ones(1) * 2,
    )


def test_compact_and_tuple_ingestion_enforce_the_same_limits_and_contexts():
    states = np.array([[0, 2], [1, 2]], dtype=np.float32)
    pack = SimpleNamespace(
        state_tensor=torch.from_numpy(states), state_array=states,
        regime_grid_ids_array=np.zeros(2), regime_grid_ids_tensor=None,
        q_table_tensor=None,
    )
    shared = {0: pack}
    fast, slow = buffer(), buffer()
    fast.shared_market_data = shared
    for reward in range(10):
        ct = _compact_record(reward)
        result = diverse.WorkerRoundResult(0, 0, 0, 0, 0, 1, compact_trajectory=ct)
        fast.add_round_records_bulk([result])
        slow.add_transition(ct.to_worker_transitions(shared)[0].transition)
    for pnl in [-0.1, 0.1]:
        ct = _compact_record(0.0, holding_return=pnl)
        fast.add_round_records_bulk([diverse.WorkerRoundResult(0, 0, 0, 0, 0, 1, compact_trajectory=ct)])
        slow.add_transition(ct.to_worker_transitions(shared)[0].transition)
    assert len(fast) == len(slow) == 6
    torch.testing.assert_close(fast.rewards[0][:6], slow.rewards[0][:6])
    assert sorted(map(len, fast.decision_groups[0].values())) == [1, 1, 4]
    assert sorted(map(len, slow.decision_groups[0].values())) == [1, 1, 4]


def test_mixed_compact_and_tuple_batch_does_not_drop_legacy_records():
    states = np.array([[0, 2], [1, 2]], dtype=np.float32)
    pack = SimpleNamespace(
        state_tensor=torch.from_numpy(states), state_array=states,
        regime_grid_ids_array=np.zeros(2), regime_grid_ids_tensor=None, q_table_tensor=None,
    )
    replay = buffer()
    replay.shared_market_data = {0: pack}
    compact = diverse.WorkerRoundResult(0, 0, 0, 0, 0, 1, compact_trajectory=_compact_record(1.0))
    legacy = diverse.WorkerRoundResult(
        1, 0, 0, 0, 0, 1,
        transitions=[diverse.WorkerTransitionRecord(0, transition(step=9, reward=-2.0))],
    )
    replay.add_round_records_bulk([compact, legacy])
    assert len(replay) == 2
    assert set(replay.rewards[0][:2, 0].tolist()) == {-2.0, 1.0}
