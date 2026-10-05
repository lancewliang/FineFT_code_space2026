from __future__ import annotations

from typing import Any
import numpy as np
import pytest
import torch
import torch.nn as nn

from model.low_level import ensemble_Qnet
from RL.DiHFT.low_level.parallel_diverse_train import DfRolloutWorkerRunner


def _create_mock_worker_config(
    state_dim: int = 10,
    action_count: int = 5,
    ensemble_number: int = 13,
    hidden_nodes: int = 32,
    time_info_dim: int = 2,
    action_persistence: int = 1,
) -> dict[str, Any]:
    return {
        "df_indices": [0],
        "train_df_by_df": {0: "df0"},
        "env_kwargs": {},
        "device": "cpu",
        "state_dict": {},
        "leverage_choices": [1],
        "position_list": [-2, -1, 0, 1, 2],
        "initial_wallet_balance": 10000.0,
        "initial_unrealized_pnL": 0.0,
        "state_dim": state_dim,
        "action_count": action_count,
        "hidden_nodes": hidden_nodes,
        "time_info_dim": time_info_dim,
        "ensemble_number": ensemble_number,
        "gamma": 0.99,
        "n_step": 1,
        "action_persistence": action_persistence,
    }


def test_targeted_subnetwork_vs_ensemble_equivalence():
    state_dim = 12
    action_count = 5
    ensemble_number = 13
    hidden_nodes = 32
    time_info_dim = 2

    torch.manual_seed(42)
    np.random.seed(42)

    model = ensemble_Qnet(
        N_STATES=state_dim,
        N_ACTIONS=action_count,
        hidden_nodes=hidden_nodes,
        TIME_INFO_DIM=time_info_dim,
        ensemble_number=ensemble_number,
        TRADING_INFO_DIM=4,
    )
    model.eval()

    cfg = _create_mock_worker_config(
        state_dim=state_dim,
        action_count=action_count,
        ensemble_number=ensemble_number,
        hidden_nodes=hidden_nodes,
        time_info_dim=time_info_dim,
    )
    runner = DfRolloutWorkerRunner(cfg)
    runner.model = model

    for trial in range(10):
        state = np.random.randn(state_dim).astype(np.float32)
        previous_action = int(np.random.randint(0, action_count))
        available_action = np.ones(action_count, dtype=np.float32)
        if np.random.uniform() < 0.5:
            masked_idx = int(np.random.randint(0, action_count))
            available_action[masked_idx] = 0.0

        funding_count_down_hour = float(np.random.randint(0, 8))
        funding_count_down_minute = float(np.random.randint(0, 60))
        trading_info = np.random.randn(4).astype(np.float32)

        info = {
            "previous_action": previous_action,
            "avaliable_action": available_action,
            "avaiable_action_list": [
                idx for idx, val in enumerate(available_action) if val > 0.5
            ],
            "funding_count_down_hour": funding_count_down_hour,
            "funding_count_down_minute": funding_count_down_minute,
            "trading_info": trading_info,
        }

        for context_index in range(ensemble_number):
            with torch.no_grad():
                st = torch.FloatTensor(state).reshape(1, -1)
                pa = torch.tensor([[previous_action]], dtype=torch.float32)
                aa = torch.as_tensor(available_action, dtype=torch.float32).unsqueeze(0)
                ti = torch.tensor(
                    [[funding_count_down_hour, funding_count_down_minute]],
                    dtype=torch.float32,
                )
                tr = torch.from_numpy(trading_info).float().reshape(1, -1)
                full_q = model(
                    state=st,
                    time=ti,
                    previous_action=pa,
                    avaliable_action=aa,
                    trading_info=tr,
                )
                expected_context_q = full_q[:, context_index, :]
                expected_action = int(torch.max(expected_context_q, 1)[1].item())
                expected_q = float(expected_context_q[0, expected_action].item())

            action, chosen_q = runner._act(
                state=state,
                info=info,
                context_index=context_index,
                epsilon=0.0,
            )

            assert action == expected_action
            assert pytest.approx(expected_q, abs=1e-5) == chosen_q


def test_irrelevant_subnets_not_invoked_during_act():
    cfg = _create_mock_worker_config(ensemble_number=5)
    runner = DfRolloutWorkerRunner(cfg)

    call_counters = [0] * len(runner.model.qnet_list)

    def make_spy_forward(idx, original_forward):
        def spy(*args, **kwargs):
            call_counters[idx] += 1
            return original_forward(*args, **kwargs)
        return spy

    for idx, qnet in enumerate(runner.model.qnet_list):
        qnet.forward = make_spy_forward(idx, qnet.forward)

    state = np.zeros(cfg["state_dim"], dtype=np.float32)
    info = {
        "previous_action": 0,
        "avaliable_action": [1] * cfg["action_count"],
        "avaiable_action_list": list(range(cfg["action_count"])),
        "funding_count_down_hour": 0.0,
        "funding_count_down_minute": 0.0,
        "trading_info": np.zeros(4, dtype=np.float32),
    }

    target_context = 2
    action, chosen_q = runner._act(
        state=state,
        info=info,
        context_index=target_context,
        epsilon=0.0,
    )

    assert call_counters[target_context] == 1
    for idx in range(len(call_counters)):
        if idx != target_context:
            assert call_counters[idx] == 0


def test_action_persistence_preserves_non_flat_action():
    cfg = _create_mock_worker_config(
        state_dim=4,
        action_count=5,
        ensemble_number=3,
        action_persistence=3,
    )
    runner = DfRolloutWorkerRunner(cfg)
    assert runner.flat_action == 2

    class DummyEnv:
        def __init__(self):
            self.step_idx = 0
            self.unrealized_pnl = 0.0
            self.wallet_balance = 10000.0

        def reset(self, initial_state=None):
            self.step_idx = 0
            return np.zeros(4, dtype=np.float32), {
                "previous_action": 2,
                "avaliable_action": [1, 1, 1, 1, 1],
                "avaiable_action_list": [0, 1, 2, 3, 4],
                "funding_count_down_hour": 0.0,
                "funding_count_down_minute": 0.0,
                "trading_info": np.zeros(4, dtype=np.float32),
                "regime_grid_id": 0,
            }

        def step(self, action):
            self.step_idx += 1
            done = self.step_idx >= 5
            return (
                np.zeros(4, dtype=np.float32),
                1.0,
                done,
                {
                    "previous_action": action,
                    "avaliable_action": [1, 1, 1, 1, 1],
                    "avaiable_action_list": [0, 1, 2, 3, 4],
                    "funding_count_down_hour": 0.0,
                    "funding_count_down_minute": 0.0,
                    "trading_info": np.zeros(4, dtype=np.float32),
                    "regime_grid_id": 0,
                },
            )

    dummy_env = DummyEnv()
    runner._get_initial_state_and_env = lambda df_index, initial_action: (
        dummy_env,
        (10000.0, 0.0, 0.0, 0.0, 1.0),
    )

    act_calls = []

    def mock_act(state, info, context_index, epsilon):
        act_calls.append(len(act_calls))
        return 4, 1.5

    runner._act = mock_act

    from RL.DiHFT.low_level.parallel_diverse_train import ExploreTask
    task = ExploreTask(
        df_index=0,
        epoch_index=0,
        context_index=1,
        initial_action=2,
        round_counter=0,
        epsilon=0.0,
    )

    result = runner.run_task(task)
    assert len(result.transitions) == 5
    actions_taken = [t.transition[2] for t in result.transitions]
    assert actions_taken == [4, 4, 4, 4, 4]
    assert len(act_calls) == 2
