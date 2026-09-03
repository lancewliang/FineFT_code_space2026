import os
from unittest.mock import MagicMock
import numpy as np
import pytest
import torch

from RL.DiHFT.low_level.evaluate_sub_agents import (
    DEFAULT_EVAL_NUM_WORKERS,
    DEFAULT_PRETRAIN_EVAL_NUM_WORKERS,
    SubAgentEvalMetric,
    SubAgentEvalTask,
    WarmupEvalMetric,
    WarmupEvalTask,
    act_test,
    evaluate_single_sub_agent_df,
    evaluate_sub_agents,
    evaluate_warmup_sub_agents,
    select_greedy_model_action,
)


class DummyEnvForPool:
    def __init__(self, df_name):
        self.df_name = df_name
        self.step_count = 0
        self.unrealized_pnl = 100.0
        self.wallet_balance = 10100.0

    def reset(self):
        self.step_count = 0
        return np.zeros(4), {
            "previous_action": 0,
            "avaliable_action": [1, 1, 1],
            "avaiable_action_list": [0, 1, 2],
            "funding_count_down_hour": 0,
            "funding_count_down_minute": 0,
            "trading_info": np.zeros(4),
        }

    def step(self, action):
        self.step_count += 1
        done = self.step_count >= 2
        return (
            np.zeros(4),
            5.0,
            done,
            {
                "previous_action": action,
                "avaliable_action": [1, 1, 1],
                "avaiable_action_list": [0, 1, 2],
                "funding_count_down_hour": 0,
                "funding_count_down_minute": 0,
                "trading_info": np.zeros(4),
            },
        )


class DummyTorchModel(torch.nn.Module):
    def __init__(self, ensemble_n=4):
        super().__init__()
        self.ensemble_n = ensemble_n
        self.linear = torch.nn.Linear(4, 3)

    def forward(self, state, time, previous_action, avaliable_action, trading_info):
        vals = [
            [1.0, 5.0, 2.0],
            [1.0, 2.0, 6.0],
            [3.0, 1.0, 2.0],
            [2.0, 4.0, 1.0],
        ][: self.ensemble_n]
        return torch.tensor([vals], dtype=torch.float32)


class FakeSubQnet(torch.nn.Module):
    def __init__(self, action_values):
        super().__init__()
        self.action_values = action_values
        self.calls = 0

    def forward(self, state, time, previous_action, avaliable_action, trading_info):
        self.calls += 1
        return torch.tensor([self.action_values], dtype=torch.float32)


class FakeEnsembleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qnet_list = torch.nn.ModuleList(
            [
                FakeSubQnet([1.0, 0.0, 0.0]),
                FakeSubQnet([0.0, 5.0, 2.0]),
                FakeSubQnet([0.0, 1.0, 4.0]),
            ]
        )


def test_default_workers_constant():
    assert DEFAULT_EVAL_NUM_WORKERS == 150
    assert DEFAULT_PRETRAIN_EVAL_NUM_WORKERS == 150


def test_evaluate_sub_agents_runs_in_process_pool(monkeypatch):
    def mock_build_initial_state(train_df, initial_action, *args, **kwargs):
        return None, None, None, f"init_{train_df}_{initial_action}"

    def mock_create_demo_env(train_df, env_kwargs, initial_state):
        return DummyEnvForPool(train_df)

    monkeypatch.setattr(
        "RL.DiHFT.low_level.evaluate_sub_agents.build_initial_state",
        mock_build_initial_state,
    )
    monkeypatch.setattr(
        "RL.DiHFT.low_level.evaluate_sub_agents.create_demo_env",
        mock_create_demo_env,
    )

    trainer = MagicMock()
    trainer.N = 2
    trainer.total_df_index_length = 3
    trainer.device = "cpu"
    trainer.leverage_choices = [1]
    trainer.position_list = [0]
    trainer.initial_wallet_balance = 10000.0
    trainer.initial_unrealized_pnL = 0.0
    trainer.eval_net = DummyTorchModel()
    trainer.writer = MagicMock()

    train_df_cache = {0: "df0", 1: "df1", 2: "df2"}

    pool_created_with_processes = []
    original_pool = evaluate_sub_agents.__globals__["get_evaluation_pool_context"]().Pool

    def tracking_pool(processes=None):
        pool_created_with_processes.append(processes)
        return original_pool(processes=processes)

    metrics = evaluate_sub_agents(
        trainer=trainer,
        train_df_cache=train_df_cache,
        env_kwargs={},
        pool_factory=tracking_pool,
    )

    assert len(pool_created_with_processes) == 1
    assert pool_created_with_processes[0] == 6

    assert len(metrics) == 6
    for m in metrics:
        assert m.initial_action == 0
        assert m.reward_sum == 10.0
        assert m.final_balance == 10200.0
        assert m.return_rate == pytest.approx(0.02, abs=1e-6)

    expected_pairs = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    actual_pairs = [(m.context_index, m.df_index) for m in metrics]
    assert actual_pairs == expected_pairs

    assert trainer.writer.add_scalar.call_count == 12


def test_evaluate_sub_agents_supports_diverse_eval_tag_prefix(monkeypatch):
    def mock_build_initial_state(train_df, initial_action, *args, **kwargs):
        return None, None, None, f"init_{train_df}_{initial_action}"

    def mock_create_demo_env(train_df, env_kwargs, initial_state):
        return DummyEnvForPool(train_df)

    monkeypatch.setattr(
        "RL.DiHFT.low_level.evaluate_sub_agents.build_initial_state",
        mock_build_initial_state,
    )
    monkeypatch.setattr(
        "RL.DiHFT.low_level.evaluate_sub_agents.create_demo_env",
        mock_create_demo_env,
    )

    trainer = MagicMock()
    trainer.N = 1
    trainer.total_df_index_length = 2
    trainer.device = "cpu"
    trainer.leverage_choices = [1]
    trainer.position_list = [0]
    trainer.initial_wallet_balance = 10000.0
    trainer.initial_unrealized_pnL = 0.0
    trainer.eval_net = DummyTorchModel(ensemble_n=1)
    trainer.writer = MagicMock()

    train_df_cache = {0: "df0", 1: "df1"}

    evaluate_sub_agents(
        trainer=trainer,
        train_df_cache=train_df_cache,
        env_kwargs={},
        tag_prefix="diverse_eval",
    )

    tags = [call.kwargs["tag"] for call in trainer.writer.add_scalar.call_args_list]
    assert "diverse_eval_return_rate_context_0" in tags
    assert "diverse_eval_reward_sum_context_0" in tags


def test_evaluate_sub_agents_respects_custom_num_workers(monkeypatch):
    def mock_build_initial_state(train_df, initial_action, *args, **kwargs):
        return None, None, None, f"init_{train_df}_{initial_action}"

    def mock_create_demo_env(train_df, env_kwargs, initial_state):
        return DummyEnvForPool(train_df)

    monkeypatch.setattr(
        "RL.DiHFT.low_level.evaluate_sub_agents.build_initial_state",
        mock_build_initial_state,
    )
    monkeypatch.setattr(
        "RL.DiHFT.low_level.evaluate_sub_agents.create_demo_env",
        mock_create_demo_env,
    )

    trainer = MagicMock()
    trainer.N = 4
    trainer.total_df_index_length = 5
    trainer.device = "cpu"
    trainer.leverage_choices = [1]
    trainer.position_list = [0]
    trainer.initial_wallet_balance = 10000.0
    trainer.initial_unrealized_pnL = 0.0
    trainer.eval_net = DummyTorchModel(ensemble_n=4)
    trainer.eval_num_workers = 8

    train_df_cache = {i: f"df{i}" for i in range(5)}

    pool_workers_record = []

    def mock_pool_factory(processes=None):
        pool_workers_record.append(processes)
        return evaluate_sub_agents.__globals__["get_evaluation_pool_context"]().Pool(
            processes=processes
        )

    metrics = evaluate_sub_agents(
        trainer=trainer,
        train_df_cache=train_df_cache,
        env_kwargs={},
        pool_factory=mock_pool_factory,
    )

    assert pool_workers_record[0] == 8
    assert len(metrics) == 20


def test_evaluate_sub_agents_rejects_non_positive_num_workers():
    trainer = MagicMock()
    trainer.N = 2
    trainer.total_df_index_length = 2
    trainer.eval_net = DummyTorchModel()
    train_df_cache = {0: "df0", 1: "df1"}

    with pytest.raises(ValueError, match="num_workers must be positive"):
        evaluate_sub_agents(
            trainer=trainer,
            train_df_cache=train_df_cache,
            env_kwargs={},
            num_workers=0,
        )

    with pytest.raises(ValueError, match="num_workers must be positive"):
        evaluate_sub_agents(
            trainer=trainer,
            train_df_cache=train_df_cache,
            env_kwargs={},
            num_workers=-5,
        )


def test_evaluate_sub_agents_returns_empty_on_invalid_configs():
    train_df_cache = {0: "df0"}

    # Missing eval_net
    trainer_no_net = MagicMock()
    trainer_no_net.eval_net = None
    assert evaluate_sub_agents(trainer_no_net, train_df_cache, {}) == []

    # Non-positive N
    trainer_bad_n = MagicMock()
    trainer_bad_n.eval_net = DummyTorchModel()
    trainer_bad_n.N = 0
    trainer_bad_n.total_df_index_length = 1
    assert evaluate_sub_agents(trainer_bad_n, train_df_cache, {}) == []

    # Non-positive total_df_index_length
    trainer_bad_df = MagicMock()
    trainer_bad_df.eval_net = DummyTorchModel()
    trainer_bad_df.N = 1
    trainer_bad_df.total_df_index_length = 0
    assert evaluate_sub_agents(trainer_bad_df, train_df_cache, {}) == []

    # Missing df in train_df_cache
    trainer_missing_df = MagicMock()
    trainer_missing_df.eval_net = DummyTorchModel()
    trainer_missing_df.N = 1
    trainer_missing_df.total_df_index_length = 2
    assert evaluate_sub_agents(trainer_missing_df, {0: "df0"}, {}) == []


def test_select_greedy_model_action_matches_test_agent_index_qnet_list_behavior():
    model = FakeEnsembleModel()
    info = {
        "previous_action": 0,
        "avaliable_action": [1, 1, 1],
        "funding_count_down_hour": 1,
        "funding_count_down_minute": 30,
        "trading_info": np.array([0.0, 0.0, 0.0, 0.0]),
    }

    action_c1 = select_greedy_model_action(
        model=model,
        state=np.zeros(4),
        info=info,
        context_index=1,
        device="cpu",
    )
    # Context 1 outputs [0.0, 5.0, 2.0], argmax is 1
    assert action_c1 == 1
    assert model.qnet_list[0].calls == 0
    assert model.qnet_list[1].calls == 1
    assert model.qnet_list[2].calls == 0

    action_c2 = act_test(
        model=model,
        state=np.zeros(4),
        info=info,
        context_index=2,
        device="cpu",
    )
    # Context 2 outputs [0.0, 1.0, 4.0], argmax is 2
    assert action_c2 == 2
    assert model.qnet_list[0].calls == 0
    assert model.qnet_list[1].calls == 1
    assert model.qnet_list[2].calls == 1


def test_select_greedy_model_action_handles_tensor_and_scalar_inputs():
    model = DummyTorchModel()
    info = {
        "previous_action": torch.tensor(1),
        "avaliable_action": torch.tensor([1, 1, 1]),
        "avaiable_action_list": [0, 1, 2],
        "funding_count_down_hour": torch.tensor(1),
        "funding_count_down_minute": torch.tensor(30),
        "trading_info": np.array([0.1, 0.2, 0.3, 0.4]),
    }
    action = select_greedy_model_action(
        model=model,
        state=np.zeros(4),
        info=info,
        context_index=0,
        device="cpu",
    )
    assert action == 1

    action_c1 = select_greedy_model_action(
        model=model,
        state=np.zeros(4),
        info=info,
        context_index=1,
        device="cpu",
    )
    assert action_c1 == 2


def test_evaluate_sub_agents_uses_independent_subprocesses(monkeypatch):
    import multiprocessing as mp

    manager = mp.Manager()
    worker_pids = manager.list()
    parent_pid = os.getpid()

    def mock_build_initial_state(train_df, initial_action, *args, **kwargs):
        worker_pids.append(os.getpid())
        return None, None, None, f"init_{train_df}_{initial_action}"

    def mock_create_demo_env(train_df, env_kwargs, initial_state):
        return DummyEnvForPool(train_df)

    monkeypatch.setattr(
        "RL.DiHFT.low_level.evaluate_sub_agents.build_initial_state",
        mock_build_initial_state,
    )
    monkeypatch.setattr(
        "RL.DiHFT.low_level.evaluate_sub_agents.create_demo_env",
        mock_create_demo_env,
    )

    trainer = MagicMock()
    trainer.N = 2
    trainer.total_df_index_length = 3
    trainer.device = "cpu"
    trainer.leverage_choices = [1]
    trainer.position_list = [0]
    trainer.initial_wallet_balance = 10000.0
    trainer.initial_unrealized_pnL = 0.0
    trainer.eval_net = DummyTorchModel()

    train_df_cache = {0: "df0", 1: "df1", 2: "df2"}

    metrics = evaluate_sub_agents(
        trainer=trainer,
        train_df_cache=train_df_cache,
        env_kwargs={},
    )

    assert len(metrics) == 6
    assert len(worker_pids) == 6
    for pid in worker_pids:
        assert pid != parent_pid


def test_evaluate_sub_agents_preserves_model_training_mode_and_gradients(monkeypatch):
    model = DummyTorchModel()
    model.train()

    dummy_input = torch.randn(2, 4)
    loss = model.linear(dummy_input).sum()
    loss.backward()

    grads_before = {
        name: p.grad.clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    assert len(grads_before) > 0

    trainer = MagicMock()
    trainer.N = 2
    trainer.total_df_index_length = 2
    trainer.eval_net = model
    trainer.device = "cpu"
    trainer.leverage_choices = [1]
    trainer.position_list = [0]
    trainer.initial_wallet_balance = 10000.0
    trainer.initial_unrealized_pnL = 0.0

    monkeypatch.setattr(
        "RL.DiHFT.low_level.evaluate_sub_agents.build_initial_state",
        lambda *args, **kwargs: (None, None, None, "init_state"),
    )
    monkeypatch.setattr(
        "RL.DiHFT.low_level.evaluate_sub_agents.create_demo_env",
        lambda train_df, kwargs, init_s: DummyEnvForPool(train_df),
    )

    evaluate_sub_agents(
        trainer=trainer,
        train_df_cache={0: "df0", 1: "df1"},
        env_kwargs={},
    )

    assert model.training is True

    for name, p in model.named_parameters():
        if name in grads_before:
            assert p.grad is not None
            assert torch.equal(p.grad, grads_before[name])


def test_run_exhaustive_warmup_evaluates_every_30_rounds(monkeypatch):
    from RL.DiHFT.low_level import parallel_pretrain as pp
    import queue

    evaluation_call_count = [0]

    def mock_eval(trainer, train_df_cache, env_kwargs):
        evaluation_call_count[0] += 1
        return [
            pp.WarmupEvalMetric(
                context_index=0,
                df_index=0,
                initial_action=0,
                reward_sum=1.0,
                final_balance=10000.0,
                return_rate=0.0,
            )
        ]

    monkeypatch.setattr(pp, "evaluate_warmup_sub_agents", mock_eval)
    monkeypatch.setattr(pp, "evaluate_sub_agents", mock_eval)

    class DummyInputQueue:
        def __init__(self, result_queue):
            self.result_queue = result_queue

        def put(self, message):
            self.result_queue.put(
                pp.PretrainCollectResult(
                    df_index=0,
                    initial_action=message.initial_action,
                    rollout_index=message.rollout_index,
                    transitions=[("s", {}, 0, 1.0, "s_", {}, True)],
                    reward_sum=1.0,
                    final_balance=10000.0,
                    transition_count=1,
                )
            )

    def mock_start_workers(tr, train_df_cache, env_kwargs, q_table_cache):
        tr.worker_result_queue = queue.Queue()
        tr.worker_input_queues = {0: DummyInputQueue(tr.worker_result_queue)}

    monkeypatch.setattr(pp, "start_pretrain_collect_workers", mock_start_workers)
    monkeypatch.setattr(pp, "update_pretrain", lambda *args, **kwargs: (1.0, 0.5, 0.5))

    trainer = MagicMock()
    trainer.total_df_index_length = 1
    trainer.position_choices = 1
    trainer.initial_wallet_balance = 10000.0
    trainer.pretrain_epoch = 65  # 65 epochs should trigger at epoch 30 and 60
    trainer.update_times = 1
    trainer.batch_size = 1

    class DummyBuffer:
        def __len__(self):
            return 10

        def sample(self):
            return ("s", {}, "a", "r", "s_", {}, "d")

        def add(self, *transition):
            pass

    summary, _ = pp.run_exhaustive_warmup(
        trainer=trainer,
        q_table_cache={0: None},
        train_df_cache={0: None},
        env_kwargs={},
        buffer_pretrain=DummyBuffer(),
        step_counter_pretrain=0,
    )

    # 65 epochs -> triggered at epoch 30 and epoch 60
    assert evaluation_call_count[0] == 2
    assert len(summary["eval_metrics"]) == 1

