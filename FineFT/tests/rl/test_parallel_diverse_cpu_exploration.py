import queue
from unittest.mock import MagicMock
import numpy as np
import pytest
import torch

from RL.DiHFT.low_level import parallel_diverse_train as pdt
from RL.util.regime_stratified_replay_buffer import RegimeStratifiedReplayBuffer


def test_start_parallel_workers_forces_cpu_device_and_uses_diverse_num_workers(monkeypatch):
    """Workers must be configured with device=cpu and count matching trainer.diverse_num_workers."""
    class FakeQueue:
        pass

    class FakeContext:
        def Queue(self):
            return FakeQueue()

        def Process(self, target, args):
            process = MagicMock()
            process.worker_config = args[0]
            process.input_queue = args[1]
            return process

    monkeypatch.setattr(pdt, "create_worker_context", lambda: FakeContext())
    monkeypatch.setattr(pdt, "build_effective_df_indices", lambda n: list(range(n)))

    trainer = MagicMock()
    trainer.total_df_index_length = 2
    trainer.diverse_num_workers = 4
    trainer.device = "cuda"
    trainer.leverage_choices = [1]
    trainer.position_list = [0.0]
    trainer.initial_wallet_balance = 10000.0
    trainer.initial_unrealized_pnL = 0.0
    trainer.tech_indicator_list = ["feat_1", "feat_2"]
    trainer.N_ACTIONS = 3
    trainer.hidden_nodes = 16
    trainer.time_info_dim = 2
    trainer.N = 2
    trainer.gamma = 0.99
    trainer.n_step = 1

    train_df_cache = {0: "df0", 1: "df1"}

    pdt.start_parallel_workers(trainer, train_df_cache, {})

    assert len(trainer.worker_processes) == 4
    for process in trainer.worker_processes:
        assert process.worker_config["device"] == "cpu"


def test_df_rollout_worker_runner_forces_cpu_device_and_single_thread():
    """DfRolloutWorkerRunner must place runner.device on cpu, model on CPU, and threads to 1."""
    worker_config = {
        "df_indices": [0],
        "train_df_by_df": {0: "df0"},
        "env_kwargs": {},
        "device": "cuda",
        "state_dict": {},
        "leverage_choices": [1],
        "position_list": [0.0],
        "initial_wallet_balance": 10000.0,
        "initial_unrealized_pnL": 0.0,
        "state_dim": 2,
        "action_count": 3,
        "hidden_nodes": 16,
        "time_info_dim": 2,
        "ensemble_number": 2,
        "gamma": 0.99,
        "n_step": 1,
    }
    runner = pdt.DfRolloutWorkerRunner(worker_config)
    assert runner.device == "cpu"
    assert next(runner.model.parameters()).device.type == "cpu"
    assert torch.get_num_threads() == 1


def test_df_rollout_worker_runner_run_task_atomic(monkeypatch):
    """run_task must reset and run full episode atomically with ExploreTask."""
    class FakeEnv:
        def __init__(self):
            self.step_count = 0
            self.unrealized_pnl = 0.0
            self.wallet_balance = 1000.0

        def reset(self):
            return np.array([0.1, 0.2]), {
                "previous_action": 0,
                "avaliable_action": [1, 1, 1],
                "avaiable_action_list": [0, 1, 2],
                "funding_count_down_hour": 0,
                "funding_count_down_minute": 0,
                "trading_info": np.zeros(4, dtype=np.float32),
                "q_value": [1.0, 0.0, 0.0],
            }

        def step(self, action):
            self.step_count += 1
            done = self.step_count >= 2
            return (
                np.array([0.3, 0.4]),
                2.0,
                done,
                {
                    "previous_action": action,
                    "avaliable_action": [1, 1, 1],
                    "avaiable_action_list": [0, 1, 2],
                    "funding_count_down_hour": 0,
                    "funding_count_down_minute": 0,
                    "trading_info": np.zeros(4, dtype=np.float32),
                    "q_value": [1.0, 0.0, 0.0],
                },
            )

    monkeypatch.setattr(pdt, "build_initial_state", lambda *args, **kwargs: (None, None, None, "init"))
    monkeypatch.setattr(pdt, "create_demo_env", lambda *args, **kwargs: FakeEnv())

    worker_config = {
        "df_indices": [0],
        "train_df_by_df": {0: "df0"},
        "env_kwargs": {},
        "device": "cpu",
        "state_dict": {},
        "leverage_choices": [1],
        "position_list": [0.0],
        "initial_wallet_balance": 10000.0,
        "initial_unrealized_pnL": 0.0,
        "state_dim": 2,
        "action_count": 3,
        "hidden_nodes": 16,
        "time_info_dim": 2,
        "ensemble_number": 2,
        "gamma": 0.99,
        "n_step": 1,
    }
    runner = pdt.DfRolloutWorkerRunner(worker_config)

    task = pdt.ExploreTask(
        df_index=0,
        epoch_index=3,
        context_index=1,
        initial_action=0,
        round_counter=7,
        epsilon=0.0,
    )
    result = runner.run_task(task)

    assert result.done is True
    assert result.df_index == 0
    assert result.epoch_index == 3
    assert result.context_index == 1
    assert result.round_counter == 7
    assert result.worker_steps == 2
    assert len(result.transitions) == 2


def test_run_epoch_exploration_task_pool_dispatch_and_collection(monkeypatch):
    """run_epoch_exploration dispatches all tasks to task pool and gathers results."""
    dispatched_tasks = []

    class MockTaskQueue:
        def put(self, item):
            if isinstance(item, pdt.ExploreTask):
                dispatched_tasks.append(item)

        def empty(self):
            return True

    result_queue = queue.Queue()

    def mock_start_workers(tr, train_df_cache, env_kwargs):
        task_q = MockTaskQueue()
        tr.worker_task_queue = task_q
        tr.worker_result_queue = result_queue
        tr.worker_input_queues = {0: task_q, 1: task_q}
        tr.worker_processes = []

    monkeypatch.setattr(pdt, "start_parallel_workers", mock_start_workers)
    monkeypatch.setattr(pdt, "shutdown_exploration_workers", lambda tr: None)

    trainer = MagicMock()
    trainer.total_df_index_length = 2
    trainer.N = 2
    trainer.position_choices = 2
    trainer.epsilon = 0.1
    trainer.buffer_size = 1000

    buffer_diverse = RegimeStratifiedReplayBuffer(
        total_buffer_size=900,
        batch_size=32,
        device="cpu",
        seed=42,
        num_grids=9,
    )

    # Pre-populate result_queue so get() succeeds for all expected tasks:
    # 2 contexts * 2 actions * 2 dfs = 8 tasks
    for i in range(8):
        result_queue.put(
            pdt.WorkerRoundResult(
                df_index=i % 2,
                epoch_index=0,
                context_index=i // 4,
                initial_action=(i // 2) % 2,
                round_counter=i,
                worker_steps=1,
                transitions=[
                    pdt.WorkerTransitionRecord(
                        step_index=0,
                        transition=(
                            np.array([float(i), 0.0]),
                            {"previous_action": 0, "regime_grid_id": 0, "trading_info": np.zeros(4)},
                            0,
                            1.0,
                            np.array([float(i) + 0.5, 0.0]),
                            {"previous_action": 0, "regime_grid_id": 0, "trading_info": np.zeros(4)},
                            True,
                        ),
                    )
                ],
                rollout_metrics=[
                    pdt.RolloutMetrics(
                        epoch_index=0,
                        context_index=i // 4,
                        initial_action=(i // 2) % 2,
                        df_index=i % 2,
                        transition_count=1,
                        reward_sum=1.0,
                        final_balance=100.0,
                        return_rate=0.01,
                    )
                ],
                done=True,
            )
        )

    epoch_metrics, steps, rounds = pdt.run_epoch_exploration(
        trainer=trainer,
        epoch_index=0,
        train_df_cache={},
        env_kwargs={},
        buffer_diverse=buffer_diverse,
        step_counter_diverse=0,
        round_counter=0,
        diverse_rollout_latest_metrics_by_df={},
    )

    assert len(dispatched_tasks) == 8
    assert len(epoch_metrics) == 8
    assert steps == 8
    assert rounds == 8
    assert len(buffer_diverse) == 8


def test_parallel_parser_diverse_num_workers_flag():
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap

    args_default = pwap.parser.parse_args([])
    assert args_default.diverse_num_workers == 96

    args_custom = pwap.parser.parse_args(["--diverse_num_workers", "64"])
    assert args_custom.diverse_num_workers == 64


def test_weighted_contexts_dqn_validates_diverse_num_workers(monkeypatch):
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap

    mock_args = MagicMock()
    mock_args.seed = 42
    mock_args.experiment_name = "test"
    mock_args.result_path = "/tmp"
    mock_args.dataset_name = "fu"
    mock_args.tau = 0.005
    mock_args.batch_size = 32
    mock_args.update_times = 1
    mock_args.gamma = 0.99
    mock_args.epsilon_init = 1.0
    mock_args.epsilon_min = 0.1
    mock_args.epsilon_step = 1000
    mock_args.rollout_steps = 100
    mock_args.n_step = 1
    mock_args.buffer_size = 1000
    mock_args.curriculum_block_epochs = 3
    mock_args.lr_init = 0.001
    mock_args.lr_min = 0.0001
    mock_args.lr_step = 1000
    mock_args.num_sample = 12
    mock_args.num_epoch = 12
    mock_args.base_path = "dataset/10min"
    mock_args.pretrain_num_workers = 20
    mock_args.diverse_num_workers = 0
    mock_args.eval_num_workers = 20
    mock_args.neighbor_size = 1
    mock_args.max_holding_number = 1.0
    mock_args.position_choices = 3
    mock_args.order_book_depth = 5
    mock_args.leverage_choices = [1]
    mock_args.long_estimated_rate = 0.0
    mock_args.short_estimated_rate = 0.0
    mock_args.transcation_cost = 0.0005
    mock_args.early_stop = 2
    mock_args.initial_wallet_balance = 10000.0
    mock_args.initial_margin = 0.0
    mock_args.initial_unrealized_pnL = 0.0
    mock_args.initial_position = 0.0
    mock_args.initial_leverage = 1
    mock_args.time_info_dim = 2
    mock_args.hidden_nodes = 16
    mock_args.N = 2
    mock_args.outer_bond = 4.0
    mock_args.reachout_index = 1
    mock_args.if_use_hubber_loss = True
    mock_args.ada_init = 96.0
    mock_args.ada_min = 0.1
    mock_args.ada_step = 1000
    mock_args.pretrain_epoch = 0
    mock_args.load_pretrain_model = True
    mock_args.allow_reverse_position = True
    mock_args.enable_limit_reward = False
    mock_args.limit_hold_bonus = 0.0
    mock_args.limit_stay_bonus = 0.0
    mock_args.limit_reverse_penalty = 0.0
    mock_args.near_limit_threshold = 0.0

    monkeypatch.setattr(pwap, "build_training_data_paths", lambda *a, **k: {
        "train_data_path": "/tmp",
        "state_features_path": "/tmp/f.npy",
        "maintenance_margin_ratio_path": "/tmp/m.npy",
    })
    monkeypatch.setattr(pwap, "count_training_data_files", lambda *a, **k: 2)
    monkeypatch.setattr(np, "load", lambda *a, **k: MagicMock(item=lambda: {}))

    with pytest.raises(ValueError, match="diverse_num_workers must be positive"):
        pwap.Weighted_Contexts_DQN(mock_args)
