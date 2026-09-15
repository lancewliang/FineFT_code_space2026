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
    trainer.dataset_name = "test_fu"
    trainer.experiment_name = "test_exp"
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
        assert process.worker_config["log_file_path"].endswith("advantage.log")


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
    mock_args.num_sample = 15
    mock_args.num_epoch = 15
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


def test_shutdown_exploration_workers_enqueues_shutdown_message_and_cleans_up():
    """shutdown_exploration_workers must send ShutdownWorker per process and join them."""
    import queue
    from RL.DiHFT.low_level.parallel_weight_advantage_pretrain import ShutdownWorker

    class FakeProcess:
        def __init__(self, pid, stuck=False):
            self.pid = pid
            self.stuck = stuck
            self.join_calls = 0
            self.terminated = False

        def join(self, timeout=None):
            self.join_calls += 1

        def is_alive(self):
            return self.stuck

        def terminate(self):
            self.terminated = True

    trainer = MagicMock()
    proc1 = FakeProcess(101)
    proc2 = FakeProcess(102)
    trainer.worker_processes = [proc1, proc2]
    task_queue = queue.Queue()
    trainer.worker_task_queue = task_queue
    trainer.worker_input_queues = {0: task_queue}
    trainer.worker_result_queue = queue.Queue()

    pdt.shutdown_exploration_workers(trainer)

    messages = []
    while not task_queue.empty():
        messages.append(task_queue.get())
    assert len(messages) == 2
    assert all(isinstance(msg, ShutdownWorker) for msg in messages)

    assert proc1.join_calls == 1
    assert proc2.join_calls == 1
    assert trainer.worker_processes == []
    assert trainer.worker_input_queues == {}
    assert trainer.worker_task_queue is None
    assert trainer.worker_result_queue is None


def test_run_epoch_exploration_logs_exception_on_failure(monkeypatch):
    """run_epoch_exploration must log exception and shut down workers if an error occurs."""
    import queue
    from RL.DiHFT.low_level.parallel_weight_advantage_pretrain import WorkerErrorMessage

    shutdown_called = []
    logger_exceptions = []

    result_queue = queue.Queue()
    # Put an error message into result_queue to trigger raise_for_worker_error
    result_queue.put(
        WorkerErrorMessage(
            df_index=0,
            epoch_index=0,
            context_index=0,
            initial_action=0,
            round_counter=0,
            traceback="simulated worker error traceback",
        )
    )

    def mock_start_workers(tr, train_df_cache, env_kwargs):
        task_q = queue.Queue()
        tr.worker_task_queue = task_q
        tr.worker_result_queue = result_queue
        tr.worker_input_queues = {0: task_q}
        tr.worker_processes = []

    monkeypatch.setattr(pdt, "start_parallel_workers", mock_start_workers)
    monkeypatch.setattr(pdt, "shutdown_exploration_workers", lambda tr: shutdown_called.append(True))
    monkeypatch.setattr(pdt.logger, "exception", lambda msg, *args: logger_exceptions.append(msg))

    trainer = MagicMock()
    trainer.total_df_index_length = 1
    trainer.N = 1
    trainer.position_choices = 1
    trainer.epsilon = 0.1
    trainer.buffer_size = 1000

    buffer_diverse = RegimeStratifiedReplayBuffer(
        total_buffer_size=900,
        batch_size=32,
        device="cpu",
        seed=42,
        num_grids=9,
    )

    with pytest.raises(RuntimeError, match="worker_error df_index=0"):
        pdt.run_epoch_exploration(
            trainer=trainer,
            epoch_index=0,
            train_df_cache={},
            env_kwargs={},
            buffer_diverse=buffer_diverse,
            step_counter_diverse=0,
            round_counter=0,
            diverse_rollout_latest_metrics_by_df={},
        )

    assert len(logger_exceptions) == 1
    assert "epoch exploration failed" in logger_exceptions[0]
    assert len(shutdown_called) == 1


def test_df_rollout_worker_captures_init_failure_in_result_queue():
    """If runner_factory raises before receiving any message, worker puts WorkerErrorMessage."""
    import queue
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap

    input_queue = queue.Queue()
    result_queue = queue.Queue()

    def faulty_runner_factory(config):
        raise RuntimeError("simulated runner factory init failure")

    worker_config = {"runner_factory": faulty_runner_factory}

    pwap.df_rollout_worker(worker_config, input_queue, result_queue)

    assert not result_queue.empty()
    err = result_queue.get()
    assert isinstance(err, pwap.WorkerErrorMessage)
    assert err.df_index == -1
    assert "simulated runner factory init failure" in err.traceback


def test_uncaught_exception_handler_logs_critical(monkeypatch):
    """_handle_uncaught_exception must log critical and call sys.__excepthook__."""
    import sys
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap

    critical_logs = []
    excepthook_called = []

    monkeypatch.setattr(pwap.logger, "critical", lambda msg, *a, **k: critical_logs.append((msg, k)))
    monkeypatch.setattr(sys, "__excepthook__", lambda *a: excepthook_called.append(a))

    try:
        raise ValueError("test uncaught exception")
    except ValueError as e:
        exc_type, exc_val, exc_tb = sys.exc_info()
        pwap._handle_uncaught_exception(exc_type, exc_val, exc_tb)

    assert len(critical_logs) == 1
    assert "Uncaught exception in main process" in critical_logs[0][0]
    assert len(excepthook_called) == 1


def test_configure_worker_logger_attaches_file_handler_with_worker_id(tmp_path):
    """configure_worker_logger sets up file handler formatted with [worker-{id}]."""
    import logging
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap

    log_file = tmp_path / "worker_test.log"
    pwap.configure_worker_logger(str(log_file), worker_id=42)

    root_logger = logging.getLogger()
    matching_handlers = [
        h for h in root_logger.handlers
        if isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file)
    ]
    assert len(matching_handlers) == 1
    handler = matching_handlers[0]

    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="test worker message",
        args=(),
        exc_info=None,
    )
    formatted = handler.formatter.format(record)
    assert "[worker-42]" in formatted
    assert "test worker message" in formatted


def test_df_rollout_worker_writes_to_log_file_on_task(tmp_path, monkeypatch):
    """df_rollout_worker writes task start logs to the configured log_file_path."""
    import queue
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap

    log_file = tmp_path / "advantage.log"
    input_queue = queue.Queue()
    result_queue = queue.Queue()

    class FakeRunner:
        def __init__(self, config):
            pass

        def run_task(self, task):
            return "fake_round_result"

    worker_config = {
        "worker_id": 7,
        "log_file_path": str(log_file),
        "runner_factory": FakeRunner,
    }

    task = pdt.ExploreTask(
        df_index=1,
        epoch_index=0,
        context_index=2,
        initial_action=0,
        round_counter=5,
        epsilon=0.1,
    )
    input_queue.put(task)
    input_queue.put(pwap.ShutdownWorker())

    pwap.df_rollout_worker(worker_config, input_queue, result_queue)

    assert result_queue.get() == "fake_round_result"

    with open(log_file, "r") as f:
        log_content = f.read()

    assert "[worker-7]" in log_content
    assert "worker task started" in log_content
    assert "df_index=1" in log_content


def test_shm_cache_optimization_for_worker_pool(tmp_path, monkeypatch):
    """start_parallel_workers uses shm cache paths and cleans them up upon shutdown."""
    import os
    import pickle
    import queue
    import pandas as pd
    from unittest.mock import MagicMock

    class FakeQueue:
        pass

    class FakeContext:
        def Queue(self):
            return FakeQueue()

        def Process(self, target, args):
            process = MagicMock()
            process.worker_config = args[0]
            process.is_alive.return_value = False
            return process

    monkeypatch.setattr(pdt, "create_worker_context", lambda: FakeContext())
    monkeypatch.setattr(pdt, "build_effective_df_indices", lambda n: list(range(n)))

    sample_df = pd.DataFrame({"mark_price": [100.0, 101.0], "feat_1": [1.0, 2.0], "feat_2": [3.0, 4.0]})
    train_df_cache = {0: sample_df}

    trainer = MagicMock()
    trainer.dataset_name = "test_shm_fu"
    trainer.experiment_name = "test_shm_exp"
    trainer.total_df_index_length = 1
    trainer.diverse_num_workers = 2
    trainer.device = "cpu"
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
    trainer.shm_df_cache_path = None
    trainer.shm_model_path = None

    eval_net = pdt.ensemble_Qnet(
        N_STATES=2,
        N_ACTIONS=3,
        hidden_nodes=16,
        TIME_INFO_DIM=2,
        ensemble_number=2,
    )
    trainer.eval_net = eval_net

    pdt.start_parallel_workers(trainer, train_df_cache, {})

    assert trainer.shm_df_cache_path is not None
    assert trainer.shm_model_path is not None
    assert os.path.exists(trainer.shm_df_cache_path)
    assert os.path.exists(trainer.shm_model_path)

    for process in trainer.worker_processes:
        cfg = process.worker_config
        assert "train_df_cache_path" in cfg
        assert "state_dict_path" in cfg
        assert cfg["train_df_cache_path"] == trainer.shm_df_cache_path
        assert cfg["state_dict_path"] == trainer.shm_model_path

        # Verify runner loads correctly from shm cache path
        runner = pdt.DfRolloutWorkerRunner(cfg)
        assert 0 in runner.train_df_by_df
        assert len(runner.train_df_by_df[0]) == 2

    # Verify shutdown removes shm cache files
    task_q = queue.Queue()
    trainer.worker_task_queue = task_q
    df_path = trainer.shm_df_cache_path
    model_path = trainer.shm_model_path

    pdt.shutdown_exploration_workers(trainer)

    assert not os.path.exists(df_path)
    assert not os.path.exists(model_path)
    assert trainer.shm_df_cache_path is None
    assert trainer.shm_model_path is None


def test_df_rollout_worker_exits_on_foreign_module_shutdown_worker():
    """df_rollout_worker must exit cleanly when ShutdownWorker comes from a foreign module (__main__)."""
    import queue
    from dataclasses import dataclass
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap

    @dataclass(frozen=True)
    class ForeignShutdownWorker:
        pass
    ForeignShutdownWorker.__name__ = "ShutdownWorker"

    input_queue = queue.Queue()
    result_queue = queue.Queue()
    input_queue.put(ForeignShutdownWorker())

    worker_config = {"runner_factory": lambda cfg: None}
    pwap.df_rollout_worker(worker_config, input_queue, result_queue)
    assert result_queue.empty()


def test_df_rollout_worker_exception_handler_does_not_crash_on_message_without_df_index():
    """If an exception occurs when message has no df_index, WorkerErrorMessage is created without AttributeError."""
    import queue
    from dataclasses import dataclass
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap

    @dataclass(frozen=True)
    class UnknownMessageWithoutDfIndex:
        pass

    input_queue = queue.Queue()
    result_queue = queue.Queue()
    input_queue.put(UnknownMessageWithoutDfIndex())

    worker_config = {"runner_factory": lambda cfg: None}
    pwap.df_rollout_worker(worker_config, input_queue, result_queue)

    assert not result_queue.empty()
    err = result_queue.get()
    assert isinstance(err, pwap.WorkerErrorMessage)
    assert err.df_index == -1
    assert "unknown worker message type: UnknownMessageWithoutDfIndex" in err.traceback
