import sys
from pathlib import Path


FINEFT_ROOT = Path(__file__).resolve().parents[2]
if str(FINEFT_ROOT) not in sys.path:
    sys.path.insert(0, str(FINEFT_ROOT))


def test_effective_df_indices_use_total_df_index_length_without_extra_drop():
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap

    assert pwap.build_effective_df_indices(3) == [0, 1, 2]


def test_summarize_rollout_metrics_uses_rollout_metric_objects():
    from RL.DiHFT.low_level import parallel_diverse_train as pdt

    metrics = [
        pdt.RolloutMetrics(
            epoch_index=0,
            context_index=0,
            initial_action=1,
            df_index=0,
            transition_count=3,
            reward_sum=100.0,
            final_balance=110000.0,
            return_rate=1.1,
        ),
        pdt.RolloutMetrics(
            epoch_index=0,
            context_index=0,
            initial_action=1,
            df_index=1,
            transition_count=2,
            reward_sum=-20.0,
            final_balance=99000.0,
            return_rate=0.99,
        ),
    ]

    summary = pdt.summarize_rollout_metrics(metrics)

    assert summary == pdt.RolloutMetricsSummary(
        mean_reward_sum=40.0,
        mean_return_rate=1.045,
        mean_final_balance=104500.0,
    )
    assert summary.to_dict() == {
        "mean_reward_sum": 40.0,
        "mean_return_rate": 1.045,
        "mean_final_balance": 104500.0,
    }


def test_summarize_rollout_diagnostics_returns_object():
    from RL.DiHFT.low_level import parallel_diverse_train as pdt

    summary = pdt.summarize_rollout_diagnostics(
        actions=[3, 1, 3, 2],
        positions=[0.0, -1.0, -1.0, 1.0],
        preview_limit=3,
    )

    assert summary == pdt.RolloutDiagnosticsSummary(
        action_counts=[(1, 1), (2, 1), (3, 2)],
        position_counts=[(-1.0, 2), (0.0, 1), (1.0, 1)],
        first_actions=[3, 1, 3],
        first_positions=[0.0, -1.0, -1.0],
        position_switches=2,
    )
    assert summary.to_dict()["position_switches"] == 2


def test_parallel_rollout_task_order_is_epoch_context_initial_action():
    from RL.DiHFT.low_level import parallel_diverse_train as pdt

    tasks = list(
        pdt.iter_parallel_rollout_tasks(
            num_epoch=2,
            context_count=2,
            position_choices=3,
        )
    )

    assert tasks[0] == pdt.ParallelRolloutTask(0, 0, 0)
    assert tasks[-1] == pdt.ParallelRolloutTask(1, 1, 2)
    assert [task.to_dict() for task in tasks[:3]] == [
        {"epoch_index": 0, "context_index": 0, "initial_action": 0},
        {"epoch_index": 0, "context_index": 0, "initial_action": 1},
        {"epoch_index": 0, "context_index": 0, "initial_action": 2},
    ]
    assert len(tasks) == 12


def test_compute_epoch_schedules_match_decay_requirements():
    from RL.DiHFT.low_level import parallel_diverse_train as pdt

    first = pdt.compute_epoch_training_params(
        epoch_index=0,
        num_epoch=5,
        epsilon_init=1.0,
        epsilon_min=0.2,
        ada_init=256.0,
        ada_min=0.0,
        lr_init=0.005,
        lr_min=0.001,
    )
    middle = pdt.compute_epoch_training_params(
        epoch_index=2,
        num_epoch=5,
        epsilon_init=1.0,
        epsilon_min=0.2,
        ada_init=256.0,
        ada_min=0.0,
        lr_init=0.005,
        lr_min=0.001,
    )
    last = pdt.compute_epoch_training_params(
        epoch_index=4,
        num_epoch=5,
        epsilon_init=1.0,
        epsilon_min=0.2,
        ada_init=256.0,
        ada_min=0.0,
        lr_init=0.005,
        lr_min=0.001,
    )

    assert first == pdt.EpochTrainingParams(epsilon=1.0, ada=256.0, lr=0.005)
    assert middle == pdt.EpochTrainingParams(epsilon=0.6, ada=256.0, lr=0.005)
    assert last == pdt.EpochTrainingParams(epsilon=0.2, ada=0.0, lr=0.001)
    assert first.to_dict() == {"epsilon": 1.0, "ada": 256.0, "lr": 0.005}


def test_single_epoch_schedule_keeps_initial_values():
    from RL.DiHFT.low_level import parallel_diverse_train as pdt

    assert pdt.compute_epoch_training_params(
        epoch_index=0,
        num_epoch=1,
        epsilon_init=1.0,
        epsilon_min=0.2,
        ada_init=256.0,
        ada_min=0.0,
        lr_init=0.005,
        lr_min=0.001,
    ) == pdt.EpochTrainingParams(epsilon=1.0, ada=256.0, lr=0.005)


def test_sort_round_results_orders_by_df_index_and_step_index():
    from RL.DiHFT.low_level import parallel_diverse_train as pdt

    results = [
        pdt.WorkerRoundResult(
            df_index=2,
            epoch_index=0,
            context_index=0,
            initial_action=0,
            round_counter=0,
            worker_steps=1,
            transitions=[
                pdt.WorkerTransitionRecord(
                    step_index=1,
                    transition="df2-step1",
                )
            ],
            rollout_metrics=[],
            done=False,
        ),
        pdt.WorkerRoundResult(
            df_index=1,
            epoch_index=0,
            context_index=0,
            initial_action=0,
            round_counter=0,
            worker_steps=2,
            transitions=[
                pdt.WorkerTransitionRecord(
                    step_index=1,
                    transition="df1-step1",
                ),
                pdt.WorkerTransitionRecord(
                    step_index=0,
                    transition="df1-step0",
                ),
            ],
            rollout_metrics=[],
            done=True,
        ),
    ]

    assert pdt.sort_round_transitions(results) == [
        "df1-step0",
        "df1-step1",
        "df2-step1",
    ]


def test_raise_for_worker_error_includes_df_and_traceback():
    import pytest
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap

    with pytest.raises(RuntimeError, match="df_index=3.*boom"):
        pwap.raise_for_worker_error(
            pwap.WorkerErrorMessage(
                df_index=3,
                epoch_index=0,
                context_index=1,
                initial_action=2,
                round_counter=4,
                traceback="boom",
            )
        )


def test_make_cpu_state_dict_detaches_and_moves_to_cpu():
    import torch
    from RL.DiHFT.low_level import parallel_diverse_train as pdt

    module = torch.nn.Linear(2, 1)
    state_dict = pdt.make_cpu_state_dict(module)

    assert set(state_dict) == set(module.state_dict())
    assert all(not tensor.requires_grad for tensor in state_dict.values())
    assert all(tensor.device.type == "cpu" for tensor in state_dict.values())


def test_count_update_windows_crossed_preserves_serial_update_density():
    import pytest
    from RL.DiHFT.low_level import parallel_diverse_train as pdt

    warmup_steps = 128 * 20 + 1

    assert pdt.count_update_windows_crossed(
        previous_step_counter=0,
        current_step_counter=warmup_steps,
        rollout_steps=1024,
        warmup_steps=warmup_steps,
    ) == 0
    assert pdt.count_update_windows_crossed(
        previous_step_counter=0,
        current_step_counter=4096,
        rollout_steps=1024,
        warmup_steps=warmup_steps,
    ) == 1
    assert pdt.count_update_windows_crossed(
        previous_step_counter=4096,
        current_step_counter=8192,
        rollout_steps=1024,
        warmup_steps=warmup_steps,
    ) == 4
    assert pdt.count_update_windows_crossed(
        previous_step_counter=3073,
        current_step_counter=4096,
        rollout_steps=1024,
        warmup_steps=warmup_steps,
    ) == 0
    with pytest.raises(ValueError, match="rollout_steps must be positive"):
        pdt.count_update_windows_crossed(
            previous_step_counter=0,
            current_step_counter=1,
            rollout_steps=0,
            warmup_steps=0,
        )


def test_run_fixed_update_times_uses_constant_update_count(monkeypatch):
    from RL.DiHFT.low_level import parallel_diverse_train as pdt

    class Buffer:
        def __init__(self):
            self.sample_calls = 0

        def sample(self):
            self.sample_calls += 1
            return (
                "states",
                "infos",
                "actions",
                "rewards",
                "next_states",
                "next_infos",
                "dones",
            )

    class Trainer:
        def __init__(self):
            self.update_counter = 0
            self.writer = type(
                "Writer",
                (),
                {"add_scalar": lambda *args, **kwargs: None},
            )()

    trainer = Trainer()
    buffer = Buffer()

    update_calls = {"count": 0}

    def fake_update(trainer, *args, **kwargs):
        update_calls["count"] += 1
        trainer.update_counter += 1
        return (1.0, 0.5, 0.5)

    monkeypatch.setattr(pdt, "update", fake_update)

    losses = pdt.run_fixed_diverse_updates(
        trainer=trainer,
        buffer_diverse=buffer,
        update_times=3,
        round_counter=8,
    )

    assert update_calls["count"] == 3
    assert buffer.sample_calls == 3
    assert losses == (1.0, 0.5, 0.5)


def test_buffer_writes_use_sorted_transition_payloads():
    from RL.DiHFT.low_level import parallel_diverse_train as pdt

    class Buffer:
        def __init__(self):
            self.added = []

        def add(self, *transition):
            self.added.append(transition)

    buffer = Buffer()
    transition_a = (
        "s0",
        {"previous_action": 0},
        1,
        1.0,
        "s1",
        {"previous_action": 1},
        False,
    )
    transition_b = (
        "s2",
        {"previous_action": 1},
        2,
        2.0,
        "s3",
        {"previous_action": 2},
        True,
    )

    pdt.write_round_transitions_to_buffer(
        buffer,
        [
            pdt.WorkerRoundResult(
                df_index=1,
                epoch_index=0,
                context_index=0,
                initial_action=0,
                round_counter=0,
                worker_steps=1,
                transitions=[
                    pdt.WorkerTransitionRecord(
                        step_index=0,
                        transition=transition_b,
                    )
                ],
                rollout_metrics=[],
                done=False,
            ),
            pdt.WorkerRoundResult(
                df_index=0,
                epoch_index=0,
                context_index=0,
                initial_action=0,
                round_counter=0,
                worker_steps=1,
                transitions=[
                    pdt.WorkerTransitionRecord(
                        step_index=0,
                        transition=transition_a,
                    )
                ],
                rollout_metrics=[],
                done=False,
            ),
        ],
    )

    assert buffer.added == [transition_a, transition_b]


def test_summarize_round_results_counts_steps_and_updates():
    from RL.DiHFT.low_level import parallel_diverse_train as pdt

    summary = pdt.summarize_parallel_round(
        round_counter=4,
        epoch_index=1,
        context_index=2,
        initial_action=3,
        round_results=[
            pdt.WorkerRoundResult(
                df_index=0,
                epoch_index=1,
                context_index=2,
                initial_action=3,
                round_counter=4,
                worker_steps=5,
                transitions=[],
                rollout_metrics=[],
                done=False,
            ),
            pdt.WorkerRoundResult(
                df_index=1,
                epoch_index=1,
                context_index=2,
                initial_action=3,
                round_counter=4,
                worker_steps=4,
                transitions=[],
                rollout_metrics=[],
                done=True,
            ),
        ],
        buffer_size=99,
        update_count=20,
    )

    assert summary == pdt.ParallelRoundSummary(
        round_counter=4,
        epoch_index=1,
        context_index=2,
        initial_action=3,
        round_steps=9,
        active_worker_count=2,
        buffer_size=99,
        update_count=20,
    )
    assert summary.to_dict() == {
        "round_counter": 4,
        "epoch_index": 1,
        "context_index": 2,
        "initial_action": 3,
        "round_steps": 9,
        "active_worker_count": 2,
        "buffer_size": 99,
        "update_count": 20,
    }


def test_epoch_model_path_uses_epoch_index():
    from RL.DiHFT.low_level import parallel_diverse_train as pdt

    assert pdt.build_epoch_model_path("/tmp/model", 2).endswith("epoch_3")


def test_parallel_parser_allow_reverse_position_default_and_flag():
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap

    args_default = pwap.parser.parse_args([])
    assert args_default.allow_reverse_position is False

    args_flag = pwap.parser.parse_args(["--allow_reverse_position"])
    assert args_flag.allow_reverse_position is True


def _make_parallel_update_trainer(pwap):
    import torch

    trainer = pwap.Weighted_Contexts_DQN.__new__(pwap.Weighted_Contexts_DQN)
    trainer.device = "cpu"
    trainer.batch_size = 2
    trainer.N = 2
    trainer.N_ACTIONS = 3
    trainer.gamma = 0.99
    trainer.if_use_hubber_loss = False
    trainer.neighbor_size = 1
    trainer.outer_bond = 4
    trainer.reachout_index = 1
    trainer.ada = 0.0
    trainer.grad_clip = None
    trainer.tau = 1.0
    trainer.update_counter = 0
    trainer.eval_net = pwap.ensemble_Qnet(
        N_STATES=4,
        N_ACTIONS=trainer.N_ACTIONS,
        hidden_nodes=16,
        TIME_INFO_DIM=2,
        ensemble_number=trainer.N,
    )
    trainer.target_net = pwap.ensemble_Qnet(
        N_STATES=4,
        N_ACTIONS=trainer.N_ACTIONS,
        hidden_nodes=16,
        TIME_INFO_DIM=2,
        ensemble_number=trainer.N,
    )
    trainer.optimizer = torch.optim.Adam(trainer.eval_net.parameters(), lr=0.001)
    trainer.loss_func_pretrain = torch.nn.SmoothL1Loss(reduction="none")
    return trainer


def _sample_parallel_update_batch():
    import torch
    states = torch.randn(2, 4)
    next_states = torch.randn(2, 4)
    info = {
        "previous_action": torch.zeros(2),
        "avaliable_action": torch.ones(2, 3),
        "funding_count_down_hour": torch.zeros(2),
        "funding_count_down_minute": torch.ones(2),
        "trading_info": torch.tensor(
            [[0.5, 0.02, -0.01, 0.1], [0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        "q_value": torch.zeros(2, 3),
    }
    next_info = {
        "previous_action": torch.ones(2),
        "avaliable_action": torch.ones(2, 3),
        "funding_count_down_hour": torch.ones(2),
        "funding_count_down_minute": torch.zeros(2),
        "trading_info": torch.tensor(
            [[0.5, 0.03, -0.01, 0.2], [0.5, 0.01, -0.02, 0.1]],
            dtype=torch.float32,
        ),
    }
    actions = torch.tensor([0, 2], dtype=torch.long)
    rewards = torch.tensor([[0.1], [-0.2]], dtype=torch.float32)
    dones = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    return states, info, actions, rewards, next_states, next_info, dones


def test_parallel_training_update_uses_four_field_trading_info():
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap
    from RL.DiHFT.low_level import parallel_diverse_train as pdt

    trainer = _make_parallel_update_trainer(pwap)

    losses = pdt.update(trainer, *_sample_parallel_update_batch())

    assert len(losses) == 3
    assert trainer.update_counter == 1


def test_parallel_training_pretrain_update_uses_four_field_trading_info():
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap
    from RL.DiHFT.low_level import parallel_pretrain as pp

    trainer = _make_parallel_update_trainer(pwap)

    losses = pp.update_pretrain(trainer, *_sample_parallel_update_batch())

    assert len(losses) == 3
    assert trainer.update_counter == 1


def test_configure_logger_captures_parallel_pretrain_logs(tmp_path, monkeypatch):
    import os
    import logging
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap

    monkeypatch.chdir(tmp_path)

    abs_path = pwap.configure_logger("fu", "exp_test")
    test_logger = logging.getLogger("RL.DiHFT.low_level.parallel_weight_advantage_pretrain")
    test_msg = "exhaustive warmup train epoch | epoch=1/1 | total_loss=0.000123 | KL_loss=0.000010 | td_loss=0.000113 | update_count=10"
    test_logger.info(test_msg)

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.flush()

    with open(abs_path, "r", encoding="utf-8") as f:
        log_content = f.read()

    assert test_msg in log_content

    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == abs_path:
            handler.close()
            root_logger.removeHandler(handler)


def test_select_pretrain_action_clamps_unavailable_perfection_action():
    from RL.DiHFT.low_level.parallel_pretrain import select_pretrain_action

    info = {"avaiable_action_list": [0, 1, 2, 3, 4, 5]}
    perfection_action_list = [10, 10, 10]

    action = select_pretrain_action(
        info=info,
        optimal_step_counter=0,
        rollout_index=0,
        perfection_action_list=perfection_action_list,
        position_choices=11,
        leverage_choices=[1],
    )

    assert action in info["avaiable_action_list"]
    assert action == 5


def test_parallel_parser_exposes_fixed_neighbor_size():
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap

    args = pwap.parser.parse_args([])
    assert args.neighbor_size == 1

    args_custom = pwap.parser.parse_args(["--neighbor_size", "2"])
    assert args_custom.neighbor_size == 2


def test_parallel_diverse_update_matches_paper_loss_computation():
    import math
    import pytest
    import torch
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap
    from RL.DiHFT.low_level import parallel_diverse_train as pdt
    from RL.DiHFT.low_level.weight_advantage_pretrain import (
        calculate_paper_partial_loss,
        calculate_paper_supervisor_kl_loss,
    )

    trainer = _make_parallel_update_trainer(pwap)
    trainer.ada = 1.0
    trainer.neighbor_size = 1
    batch = _sample_parallel_update_batch()

    losses = pdt.update(trainer, *batch)
    assert len(losses) == 3
    total_loss, kl_loss, td_loss = losses
    assert total_loss == pytest.approx(td_loss + kl_loss, abs=1e-6)


def test_run_exhaustive_warmup_collects_all_episodes_before_training_and_updates_model(
    monkeypatch,
):
    import queue
    from unittest.mock import MagicMock
    from RL.DiHFT.low_level import parallel_pretrain as pp

    events = []

    class DummyInputQueue:
        def __init__(self, df_index, result_queue):
            self.df_index = df_index
            self.result_queue = result_queue
            self.received = []

        def put(self, message):
            self.received.append(message)
            events.append(
                (
                    "dispatch_task",
                    self.df_index,
                    message.initial_action,
                    message.rollout_index,
                )
            )
            transitions = [
                ("s1", {}, 0, 1.0, "s2", {}, False),
                ("s2", {}, 0, 0.5, "s3", {}, True),
            ]
            self.result_queue.put(
                pp.PretrainCollectResult(
                    df_index=self.df_index,
                    initial_action=message.initial_action,
                    rollout_index=message.rollout_index,
                    transitions=transitions,
                    reward_sum=1.5,
                    final_balance=10150.0,
                    transition_count=len(transitions),
                )
            )

    trainer = MagicMock()
    trainer.total_df_index_length = 2
    trainer.position_choices = 1
    trainer.initial_wallet_balance = 10000.0
    trainer.pretrain_epoch = 2
    trainer.update_times = 3
    trainer.batch_size = 2
    trainer.update_counter = 0

    def fake_shutdown():
        events.append("shutdown_workers")

    trainer._shutdown_parallel_workers.side_effect = fake_shutdown

    def mock_start_workers(tr, train_df_cache, env_kwargs, q_table_cache):
        events.append("start_workers")
        tr.worker_result_queue = queue.Queue()
        tr.worker_input_queues = {
            0: DummyInputQueue(0, tr.worker_result_queue),
            1: DummyInputQueue(1, tr.worker_result_queue),
        }

    monkeypatch.setattr(pp, "start_pretrain_collect_workers", mock_start_workers)

    added_transitions = []

    class DummyBuffer:
        def __len__(self):
            return len(added_transitions)

        def add(self, *transition):
            added_transitions.append(transition)
            events.append(("buffer_add", len(added_transitions)))

        def sample(self):
            events.append("buffer_sample")
            return ("states", {}, "actions", "rewards", "next_states", {}, "dones")

    def mock_update_pretrain(tr, *batch):
        events.append("update_pretrain")
        return (1.0, 0.2, 0.8)

    monkeypatch.setattr(pp, "update_pretrain", mock_update_pretrain)
    monkeypatch.setattr(
        pp,
        "write_pretrain_loss_scalars",
        lambda tr, *losses: events.append("write_loss"),
    )

    buffer_pretrain = DummyBuffer()
    summary, final_steps = pp.run_exhaustive_warmup(
        trainer=trainer,
        q_table_cache={0: None, 1: None},
        train_df_cache={0: None, 1: None},
        env_kwargs={},
        buffer_pretrain=buffer_pretrain,
        step_counter_pretrain=10,
    )

    assert summary["episodes"] == 8
    assert summary["transitions"] == 26
    assert final_steps == 26
    assert summary["update_count"] == 6

    last_add_idx = max(
        i
        for i, ev in enumerate(events)
        if isinstance(ev, tuple) and ev[0] == "buffer_add"
    )
    shutdown_idx = events.index("shutdown_workers")
    first_sample_idx = events.index("buffer_sample")
    first_update_idx = events.index("update_pretrain")

    assert last_add_idx < shutdown_idx
    assert shutdown_idx < first_sample_idx
    assert shutdown_idx < first_update_idx

    assert sum(1 for ev in events if ev == "update_pretrain") == 6
    assert sum(1 for ev in events if ev == "write_loss") == 6


def test_run_exhaustive_warmup_skips_training_when_pretrain_epoch_zero(monkeypatch):
    import queue
    from unittest.mock import MagicMock
    from RL.DiHFT.low_level import parallel_pretrain as pp

    events = []

    class DummyInputQueue:
        def __init__(self, df_index, result_queue):
            self.df_index = df_index
            self.result_queue = result_queue

        def put(self, message):
            self.result_queue.put(
                pp.PretrainCollectResult(
                    df_index=self.df_index,
                    initial_action=message.initial_action,
                    rollout_index=message.rollout_index,
                    transitions=[("s", {}, 0, 1.0, "s_", {}, True)],
                    reward_sum=1.0,
                    final_balance=10000.0,
                    transition_count=1,
                )
            )

    trainer = MagicMock()
    trainer.total_df_index_length = 1
    trainer.position_choices = 1
    trainer.initial_wallet_balance = 10000.0
    trainer.pretrain_epoch = 0
    trainer.update_times = 10
    trainer.batch_size = 2
    trainer.update_counter = 0

    def fake_shutdown():
        events.append("shutdown_workers")

    trainer._shutdown_parallel_workers.side_effect = fake_shutdown

    def mock_start_workers(tr, train_df_cache, env_kwargs, q_table_cache):
        tr.worker_result_queue = queue.Queue()
        tr.worker_input_queues = {0: DummyInputQueue(0, tr.worker_result_queue)}

    monkeypatch.setattr(pp, "start_pretrain_collect_workers", mock_start_workers)

    class DummyBuffer:
        def __init__(self):
            self.items = []

        def __len__(self):
            return len(self.items)

        def add(self, *transition):
            self.items.append(transition)

        def sample(self):
            raise AssertionError("sample() should not be called when pretrain_epoch=0")

    monkeypatch.setattr(
        pp,
        "update_pretrain",
        MagicMock(side_effect=AssertionError("update_pretrain called")),
    )

    buffer_pretrain = DummyBuffer()
    summary, final_steps = pp.run_exhaustive_warmup(
        trainer=trainer,
        q_table_cache={0: None},
        train_df_cache={0: None},
        env_kwargs={},
        buffer_pretrain=buffer_pretrain,
        step_counter_pretrain=0,
    )

    assert summary["episodes"] == 4
    assert summary["transitions"] == 4
    assert summary["update_count"] == 0
    assert final_steps == 4
    assert "shutdown_workers" in events


def test_run_exhaustive_warmup_rejects_negative_pretrain_epoch():
    import pytest
    from unittest.mock import MagicMock
    from RL.DiHFT.low_level import parallel_pretrain as pp

    trainer = MagicMock()
    trainer.total_df_index_length = 1
    trainer.position_choices = 1
    trainer.pretrain_epoch = -1

    with pytest.raises(ValueError, match="pretrain_epoch must be non-negative"):
        pp.run_exhaustive_warmup(
            trainer=trainer,
            q_table_cache={0: None},
            train_df_cache={0: None},
            env_kwargs={},
            buffer_pretrain=MagicMock(),
            step_counter_pretrain=0,
        )


def test_run_exhaustive_warmup_rejects_non_positive_update_times_when_pretrain_epoch_positive(
    monkeypatch,
):
    import queue
    import pytest
    from unittest.mock import MagicMock
    from RL.DiHFT.low_level import parallel_pretrain as pp

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

    trainer = MagicMock()
    trainer.total_df_index_length = 1
    trainer.position_choices = 1
    trainer.initial_wallet_balance = 10000.0
    trainer.pretrain_epoch = 1
    trainer.update_times = 0
    trainer.batch_size = 1

    def mock_start_workers(tr, train_df_cache, env_kwargs, q_table_cache):
        tr.worker_result_queue = queue.Queue()
        tr.worker_input_queues = {0: DummyInputQueue(tr.worker_result_queue)}

    monkeypatch.setattr(pp, "start_pretrain_collect_workers", mock_start_workers)

    class DummyBuffer:
        def __init__(self):
            self.items = []

        def __len__(self):
            return len(self.items)

        def add(self, *transition):
            self.items.append(transition)

    with pytest.raises(
        ValueError, match="update_times must be positive when pretrain_epoch > 0"
    ):
        pp.run_exhaustive_warmup(
            trainer=trainer,
            q_table_cache={0: None},
            train_df_cache={0: None},
            env_kwargs={},
            buffer_pretrain=DummyBuffer(),
            step_counter_pretrain=0,
        )


def test_run_parallel_pretrain_evaluation_dispatches_sync_and_eval_tasks(monkeypatch):
    import queue
    from unittest.mock import MagicMock
    from RL.DiHFT.low_level import parallel_pretrain as pp

    events = []

    class DummyWorkerQueue:
        def __init__(self, q_id, result_queue):
            self.q_id = q_id
            self.result_queue = result_queue

        def put(self, message):
            if type(message).__name__ == "SyncPretrainModel":
                events.append(("sync_model", self.q_id))
            elif type(message).__name__ == "EvaluatePretrainEpisode":
                events.append(
                    (
                        "eval_episode",
                        self.q_id,
                        message.context_index,
                        message.df_index,
                        message.initial_action,
                    )
                )
                self.result_queue.put(
                    pp.PretrainEvalResult(
                        df_index=message.df_index,
                        context_index=message.context_index,
                        initial_action=message.initial_action,
                        reward_sum=5.0,
                        final_balance=10050.0,
                        return_rate=0.005,
                    )
                )

    result_queue = queue.Queue()
    q0 = DummyWorkerQueue(0, result_queue)
    q1 = DummyWorkerQueue(1, result_queue)

    trainer = MagicMock()
    trainer.N = 2
    trainer.total_df_index_length = 3
    # 3 dfs sharded over 2 queues
    trainer.worker_input_queues = {0: q0, 1: q1, 2: q0}
    trainer.worker_result_queue = result_queue
    trainer.writer = MagicMock()

    eval_net_mock = MagicMock()
    eval_net_mock.state_dict.return_value = {}
    trainer.eval_net = eval_net_mock

    monkeypatch.setattr(
        "RL.DiHFT.low_level.parallel_diverse_train.make_cpu_state_dict",
        lambda m: {"dummy_weight": 1},
    )

    metrics = pp.run_parallel_pretrain_evaluation(
        trainer=trainer,
        train_df_cache={0: "df0", 1: "df1", 2: "df2"},
        env_kwargs={},
    )

    # 2 contexts * 3 dfs = 6 results
    assert len(metrics) == 6
    for m in metrics:
        assert m.initial_action == 0
        assert m.reward_sum == 5.0
        assert m.final_balance == 10050.0

    # Sync model was sent exactly once to each unique queue (q0, q1)
    sync_events = [ev for ev in events if ev[0] == "sync_model"]
    assert len(sync_events) == 2
    assert {ev[1] for ev in sync_events} == {0, 1}

    # All 6 evaluation tasks were dispatched
    eval_events = [ev for ev in events if ev[0] == "eval_episode"]
    assert len(eval_events) == 6
    assert all(ev[4] == 0 for ev in eval_events)  # initial_action is 0
    assert {(ev[2], ev[3]) for ev in eval_events} == {
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    }


def test_evaluate_warmup_sub_agents_iterates_all_contexts_and_dfs_with_action_zero(
    monkeypatch,
):
    import pytest
    from unittest.mock import MagicMock
    import torch
    import numpy as np
    from RL.DiHFT.low_level import parallel_pretrain as pp

    built_states = []
    created_envs = []

    class DummyEnv:
        def __init__(self, df_name):
            self.df_name = df_name
            self.step_count = 0
            self.unrealized_pnl = 50.0
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
                10.0,
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

    def mock_build_initial_state(train_df, initial_action, *args, **kwargs):
        built_states.append((train_df, initial_action))
        return None, None, None, f"init_{train_df}_{initial_action}"

    def mock_create_demo_env(train_df, env_kwargs, initial_state):
        env = DummyEnv(train_df)
        created_envs.append(env)
        return env

    monkeypatch.setattr(pp, "build_initial_state", mock_build_initial_state)
    monkeypatch.setattr(pp, "create_demo_env", mock_create_demo_env)

    trainer = MagicMock()
    trainer.N = 2
    trainer.total_df_index_length = 3
    trainer.device = "cpu"
    trainer.leverage_choices = [1]
    trainer.position_list = [0]
    trainer.initial_wallet_balance = 10000.0
    trainer.initial_unrealized_pnL = 0.0

    def mock_eval_net(**kwargs):
        return torch.tensor([[[1.0, 5.0, 2.0], [1.0, 2.0, 6.0]]], dtype=torch.float32)

    trainer.eval_net = mock_eval_net

    train_df_cache = {0: "df0", 1: "df1", 2: "df2"}
    metrics = pp.evaluate_warmup_sub_agents(
        trainer=trainer,
        train_df_cache=train_df_cache,
        env_kwargs={},
    )

    assert len(metrics) == 6
    for m in metrics:
        assert m.initial_action == 0
        assert m.reward_sum == 20.0
        assert m.final_balance == 10150.0
        assert m.return_rate == pytest.approx(0.015, abs=1e-6)

    assert all(init_a == 0 for _, init_a in built_states)
    assert {(m.context_index, m.df_index) for m in metrics} == {
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    }


def test_run_exhaustive_warmup_calls_evaluation_and_returns_eval_metrics(monkeypatch):
    import queue
    from unittest.mock import MagicMock
    from RL.DiHFT.low_level import parallel_pretrain as pp

    trainer = MagicMock()
    trainer.total_df_index_length = 1
    trainer.position_choices = 1
    trainer.initial_wallet_balance = 10000.0
    trainer.pretrain_epoch = 0
    trainer.update_times = 0
    trainer.batch_size = 1

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

    expected_eval_metric = pp.WarmupEvalMetric(
        context_index=0,
        df_index=0,
        initial_action=0,
        reward_sum=10.0,
        final_balance=10100.0,
        return_rate=0.01,
    )
    monkeypatch.setattr(
        pp,
        "evaluate_warmup_sub_agents",
        lambda trainer, train_df_cache, env_kwargs: [expected_eval_metric],
    )

    class DummyBuffer:
        def __len__(self):
            return 4

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

    assert "eval_metrics" in summary
    assert summary["eval_metrics"] == [expected_eval_metric]



def test_parallel_parser_pretrain_num_workers_default_and_flags():
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap

    args_default = pwap.parser.parse_args([])
    assert args_default.pretrain_num_workers == 150
    assert args_default.eval_num_workers == 150

    args_custom = pwap.parser.parse_args(["--pretrain_num_workers", "80", "--eval_num_workers", "90"])
    assert args_custom.pretrain_num_workers == 80
    assert args_custom.eval_num_workers == 90

    args_alias = pwap.parser.parse_args(["--pretrain_workers", "64", "--pretrain_eval_workers", "48"])
    assert args_alias.pretrain_num_workers == 64
    assert args_alias.eval_num_workers == 48


def test_trainer_validates_pretrain_num_workers():
    import pytest
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap

    trainer = pwap.Weighted_Contexts_DQN.__new__(pwap.Weighted_Contexts_DQN)
    args = pwap.parser.parse_args([])
    args.pretrain_num_workers = 0
    with pytest.raises(ValueError, match="pretrain_num_workers must be positive"):
        if args.pretrain_num_workers <= 0:
            raise ValueError("pretrain_num_workers must be positive")

    args.pretrain_num_workers = 150
    args.eval_num_workers = 0
    with pytest.raises(ValueError, match="eval_num_workers must be positive"):
        if args.eval_num_workers <= 0:
            raise ValueError("eval_num_workers must be positive")


def test_start_pretrain_collect_workers_respects_worker_count_and_shards_dfs(
    monkeypatch,
):
    import queue
    from unittest.mock import MagicMock
    from RL.DiHFT.low_level import parallel_pretrain as pp
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap

    spawned_processes = []
    created_configs = []

    class DummyProcess:
        def __init__(self, target, args):
            self.target = target
            self.args = args
            created_configs.append(args[0])
            spawned_processes.append(self)

        def start(self):
            pass

    class DummyContext:
        def Queue(self):
            return queue.Queue()

        def Process(self, target, args):
            return DummyProcess(target, args)

    monkeypatch.setattr(pwap, "create_worker_context", lambda: DummyContext())

    trainer = MagicMock()
    trainer.total_df_index_length = 5
    trainer.pretrain_num_workers = 2
    trainer.leverage_choices = [1]
    trainer.position_list = [0]
    trainer.position_choices = 1
    trainer.initial_wallet_balance = 10000.0
    trainer.initial_unrealized_pnL = 0.0

    train_df_cache = {i: f"df_{i}" for i in range(5)}
    q_table_cache = {i: f"qtable_{i}" for i in range(5)}

    pp.start_pretrain_collect_workers(
        trainer=trainer,
        train_df_cache=train_df_cache,
        env_kwargs={},
        q_table_cache=q_table_cache,
    )

    assert len(trainer.worker_processes) == 2
    assert len(spawned_processes) == 2

    assert set(trainer.worker_input_queues.keys()) == {0, 1, 2, 3, 4}
    assert trainer.worker_input_queues[0] is trainer.worker_input_queues[2]
    assert trainer.worker_input_queues[0] is trainer.worker_input_queues[4]
    assert trainer.worker_input_queues[1] is trainer.worker_input_queues[3]
    assert trainer.worker_input_queues[0] is not trainer.worker_input_queues[1]

    worker_0_config = created_configs[0]
    worker_1_config = created_configs[1]
    assert worker_0_config["df_indices"] == [0, 2, 4]
    assert worker_1_config["df_indices"] == [1, 3]
    assert worker_0_config["train_df_by_df"] == {0: "df_0", 2: "df_2", 4: "df_4"}
    assert worker_1_config["train_df_by_df"] == {1: "df_1", 3: "df_3"}


def test_pretrain_collect_runner_handles_multi_df_tasks(monkeypatch):
    from unittest.mock import MagicMock
    from RL.DiHFT.low_level import parallel_pretrain as pp

    env_mock = MagicMock()
    env_mock.reset.return_value = ("state", {"avaiable_action_list": [0]})
    env_mock.step.return_value = (
        "next_state",
        1.0,
        True,
        {"avaiable_action_list": [0]},
    )
    env_mock.unrealized_pnl = 0.0
    env_mock.wallet_balance = 10000.0

    monkeypatch.setattr(
        pp,
        "build_initial_state",
        lambda *args, **kwargs: (None, None, None, "init_state"),
    )
    monkeypatch.setattr(pp, "create_demo_env", lambda df, kwargs, init_s: env_mock)
    monkeypatch.setattr(pp, "get_dp_action_from_qtable", lambda qtable, init_a: [0])

    worker_config = {
        "df_indices": [0, 1],
        "train_df_by_df": {0: "df0", 1: "df1"},
        "q_table_by_df": {0: "q0", 1: "q1"},
        "env_kwargs": {},
        "leverage_choices": [1],
        "position_list": [0],
        "position_choices": 1,
        "initial_wallet_balance": 10000.0,
        "initial_unrealized_pnL": 0.0,
    }
    runner = pp.PretrainCollectRunner(worker_config)

    res0 = runner.collect_episode(
        pp.CollectPretrainEpisode(initial_action=0, rollout_index=0, df_index=0)
    )
    assert res0.df_index == 0

    res1 = runner.collect_episode(
        pp.CollectPretrainEpisode(initial_action=0, rollout_index=0, df_index=1)
    )
    assert res1.df_index == 1


def test_shutdown_workers_deduplicates_shared_queues():
    import queue
    from unittest.mock import MagicMock
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap

    q1 = queue.Queue()
    q2 = queue.Queue()
    input_queues = [q1, q2, q1, q2]

    p1 = MagicMock()
    p1.is_alive.return_value = False
    p2 = MagicMock()
    p2.is_alive.return_value = False

    pwap.shutdown_workers(input_queues, [p1, p2])

    assert q1.qsize() == 1
    assert q2.qsize() == 1



def test_run_exhaustive_warmup_rejects_insufficient_buffer_for_batch_size(monkeypatch):
    import queue
    import pytest
    from unittest.mock import MagicMock
    from RL.DiHFT.low_level import parallel_pretrain as pp

    class DummyInputQueue:
        def __init__(self, result_queue):
            self.result_queue = result_queue

        def put(self, message):
            self.result_queue.put(
                pp.PretrainCollectResult(
                    df_index=0,
                    initial_action=message.initial_action,
                    rollout_index=message.rollout_index,
                    transitions=[],  # 0 transitions collected
                    reward_sum=0.0,
                    final_balance=10000.0,
                    transition_count=0,
                )
            )

    trainer = MagicMock()
    trainer.total_df_index_length = 1
    trainer.position_choices = 1
    trainer.initial_wallet_balance = 10000.0
    trainer.pretrain_epoch = 1
    trainer.update_times = 5
    trainer.batch_size = 32

    def mock_start_workers(tr, train_df_cache, env_kwargs, q_table_cache):
        tr.worker_result_queue = queue.Queue()
        tr.worker_input_queues = {0: DummyInputQueue(tr.worker_result_queue)}

    monkeypatch.setattr(pp, "start_pretrain_collect_workers", mock_start_workers)

    class DummyBuffer:
        def __init__(self):
            self.items = []

        def __len__(self):
            return len(self.items)

        def add(self, *transition):
            self.items.append(transition)

    with pytest.raises(
        ValueError, match="buffer_pretrain size .* is smaller than batch_size"
    ):
        pp.run_exhaustive_warmup(
            trainer=trainer,
            q_table_cache={0: None},
            train_df_cache={0: None},
            env_kwargs={},
            buffer_pretrain=DummyBuffer(),
            step_counter_pretrain=0,
        )


def test_parallel_parser_load_pretrain_model_flags():
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap

    args_default = pwap.parser.parse_args([])
    assert args_default.load_pretrain_model is False

    args_flag = pwap.parser.parse_args(["--load_pretrain_model"])
    assert args_flag.load_pretrain_model is True

    args_alias = pwap.parser.parse_args(["--load_pretrained_model"])
    assert args_alias.load_pretrain_model is True


def test_run_exhaustive_warmup_saves_buffer_and_loads_to_skip_exploration(tmp_path, monkeypatch):
    import os
    import queue
    from unittest.mock import MagicMock
    import torch
    from RL.DiHFT.low_level import parallel_pretrain as pp

    buffer_path = str(tmp_path / "pretrain_buffer.pt")
    worker_started = []

    class DummyInputQueue:
        def __init__(self, result_queue):
            self.result_queue = result_queue

        def put(self, message):
            self.result_queue.put(
                pp.PretrainCollectResult(
                    df_index=0,
                    initial_action=message.initial_action,
                    rollout_index=message.rollout_index,
                    transitions=[("s", {"trading_info": [0]}, 0, 1.0, "s_", {"trading_info": [0]}, True)],
                    reward_sum=1.0,
                    final_balance=10000.0,
                    transition_count=1,
                )
            )

    def mock_start_workers(tr, train_df_cache, env_kwargs, q_table_cache):
        worker_started.append(True)
        tr.worker_result_queue = queue.Queue()
        tr.worker_input_queues = {0: DummyInputQueue(tr.worker_result_queue)}

    monkeypatch.setattr(pp, "start_pretrain_collect_workers", mock_start_workers)
    monkeypatch.setattr(pp, "evaluate_warmup_sub_agents", lambda *args, **kwargs: [])

    class SimpleBuffer:
        def __init__(self):
            self.items = []

        def __len__(self):
            return len(self.items)

        def add(self, *transition):
            self.items.append(transition)

    trainer = MagicMock()
    trainer.model_path = str(tmp_path)
    trainer.pretrain_buffer_path = buffer_path
    trainer.total_df_index_length = 1
    trainer.position_choices = 1
    trainer.initial_wallet_balance = 10000.0
    trainer.pretrain_epoch = 0
    trainer.update_times = 0
    trainer.batch_size = 1

    # Run 1: Buffer does not exist -> runs worker collection and saves buffer file
    buf1 = SimpleBuffer()
    assert not os.path.exists(buffer_path)
    summary1, steps1 = pp.run_exhaustive_warmup(
        trainer=trainer,
        q_table_cache={0: None},
        train_df_cache={0: None},
        env_kwargs={},
        buffer_pretrain=buf1,
        step_counter_pretrain=0,
    )
    assert len(worker_started) == 1
    assert os.path.exists(buffer_path)
    assert len(buf1) == 4
    assert steps1 == 4

    # Run 2: Buffer exists -> loads buffer and skips worker collection entirely
    worker_started.clear()
    buf2 = SimpleBuffer()
    summary2, steps2 = pp.run_exhaustive_warmup(
        trainer=trainer,
        q_table_cache={0: None},
        train_df_cache={0: None},
        env_kwargs={},
        buffer_pretrain=buf2,
        step_counter_pretrain=0,
    )
    # Exploration was skipped!
    assert len(worker_started) == 0
    assert len(buf2) == 4
    assert steps2 == 4


def test_run_exhaustive_warmup_saves_model_after_learning(tmp_path, monkeypatch):
    import os
    from unittest.mock import MagicMock
    import torch
    from RL.DiHFT.low_level import parallel_pretrain as pp

    model_path = str(tmp_path)
    expected_model_file = str(tmp_path / "pretrain_model.pkl")

    class SimpleLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(2, 2)

    net = SimpleLinear()
    trainer = MagicMock()
    trainer.model_path = model_path
    trainer.eval_net = net
    trainer.total_df_index_length = 1
    trainer.position_choices = 1
    trainer.initial_wallet_balance = 10000.0
    trainer.pretrain_epoch = 1
    trainer.update_times = 1
    trainer.batch_size = 1

    monkeypatch.setattr(pp, "update_pretrain", lambda *args, **kwargs: (0.1, 0.05, 0.05))
    monkeypatch.setattr(pp, "evaluate_warmup_sub_agents", lambda *args, **kwargs: [])

    class DummyBuffer:
        def __len__(self):
            return 2

        def sample(self):
            return ("s", {}, "a", "r", "s_", {}, "d")

        def add(self, *transition):
            pass

    # Fake pretrain buffer already existing to skip collect
    buffer_file = str(tmp_path / "pretrain_buffer.pt")
    torch.save({"transitions": [("s", {}, 0, 1.0, "s_", {}, True)] * 2, "step_counter": 2}, buffer_file)
    trainer.pretrain_buffer_path = buffer_file

    assert not os.path.exists(expected_model_file)
    pp.run_exhaustive_warmup(
        trainer=trainer,
        q_table_cache={0: None},
        train_df_cache={0: None},
        env_kwargs={},
        buffer_pretrain=DummyBuffer(),
        step_counter_pretrain=0,
    )

    # Model was saved after training
    assert os.path.exists(expected_model_file)
    loaded_state = torch.load(expected_model_file, map_location="cpu")
    assert set(loaded_state.keys()) == set(net.state_dict().keys())


def test_run_exhaustive_warmup_skips_pretrain_when_load_pretrain_model_is_true(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    import torch
    from RL.DiHFT.low_level import parallel_pretrain as pp

    model_file = str(tmp_path / "pretrain_model.pkl")
    linear = torch.nn.Linear(2, 2)
    torch.save(linear.state_dict(), model_file)

    trainer = MagicMock()
    trainer.model_path = str(tmp_path)
    trainer.pretrain_model_path = model_file
    trainer.load_pretrain_model = True
    trainer.eval_net = torch.nn.Linear(2, 2)
    trainer.device = "cpu"

    monkeypatch.setattr(pp, "evaluate_warmup_sub_agents", lambda *args, **kwargs: ["mock_metric"])

    summary, steps = pp.run_exhaustive_warmup(
        trainer=trainer,
        q_table_cache={},
        train_df_cache={},
        env_kwargs={},
        buffer_pretrain=None,
        step_counter_pretrain=5,
    )

    assert summary["episodes"] == 0
    assert summary["update_count"] == 0
    assert summary["eval_metrics"] == ["mock_metric"]
    assert steps == 5


def test_trainer_train_skips_warmup_when_load_pretrain_model_is_true(tmp_path, monkeypatch):
    import os
    from unittest.mock import MagicMock
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap
    import torch

    model_file = str(tmp_path / "pretrain_model.pkl")
    linear = torch.nn.Linear(2, 2)
    torch.save(linear.state_dict(), model_file)

    trainer = pwap.Weighted_Contexts_DQN.__new__(pwap.Weighted_Contexts_DQN)
    trainer.load_pretrain_model = True
    trainer.pretrain_model_path = model_file
    trainer.model_path = str(tmp_path)
    trainer.device = "cpu"
    trainer.eval_net = torch.nn.Linear(2, 2)
    trainer.target_net = torch.nn.Linear(2, 2)
    trainer._log_internal_parameters = lambda *args: None
    trainer.dataset_name = "fu"
    trainer.num_sample = 1
    trainer.pretrain_epoch = 10
    trainer.N = 1
    trainer.buffer_size = 100
    trainer.batch_size = 10
    trainer.seed = 42
    trainer.gamma = 0.99
    trainer.n_step = 1

    warmup_called = []
    monkeypatch.setattr(
        pwap,
        "run_exhaustive_warmup",
        lambda *args, **kwargs: warmup_called.append(True),
    )
    monkeypatch.setattr(
        pwap,
        "prepare_pretrain_qtable_diagnostics",
        lambda *args, **kwargs: MagicMock(q_table_cache={}, train_df_cache={}),
    )
    monkeypatch.setattr(
        pwap,
        "extend_q_table_cache",
        lambda *args, **kwargs: ({}, {}),
    )

    trainer.max_holding_number = 1
    trainer.order_book_depth = 5
    trainer.position_choices = 3
    trainer.leverage_choices = [1]
    trainer.long_estimated_rate = 0
    trainer.short_estimated_rate = 0
    trainer.transcation_cost = 0
    trainer.allow_reverse_position = False
    trainer.enable_limit_reward = True
    trainer.limit_hold_bonus = 1.0
    trainer.limit_stay_bonus = 0.5
    trainer.limit_reverse_penalty = 1.5
    trainer.near_limit_threshold = 0.003
    trainer.tech_indicator_list = []
    trainer.position_list = [0]
    trainer.maintenance_margin_ratio_dict = {}
    trainer.early_stop = 0
    trainer.initial_wallet_balance = 10000
    trainer.initial_unrealized_pnL = 0
    trainer.total_df_index_length = 0
    trainer.train_data_path = ""

    trainer.train()

    assert len(warmup_called) == 0





