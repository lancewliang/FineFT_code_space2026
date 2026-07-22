import sys
from pathlib import Path


FINEFT_ROOT = Path(__file__).resolve().parents[2]
if str(FINEFT_ROOT) not in sys.path:
    sys.path.insert(0, str(FINEFT_ROOT))


def test_effective_df_indices_use_total_df_index_length_without_extra_drop():
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap

    assert pwap.build_effective_df_indices(3) == [0, 1, 2]


def test_summarize_rollout_metrics_uses_rollout_metric_objects():
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap

    metrics = [
        pwap.RolloutMetrics(
            epoch_index=0,
            context_index=0,
            initial_action=1,
            df_index=0,
            transition_count=3,
            reward_sum=100.0,
            final_balance=110000.0,
            return_rate=1.1,
        ),
        pwap.RolloutMetrics(
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

    summary = pwap.summarize_rollout_metrics(metrics)

    assert summary == pwap.RolloutMetricsSummary(
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
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap

    summary = pwap.summarize_rollout_diagnostics(
        actions=[3, 1, 3, 2],
        positions=[0.0, -1.0, -1.0, 1.0],
        preview_limit=3,
    )

    assert summary == pwap.RolloutDiagnosticsSummary(
        action_counts=[(1, 1), (2, 1), (3, 2)],
        position_counts=[(-1.0, 2), (0.0, 1), (1.0, 1)],
        first_actions=[3, 1, 3],
        first_positions=[0.0, -1.0, -1.0],
        position_switches=2,
    )
    assert summary.to_dict()["position_switches"] == 2


def test_parallel_rollout_task_order_is_epoch_context_initial_action():
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap

    tasks = list(
        pwap.iter_parallel_rollout_tasks(
            num_epoch=2,
            context_count=2,
            position_choices=3,
        )
    )

    assert tasks[0] == pwap.ParallelRolloutTask(0, 0, 0)
    assert tasks[-1] == pwap.ParallelRolloutTask(1, 1, 2)
    assert [task.to_dict() for task in tasks[:3]] == [
        {"epoch_index": 0, "context_index": 0, "initial_action": 0},
        {"epoch_index": 0, "context_index": 0, "initial_action": 1},
        {"epoch_index": 0, "context_index": 0, "initial_action": 2},
    ]
    assert len(tasks) == 12


def test_compute_epoch_schedules_match_decay_requirements():
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap

    first = pwap.compute_epoch_training_params(
        epoch_index=0,
        num_epoch=5,
        epsilon_init=1.0,
        epsilon_min=0.2,
        ada_init=256.0,
        ada_min=0.0,
        lr_init=0.005,
        lr_min=0.001,
    )
    middle = pwap.compute_epoch_training_params(
        epoch_index=2,
        num_epoch=5,
        epsilon_init=1.0,
        epsilon_min=0.2,
        ada_init=256.0,
        ada_min=0.0,
        lr_init=0.005,
        lr_min=0.001,
    )
    last = pwap.compute_epoch_training_params(
        epoch_index=4,
        num_epoch=5,
        epsilon_init=1.0,
        epsilon_min=0.2,
        ada_init=256.0,
        ada_min=0.0,
        lr_init=0.005,
        lr_min=0.001,
    )

    assert first == pwap.EpochTrainingParams(epsilon=1.0, ada=256.0, lr=0.005)
    assert middle == pwap.EpochTrainingParams(epsilon=0.6, ada=256.0, lr=0.005)
    assert last == pwap.EpochTrainingParams(epsilon=0.2, ada=0.0, lr=0.001)
    assert first.to_dict() == {"epsilon": 1.0, "ada": 256.0, "lr": 0.005}


def test_single_epoch_schedule_keeps_initial_values():
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap

    assert pwap.compute_epoch_training_params(
        epoch_index=0,
        num_epoch=1,
        epsilon_init=1.0,
        epsilon_min=0.2,
        ada_init=256.0,
        ada_min=0.0,
        lr_init=0.005,
        lr_min=0.001,
    ) == pwap.EpochTrainingParams(epsilon=1.0, ada=256.0, lr=0.005)


def test_sort_round_results_orders_by_df_index_and_step_index():
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap

    results = [
        pwap.WorkerRoundResult(
            df_index=2,
            epoch_index=0,
            context_index=0,
            initial_action=0,
            round_counter=0,
            worker_steps=1,
            transitions=[
                pwap.WorkerTransitionRecord(
                    step_index=1,
                    transition="df2-step1",
                )
            ],
            rollout_metrics=[],
            done=False,
        ),
        pwap.WorkerRoundResult(
            df_index=1,
            epoch_index=0,
            context_index=0,
            initial_action=0,
            round_counter=0,
            worker_steps=2,
            transitions=[
                pwap.WorkerTransitionRecord(
                    step_index=1,
                    transition="df1-step1",
                ),
                pwap.WorkerTransitionRecord(
                    step_index=0,
                    transition="df1-step0",
                ),
            ],
            rollout_metrics=[],
            done=True,
        ),
    ]

    assert pwap.sort_round_transitions(results) == [
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
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap

    module = torch.nn.Linear(2, 1)
    state_dict = pwap.make_cpu_state_dict(module)

    assert set(state_dict) == set(module.state_dict())
    assert all(not tensor.requires_grad for tensor in state_dict.values())
    assert all(tensor.device.type == "cpu" for tensor in state_dict.values())


def test_run_fixed_update_times_uses_constant_update_count():
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap

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
            self.update_calls = 0
            self.update_counter = 0
            self.writer = type(
                "Writer",
                (),
                {"add_scalar": lambda *args, **kwargs: None},
            )()

        def update(self, *args):
            self.update_calls += 1
            self.update_counter += 1
            return (1.0, 0.5, 0.5)

    trainer = Trainer()
    buffer = Buffer()

    losses = pwap.run_fixed_diverse_updates(
        trainer=trainer,
        buffer_diverse=buffer,
        update_times=3,
        round_counter=8,
    )

    assert trainer.update_calls == 3
    assert buffer.sample_calls == 3
    assert losses == (1.0, 0.5, 0.5)


def test_buffer_writes_use_sorted_transition_payloads():
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap

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

    pwap.write_round_transitions_to_buffer(
        buffer,
        [
            pwap.WorkerRoundResult(
                df_index=1,
                epoch_index=0,
                context_index=0,
                initial_action=0,
                round_counter=0,
                worker_steps=1,
                transitions=[
                    pwap.WorkerTransitionRecord(
                        step_index=0,
                        transition=transition_b,
                    )
                ],
                rollout_metrics=[],
                done=False,
            ),
            pwap.WorkerRoundResult(
                df_index=0,
                epoch_index=0,
                context_index=0,
                initial_action=0,
                round_counter=0,
                worker_steps=1,
                transitions=[
                    pwap.WorkerTransitionRecord(
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
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap

    summary = pwap.summarize_parallel_round(
        round_counter=4,
        epoch_index=1,
        context_index=2,
        initial_action=3,
        round_results=[
            pwap.WorkerRoundResult(
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
            pwap.WorkerRoundResult(
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

    assert summary == pwap.ParallelRoundSummary(
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
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap

    assert pwap.build_epoch_model_path("/tmp/model", 2).endswith("epoch_3")
