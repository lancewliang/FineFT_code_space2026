import sys
from pathlib import Path


FINEFT_ROOT = Path(__file__).resolve().parents[2]
if str(FINEFT_ROOT) not in sys.path:
    sys.path.insert(0, str(FINEFT_ROOT))


def test_effective_df_indices_use_total_df_index_length_without_extra_drop():
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap

    assert pwap.build_effective_df_indices(3) == [0, 1, 2]


def test_parallel_rollout_task_order_is_epoch_context_initial_action():
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap

    tasks = list(
        pwap.iter_parallel_rollout_tasks(
            num_epoch=2,
            context_count=2,
            position_choices=3,
        )
    )

    assert tasks == [
        {"epoch_index": 0, "context_index": 0, "initial_action": 0},
        {"epoch_index": 0, "context_index": 0, "initial_action": 1},
        {"epoch_index": 0, "context_index": 0, "initial_action": 2},
        {"epoch_index": 0, "context_index": 1, "initial_action": 0},
        {"epoch_index": 0, "context_index": 1, "initial_action": 1},
        {"epoch_index": 0, "context_index": 1, "initial_action": 2},
        {"epoch_index": 1, "context_index": 0, "initial_action": 0},
        {"epoch_index": 1, "context_index": 0, "initial_action": 1},
        {"epoch_index": 1, "context_index": 0, "initial_action": 2},
        {"epoch_index": 1, "context_index": 1, "initial_action": 0},
        {"epoch_index": 1, "context_index": 1, "initial_action": 1},
        {"epoch_index": 1, "context_index": 1, "initial_action": 2},
    ]


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

    assert first == {"epsilon": 1.0, "ada": 256.0, "lr": 0.005}
    assert middle == {"epsilon": 0.6, "ada": 256.0, "lr": 0.005}
    assert last == {"epsilon": 0.2, "ada": 0.0, "lr": 0.001}


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
    ) == {"epsilon": 1.0, "ada": 256.0, "lr": 0.005}


def test_sort_round_results_orders_by_df_index_and_step_index():
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap

    results = [
        {
            "type": "round_result",
            "df_index": 2,
            "worker_steps": 1,
            "transitions": [{"step_index": 1, "transition": "df2-step1"}],
            "rollout_metrics": [],
            "done": False,
        },
        {
            "type": "round_result",
            "df_index": 1,
            "worker_steps": 2,
            "transitions": [
                {"step_index": 1, "transition": "df1-step1"},
                {"step_index": 0, "transition": "df1-step0"},
            ],
            "rollout_metrics": [],
            "done": True,
        },
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
            {
                "type": "worker_error",
                "df_index": 3,
                "epoch_index": 0,
                "context_index": 1,
                "initial_action": 2,
                "round_counter": 4,
                "traceback": "boom",
            }
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
            {
                "type": "round_result",
                "df_index": 1,
                "worker_steps": 1,
                "transitions": [{"step_index": 0, "transition": transition_b}],
            },
            {
                "type": "round_result",
                "df_index": 0,
                "worker_steps": 1,
                "transitions": [{"step_index": 0, "transition": transition_a}],
            },
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
            {"df_index": 0, "worker_steps": 5, "done": False},
            {"df_index": 1, "worker_steps": 4, "done": True},
        ],
        buffer_size=99,
        update_count=20,
    )

    assert summary == {
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
