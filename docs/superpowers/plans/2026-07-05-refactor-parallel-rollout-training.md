# Refactor Parallel Rollout Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `parallel_weight_advantage_pretrain.py` so diverse rollout exploration runs through df-bound worker processes while training updates remain serial in the main process.

**Architecture:** Keep full-df warmup unchanged. Add deterministic scheduling helpers, spawn-based df workers, Queue messages, and a main-process round loop that writes worker transitions into `buffer_diverse` before running fixed `update_times` updates. Epoch-level `epsilon / ada / lr` schedules are computed once per epoch and reused across all contexts and rounds in that epoch.

**Tech Stack:** Python, PyTorch, `torch.multiprocessing`, pytest, OpenSpec.

**Hard Constraints:**
- Do not modify `FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`.
- Do not modify `FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py`.
- Keep the serial training version unchanged; implement only the parallel training version.

**Traceability (sddflow):**
- plan-ready: `openspec/changes/refactor-parallel-rollout-training/plan-ready.md`
- tasks: `openspec/changes/refactor-parallel-rollout-training/tasks.md`
- plan: `docs/superpowers/plans/2026-07-05-refactor-parallel-rollout-training.md`

---

### Task 1: Parallel rollout scheduling

> **trace:** plan-ready.md -> `### Task 1: Parallel rollout scheduling` | tasks.md -> `- [ ] 1.0 Complete parallel rollout scheduling.`
> **sync:** tasks.md -> `- [ ] 1.0 Complete parallel rollout scheduling.` | plan-ready.md -> `### Task 1: Parallel rollout scheduling`

**Files:**
- Modify: `FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py`
- Test: `FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py`

- [x] **Step 1: Write scheduling helper tests**

Add `FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py`:

```python
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
```

- [x] **Step 2: Run scheduling tests and verify they fail**

Run: `conda activate finetf && pytest FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py -q`

Expected: FAIL with `AttributeError` for missing `build_effective_df_indices`, `iter_parallel_rollout_tasks`, or `compute_epoch_training_params`.

- [x] **Step 3: Add scheduling helpers and epoch CLI wiring**

In `FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py`, add `--num_epoch` near the existing `--num_sample` argument and keep `num_sample` as a compatibility fallback:

```python
parser.add_argument(
    "--num_epoch",
    type=int,
    default=None,
    help="number of parallel diverse-training epochs; one epoch explores every effective df once",
)
```

Add these module-level helpers before `class Weighted_Contexts_DQN`:

```python
def build_effective_df_indices(total_df_index_length):
    return list(range(total_df_index_length))


def iter_parallel_rollout_tasks(num_epoch, context_count, position_choices):
    for epoch_index in range(num_epoch):
        for context_index in range(context_count):
            for initial_action in range(position_choices):
                yield {
                    "epoch_index": epoch_index,
                    "context_index": context_index,
                    "initial_action": initial_action,
                }


def _linear_value(start, end, index, total_count):
    if total_count <= 1:
        return float(start)
    progress = min(max(index, 0), total_count - 1) / float(total_count - 1)
    return float(max(end, start - (start - end) * progress))


def _held_then_linear_value(start, end, epoch_index, num_epoch):
    if num_epoch <= 1:
        return float(start)
    hold_epochs = num_epoch // 2
    if epoch_index < hold_epochs:
        return float(start)
    decay_epochs = max(num_epoch - hold_epochs - 1, 1)
    decay_index = min(max(epoch_index - hold_epochs, 0), decay_epochs)
    return float(max(end, start - (start - end) * decay_index / float(decay_epochs)))


def compute_epoch_training_params(
    epoch_index,
    num_epoch,
    epsilon_init,
    epsilon_min,
    ada_init,
    ada_min,
    lr_init,
    lr_min,
):
    return {
        "epsilon": _linear_value(epsilon_init, epsilon_min, epoch_index, num_epoch),
        "ada": _held_then_linear_value(ada_init, ada_min, epoch_index, num_epoch),
        "lr": _held_then_linear_value(lr_init, lr_min, epoch_index, num_epoch),
    }
```

In `Weighted_Contexts_DQN.__init__`, after `self.num_sample = args.num_sample`, add:

```python
self.num_epoch = args.num_epoch if args.num_epoch is not None else args.num_sample
```

- [x] **Step 4: Run scheduling tests and verify they pass**

Run: `conda activate finetf && pytest FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py -q`

Expected: PASS for the scheduling tests.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 2: Worker process protocol

> **trace:** plan-ready.md -> `### Task 2: Worker process protocol` | tasks.md -> `- [ ] 2.0 Complete worker process protocol.`
> **sync:** tasks.md -> `- [ ] 2.0 Complete worker process protocol.` | plan-ready.md -> `### Task 2: Worker process protocol`

**Files:**
- Modify: `FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py`
- Test: `FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py`

- [x] **Step 1: Write worker protocol tests**

Append to `FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py`:

```python
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
```

- [x] **Step 2: Run worker protocol tests and verify they fail**

Run: `conda activate finetf && pytest FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py -q`

Expected: FAIL with missing worker protocol helper functions.

- [x] **Step 3: Add protocol helpers and worker entrypoint**

In `parallel_weight_advantage_pretrain.py`, add imports near the top:

```python
import traceback
import torch.multiprocessing as tmp
```

Add helpers near the scheduling helpers:

```python
def make_cpu_state_dict(module):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


def sort_round_transitions(round_results):
    ordered = []
    for result in sorted(round_results, key=lambda item: item["df_index"]):
        ordered.extend(
            item["transition"]
            for item in sorted(
                result.get("transitions", []),
                key=lambda transition: transition["step_index"],
            )
        )
    return ordered


def raise_for_worker_error(message):
    if message.get("type") != "worker_error":
        return
    raise RuntimeError(
        "worker_error df_index={df_index} epoch_index={epoch_index} "
        "context_index={context_index} initial_action={initial_action} "
        "round_counter={round_counter}: {traceback}".format(**message)
    )
```

Add a worker entrypoint that handles message routing and reports tracebacks:

```python
def df_rollout_worker(worker_config, input_queue, result_queue):
    df_index = worker_config["df_index"]
    try:
        runner = worker_config["runner_factory"](worker_config)
        while True:
            message = input_queue.get()
            message_type = message["type"]
            if message_type == "shutdown":
                return
            if message_type == "reset_task":
                runner.reset_task(message)
                continue
            if message_type == "explore_round":
                result = runner.explore_round(message)
                result_queue.put(result)
                continue
            raise ValueError("unknown worker message type: {}".format(message_type))
    except Exception:
        result_queue.put(
            {
                "type": "worker_error",
                "df_index": df_index,
                "epoch_index": locals().get("message", {}).get("epoch_index", -1),
                "context_index": locals().get("message", {}).get("context_index", -1),
                "initial_action": locals().get("message", {}).get("initial_action", -1),
                "round_counter": locals().get("message", {}).get("round_counter", -1),
                "traceback": traceback.format_exc(),
            }
        )
```

Add a runner class that keeps env state across rounds for one df:

```python
class DfRolloutWorkerRunner:
    def __init__(self, worker_config):
        self.df_index = worker_config["df_index"]
        self.train_df = worker_config["train_df"]
        self.env_kwargs = worker_config["env_kwargs"]
        self.model_factory = worker_config["model_factory"]
        self.device = worker_config["device"]
        self.leverage_choices = worker_config["leverage_choices"]
        self.position_list = worker_config["position_list"]
        self.initial_wallet_balance = worker_config["initial_wallet_balance"]
        self.initial_unrealized_pnL = worker_config["initial_unrealized_pnL"]
        self.model = self.model_factory().to(self.device)
        self.env = None
        self.state = None
        self.info = None
        self.done = True
        self.reward_sum = 0.0
        self.transition_count = 0

    def reset_task(self, message):
        _, _, _, initial_state = build_initial_state(
            self.train_df,
            message["initial_action"],
            self.leverage_choices,
            self.position_list,
            self.initial_wallet_balance,
            self.initial_unrealized_pnL,
        )
        self.env = create_demo_env(self.train_df, self.env_kwargs, initial_state)
        self.state, self.info = self.env.reset()
        self.done = False
        self.reward_sum = 0.0
        self.transition_count = 0

    def _act(self, state, info, context_index, epsilon):
        if np.random.uniform() <= epsilon:
            return np.random.choice(info["avaiable_action_list"])
        with torch.no_grad():
            state_tensor = torch.unsqueeze(torch.FloatTensor(state).reshape(-1), 0).to(self.device)
            previous_action = torch.unsqueeze(
                torch.tensor([info["previous_action"]]).float().to(self.device), 0
            )
            avaliable_action = torch.unsqueeze(
                torch.tensor(info["avaliable_action"]).to(self.device), 0
            )
            hour_count_down = torch.unsqueeze(
                torch.tensor([info["funding_count_down_hour"]]).float().to(self.device), 0
            )
            minute_count_down = torch.unsqueeze(
                torch.tensor([info["funding_count_down_minute"]]).float().to(self.device), 0
            )
            time_input = torch.cat([hour_count_down, minute_count_down], dim=1)
            q_values = self.model(
                state=state_tensor,
                time=time_input,
                previous_action=previous_action,
                avaliable_action=avaliable_action,
            )
            return int(torch.max(q_values[:, context_index, :], 1)[1].data.cpu().numpy()[0])

    def explore_round(self, message):
        self.model.load_state_dict(message["state_dict"])
        self.model.eval()
        transitions = []
        step_index = 0
        while not self.done and step_index < message["rollout_steps"]:
            action = self._act(
                self.state,
                self.info,
                message["context_index"],
                message["epsilon"],
            )
            next_state, reward, done, next_info = self.env.step(action)
            transitions.append(
                {
                    "step_index": self.transition_count,
                    "transition": (
                        self.state,
                        self.info,
                        action,
                        reward,
                        next_state,
                        next_info,
                        done,
                    ),
                }
            )
            self.reward_sum += reward
            self.transition_count += 1
            step_index += 1
            self.state, self.info, self.done = next_state, next_info, done
        final_balance = self.env.unrealized_pnl + self.env.wallet_balance
        return {
            "type": "round_result",
            "df_index": self.df_index,
            "epoch_index": message["epoch_index"],
            "context_index": message["context_index"],
            "initial_action": message["initial_action"],
            "round_counter": message["round_counter"],
            "worker_steps": len(transitions),
            "transitions": transitions,
            "rollout_metrics": [
                {
                    "epoch_index": message["epoch_index"],
                    "context_index": message["context_index"],
                    "initial_action": message["initial_action"],
                    "df_index": self.df_index,
                    "transition_count": self.transition_count,
                    "reward_sum": float(self.reward_sum),
                    "final_balance": float(final_balance),
                    "return_rate": float(final_balance / (self.initial_wallet_balance + 1e-12) - 1),
                }
            ],
            "done": self.done,
            "progress": {"transition_count": self.transition_count},
        }
```

- [x] **Step 4: Add minimal worker pool lifecycle helpers**

Add:

```python
def create_worker_context():
    return tmp.get_context("spawn")


def shutdown_workers(input_queues, processes):
    for queue in input_queues:
        queue.put({"type": "shutdown"})
    for process in processes:
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
```

- [x] **Step 5: Run worker protocol tests and verify they pass**

Run: `conda activate finetf && pytest FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py -q`

Expected: PASS for scheduling and protocol helper tests.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 3: Main-process serial training integration

> **trace:** plan-ready.md -> `### Task 3: Main-process serial training integration` | tasks.md -> `- [ ] 3.0 Complete main-process serial training integration.`
> **sync:** tasks.md -> `- [ ] 3.0 Complete main-process serial training integration.` | plan-ready.md -> `### Task 3: Main-process serial training integration`

**Files:**
- Modify: `FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py`
- Test: `FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py`

- [x] **Step 1: Write main-process integration helper tests**

Append to `FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py`:

```python
def test_run_fixed_update_times_uses_constant_update_count():
    from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap

    class Buffer:
        def __init__(self):
            self.sample_calls = 0

        def sample(self):
            self.sample_calls += 1
            return ("states", "infos", "actions", "rewards", "next_states", "next_infos", "dones")

    class Trainer:
        def __init__(self):
            self.update_calls = 0
            self.update_counter = 0
            self.writer = type("Writer", (), {"add_scalar": lambda *args, **kwargs: None})()

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
    transition_a = ("s0", {"previous_action": 0}, 1, 1.0, "s1", {"previous_action": 1}, False)
    transition_b = ("s2", {"previous_action": 1}, 2, 2.0, "s3", {"previous_action": 2}, True)

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
```

- [x] **Step 2: Run integration helper tests and verify they fail**

Run: `conda activate finetf && pytest FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py -q`

Expected: FAIL with missing `run_fixed_diverse_updates` and `write_round_transitions_to_buffer`.

- [x] **Step 3: Add main-process helper functions**

Add to `parallel_weight_advantage_pretrain.py`:

```python
def write_round_transitions_to_buffer(buffer_diverse, round_results):
    for transition in sort_round_transitions(round_results):
        buffer_diverse.add(*transition)


def run_fixed_diverse_updates(trainer, buffer_diverse, update_times, round_counter):
    last_losses = None
    for _ in range(update_times):
        (
            states,
            infos,
            actions,
            rewards,
            next_states,
            next_infos,
            dones,
        ) = buffer_diverse.sample()
        last_losses = trainer.update(
            states,
            infos,
            actions,
            rewards,
            next_states,
            next_infos,
            dones,
        )
        total_loss, KL_loss, td_loss = last_losses
        trainer.writer.add_scalar("total_loss", total_loss, round_counter)
        trainer.writer.add_scalar("KL_loss", KL_loss, round_counter)
        trainer.writer.add_scalar("td_loss", td_loss, round_counter)
    return last_losses
```

- [x] **Step 4: Replace diverse rollout loop with dispatch method**

Inside `Weighted_Contexts_DQN`, add a method that owns the new diverse-training path:

```python
def _run_parallel_diverse_training(
    self,
    train_df_cache,
    env_kwargs,
    buffer_diverse,
    step_counter_diverse,
):
    round_counter = 0
    for epoch_index in range(self.num_epoch):
        params = compute_epoch_training_params(
            epoch_index=epoch_index,
            num_epoch=self.num_epoch,
            epsilon_init=self.epsilon_init,
            epsilon_min=self.epsilon_min,
            ada_init=self.ada_init,
            ada_min=self.ada_min,
            lr_init=self.lr_init,
            lr_min=self.lr_min,
        )
        self.epsilon = params["epsilon"]
        self.ada = params["ada"]
        self.lr = params["lr"]
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr

        for context_index in range(self.N):
            for initial_action in range(self.position_choices):
                round_counter, step_counter_diverse = self._run_parallel_rollout_task(
                    epoch_index=epoch_index,
                    context_index=context_index,
                    initial_action=initial_action,
                    train_df_cache=train_df_cache,
                    env_kwargs=env_kwargs,
                    buffer_diverse=buffer_diverse,
                    step_counter_diverse=step_counter_diverse,
                    round_counter=round_counter,
                )
        self._save_parallel_epoch_model(epoch_index)
    return step_counter_diverse
```

In `train()`, keep qtable diagnostics and `_run_full_df_warmup(...)`, then replace the old `for sample in range(self.num_sample):` diverse-training block with:

```python
step_counter_diverse = self._run_parallel_diverse_training(
    train_df_cache=train_df_cache,
    env_kwargs=env_kwargs,
    buffer_diverse=buffer_diverse,
    step_counter_diverse=step_counter_diverse,
)
```

- [x] **Step 5: Add round dispatch method with deterministic writes**

Add this method to `Weighted_Contexts_DQN`:

```python
def _run_parallel_rollout_task(
    self,
    epoch_index,
    context_index,
    initial_action,
    train_df_cache,
    env_kwargs,
    buffer_diverse,
    step_counter_diverse,
    round_counter,
):
    active_df_indices = set(build_effective_df_indices(self.total_df_index_length))
    self._reset_worker_task(epoch_index, context_index, initial_action, active_df_indices)
    while active_df_indices:
        state_dict = make_cpu_state_dict(self.eval_net)
        self._send_worker_rounds(
            active_df_indices=active_df_indices,
            epoch_index=epoch_index,
            context_index=context_index,
            initial_action=initial_action,
            round_counter=round_counter,
            state_dict=state_dict,
        )
        round_results = self._collect_worker_rounds(active_df_indices, round_counter)
        for result in round_results:
            raise_for_worker_error(result)
        write_round_transitions_to_buffer(buffer_diverse, round_results)
        round_steps = sum(result["worker_steps"] for result in round_results)
        step_counter_diverse += round_steps
        if step_counter_diverse > (self.batch_size * self.update_times + self.n_step):
            run_fixed_diverse_updates(self, buffer_diverse, self.update_times, round_counter)
        active_df_indices = {
            result["df_index"]
            for result in round_results
            if not result.get("done", False)
        }
        round_counter += 1
    return round_counter, step_counter_diverse
```

Add the queue methods used by `_run_parallel_rollout_task`:

```python
def _reset_worker_task(self, epoch_index, context_index, initial_action, active_df_indices):
    for df_index in sorted(active_df_indices):
        self.worker_input_queues[df_index].put(
            {
                "type": "reset_task",
                "epoch_index": epoch_index,
                "context_index": context_index,
                "initial_action": initial_action,
            }
        )


def _send_worker_rounds(
    self,
    active_df_indices,
    epoch_index,
    context_index,
    initial_action,
    round_counter,
    state_dict,
):
    for df_index in sorted(active_df_indices):
        self.worker_input_queues[df_index].put(
            {
                "type": "explore_round",
                "epoch_index": epoch_index,
                "context_index": context_index,
                "initial_action": initial_action,
                "round_counter": round_counter,
                "state_dict": state_dict,
                "epsilon": self.epsilon,
                "rollout_steps": self.rollout_steps,
            }
        )


def _collect_worker_rounds(self, active_df_indices, round_counter):
    expected_count = len(active_df_indices)
    results = []
    while len(results) < expected_count:
        message = self.worker_result_queue.get()
        if message.get("round_counter") != round_counter:
            raise RuntimeError(
                "unexpected worker round_counter={} expected={}".format(
                    message.get("round_counter"),
                    round_counter,
                )
            )
        if message.get("df_index") not in active_df_indices:
            raise RuntimeError(
                "unexpected worker df_index={} active={}".format(
                    message.get("df_index"),
                    sorted(active_df_indices),
                )
            )
        results.append(message)
    return sorted(results, key=lambda result: result["df_index"])
```

- [x] **Step 6: Run integration helper tests**

Run: `conda activate finetf && pytest FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py -q`

Expected: PASS for scheduling, protocol, and main-process helper tests.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 4: Logging, saving, and verification

> **trace:** plan-ready.md -> `### Task 4: Logging, saving, and verification` | tasks.md -> `- [ ] 4.0 Complete logging, saving, and verification.`
> **sync:** tasks.md -> `- [ ] 4.0 Complete logging, saving, and verification.` | plan-ready.md -> `### Task 4: Logging, saving, and verification`

**Files:**
- Modify: `FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py`
- Test: `FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py`
- Verify: `openspec/changes/refactor-parallel-rollout-training/tasks.md`

- [x] **Step 1: Write logging and save helper tests**

Append to `FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py`:

```python
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
```

- [x] **Step 2: Run logging helper tests and verify they fail**

Run: `conda activate finetf && pytest FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py -q`

Expected: FAIL with missing `summarize_parallel_round` and `build_epoch_model_path`.

- [x] **Step 3: Add logging and epoch save helpers**

Add to `parallel_weight_advantage_pretrain.py`:

```python
def summarize_parallel_round(
    round_counter,
    epoch_index,
    context_index,
    initial_action,
    round_results,
    buffer_size,
    update_count,
):
    return {
        "round_counter": int(round_counter),
        "epoch_index": int(epoch_index),
        "context_index": int(context_index),
        "initial_action": int(initial_action),
        "round_steps": int(sum(result["worker_steps"] for result in round_results)),
        "active_worker_count": int(len(round_results)),
        "buffer_size": int(buffer_size),
        "update_count": int(update_count),
    }


def build_epoch_model_path(model_path, epoch_index):
    return os.path.join(model_path, "epoch_{}".format(epoch_index + 1))
```

Add `_save_parallel_epoch_model` to `Weighted_Contexts_DQN`:

```python
def _save_parallel_epoch_model(self, epoch_index):
    epoch_path = build_epoch_model_path(self.model_path, epoch_index)
    if not os.path.exists(epoch_path):
        os.makedirs(epoch_path)
    torch.save(
        self.eval_net.state_dict(),
        os.path.join(epoch_path, "trained_model.pkl"),
    )
    logger.info(
        "第 %d 轮 epoch 训练完成 | 模型已保存至=%s",
        epoch_index + 1,
        epoch_path,
    )
```

- [x] **Step 4: Log rollout and round summaries from dispatch**

In `_run_parallel_rollout_task`, after `round_results` are collected and updates have run, add:

```python
round_summary = summarize_parallel_round(
    round_counter=round_counter,
    epoch_index=epoch_index,
    context_index=context_index,
    initial_action=initial_action,
    round_results=round_results,
    buffer_size=len(buffer_diverse),
    update_count=self.update_times,
)
logger.info(
    "parallel rollout round complete | round_counter=%d | epoch_index=%d | "
    "context_index=%d | initial_action=%d | round_steps=%d | "
    "active_worker_count=%d | buffer_size=%d | update_count=%d",
    round_summary["round_counter"],
    round_summary["epoch_index"],
    round_summary["context_index"],
    round_summary["initial_action"],
    round_summary["round_steps"],
    round_summary["active_worker_count"],
    round_summary["buffer_size"],
    round_summary["update_count"],
)
```

For each worker `rollout_metrics` item, log:

```python
for result in round_results:
    for metrics in result.get("rollout_metrics", []):
        logger.info(
            "parallel rollout metrics | epoch_index=%d | context_index=%d | "
            "initial_action=%d | df_index=%d | transition_count=%d | "
            "reward_sum=%.4f | final_balance=%.4f | return_rate=%.6f",
            metrics["epoch_index"],
            metrics["context_index"],
            metrics["initial_action"],
            metrics["df_index"],
            metrics["transition_count"],
            metrics["reward_sum"],
            metrics["final_balance"],
            metrics["return_rate"],
        )
```

- [x] **Step 5: Run focused tests, py_compile, and OpenSpec validation**

Run:

```bash
conda activate finetf && pytest FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py -q
```

Expected: PASS.

Run:

```bash
conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py
```

Expected: exit code 0.

Run:

```bash
openspec validate refactor-parallel-rollout-training --strict
```

Expected: `Change 'refactor-parallel-rollout-training' is valid`.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）
