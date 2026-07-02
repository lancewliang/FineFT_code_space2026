# add-full-df-warmup-pretrain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a default-on full-df expert warmup stage before diverse training so every training `df_index` is pretrained once from an empty initial position.

**Architecture:** Keep qtable/df cache construction in `pretrain_qtable_diagnostics.py` and keep all network-updating training side effects in `weight_advantage_pretrain.py`. The trainer will extend existing diagnostics caches to cover all `df_index`, resolve the empty-position action from action semantics, run a full-df warmup loop before the sample loop, and then continue with the existing sample-level pretrain/diverse flow.

**Tech Stack:** Python, PyTorch, pandas, pytest, OpenSpec, existing FineFT RL utilities.

**Traceability (sddflow):**
- plan-ready: `openspec/changes/add-full-df-warmup-pretrain/plan-ready.md`
- tasks: `openspec/changes/add-full-df-warmup-pretrain/tasks.md`
- plan: `docs/superpowers/plans/2026-07-01-add-full-df-warmup-pretrain.md`

---

### Task 1: Full-df warmup implementation

> **trace:** plan-ready.md → `### Task 1: Full-df warmup implementation` | tasks.md → `- [ ] 1.0 Complete full-df warmup implementation.`
> **sync:** tasks.md → `- [ ] 1.0 Complete full-df warmup implementation.` | plan-ready.md → `### Task 1: Full-df warmup implementation`

**Files:**
- Modify: `FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py`
- Modify: `FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`
- Test: `FineFT/tests/rl/test_pretrain_qtable_diagnostics.py`
- Test: `FineFT/tests/rl/test_weight_advantage_pretrain_logging.py`

- [x] **Step 1: Add failing tests for cache extension**

Append this test to `FineFT/tests/rl/test_pretrain_qtable_diagnostics.py`:

```python
def test_extend_q_table_cache_only_computes_missing_df_indices(monkeypatch, tmp_path):
    from RL.DiHFT.low_level import pretrain_qtable_diagnostics as diag

    train_data_path = tmp_path / "train"
    train_data_path.mkdir()
    pd.DataFrame({"mark_price": [10.0]}).to_feather(train_data_path / "df_0.feather")
    pd.DataFrame({"mark_price": [20.0]}).to_feather(train_data_path / "df_1.feather")
    pd.DataFrame({"mark_price": [30.0]}).to_feather(train_data_path / "df_2.feather")

    existing_q_table = np.ones((1, 3, 3))
    existing_df = pd.DataFrame({"mark_price": [10.0]})
    computed = []

    def fake_create_q_table_from_df(df, **kwargs):
        computed.append(float(df["mark_price"].iloc[0]))
        return np.zeros((len(df), 3, 3)) + float(df["mark_price"].iloc[0])

    monkeypatch.setattr(
        diag, "create_optimal_q_table_from_df", fake_create_q_table_from_df
    )

    q_table_cache, train_df_cache = diag.extend_q_table_cache(
        df_indices=[0, 1, 2],
        train_data_path=str(train_data_path),
        qtable_kwargs={},
        q_table_cache={0: existing_q_table},
        train_df_cache={0: existing_df},
        process_count=1,
    )

    assert q_table_cache[0] is existing_q_table
    assert train_df_cache[0] is existing_df
    assert sorted(q_table_cache) == [0, 1, 2]
    assert sorted(train_df_cache) == [0, 1, 2]
    assert computed == [20.0, 30.0]
```

- [x] **Step 2: Run cache extension test and verify it fails**

Run:

```bash
conda activate finetf && pytest FineFT/tests/rl/test_pretrain_qtable_diagnostics.py::test_extend_q_table_cache_only_computes_missing_df_indices -q
```

Expected: FAIL with `AttributeError: module 'RL.DiHFT.low_level.pretrain_qtable_diagnostics' has no attribute 'extend_q_table_cache'`.

- [x] **Step 3: Implement cache extension helper**

In `FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py`, add this function after `build_q_table_cache(...)`:

```python
def extend_q_table_cache(
    df_indices,
    train_data_path,
    qtable_kwargs,
    q_table_cache=None,
    train_df_cache=None,
    process_count=None,
):
    q_table_cache = dict(q_table_cache or {})
    train_df_cache = dict(train_df_cache or {})
    missing_df_indices = [
        df_index
        for df_index in sorted(set(df_indices))
        if df_index not in q_table_cache or df_index not in train_df_cache
    ]
    if not missing_df_indices:
        return q_table_cache, train_df_cache

    missing_plan = [(df_index, 0) for df_index in missing_df_indices]
    missing_q_table_cache, missing_train_df_cache = build_q_table_cache(
        missing_plan,
        train_data_path,
        qtable_kwargs,
        process_count=process_count,
    )
    q_table_cache.update(missing_q_table_cache)
    train_df_cache.update(missing_train_df_cache)
    return q_table_cache, train_df_cache
```

- [x] **Step 4: Run cache extension test and verify it passes**

Run:

```bash
conda activate finetf && pytest FineFT/tests/rl/test_pretrain_qtable_diagnostics.py::test_extend_q_table_cache_only_computes_missing_df_indices -q
```

Expected: PASS.

- [x] **Step 5: Add failing tests for CLI defaults and empty action resolution**

Append these tests to `FineFT/tests/rl/test_weight_advantage_pretrain_logging.py`:

```python
def test_parser_defaults_enable_full_df_warmup_and_zero_sample_pretrain():
    from RL.DiHFT.low_level import weight_advantage_pretrain as wap

    args = wap.parser.parse_args([])

    assert args.full_df_warmup is True
    assert args.pretrain_epoch == 0


def test_parser_can_disable_full_df_warmup():
    from RL.DiHFT.low_level import weight_advantage_pretrain as wap

    args = wap.parser.parse_args(["--no_full_df_warmup"])

    assert args.full_df_warmup is False


def test_resolve_empty_initial_action_uses_action_mapping():
    trainer = Weighted_Contexts_DQN.__new__(Weighted_Contexts_DQN)
    trainer.position_choices = 5
    trainer.leverage_choices = [1, 2]
    trainer.position_list = [-8.0, -4.0, 0, 4.0, 8.0]

    assert trainer._resolve_empty_initial_action() == 4


def test_resolve_empty_initial_action_raises_when_no_empty_position():
    trainer = Weighted_Contexts_DQN.__new__(Weighted_Contexts_DQN)
    trainer.position_choices = 2
    trainer.leverage_choices = [1]
    trainer.position_list = [-8.0, 8.0]

    try:
        trainer._resolve_empty_initial_action()
    except ValueError as exc:
        assert "empty position action" in str(exc)
    else:
        raise AssertionError("expected ValueError")
```

- [x] **Step 6: Run CLI and empty action tests and verify they fail**

Run:

```bash
conda activate finetf && pytest FineFT/tests/rl/test_weight_advantage_pretrain_logging.py::test_parser_defaults_enable_full_df_warmup_and_zero_sample_pretrain FineFT/tests/rl/test_weight_advantage_pretrain_logging.py::test_parser_can_disable_full_df_warmup FineFT/tests/rl/test_weight_advantage_pretrain_logging.py::test_resolve_empty_initial_action_uses_action_mapping FineFT/tests/rl/test_weight_advantage_pretrain_logging.py::test_resolve_empty_initial_action_raises_when_no_empty_position -q
```

Expected: FAIL because `full_df_warmup` and `_resolve_empty_initial_action()` do not exist yet, and `pretrain_epoch` still defaults to `2`.

- [x] **Step 7: Add CLI switches and empty action resolver**

In `FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`, update imports near the existing futures util import:

```python
from env.env_class.futures_util import (
    get_dp_action_from_qtable,
    map_action_to_position_leverage,
)
```

Update the pretrain parser section:

```python
parser.add_argument(
    "--pretrain_epoch",
    type=int,
    default=0,
    help="the number of sample-level pretrain epochs after full df warmup",
)
parser.add_argument(
    "--full_df_warmup",
    dest="full_df_warmup",
    action="store_true",
    default=True,
    help="run one empty-position pretrain warmup for every training df before sample loop",
)
parser.add_argument(
    "--no_full_df_warmup",
    dest="full_df_warmup",
    action="store_false",
    help="disable full df warmup before sample loop",
)
```

In `Weighted_Contexts_DQN.__init__`, after `self.pretrain_epoch = args.pretrain_epoch`, add:

```python
self.full_df_warmup = args.full_df_warmup
```

Add this method near `_set_initial_state_from_action(...)`:

```python
def _resolve_empty_initial_action(self):
    for action in range(self.N_ACTIONS):
        position, _ = map_action_to_position_leverage(
            action,
            self.leverage_choices,
            self.position_list,
        )
        if position == 0:
            return action
    raise ValueError(
        "Unable to resolve empty position action from position_list={} and "
        "leverage_choices={}".format(self.position_list, self.leverage_choices)
    )
```

- [x] **Step 8: Run CLI and empty action tests and verify they pass**

Run:

```bash
conda activate finetf && pytest FineFT/tests/rl/test_weight_advantage_pretrain_logging.py::test_parser_defaults_enable_full_df_warmup_and_zero_sample_pretrain FineFT/tests/rl/test_weight_advantage_pretrain_logging.py::test_parser_can_disable_full_df_warmup FineFT/tests/rl/test_weight_advantage_pretrain_logging.py::test_resolve_empty_initial_action_uses_action_mapping FineFT/tests/rl/test_weight_advantage_pretrain_logging.py::test_resolve_empty_initial_action_raises_when_no_empty_position -q
```

Expected: PASS.

- [x] **Step 9: Add failing trainer test for one full-df warmup per df**

Append this test to `FineFT/tests/rl/test_weight_advantage_pretrain_logging.py`:

```python
def test_run_full_df_warmup_updates_once_per_df(monkeypatch):
    import pandas as pd
    from RL.DiHFT.low_level import weight_advantage_pretrain as wap

    class TinyBuffer:
        def __init__(self):
            self.added = []

        def add(self, s, info, a, r, s_, info_, done):
            self.added.append((a, r, done))

        def sample(self):
            raise AssertionError("sample should not be called with large batch threshold")

    class TinyEnv:
        def __init__(self):
            self.wallet_balance = 100000.0
            self.unrealized_pnl = 0.0
            self._step = 0

        def reset(self):
            self._step = 0
            return "s0", {
                "avaiable_action_list": [0, 1, 2],
                "previous_action": 0,
                "personal_state": [0, 0, 0, 0, 0.0],
            }

        def step(self, action):
            self._step += 1
            done = self._step == 1
            return "s1", 1.0, done, {
                "avaiable_action_list": [0, 1, 2],
                "previous_action": action,
                "personal_state": [0, 0, 0, 0, 0.0],
            }

    trainer = Weighted_Contexts_DQN.__new__(Weighted_Contexts_DQN)
    trainer.full_df_warmup = True
    trainer.total_df_index_length = 3
    trainer.position_choices = 3
    trainer.leverage_choices = [1]
    trainer.position_list = [-1.0, 0, 1.0]
    trainer.N_ACTIONS = 3
    trainer.initial_wallet_balance = 100000.0
    trainer.initial_unrealized_pnL = 0.0
    trainer.batch_size = 999
    trainer.update_times = 1
    trainer.n_step = 1
    trainer.rollout_steps = 1024
    trainer.writer = type("Writer", (), {"add_scalar": lambda *args, **kwargs: None})()
    trainer.update_counter = 0
    trainer._set_initial_state_from_action = lambda train_df, action: setattr(
        trainer, "initial_state", ("state", int(action))
    )
    trainer.act_multi_styles_pretrain = (
        lambda info, optimal_step_counter, rollout_index: rollout_index
    )
    trainer.update_pretrain = lambda *args, **kwargs: (0.0, 0.0, 0.0)

    created_envs = []

    def fake_create_demo_env(train_df, env_kwargs, initial_state):
        created_envs.append((float(train_df["mark_price"].iloc[0]), initial_state))
        return TinyEnv()

    monkeypatch.setattr(wap, "create_demo_env", fake_create_demo_env)
    monkeypatch.setattr(
        wap, "get_dp_action_from_qtable", lambda q_table, initial_action: [initial_action]
    )

    train_df_cache = {
        0: pd.DataFrame({"mark_price": [10.0]}),
        1: pd.DataFrame({"mark_price": [20.0]}),
        2: pd.DataFrame({"mark_price": [30.0]}),
    }
    q_table_cache = {0: "q0", 1: "q1", 2: "q2"}
    buffer_pretrain = TinyBuffer()

    summary, step_counter = trainer._run_full_df_warmup(
        q_table_cache=q_table_cache,
        train_df_cache=train_df_cache,
        env_kwargs={},
        buffer_pretrain=buffer_pretrain,
        step_counter_pretrain=0,
    )

    assert [item[0] for item in created_envs] == [10.0, 20.0, 30.0]
    assert [item[1][1] for item in created_envs] == [1, 1, 1]
    assert step_counter == 12
    assert len(buffer_pretrain.added) == 12
    assert summary["df_count"] == 3
```

- [x] **Step 10: Run full-df warmup trainer test and verify it fails**

Run:

```bash
conda activate finetf && pytest FineFT/tests/rl/test_weight_advantage_pretrain_logging.py::test_run_full_df_warmup_updates_once_per_df -q
```

Expected: FAIL with `AttributeError: 'Weighted_Contexts_DQN' object has no attribute '_run_full_df_warmup'`.

- [x] **Step 11: Import cache extension and add full-df warmup helper methods**

In `FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`, update the diagnostics import:

```python
from RL.DiHFT.low_level.pretrain_qtable_diagnostics import (
    build_initial_state,
    create_demo_env,
    extend_q_table_cache,
    prepare_pretrain_qtable_diagnostics,
)
```

Add these methods to `Weighted_Contexts_DQN` after `_set_initial_state_from_action(...)`:

```python
def _write_pretrain_loss_scalars(self, total_loss, KL_loss, td_loss):
    self.writer.add_scalar(
        tag="total_loss",
        scalar_value=total_loss,
        global_step=self.update_counter,
        walltime=None,
    )
    self.writer.add_scalar(
        tag="KL_loss",
        scalar_value=KL_loss,
        global_step=self.update_counter,
        walltime=None,
    )
    self.writer.add_scalar(
        tag="td_loss",
        scalar_value=td_loss,
        global_step=self.update_counter,
        walltime=None,
    )

def _run_pretrain_updates_if_ready(self, buffer_pretrain, step_counter_pretrain):
    if not (
        step_counter_pretrain > (self.batch_size * self.update_times + self.n_step)
        and step_counter_pretrain % self.rollout_steps == 1
    ):
        return None
    last_losses = None
    for _ in range(self.update_times):
        (
            states,
            infos,
            actions,
            rewards,
            next_states,
            next_infos,
            dones,
        ) = buffer_pretrain.sample()
        last_losses = self.update_pretrain(
            states,
            infos,
            actions,
            rewards,
            next_states,
            next_infos,
            dones,
        )
        self._write_pretrain_loss_scalars(*last_losses)
    return last_losses

def _run_full_df_warmup(
    self,
    q_table_cache,
    train_df_cache,
    env_kwargs,
    buffer_pretrain,
    step_counter_pretrain,
):
    if not self.full_df_warmup:
        logger.info("full-df warmup disabled")
        return {"df_count": 0, "reward_sum": 0.0, "update_count": 0}, step_counter_pretrain
    if self.total_df_index_length <= 0:
        raise ValueError("full-df warmup requires total_df_index_length > 0")

    empty_initial_action = self._resolve_empty_initial_action()
    logger.info(
        "full-df warmup start | df_count=%d | empty_initial_action=%d",
        self.total_df_index_length,
        empty_initial_action,
    )
    total_reward_sum = 0.0
    update_count_before = self.update_counter

    for df_index in range(self.total_df_index_length):
        train_df = train_df_cache[df_index]
        q_table = q_table_cache[df_index]
        self._set_initial_state_from_action(train_df, empty_initial_action)
        env = create_demo_env(train_df, env_kwargs, self.initial_state)
        self.perfection_action_list = get_dp_action_from_qtable(
            q_table,
            empty_initial_action,
        )
        df_reward_sum = 0.0

        for rollout_index in range(4):
            s, info = env.reset()
            optimal_step_counter = 0
            rollout_reward_sum = 0.0
            while True:
                a = self.act_multi_styles_pretrain(
                    info,
                    optimal_step_counter,
                    rollout_index,
                )
                optimal_step_counter += 1
                s_, r, done, info_ = env.step(a)
                step_counter_pretrain += 1
                buffer_pretrain.add(s, info, a, r, s_, info_, done)
                rollout_reward_sum += r
                s, info = s_, info_
                if done:
                    break
                losses = self._run_pretrain_updates_if_ready(
                    buffer_pretrain,
                    step_counter_pretrain,
                )
                if losses is not None:
                    logger.info(
                        "full-df warmup update | df_index=%d | step=%d | "
                        "total_loss=%.6f | KL_loss=%.6f | td_loss=%.6f",
                        df_index,
                        step_counter_pretrain,
                        losses[0],
                        losses[1],
                        losses[2],
                    )
            df_reward_sum += rollout_reward_sum

        final_balance = env.unrealized_pnl + env.wallet_balance
        return_rate = final_balance / self.initial_wallet_balance
        if df_reward_sum <= 0:
            logger.warning(
                "full-df warmup unprofitable | df_index=%d | reward_sum=%.4f | "
                "final_balance=%.4f | return_rate=%.6f",
                df_index,
                df_reward_sum,
                final_balance,
                return_rate,
            )
        else:
            logger.info(
                "full-df warmup df complete | df_index=%d | reward_sum=%.4f | "
                "final_balance=%.4f | return_rate=%.6f",
                df_index,
                df_reward_sum,
                final_balance,
                return_rate,
            )
        total_reward_sum += df_reward_sum

    update_count = self.update_counter - update_count_before
    logger.info(
        "full-df warmup complete | df_count=%d | reward_sum=%.4f | update_count=%d",
        self.total_df_index_length,
        total_reward_sum,
        update_count,
    )
    return {
        "df_count": self.total_df_index_length,
        "reward_sum": total_reward_sum,
        "update_count": update_count,
    }, step_counter_pretrain
```

- [x] **Step 12: Run full-df warmup trainer test and verify it passes**

Run:

```bash
conda activate finetf && pytest FineFT/tests/rl/test_weight_advantage_pretrain_logging.py::test_run_full_df_warmup_updates_once_per_df -q
```

Expected: PASS.

- [x] **Step 13: Wire full-df warmup into `train()` before the sample loop**

In `FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`, immediately after the existing `prepare_pretrain_qtable_diagnostics(...)` call and before `for sample in range(self.num_sample):`, add:

```python
        if self.full_df_warmup:
            q_table_cache, train_df_cache = extend_q_table_cache(
                df_indices=range(self.total_df_index_length),
                train_data_path=self.train_data_path,
                qtable_kwargs=qtable_kwargs,
                q_table_cache=q_table_cache,
                train_df_cache=train_df_cache,
            )
        _, step_counter_pretrain = self._run_full_df_warmup(
            q_table_cache=q_table_cache,
            train_df_cache=train_df_cache,
            env_kwargs=env_kwargs,
            buffer_pretrain=buffer_pretrain,
            step_counter_pretrain=step_counter_pretrain,
        )
```

Replace the three repeated `self.writer.add_scalar(...)` calls inside the existing sample-level pretrain update block with:

```python
                                self._write_pretrain_loss_scalars(
                                    total_loss,
                                    KL_loss,
                                    td_loss,
                                )
```

This keeps the existing sample-level behavior but shares the scalar writer with full-df warmup.

- [x] **Step 14: Run the focused RL tests**

Run:

```bash
conda activate finetf && pytest FineFT/tests/rl/test_pretrain_qtable_diagnostics.py FineFT/tests/rl/test_weight_advantage_pretrain_logging.py -q
```

Expected: PASS.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 2: Verification

> **trace:** plan-ready.md → `### Task 2: Verification` | tasks.md → `- [ ] 2.0 Complete verification.`
> **sync:** tasks.md → `- [ ] 2.0 Complete verification.` | plan-ready.md → `### Task 2: Verification`

**Files:**
- Modify: `openspec/changes/add-full-df-warmup-pretrain/tasks.md`
- Modify: `openspec/changes/add-full-df-warmup-pretrain/plan-ready.md`
- Modify: `docs/superpowers/plans/2026-07-01-add-full-df-warmup-pretrain.md`

- [x] **Step 1: Run focused tests**

Run:

```bash
conda activate finetf && pytest FineFT/tests/rl/test_pretrain_qtable_diagnostics.py FineFT/tests/rl/test_weight_advantage_pretrain_logging.py -q
```

Expected: PASS.

- [x] **Step 2: Run Python syntax check**

Run:

```bash
conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py
```

Expected: PASS with no output.

- [x] **Step 3: Run OpenSpec strict validation**

Run:

```bash
openspec validate add-full-df-warmup-pretrain --strict
```

Expected: `Change 'add-full-df-warmup-pretrain' is valid`.

- [x] **Step 4: Inspect implementation evidence**

Run:

```bash
rg -n "full_df_warmup|no_full_df_warmup|_resolve_empty_initial_action|_run_full_df_warmup|extend_q_table_cache|pretrain_epoch\".*default=0|update_pretrain\\(" FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py FineFT/tests/rl/test_weight_advantage_pretrain_logging.py FineFT/tests/rl/test_pretrain_qtable_diagnostics.py
```

Expected: output shows parser switches and default `pretrain_epoch=0`; `_resolve_empty_initial_action` and `_run_full_df_warmup` in `weight_advantage_pretrain.py`; `extend_q_table_cache` in `pretrain_qtable_diagnostics.py`; focused tests for these behaviors.

- [x] **Step 5: Mark verification task complete after all checks pass**

In `openspec/changes/add-full-df-warmup-pretrain/tasks.md`, update:

```markdown
- [x] 2.0 Complete verification.
- [x] 2.1 Add focused tests for defaults, disable switch, empty-position action resolution, cache reuse, and one-warmup-per-df behavior.
- [x] 2.2 Run `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py`.
- [x] 2.3 Run `openspec validate add-full-df-warmup-pretrain --strict`.
```

to:

```markdown
- [x] 2.0 Complete verification.
- [x] 2.1 Add focused tests for defaults, disable switch, empty-position action resolution, cache reuse, and one-warmup-per-df behavior.
- [x] 2.2 Run `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py`.
- [x] 2.3 Run `openspec validate add-full-df-warmup-pretrain --strict`.
```

Also mark Task 2 complete in this file and `openspec/changes/add-full-df-warmup-pretrain/plan-ready.md` only after Steps 1-4 pass.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）
