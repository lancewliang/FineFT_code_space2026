# Pretrain Qtable Profit Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Precompute qtables for the actual `weight_advantage_pretrain.py` sample plan with multiprocessing, independently validate every sample, and export per-sample DP-path CSV diagnostics before training starts.

**Architecture:** Move qtable diagnostic logic into `FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py`. The module builds the sample plan, computes unique `df_index` qtables in worker processes, then performs stable sample-ordered env replay and CSV export in the main process; `weight_advantage_pretrain.py` consumes the returned plan, qtable cache, and df cache.

**Tech Stack:** Python 3.10, multiprocessing, pandas, NumPy, existing FineFT demo environment utilities, existing qtable utilities, OpenSpec.

**Traceability (sddflow):**
- plan-ready: `openspec/changes/add-pretrain-qtable-profit-check/plan-ready.md`
- tasks: `openspec/changes/add-pretrain-qtable-profit-check/tasks.md`
- plan: `docs/superpowers/plans/2026-06-29-add-pretrain-qtable-profit-check.md`

---

## File Structure

- `FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py`: new independent diagnostics module. It owns sample plan generation, multiprocessing qtable calculation, qtable/df cache construction, DP path replay, sample-level logging, and per-sample CSV export.
- `FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`: training script. It imports the diagnostics module, calls it before `for sample in range(self.num_sample)`, consumes `sample_plan`, `q_table_cache`, and `train_df_cache`, and reuses cached qtables in pretrain.
- `openspec/changes/add-pretrain-qtable-profit-check/tasks.md`: build stage will mark task checkboxes when complete.
- `openspec/changes/add-pretrain-qtable-profit-check/plan-ready.md`: build stage will mark task checkboxes when complete.

### Task 1: Sample plan and qtable cache implementation

> **trace:** plan-ready.md -> `### Task 1: Sample plan and qtable cache implementation` | tasks.md -> `- [ ] 1.0 Complete sample plan, qtable cache, multiprocessing, and CSV diagnostics implementation.`
> **sync:** tasks.md -> `- [ ] 1.0 Complete sample plan, qtable cache, multiprocessing, and CSV diagnostics implementation.` | plan-ready.md -> `### Task 1: Sample plan and qtable cache implementation`

**Files:**
- Create: `FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py`
- Modify: `FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`

- [x] **Step 1: Create the diagnostics module imports and sample plan helper**

Create `FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py` with this content:

```python
import multiprocessing as mp
import os
import random

import numpy as np
import pandas as pd

from env.env_class.futures_util import (
    create_optimal_q_table_from_df,
    get_dp_action_from_qtable,
    map_action_to_position_leverage,
)
from env.env_initiate.demo_initiate import initiate_demo_env


def build_sample_plan(num_sample, total_df_index_length, position_choices):
    sample_plan = []
    for _ in range(num_sample):
        df_index = random.choices(range(total_df_index_length), k=1)[0]
        initial_action = random.choices(range(position_choices), k=1)[0]
        sample_plan.append((df_index, initial_action))
    return sample_plan
```

- [x] **Step 2: Add qtable worker and multiprocessing cache builder**

Append this code to `pretrain_qtable_diagnostics.py`:

```python
def _create_q_table_worker(args):
    df_index, train_data_path, qtable_kwargs = args
    df_path = os.path.join(train_data_path, "df_{}.feather".format(df_index))
    train_df = pd.read_feather(df_path)
    q_table = create_optimal_q_table_from_df(df=train_df, **qtable_kwargs)
    return df_index, train_df, q_table


def build_q_table_cache(sample_plan, train_data_path, qtable_kwargs, process_count=None):
    unique_df_indices = sorted({df_index for df_index, _ in sample_plan})
    if process_count is None:
        process_count = min(len(unique_df_indices), os.cpu_count() or 1)
    process_count = max(1, process_count)

    worker_args = [
        (df_index, train_data_path, qtable_kwargs)
        for df_index in unique_df_indices
    ]
    if process_count == 1:
        results = [_create_q_table_worker(args) for args in worker_args]
    else:
        with mp.Pool(processes=process_count) as pool:
            results = pool.map(_create_q_table_worker, worker_args)

    train_df_cache = {}
    q_table_cache = {}
    for df_index, train_df, q_table in results:
        train_df_cache[df_index] = train_df
        q_table_cache[df_index] = q_table
    return q_table_cache, train_df_cache
```

- [x] **Step 3: Add initial-state and env helper functions**

Append this code to `pretrain_qtable_diagnostics.py`:

```python
def build_initial_state(
    train_df,
    initial_action,
    leverage_choices,
    position_list,
    initial_wallet_balance,
    initial_unrealized_pnl,
):
    initial_position, initial_leverage = map_action_to_position_leverage(
        initial_action, leverage_choices, position_list
    )
    current_markprice = train_df["mark_price"].values[0]
    initial_margin = np.abs(initial_position * current_markprice / initial_leverage)
    initial_state = (
        initial_wallet_balance,
        initial_margin,
        initial_unrealized_pnl,
        initial_position,
        initial_leverage,
    )
    return initial_position, initial_leverage, initial_margin, initial_state


def create_demo_env(train_df, env_kwargs, initial_state):
    return initiate_demo_env(
        df=train_df,
        feature_list=env_kwargs["feature_list"],
        max_holding_number=env_kwargs["max_holding_number"],
        order_book_depth=env_kwargs["order_book_depth"],
        position_choices=env_kwargs["position_choices"],
        leverage_choice=env_kwargs["leverage_choices"],
        long_estimated_rate=env_kwargs["long_estimated_rate"],
        short_estimated_rate=env_kwargs["short_estimated_rate"],
        commission_rate=env_kwargs["commission_rate"],
        maintenance_margin_ratio_dict=env_kwargs["maintenance_margin_ratio_dict"],
        early_stop=env_kwargs["early_stop"],
        initial_state=initial_state,
        gamma=env_kwargs["gamma"],
        max_punishment=1e10,
    )
```

- [x] **Step 4: Add CSV row helpers and DP replay**

Append this code to `pretrain_qtable_diagnostics.py`:

```python
def _value_from_row(row, column):
    return row[column] if column in row.index else np.nan


def _diagnostic_row(
    train_df,
    env,
    sample_index,
    df_index,
    initial_action,
    step_index,
    action,
    previous_action,
    reward,
    cumulative_profit,
    previous_slippage_sum,
):
    source_row = train_df.iloc[min(step_index, len(train_df) - 1)]
    step_slippage = env.slippage_sum - previous_slippage_sum
    return {
        "sample_index": sample_index + 1,
        "df_index": df_index,
        "initial_action": initial_action,
        "step_index": step_index,
        "timestamp": _value_from_row(source_row, "timestamp"),
        "open": _value_from_row(source_row, "open"),
        "high": _value_from_row(source_row, "high"),
        "low": _value_from_row(source_row, "low"),
        "close": _value_from_row(source_row, "close"),
        "volume": _value_from_row(source_row, "volume"),
        "mark_price": _value_from_row(source_row, "mark_price"),
        "action": action,
        "previous_action": previous_action,
        "position": env.position,
        "leverage": env.leverage,
        "commission_rate": env.commission_rate,
        "step_slippage": step_slippage,
        "step_reward": reward,
        "cumulative_profit": cumulative_profit,
        "profitable": cumulative_profit > 0,
    }


def evaluate_and_export_sample(
    sample_index,
    df_index,
    initial_action,
    train_df,
    q_table,
    env_kwargs,
    output_dir,
):
    _, _, _, initial_state = build_initial_state(
        train_df,
        initial_action,
        env_kwargs["leverage_choices"],
        env_kwargs["position_list"],
        env_kwargs["initial_wallet_balance"],
        env_kwargs["initial_unrealized_pnl"],
    )
    env = create_demo_env(train_df, env_kwargs, initial_state)
    action_list = get_dp_action_from_qtable(q_table, initial_action)
    _, info = env.reset()
    cumulative_profit = 0
    previous_slippage_sum = env.slippage_sum
    rows = []

    for step_index, action in enumerate(action_list):
        previous_action = info["previous_action"]
        _, reward, done, info = env.step(action)
        cumulative_profit += reward
        rows.append(
            _diagnostic_row(
                train_df,
                env,
                sample_index,
                df_index,
                initial_action,
                step_index,
                action,
                previous_action,
                reward,
                cumulative_profit,
                previous_slippage_sum,
            )
        )
        previous_slippage_sum = env.slippage_sum
        if done:
            break

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(
        output_dir,
        "sample_{:04d}_df_{}_initial_action_{}.csv".format(
            sample_index + 1, df_index, initial_action
        ),
    )
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return {
        "sample_index": sample_index + 1,
        "df_index": df_index,
        "initial_action": initial_action,
        "episode_reward_sum": cumulative_profit,
        "profitable": cumulative_profit > 0,
        "csv_path": csv_path,
    }
```

- [x] **Step 5: Add orchestration function for training script use**

Append this code to `pretrain_qtable_diagnostics.py`:

```python
def prepare_pretrain_qtable_diagnostics(
    num_sample,
    total_df_index_length,
    position_choices,
    train_data_path,
    qtable_kwargs,
    env_kwargs,
    output_dir,
    logger=None,
    process_count=None,
):
    sample_plan = build_sample_plan(
        num_sample, total_df_index_length, position_choices
    )
    q_table_cache, train_df_cache = build_q_table_cache(
        sample_plan,
        train_data_path,
        qtable_kwargs,
        process_count=process_count,
    )
    diagnostics = []
    for sample_index, (df_index, initial_action) in enumerate(sample_plan):
        diagnostic = evaluate_and_export_sample(
            sample_index,
            df_index,
            initial_action,
            train_df_cache[df_index],
            q_table_cache[df_index],
            env_kwargs,
            output_dir,
        )
        diagnostics.append(diagnostic)
        message = (
            "qtable诊断 | sample={sample_index} | df_index={df_index} | "
            "initial_action={initial_action} | episode_reward_sum={episode_reward_sum:.4f} | "
            "profitable={profitable} | csv_path={csv_path}"
        ).format(**diagnostic)
        if logger is not None:
            logger.info(message)
        print(message.replace(" | ", " "))
    return sample_plan, q_table_cache, train_df_cache, diagnostics
```

- [x] **Step 6: Import diagnostics orchestration in the training script**

In `FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`, add this import near the other low-level imports:

```python
from RL.DiHFT.low_level.pretrain_qtable_diagnostics import (
    prepare_pretrain_qtable_diagnostics,
    build_initial_state,
    create_demo_env,
)
```

- [x] **Step 7: Add a small method to apply initial state in the trainer**

Inside `Weighted_Contexts_DQN`, insert this method after `act_multi_styles_pretrain()`:

```python
    def _set_initial_state_from_action(self, train_df, initial_action):
        (
            self.initial_position,
            self.initial_leverage,
            self.initial_margin,
            self.initial_state,
        ) = build_initial_state(
            train_df,
            initial_action,
            self.leverage_choices,
            self.position_list,
            self.initial_wallet_balance,
            self.initial_unrealized_pnL,
        )
```

- [x] **Step 8: Build plan/cache/CSV diagnostics before the sample loop**

In `train()`, replace:

```python
        step_counter_pretrain = 0
        step_counter_diverse = 0
        for sample in range(self.num_sample):
```

with:

```python
        step_counter_pretrain = 0
        step_counter_diverse = 0
        qtable_diagnostics_dir = os.path.join(self.model_path, "qtable_diagnostics")
        qtable_kwargs = {
            "max_holding_number": self.max_holding_number,
            "order_book_depth": self.order_book_depth,
            "position_choices": self.position_choices,
            "leverage_choice": self.leverage_choices,
            "long_estimated_rate": self.long_estimated_rate,
            "short_estimated_rate": self.short_estimated_rate,
            "commission_rate": self.transcation_cost,
            "max_punishment": 1e10,
            "gamma": 1,
        }
        env_kwargs = {
            "feature_list": self.tech_indicator_list,
            "max_holding_number": self.max_holding_number,
            "order_book_depth": self.order_book_depth,
            "position_choices": self.position_choices,
            "leverage_choices": self.leverage_choices,
            "position_list": self.position_list,
            "long_estimated_rate": self.long_estimated_rate,
            "short_estimated_rate": self.short_estimated_rate,
            "commission_rate": self.transcation_cost,
            "maintenance_margin_ratio_dict": self.maintenance_margin_ratio_dict,
            "early_stop": self.early_stop,
            "gamma": self.gamma,
            "initial_wallet_balance": self.initial_wallet_balance,
            "initial_unrealized_pnl": self.initial_unrealized_pnL,
        }
        sample_plan, q_table_cache, train_df_cache, _ = (
            prepare_pretrain_qtable_diagnostics(
                num_sample=self.num_sample,
                total_df_index_length=self.total_df_index_length,
                position_choices=self.position_choices,
                train_data_path=self.train_data_path,
                qtable_kwargs=qtable_kwargs,
                env_kwargs=env_kwargs,
                output_dir=qtable_diagnostics_dir,
                logger=logger,
            )
        )
        for sample in range(self.num_sample):
```

- [x] **Step 9: Make the training loop consume cached plan and df**

Inside the `for sample in range(self.num_sample):` loop, replace:

```python
            df_index = random.choices(range(self.total_df_index_length), k=1)[0]
            initial_action = random.choices(range(self.position_choices), k=1)[0]
```

with:

```python
            df_index, initial_action = sample_plan[sample]
```

Then replace the current `self.train_df = pd.read_feather(...)` line and initial-state block through `env = initiate_demo_env(...)` with:

```python
            self.train_df = train_df_cache[df_index]
            self._set_initial_state_from_action(self.train_df, initial_action)
            print(
                "初始仓位={}, 初始杠杆={}".format(
                    self.initial_position, self.initial_leverage
                )
            )
            env = create_demo_env(self.train_df, env_kwargs, self.initial_state)
```

- [x] **Step 10: Reuse cached qtable in the pretrain branch**

Inside `if pretrain:`, replace the existing `q_table = create_optimal_q_table_from_df(...)` call with:

```python
                q_table = q_table_cache[df_index]
```

- [x] **Step 11: Run a focused syntax check**

Run:

```bash
conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py
```

Expected: command exits with status 0 and prints no Python syntax errors.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 2: Verification

> **trace:** plan-ready.md -> `### Task 2: Verification` | tasks.md -> `- [ ] 2.0 Complete verification.`
> **sync:** tasks.md -> `- [ ] 2.0 Complete verification.` | plan-ready.md -> `### Task 2: Verification`

**Files:**
- Verify: `FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`
- Verify: `FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py`
- Verify: `openspec/changes/add-pretrain-qtable-profit-check/specs/fineft-stage-i-pretrain/spec.md`
- Verify: `openspec/changes/add-pretrain-qtable-profit-check/tasks.md`

- [x] **Step 1: Run Python syntax verification in the requested conda environment**

Run:

```bash
conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py
```

Expected: command exits with status 0 and produces no `SyntaxError`.

- [x] **Step 2: Run OpenSpec strict validation**

Run:

```bash
openspec validate add-pretrain-qtable-profit-check --strict
```

Expected:

```text
Change 'add-pretrain-qtable-profit-check' is valid
```

- [x] **Step 3: Check whether local training data is available**

Run:

```bash
test -d dataset && find dataset -path '*/train/df_0.feather' -print -quit
```

Expected if data is available: prints a path ending in `/train/df_0.feather`.

Expected if data is unavailable: prints nothing. Record that the smoke run was skipped because the local dataset is absent.

- [ ] **Step 4: If data is available, run a small smoke command**

Use the dataset name from the printed path. For a path like `dataset/BTCUSDT/train/df_0.feather`, run:

```bash
conda activate finetf && python FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py --base_path dataset --dataset_name BTCUSDT --num_sample 2 --pretrain_epoch 1
```

Expected: before `===== 第 1/2 轮采样 =====`, output includes two qtable diagnostics. Each diagnostic line includes `qtable诊断`, `sample=`, `df_index=`, `initial_action=`, `episode_reward_sum=`, `profitable=`, and `csv_path=`.

Expected: `result/BTCUSDT/weights_advantage_pretrain/qtable_diagnostics/` contains two CSV files when the default `--result_path result` is used.

- [ ] **Step 5: Inspect one generated CSV**

Run:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

files = sorted(Path("result/BTCUSDT/weights_advantage_pretrain/qtable_diagnostics").glob("sample_*.csv"))
assert files, "no qtable diagnostic csv files found"
df = pd.read_csv(files[0])
required = {
    "sample_index",
    "df_index",
    "initial_action",
    "step_index",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "mark_price",
    "action",
    "previous_action",
    "position",
    "leverage",
    "commission_rate",
    "step_slippage",
    "step_reward",
    "cumulative_profit",
    "profitable",
}
missing = required.difference(df.columns)
assert not missing, sorted(missing)
assert len(df) > 0
print(files[0])
PY
```

Expected: prints the inspected CSV path and raises no assertion.

- [x] **Step 6: Confirm multiprocessing qtable calculation and cached pretrain use**

Run:

```bash
rg -n "multiprocessing|mp\\.Pool|build_q_table_cache|q_table_cache\\[df_index\\]|create_optimal_q_table_from_df|random\\.choices\\(range\\(self\\.total_df_index_length\\)|random\\.choices\\(range\\(self\\.position_choices\\)" FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py
```

Expected: output shows `mp.Pool` and `create_optimal_q_table_from_df` only in `pretrain_qtable_diagnostics.py`; output shows `q_table = q_table_cache[df_index]` in `weight_advantage_pretrain.py`; the two sample-plan `random.choices(...)` calls are in `pretrain_qtable_diagnostics.py`.

- [ ] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）
