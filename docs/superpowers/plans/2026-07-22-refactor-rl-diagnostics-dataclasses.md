# Refactor RL Diagnostics Dataclasses Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace anonymous dict business records in low-level RL diagnostics with explicit dataclass contracts while preserving existing manifest JSON and diagnostics CSV formats.

**Architecture:** Keep dataclasses inside the owning low-level modules. Convert cross-function records, worker queue payloads, diagnostic rows, metrics, and summaries to module-level dataclasses with `to_dict()` only at compatibility boundaries. Keep natural cache maps such as `df_index -> DataFrame`, `df_index -> q_table`, and `SamplePlanItem -> action list`.

**Tech Stack:** Python standard-library `dataclasses`, numpy, pandas, polars, torch multiprocessing, pytest, OpenSpec.

**Traceability (sddflow):**
- plan-ready: `openspec/changes/refactor-rl-diagnostics-dataclasses/plan-ready.md`
- tasks: `openspec/changes/refactor-rl-diagnostics-dataclasses/tasks.md`
- plan: `docs/superpowers/plans/2026-07-22-refactor-rl-diagnostics-dataclasses.md`

---

### Task 1: Update focused tests for dataclass diagnostics contracts

> **trace:** plan-ready.md -> `### Task 1: Update focused tests for dataclass diagnostics contracts` | tasks.md -> ``- [ ] 1.1 Update focused tests to require dataclass interfaces for loss NaN diagnostics, qtable diagnostics, and parallel rollout contracts while preserving legacy `.to_dict()` and file-format assertions.``
> **sync:** tasks.md -> ``- [ ] 1.1 Update focused tests to require dataclass interfaces for loss NaN diagnostics, qtable diagnostics, and parallel rollout contracts while preserving legacy `.to_dict()` and file-format assertions.`` | plan-ready.md -> `### Task 1: Update focused tests for dataclass diagnostics contracts`

**Files:**
- Modify: `FineFT/tests/rl/test_weight_advantage_pretrain_logging.py`
- Modify: `FineFT/tests/rl/test_pretrain_qtable_diagnostics.py`
- Modify: `FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py`

- [x] **Step 1: Update loss NaN diagnostics assertions to require objects**

In `FineFT/tests/rl/test_weight_advantage_pretrain_logging.py`, change `test_build_loss_nan_diagnostics_identifies_nonfinite_training_data` so it asserts attributes first and `.to_dict()` compatibility second:

```python
def test_build_loss_nan_diagnostics_identifies_nonfinite_training_data():
    import numpy as np
    import torch
    from RL.DiHFT.low_level import loss_nan_diagnostics

    diagnostics = loss_nan_diagnostics.build_loss_nan_diagnostics(
        numeric_values={
            "loss": torch.tensor(float("nan")),
            "td_loss": torch.tensor(3.0),
            "states": torch.tensor([[1.0, float("inf")]]),
        },
        info_values={
            "info": {
                "q_value": torch.tensor([[1.0, float("nan")]]),
                "funding_count_down_hour": np.array([1.0, float("-inf")]),
                "safe_value": [1.0, 2.0],
            }
        },
    )

    assert isinstance(diagnostics, loss_nan_diagnostics.LossNanDiagnostics)
    assert diagnostics.numeric["loss"].nan_count == 1
    assert diagnostics.numeric["td_loss"].finite_count == 1
    assert diagnostics.numeric["states"].inf_count == 1
    assert diagnostics.info_nonfinite == [
        loss_nan_diagnostics.NonfiniteLocation(
            path="info.q_value",
            shape=[1, 2],
            dtype="torch.float32",
            nan_count=1,
            inf_count=0,
            first_nonfinite_indices=[[0, 1]],
        ),
        loss_nan_diagnostics.NonfiniteLocation(
            path="info.funding_count_down_hour",
            shape=[2],
            dtype="float64",
            nan_count=0,
            inf_count=1,
            first_nonfinite_indices=[[1]],
        ),
    ]
    assert diagnostics.to_dict()["numeric"]["loss"]["nan_count"] == 1
    assert diagnostics.to_dict()["info_nonfinite"][1]["inf_count"] == 1
```

- [x] **Step 2: Run loss diagnostics test to verify it fails**

Run: `conda activate finetf && pytest FineFT/tests/rl/test_weight_advantage_pretrain_logging.py::test_build_loss_nan_diagnostics_identifies_nonfinite_training_data -q`

Expected: FAIL with `AttributeError` for missing `LossNanDiagnostics` or missing object attributes.

- [x] **Step 3: Update qtable tests to require sample/result objects**

In `FineFT/tests/rl/test_pretrain_qtable_diagnostics.py`, update the sample plan tests:

```python
def test_build_sample_plan_traverses_every_df_and_position():
    from RL.DiHFT.low_level import pretrain_qtable_diagnostics as diag

    sample_plan = diag.build_sample_plan(
        total_df_index_length=2,
        position_choices=3,
    )

    assert sample_plan == [
        diag.SamplePlanItem(0, 0),
        diag.SamplePlanItem(0, 1),
        diag.SamplePlanItem(0, 2),
        diag.SamplePlanItem(1, 0),
        diag.SamplePlanItem(1, 1),
        diag.SamplePlanItem(1, 2),
    ]


def test_select_sample_from_plan_randomly_picks_one_combination(monkeypatch):
    from RL.DiHFT.low_level import pretrain_qtable_diagnostics as diag

    sample_plan = [
        diag.SamplePlanItem(0, 0),
        diag.SamplePlanItem(0, 1),
        diag.SamplePlanItem(1, 0),
        diag.SamplePlanItem(1, 1),
    ]
    monkeypatch.setattr(diag.random, "choice", lambda choices: choices[3])

    assert diag.select_sample_from_plan(sample_plan) == diag.SamplePlanItem(1, 1)
```

Update every `prepare_pretrain_qtable_diagnostics(...)` unpacking to use a result object:

```python
result = diag.prepare_pretrain_qtable_diagnostics(...)
assert isinstance(result, diag.PretrainQTableDiagnosticsResult)
assert result.sample_plan == [diag.SamplePlanItem(0, 0), diag.SamplePlanItem(0, 1)]
assert list(result.q_table_cache) == [0]
assert list(result.train_df_cache) == [0]
assert result.sample_action_cache == {
    diag.SamplePlanItem(0, 0): [1, 2],
    diag.SamplePlanItem(0, 1): [1, 2],
}
assert result.diagnostics[0].to_dict()["episode_reward_sum"] == 30.0
```

Where tests monkeypatch `build_sample_plan`, return `SamplePlanItem` values:

```python
monkeypatch.setattr(
    diag,
    "build_sample_plan",
    lambda total, choices: [diag.SamplePlanItem(0, 0), diag.SamplePlanItem(0, 1)],
)
```

Where fake `build_q_table_cache` returns diagnostics, return `SampleDiagnostic` objects:

```python
return diag.QTableCacheBuildResult(
    q_table_cache={2: "q-table"},
    train_df_cache={2: "train-df"},
    diagnostics=[
        diag.SampleDiagnostic(
            df_index=3,
            initial_action=0,
            episode_reward_sum=-1.0,
            profitable=False,
            csv_path=str(output_dir_path / "df_3_initial_action_0.csv"),
            action_list=[7],
        ),
        diag.SampleDiagnostic(
            df_index=2,
            initial_action=1,
            episode_reward_sum=3.0,
            profitable=True,
            csv_path=str(output_dir_path / "df_2_initial_action_1.csv"),
            action_list=[5, 6],
        ),
    ],
)
```

- [x] **Step 4: Run qtable diagnostics tests to verify they fail**

Run: `conda activate finetf && pytest FineFT/tests/rl/test_pretrain_qtable_diagnostics.py -q`

Expected: FAIL with missing `SamplePlanItem`, `PretrainQTableDiagnosticsResult`, or old tuple/dict return assumptions.

- [x] **Step 5: Update parallel rollout tests to require dataclass contracts**

In `FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py`, update key assertions:

```python
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
```

Update epoch params assertions:

```python
assert first == pwap.EpochTrainingParams(epsilon=1.0, ada=256.0, lr=0.005)
assert first.to_dict() == {"epsilon": 1.0, "ada": 256.0, "lr": 0.005}
```

Update worker result tests to use dataclasses:

```python
results = [
    pwap.WorkerRoundResult(
        df_index=2,
        epoch_index=0,
        context_index=0,
        initial_action=0,
        round_counter=0,
        worker_steps=1,
        transitions=[
            pwap.WorkerTransitionRecord(step_index=1, transition="df2-step1")
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
            pwap.WorkerTransitionRecord(step_index=1, transition="df1-step1"),
            pwap.WorkerTransitionRecord(step_index=0, transition="df1-step0"),
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
```

Update worker error assertion:

```python
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
```

- [x] **Step 6: Run parallel rollout tests to verify they fail**

Run: `conda activate finetf && pytest FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py -q`

Expected: FAIL with missing dataclass types or old dict access in implementation.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 2: Refactor loss NaN diagnostics dataclasses

> **trace:** plan-ready.md -> `### Task 2: Refactor loss NaN diagnostics dataclasses` | tasks.md -> ``- [ ] 1.2 Refactor `loss_nan_diagnostics.py` to return dataclass diagnostics and update logging to use attributes.``
> **sync:** tasks.md -> ``- [ ] 1.2 Refactor `loss_nan_diagnostics.py` to return dataclass diagnostics and update logging to use attributes.`` | plan-ready.md -> `### Task 2: Refactor loss NaN diagnostics dataclasses`

**Files:**
- Modify: `FineFT/RL/DiHFT/low_level/loss_nan_diagnostics.py`
- Test: `FineFT/tests/rl/test_weight_advantage_pretrain_logging.py`

- [x] **Step 1: Add loss diagnostics dataclasses**

At the top of `FineFT/RL/DiHFT/low_level/loss_nan_diagnostics.py`, after imports, add:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class NumericValueSummary:
    shape: list[int]
    dtype: str
    finite_count: int
    nan_count: int
    inf_count: int
    first_nonfinite_indices: list[list[int]]
    device: str | None = None
    finite_min: float | None = None
    finite_max: float | None = None
    finite_mean: float | None = None

    def to_dict(self):
        payload = {
            "shape": self.shape,
            "dtype": self.dtype,
            "finite_count": self.finite_count,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "first_nonfinite_indices": self.first_nonfinite_indices,
        }
        if self.device is not None:
            payload["device"] = self.device
        if self.finite_min is not None:
            payload["finite_min"] = self.finite_min
            payload["finite_max"] = self.finite_max
            payload["finite_mean"] = self.finite_mean
        return payload


@dataclass(frozen=True)
class NonfiniteLocation:
    path: str
    shape: list[int]
    dtype: str
    nan_count: int
    inf_count: int
    first_nonfinite_indices: list[list[int]]

    def to_dict(self):
        return {
            "path": self.path,
            "shape": self.shape,
            "dtype": self.dtype,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "first_nonfinite_indices": self.first_nonfinite_indices,
        }


@dataclass(frozen=True)
class LossNanDiagnostics:
    numeric: dict[str, NumericValueSummary | None]
    info_nonfinite: list[NonfiniteLocation]

    def to_dict(self):
        return {
            "numeric": {
                name: summary.to_dict() if summary is not None else None
                for name, summary in self.numeric.items()
            },
            "info_nonfinite": [
                location.to_dict() for location in self.info_nonfinite
            ],
        }
```

- [x] **Step 2: Convert numeric summary builder to dataclass**

In `_summarize_numeric_value`, replace summary dict construction with `NumericValueSummary` construction. Preserve the existing finite stats logic:

```python
def _summarize_numeric_value(value, max_indices=10):
    if torch.is_tensor(value):
        data = value.detach()
        finite_mask = torch.isfinite(data)
        finite_values = data[finite_mask]
        finite_min = None
        finite_max = None
        finite_mean = None
        if finite_values.numel() > 0:
            finite_values = finite_values.float()
            finite_min = float(finite_values.min().item())
            finite_max = float(finite_values.max().item())
            finite_mean = float(finite_values.mean().item())
        return NumericValueSummary(
            shape=list(data.shape),
            dtype=str(data.dtype),
            device=str(data.device),
            finite_count=int(finite_mask.sum().item()),
            nan_count=int(torch.isnan(data).sum().item()),
            inf_count=int(torch.isinf(data).sum().item()),
            first_nonfinite_indices=torch.nonzero(
                ~finite_mask, as_tuple=False
            )[:max_indices].cpu().tolist(),
            finite_min=finite_min,
            finite_max=finite_max,
            finite_mean=finite_mean,
        )

    if isinstance(value, (int, float, np.number, np.ndarray, list, tuple)):
        try:
            data = np.asarray(value)
        except (TypeError, ValueError):
            return None
        if not np.issubdtype(data.dtype, np.number):
            return None

        finite_mask = np.isfinite(data)
        finite_values = data[finite_mask]
        finite_min = None
        finite_max = None
        finite_mean = None
        if finite_values.size > 0:
            finite_values = finite_values.astype(float)
            finite_min = float(finite_values.min())
            finite_max = float(finite_values.max())
            finite_mean = float(finite_values.mean())
        return NumericValueSummary(
            shape=list(data.shape),
            dtype=str(data.dtype),
            finite_count=int(finite_mask.sum()),
            nan_count=int(np.isnan(data).sum()),
            inf_count=int(np.isinf(data).sum()),
            first_nonfinite_indices=np.argwhere(~finite_mask)[:max_indices].tolist(),
            finite_min=finite_min,
            finite_max=finite_max,
            finite_mean=finite_mean,
        )

    return None
```

- [x] **Step 3: Convert nonfinite location and diagnostics builders**

Update `_find_nonfinite_locations` and `build_loss_nan_diagnostics`:

```python
def _find_nonfinite_locations(value, path, locations, max_items=20):
    if len(locations) >= max_items:
        return

    summary = _summarize_numeric_value(value)
    if summary is not None:
        if summary.nan_count or summary.inf_count:
            locations.append(
                NonfiniteLocation(
                    path=path,
                    shape=summary.shape,
                    dtype=summary.dtype,
                    nan_count=summary.nan_count,
                    inf_count=summary.inf_count,
                    first_nonfinite_indices=summary.first_nonfinite_indices,
                )
            )
        return

    if isinstance(value, dict):
        for key in value:
            _find_nonfinite_locations(
                value[key], f"{path}.{key}", locations, max_items=max_items
            )
            if len(locations) >= max_items:
                return
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _find_nonfinite_locations(
                item, f"{path}[{index}]", locations, max_items=max_items
            )
            if len(locations) >= max_items:
                return


def build_loss_nan_diagnostics(numeric_values, info_values, max_items=20):
    diagnostics = LossNanDiagnostics(numeric={}, info_nonfinite=[])
    for name, value in numeric_values.items():
        diagnostics.numeric[name] = _summarize_numeric_value(value)

    for name, value in info_values.items():
        _find_nonfinite_locations(
            value,
            name,
            diagnostics.info_nonfinite,
            max_items=max_items,
        )

    return diagnostics
```

- [x] **Step 4: Update logger usage to attributes**

In `log_loss_nan_diagnostics`, replace dict access with attributes:

```python
for name, summary in diagnostics.numeric.items():
    logger.error(
        "loss nan numeric | %s=%s",
        name,
        summary.to_dict() if summary is not None else None,
    )

if diagnostics.info_nonfinite:
    for location in diagnostics.info_nonfinite:
        logger.error("loss nan data nonfinite | %s", location.to_dict())
else:
    logger.error("loss nan data nonfinite | no nonfinite values found in info")
```

- [x] **Step 5: Run loss diagnostics verification**

Run: `conda activate finetf && pytest FineFT/tests/rl/test_weight_advantage_pretrain_logging.py::test_build_loss_nan_diagnostics_identifies_nonfinite_training_data -q`

Expected: PASS.

Run: `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/loss_nan_diagnostics.py`

Expected: exits with status 0.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 3: Refactor qtable diagnostics dataclasses

> **trace:** plan-ready.md -> `### Task 3: Refactor qtable diagnostics dataclasses` | tasks.md -> ``- [ ] 1.3 Refactor `pretrain_qtable_diagnostics.py` to use dataclass sample items, manifest, CSV rows, sample diagnostics, worker result, and prepare result while preserving manifest JSON and diagnostics CSV formats.``
> **sync:** tasks.md -> ``- [ ] 1.3 Refactor `pretrain_qtable_diagnostics.py` to use dataclass sample items, manifest, CSV rows, sample diagnostics, worker result, and prepare result while preserving manifest JSON and diagnostics CSV formats.`` | plan-ready.md -> `### Task 3: Refactor qtable diagnostics dataclasses`

**Files:**
- Modify: `FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py`
- Test: `FineFT/tests/rl/test_pretrain_qtable_diagnostics.py`

- [x] **Step 1: Add qtable diagnostics dataclasses**

Add these top-level definitions after constants in `pretrain_qtable_diagnostics.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class SamplePlanItem:
    df_index: int
    initial_action: int

    def to_tuple(self):
        return (self.df_index, self.initial_action)


@dataclass(frozen=True)
class QTableDiagnosticsManifest:
    diagnostic_count: int
    total_df_index_length: int
    position_choices: int
    qtable_kwargs: dict
    env_kwargs: dict

    def to_dict(self):
        return _normalize_manifest_value(
            {
                "diagnostic_count": self.diagnostic_count,
                "total_df_index_length": self.total_df_index_length,
                "position_choices": self.position_choices,
                "qtable_kwargs": self.qtable_kwargs,
                "env_kwargs": self.env_kwargs,
            }
        )


@dataclass(frozen=True)
class DiagnosticCsvRow:
    df_index: int
    initial_action: int
    step_index: int
    timestamp: object
    open: object
    high: object
    low: object
    close: object
    volume: object
    mark_price: object
    action: int
    previous_action: int
    position: float
    leverage: float
    commission_rate: float
    step_slippage: float
    step_reward: float
    cumulative_profit: float
    profitable: bool

    def to_dict(self):
        return {
            "df_index": self.df_index,
            "initial_action": self.initial_action,
            "step_index": self.step_index,
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "mark_price": self.mark_price,
            "action": self.action,
            "previous_action": self.previous_action,
            "position": self.position,
            "leverage": self.leverage,
            "commission_rate": self.commission_rate,
            "step_slippage": self.step_slippage,
            "step_reward": self.step_reward,
            "cumulative_profit": self.cumulative_profit,
            "profitable": self.profitable,
        }


@dataclass(frozen=True)
class SampleDiagnostic:
    df_index: int
    initial_action: int
    episode_reward_sum: float
    profitable: bool
    csv_path: str
    action_list: list[int]

    @property
    def sample_item(self):
        return SamplePlanItem(self.df_index, self.initial_action)

    def to_dict(self, include_action_list=True):
        payload = {
            "df_index": self.df_index,
            "initial_action": self.initial_action,
            "episode_reward_sum": self.episode_reward_sum,
            "profitable": self.profitable,
            "csv_path": self.csv_path,
        }
        if include_action_list:
            payload["action_list"] = list(self.action_list)
        return payload


@dataclass(frozen=True)
class QTableCacheBuildResult:
    q_table_cache: dict
    train_df_cache: dict
    diagnostics: list[SampleDiagnostic]


@dataclass(frozen=True)
class PretrainQTableDiagnosticsResult:
    sample_plan: list[SamplePlanItem]
    q_table_cache: dict
    train_df_cache: dict
    diagnostics: list[SampleDiagnostic]
    sample_action_cache: dict[SamplePlanItem, list[int]]
```

- [x] **Step 2: Convert sample plan and worker result functions**

Update `build_sample_plan`, `_create_q_table_worker`, and `build_q_table_cache`:

```python
def build_sample_plan(total_df_index_length, position_choices):
    return [
        SamplePlanItem(df_index, initial_action)
        for df_index in range(total_df_index_length)
        for initial_action in range(position_choices)
    ]


def _coerce_sample_item(sample_item):
    if isinstance(sample_item, SamplePlanItem):
        return sample_item
    df_index, initial_action = sample_item
    return SamplePlanItem(int(df_index), int(initial_action))


def _create_q_table_worker(args):
    (
        df_index,
        train_data_path,
        qtable_kwargs,
        sample_tasks,
        env_kwargs,
        output_dir,
    ) = args
    df_path = os.path.join(train_data_path, "df_{}.feather".format(df_index))
    train_df = pd.read_feather(df_path)
    q_table = create_optimal_q_table_from_df(df=train_df, **qtable_kwargs)
    diagnostics = []
    if sample_tasks and env_kwargs is not None and output_dir is not None:
        for initial_action in sample_tasks:
            diagnostics.append(
                evaluate_and_export_sample(
                    df_index,
                    initial_action,
                    train_df,
                    q_table,
                    env_kwargs,
                    output_dir,
                )
            )
    return QTableCacheBuildResult(
        q_table_cache={df_index: q_table},
        train_df_cache={df_index: train_df},
        diagnostics=diagnostics,
    )
```

In `build_q_table_cache`, coerce incoming plan items and return `QTableCacheBuildResult`:

```python
sample_plan = [_coerce_sample_item(item) for item in sample_plan]
unique_df_indices = sorted({item.df_index for item in sample_plan})
...
for item in sample_plan:
    sample_tasks_by_df.setdefault(item.df_index, []).append(item.initial_action)
...
q_table_cache = {}
train_df_cache = {}
diagnostics = []
for result in results:
    q_table_cache.update(result.q_table_cache)
    train_df_cache.update(result.train_df_cache)
    diagnostics.extend(result.diagnostics)
return QTableCacheBuildResult(q_table_cache, train_df_cache, diagnostics)
```

- [x] **Step 3: Convert manifest and CSV row writing**

Replace `_build_diagnostics_manifest` with an object return:

```python
def _build_diagnostics_manifest(
    diagnostic_count,
    total_df_index_length,
    position_choices,
    qtable_kwargs,
    env_kwargs,
):
    return QTableDiagnosticsManifest(
        diagnostic_count=diagnostic_count,
        total_df_index_length=total_df_index_length,
        position_choices=position_choices,
        qtable_kwargs=qtable_kwargs,
        env_kwargs=env_kwargs,
    )
```

Update `_manifest_matches` and `_write_diagnostics_manifest` to accept either the object or a dict:

```python
def _manifest_payload(manifest):
    return manifest.to_dict() if hasattr(manifest, "to_dict") else manifest


def _manifest_matches(output_dir, expected_manifest):
    manifest_path = os.path.join(output_dir, DIAGNOSTIC_MANIFEST_NAME)
    if not os.path.isfile(manifest_path):
        return False
    try:
        with open(manifest_path, "r", encoding="utf-8") as manifest_file:
            existing_manifest = json.load(manifest_file)
    except (OSError, json.JSONDecodeError):
        return False
    return existing_manifest == _manifest_payload(expected_manifest)


def _write_diagnostics_manifest(output_dir, manifest):
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, DIAGNOSTIC_MANIFEST_NAME)
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(_manifest_payload(manifest), manifest_file, sort_keys=True, indent=2)
        manifest_file.write("\n")
```

Update `_diagnostic_row` to return `DiagnosticCsvRow`, and write rows as dicts:

```python
rows.append(_diagnostic_row(...))
...
pl.DataFrame([row.to_dict() for row in rows]).write_csv(csv_path)
return SampleDiagnostic(
    df_index=df_index,
    initial_action=initial_action,
    episode_reward_sum=cumulative_profit,
    profitable=cumulative_profit > 0,
    csv_path=csv_path,
    action_list=[row.action for row in rows],
)
```

- [x] **Step 4: Convert existing-cache and prepare return path**

Update `_load_existing_diagnostics` so it builds `SamplePlanItem`, `SampleDiagnostic`, and `PretrainQTableDiagnosticsResult`:

```python
sample_plan = [
    SamplePlanItem(df_index, initial_action)
    for df_index in range(total_df_index_length)
    for initial_action in range(position_choices)
]
...
sample_item = SamplePlanItem(df_index, initial_action)
sample_action_cache[sample_item] = diagnostic_df["action"].cast(pl.Int64).to_list()
diagnostics.append(
    SampleDiagnostic(
        df_index=df_index,
        initial_action=initial_action,
        episode_reward_sum=float(episode_reward_sum),
        profitable=episode_reward_sum > 0,
        csv_path=csv_path,
        action_list=sample_action_cache[sample_item],
    )
)
...
return PretrainQTableDiagnosticsResult(
    sample_plan=sample_plan,
    q_table_cache={},
    train_df_cache=train_df_cache,
    diagnostics=diagnostics,
    sample_action_cache=sample_action_cache,
)
```

Update `prepare_pretrain_qtable_diagnostics` recomputation path:

```python
cache_result = build_q_table_cache(...)
sample_action_cache = {}
diagnostics = sorted(
    cache_result.diagnostics,
    key=lambda item: (item.df_index, item.initial_action),
)
for diagnostic in diagnostics:
    sample_action_cache[diagnostic.sample_item] = diagnostic.action_list
    message = (
        "qtable诊断 | df_index={df_index} | "
        "initial_action={initial_action} | episode_reward_sum={episode_reward_sum:.4f} | "
        "profitable={profitable} | csv_path={csv_path}"
    ).format(**diagnostic.to_dict(include_action_list=False))
    if logger is not None:
        logger.info(message)
_write_diagnostics_manifest(output_dir, manifest)
return PretrainQTableDiagnosticsResult(
    sample_plan=sample_plan,
    q_table_cache=cache_result.q_table_cache,
    train_df_cache=cache_result.train_df_cache,
    diagnostics=diagnostics,
    sample_action_cache=sample_action_cache,
)
```

- [x] **Step 5: Keep compatibility for explicit cache lookup parameters**

Update `get_sample_action_from_cache` to accept either a `SamplePlanItem` or explicit legacy parameters:

```python
def get_sample_action_from_cache(
    sample_action_cache_by_plan,
    sample_item_or_df_index,
    initial_action=None,
):
    if isinstance(sample_item_or_df_index, SamplePlanItem):
        sample_key = sample_item_or_df_index
    else:
        sample_key = SamplePlanItem(int(sample_item_or_df_index), int(initial_action))
    if sample_key not in sample_action_cache_by_plan:
        raise KeyError(
            "sample_action_cache missing for df_index={} initial_action={}".format(
                sample_key.df_index,
                sample_key.initial_action,
            )
        )
    return sample_action_cache_by_plan[sample_key]
```

- [x] **Step 6: Run qtable diagnostics verification**

Run: `conda activate finetf && pytest FineFT/tests/rl/test_pretrain_qtable_diagnostics.py -q`

Expected: PASS.

Run: `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py`

Expected: exits with status 0.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 4: Update training callers for qtable dataclass results

> **trace:** plan-ready.md -> `### Task 4: Update training callers for qtable dataclass results` | tasks.md -> ``- [ ] 1.4 Update `weight_advantage_pretrain.py` and the qtable-related portions of `parallel_weight_advantage_pretrain.py` to consume `PretrainQTableDiagnosticsResult` and `SamplePlanItem` via attributes.``
> **sync:** tasks.md -> ``- [ ] 1.4 Update `weight_advantage_pretrain.py` and the qtable-related portions of `parallel_weight_advantage_pretrain.py` to consume `PretrainQTableDiagnosticsResult` and `SamplePlanItem` via attributes.`` | plan-ready.md -> `### Task 4: Update training callers for qtable dataclass results`

**Files:**
- Modify: `FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`
- Modify: `FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py`
- Test: `FineFT/tests/rl/test_pretrain_qtable_diagnostics.py`
- Test: `FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py`

- [x] **Step 1: Update serial pretrain caller unpacking**

In `weight_advantage_pretrain.py`, replace the tuple unpacking with result attributes:

```python
diagnostics_result = prepare_pretrain_qtable_diagnostics(
    total_df_index_length=self.total_df_index_length,
    position_choices=self.position_choices,
    train_data_path=self.train_data_path,
    qtable_kwargs=qtable_kwargs,
    env_kwargs=env_kwargs,
    output_dir=qtable_diagnostics_dir,
    logger=logger,
)
sample_plan = diagnostics_result.sample_plan
q_table_cache = diagnostics_result.q_table_cache
train_df_cache = diagnostics_result.train_df_cache
sample_action_cache = diagnostics_result.sample_action_cache
```

Where a sample is selected, use the object:

```python
sample_item = select_sample_from_plan(sample_plan)
df_index = sample_item.df_index
initial_action = sample_item.initial_action
self.perfection_action_list = get_sample_action_from_cache(
    sample_action_cache,
    sample_item,
)
```

- [x] **Step 2: Update parallel pretrain caller unpacking**

In `parallel_weight_advantage_pretrain.py`, apply the same `diagnostics_result` pattern to the call near `prepare_pretrain_qtable_diagnostics(...)`. Keep `q_table_cache`, `train_df_cache`, and `sample_action_cache` variable names after assignment so the rest of the train method stays narrow.

```python
diagnostics_result = prepare_pretrain_qtable_diagnostics(
    total_df_index_length=self.total_df_index_length,
    position_choices=self.position_choices,
    train_data_path=self.train_data_path,
    qtable_kwargs=qtable_kwargs,
    env_kwargs=env_kwargs,
    output_dir=qtable_diagnostics_dir,
    logger=logger,
)
sample_plan = diagnostics_result.sample_plan
q_table_cache = diagnostics_result.q_table_cache
train_df_cache = diagnostics_result.train_df_cache
sample_action_cache = diagnostics_result.sample_action_cache
```

- [x] **Step 3: Search for stale tuple cache access**

Run: `rg -n "prepare_pretrain_qtable_diagnostics\\(|get_sample_action_from_cache\\(|sample_plan\\[|df_index, initial_action|\\[\"df_index\"\\]|\\[\"initial_action\"\\]" FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py FineFT/tests/rl`

Expected: output shows no tuple unpacking of `prepare_pretrain_qtable_diagnostics`, no sample-plan tuple destructuring, and no qtable diagnostic string-key access outside `.to_dict()` compatibility checks.

- [x] **Step 4: Run caller verification**

Run: `conda activate finetf && pytest FineFT/tests/rl/test_pretrain_qtable_diagnostics.py FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py -q`

Expected: PASS or only failures belonging to Task 5 parallel dataclass conversion.

Run: `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py`

Expected: exits with status 0.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 5: Refactor parallel rollout dataclass contracts

> **trace:** plan-ready.md -> `### Task 5: Refactor parallel rollout dataclass contracts` | tasks.md -> ``- [ ] 1.5 Refactor `parallel_weight_advantage_pretrain.py` rollout task, epoch params, worker messages/results/errors, metrics, transition records, and round summaries to dataclass objects.``
> **sync:** tasks.md -> ``- [ ] 1.5 Refactor `parallel_weight_advantage_pretrain.py` rollout task, epoch params, worker messages/results/errors, metrics, transition records, and round summaries to dataclass objects.`` | plan-ready.md -> `### Task 5: Refactor parallel rollout dataclass contracts`

**Files:**
- Modify: `FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py`
- Test: `FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py`

- [x] **Step 1: Add parallel rollout dataclasses**

Near the top of `parallel_weight_advantage_pretrain.py`, after imports and logger setup, add:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class RolloutMetrics:
    epoch_index: int
    context_index: int
    initial_action: int
    df_index: int
    transition_count: int
    reward_sum: float
    final_balance: float
    return_rate: float

    def to_dict(self):
        return {
            "epoch_index": self.epoch_index,
            "context_index": self.context_index,
            "initial_action": self.initial_action,
            "df_index": self.df_index,
            "transition_count": self.transition_count,
            "reward_sum": self.reward_sum,
            "final_balance": self.final_balance,
            "return_rate": self.return_rate,
        }


@dataclass(frozen=True)
class RolloutMetricsSummary:
    mean_return_rate: float
    mean_final_balance: float
    mean_reward_sum: float

    def to_dict(self):
        return {
            "mean_return_rate": self.mean_return_rate,
            "mean_final_balance": self.mean_final_balance,
            "mean_reward_sum": self.mean_reward_sum,
        }


@dataclass(frozen=True)
class RolloutDiagnosticsSummary:
    action_counts: list[tuple[int, int]]
    position_counts: list[tuple[float, int]]
    first_actions: list[int]
    first_positions: list[float]
    position_switches: int

    def to_dict(self):
        return {
            "action_counts": self.action_counts,
            "position_counts": self.position_counts,
            "first_actions": self.first_actions,
            "first_positions": self.first_positions,
            "position_switches": self.position_switches,
        }


@dataclass(frozen=True)
class ParallelRolloutTask:
    epoch_index: int
    context_index: int
    initial_action: int

    def to_dict(self):
        return {
            "epoch_index": self.epoch_index,
            "context_index": self.context_index,
            "initial_action": self.initial_action,
        }


@dataclass(frozen=True)
class EpochTrainingParams:
    epsilon: float
    ada: float
    lr: float

    def to_dict(self):
        return {"epsilon": self.epsilon, "ada": self.ada, "lr": self.lr}


@dataclass(frozen=True)
class ResetWorkerTask:
    epoch_index: int
    context_index: int
    initial_action: int


@dataclass(frozen=True)
class ExploreWorkerRound:
    epoch_index: int
    context_index: int
    initial_action: int
    round_counter: int
    state_dict: dict
    epsilon: float
    rollout_steps: int


@dataclass(frozen=True)
class ShutdownWorker:
    pass


@dataclass(frozen=True)
class WorkerTransitionRecord:
    step_index: int
    transition: tuple | object


@dataclass(frozen=True)
class WorkerRoundResult:
    df_index: int
    epoch_index: int
    context_index: int
    initial_action: int
    round_counter: int
    worker_steps: int
    transitions: list[WorkerTransitionRecord]
    rollout_metrics: list[RolloutMetrics]
    done: bool
    progress: dict | None = None


@dataclass(frozen=True)
class WorkerErrorMessage:
    df_index: int
    epoch_index: int
    context_index: int
    initial_action: int
    round_counter: int
    traceback: str


@dataclass(frozen=True)
class ParallelRoundSummary:
    round_counter: int
    epoch_index: int
    context_index: int
    initial_action: int
    round_steps: int
    active_worker_count: int
    buffer_size: int
    update_count: int

    def to_dict(self):
        return {
            "round_counter": self.round_counter,
            "epoch_index": self.epoch_index,
            "context_index": self.context_index,
            "initial_action": self.initial_action,
            "round_steps": self.round_steps,
            "active_worker_count": self.active_worker_count,
            "buffer_size": self.buffer_size,
            "update_count": self.update_count,
        }
```

- [x] **Step 2: Convert metric and summary helpers**

Update helper return values and readers:

```python
def summarize_rollout_metrics(metrics):
    return RolloutMetricsSummary(
        mean_return_rate=float(np.mean([item.return_rate for item in metrics])),
        mean_final_balance=float(np.mean([item.final_balance for item in metrics])),
        mean_reward_sum=float(np.mean([item.reward_sum for item in metrics])),
    )


def record_diverse_rollout_latest_metric(
    metrics_by_df,
    df_index,
    rollout_index,
    reward_sum,
    final_balance,
    return_rate,
):
    df_metrics = metrics_by_df.setdefault(int(df_index), {})
    df_metrics[int(rollout_index)] = RolloutMetrics(
        epoch_index=-1,
        context_index=int(rollout_index),
        initial_action=-1,
        df_index=int(df_index),
        transition_count=0,
        reward_sum=float(reward_sum),
        final_balance=float(final_balance),
        return_rate=float(return_rate),
    )


def log_diverse_rollout_latest_metrics(epoch_index, metrics_by_df):
    for df_index in sorted(metrics_by_df):
        for rollout_index in sorted(metrics_by_df[df_index]):
            metrics = metrics_by_df[df_index][rollout_index]
            profit_label = "盈利" if metrics.return_rate > 0 else "亏损"
            logger.info(
                "第 %d 轮 epoch 训练完成 | 多样化训练最新明细 | "
                "df_index=%d | rollout_index=%d | 累计奖励=%.4f | "
                "最终余额=%.4f | 收益率=%.6f | %s",
                epoch_index,
                df_index,
                rollout_index,
                metrics.reward_sum,
                metrics.final_balance,
                metrics.return_rate,
                profit_label,
            )
```

Update `summarize_rollout_diagnostics` and `compute_epoch_training_params`:

```python
return RolloutDiagnosticsSummary(...)
return EpochTrainingParams(
    epsilon=_linear_value(epsilon_init, epsilon_min, epoch_index, num_epoch),
    ada=_held_then_linear_value(ada_init, ada_min, epoch_index, num_epoch),
    lr=_held_then_linear_value(lr_init, lr_min, epoch_index, num_epoch),
)
```

- [x] **Step 3: Convert task iterator and sorting/error helpers**

Update task iterator and sorting:

```python
def iter_parallel_rollout_tasks(num_epoch, context_count, position_choices):
    for epoch_index in range(num_epoch):
        for context_index in range(context_count):
            for initial_action in range(position_choices):
                yield ParallelRolloutTask(
                    epoch_index=epoch_index,
                    context_index=context_index,
                    initial_action=initial_action,
                )


def sort_round_transitions(round_results):
    ordered = []
    for result in sorted(round_results, key=lambda item: item.df_index):
        ordered.extend(
            item.transition
            for item in sorted(
                result.transitions,
                key=lambda transition: transition.step_index,
            )
        )
    return ordered


def raise_for_worker_error(message):
    if not isinstance(message, WorkerErrorMessage):
        return
    raise RuntimeError(
        "worker_error df_index={} epoch_index={} context_index={} "
        "initial_action={} round_counter={}: {}".format(
            message.df_index,
            message.epoch_index,
            message.context_index,
            message.initial_action,
            message.round_counter,
            message.traceback,
        )
    )
```

- [x] **Step 4: Convert worker queue message handling**

Update `df_rollout_worker` and `DfRolloutWorkerRunner`:

```python
def df_rollout_worker(worker_config, input_queue, result_queue):
    df_index = worker_config["df_index"]
    message = None
    try:
        runner_factory = worker_config.get("runner_factory", DfRolloutWorkerRunner)
        runner = runner_factory(worker_config)
        while True:
            message = input_queue.get()
            if isinstance(message, ShutdownWorker):
                return
            if isinstance(message, ResetWorkerTask):
                runner.reset_task(message)
                continue
            if isinstance(message, ExploreWorkerRound):
                result_queue.put(runner.explore_round(message))
                continue
            raise ValueError("unknown worker message type: {}".format(type(message).__name__))
    except Exception:
        result_queue.put(
            WorkerErrorMessage(
                df_index=df_index,
                epoch_index=getattr(message, "epoch_index", -1),
                context_index=getattr(message, "context_index", -1),
                initial_action=getattr(message, "initial_action", -1),
                round_counter=getattr(message, "round_counter", -1),
                traceback=traceback.format_exc(),
            )
        )
```

Update `reset_task` and `explore_round` to read attributes:

```python
message.initial_action
message.state_dict
message.rollout_steps
message.context_index
message.epsilon
```

Return `WorkerRoundResult` from `explore_round`:

```python
return WorkerRoundResult(
    df_index=self.df_index,
    epoch_index=message.epoch_index,
    context_index=message.context_index,
    initial_action=message.initial_action,
    round_counter=message.round_counter,
    worker_steps=len(transitions),
    transitions=transitions,
    rollout_metrics=[
        RolloutMetrics(
            epoch_index=message.epoch_index,
            context_index=message.context_index,
            initial_action=message.initial_action,
            df_index=self.df_index,
            transition_count=self.transition_count,
            reward_sum=float(self.reward_sum),
            final_balance=float(final_balance),
            return_rate=float(
                final_balance / (self.initial_wallet_balance + 1e-12) - 1
            ),
        )
    ],
    done=self.done,
    progress={"transition_count": self.transition_count},
)
```

When appending transitions inside `explore_round`, use:

```python
transitions.append(
    WorkerTransitionRecord(
        step_index=self.transition_count,
        transition=(
            self.state,
            self.info,
            action,
            reward,
            next_state,
            next_info,
            done,
        ),
    )
)
```

- [x] **Step 5: Convert trainer worker send/collect and round logging**

Replace queue sends:

```python
self.worker_input_queues[df_index].put(
    ResetWorkerTask(
        epoch_index=epoch_index,
        context_index=context_index,
        initial_action=initial_action,
    )
)
...
self.worker_input_queues[df_index].put(
    ExploreWorkerRound(
        epoch_index=epoch_index,
        context_index=context_index,
        initial_action=initial_action,
        round_counter=round_counter,
        state_dict=state_dict,
        epsilon=self.epsilon,
        rollout_steps=self.rollout_steps,
    )
)
```

Update shutdown:

```python
for queue in input_queues:
    queue.put(ShutdownWorker())
```

Update `_collect_worker_rounds`:

```python
message = self.worker_result_queue.get()
if isinstance(message, WorkerErrorMessage):
    return [message]
if message.round_counter != round_counter:
    raise RuntimeError(
        "unexpected worker round_counter={} expected={}".format(
            message.round_counter,
            round_counter,
        )
    )
if message.df_index not in active_df_indices:
    raise RuntimeError(
        "unexpected worker df_index={} active={}".format(
            message.df_index,
            sorted(active_df_indices),
        )
    )
results.append(message)
return sorted(results, key=lambda result: result.df_index)
```

Update round summary and logging:

```python
round_steps = sum(result.worker_steps for result in round_results)
...
round_summary = summarize_parallel_round(...)
logger.info(..., round_summary.round_counter, round_summary.epoch_index, ...)
for result in round_results:
    for metrics in result.rollout_metrics:
        logger.info(..., metrics.epoch_index, metrics.context_index, ...)
active_df_indices = {
    result.df_index
    for result in round_results
    if not result.done
}
```

Update `summarize_parallel_round`:

```python
return ParallelRoundSummary(
    round_counter=int(round_counter),
    epoch_index=int(epoch_index),
    context_index=int(context_index),
    initial_action=int(initial_action),
    round_steps=int(sum(result.worker_steps for result in round_results)),
    active_worker_count=int(len(round_results)),
    buffer_size=int(buffer_size),
    update_count=int(update_count),
)
```

- [x] **Step 6: Update epoch parameter usage**

In `_run_parallel_diverse_training`, replace dict access:

```python
self.epsilon = params.epsilon
self.ada = params.ada
self.lr = params.lr
```

- [x] **Step 7: Run parallel rollout verification**

Run: `conda activate finetf && pytest FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py -q`

Expected: PASS.

Run: `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py`

Expected: exits with status 0.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 6: Run focused verification

> **trace:** plan-ready.md -> `### Task 6: Run focused verification` | tasks.md -> `- [ ] 1.6 Run focused tests, Python compilation, and OpenSpec strict validation.`
> **sync:** tasks.md -> `- [ ] 1.6 Run focused tests, Python compilation, and OpenSpec strict validation.` | plan-ready.md -> `### Task 6: Run focused verification`

**Files:**
- Modify: `openspec/changes/refactor-rl-diagnostics-dataclasses/tasks.md`
- Modify: `openspec/changes/refactor-rl-diagnostics-dataclasses/plan-ready.md`
- Modify: `docs/superpowers/plans/2026-07-22-refactor-rl-diagnostics-dataclasses.md`

- [x] **Step 1: Run focused tests**

Run:

```bash
conda activate finetf && pytest FineFT/tests/rl/test_weight_advantage_pretrain_logging.py FineFT/tests/rl/test_pretrain_qtable_diagnostics.py FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py -q
```

Expected: PASS.

- [x] **Step 2: Run Python compilation checks**

Run:

```bash
conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/loss_nan_diagnostics.py FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py
```

Expected: exits with status 0 and prints no syntax errors.

- [x] **Step 3: Run OpenSpec strict validation**

Run:

```bash
openspec validate refactor-rl-diagnostics-dataclasses --strict
```

Expected: `Change 'refactor-rl-diagnostics-dataclasses' is valid`.

- [x] **Step 4: Check for stale anonymous business-record access**

Run:

```bash
rg -n "\\[\"(df_index|initial_action|worker_steps|round_counter|return_rate|reward_sum|final_balance|episode_reward_sum|profitable|csv_path|numeric|info_nonfinite)\"\\]" FineFT/RL/DiHFT/low_level/loss_nan_diagnostics.py FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py
```

Expected: output is limited to legitimate external dicts such as `env_kwargs`, `info`, `worker_config`, `qtable_kwargs`, `state_dict`, and `.to_dict()` compatibility tests. Any result touching dataclass-owned fields should be changed to attribute access.

- [x] **Step 5: Update task checkboxes after implementation**

After Tasks 1-5 have passed, update:

```markdown
openspec/changes/refactor-rl-diagnostics-dataclasses/tasks.md
openspec/changes/refactor-rl-diagnostics-dataclasses/plan-ready.md
docs/superpowers/plans/2026-07-22-refactor-rl-diagnostics-dataclasses.md
```

Change the corresponding task-level checkboxes from `[ ]` to `[x]` only for tasks whose tests and verification commands passed.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）
