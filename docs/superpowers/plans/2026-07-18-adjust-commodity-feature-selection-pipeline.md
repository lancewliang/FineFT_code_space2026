# Adjust Commodity Feature Selection Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move commodity feature selection after dataset split, let train produce the only final `state_features.npy`, run valid as evaluation/reporting only, and run scale-save from split-stage inputs using the train feature list.

**Architecture:** Keep existing contract-level preprocessing through `ALL_FEATURE` unchanged. Add a focused `operator_futures.feature_selection.muti_contract` package for split-input feature metrics, train filtering, valid reporting, and manifests. Keep `fu_full_process.sh` as orchestration only, and use `muti_contract_scale_save.py` as the commodity split-stage batch scale-save source of truth while preserving old `scale_save.py` / `IC_RESULT` behavior for old callers.

**Tech Stack:** Bash, Python, Polars, NumPy, pytest, CatBoost when installed, OpenSpec.

**Traceability (sddflow):**
- plan-ready: `openspec/changes/adjust-commodity-feature-selection-pipeline/plan-ready.md`
- tasks: `openspec/changes/adjust-commodity-feature-selection-pipeline/tasks.md`
- plan: `docs/superpowers/plans/2026-07-18-adjust-commodity-feature-selection-pipeline.md`

---

### Task 1: Add full-process ordering tests

> **trace:** plan-ready.md → `### Task 1: Add full-process ordering tests` | tasks.md → ``- [ ] 1.1 Add focused tests for the split-after-merge-clean full-process order: all contracts run through `merge_clean`, `dataset_split` runs once, `feature_selection_train` then `feature_selection_valid` run once, per-contract `scale_save` runs after valid feature selection, and old immediate post-`merge_clean` scale-save ordering is rejected.``
> **sync:** tasks.md → ``- [ ] 1.1 Add focused tests for the split-after-merge-clean full-process order: all contracts run through `merge_clean`, `dataset_split` runs once, `feature_selection_train` then `feature_selection_valid` run once, per-contract `scale_save` runs after valid feature selection, and old immediate post-`merge_clean` scale-save ordering is rejected.`` | plan-ready.md → `### Task 1: Add full-process ordering tests`

**Files:**
- Modify: `data_preprocess/tests/test_commodity_main_contract_cli.py`

- [x] **Step 1: Update the step-log stub test with feature-selection stubs**

In `data_preprocess/tests/test_commodity_main_contract_cli.py`, inside `test_commodity_full_process_writes_step_logs_and_preserves_child_log_paths`, replace the existing stub block that defines `run_commodity_scale_save`, `run_commodity_dataset_split`, and `run_commodity_maintenance_margin_dict` with this block:

```bash
run_commodity_feature_selection() {
    local stage=$1
    local split_root=$2
    local target_freq=$3
    local symbol=$4
    echo "feature_selection:${stage}:${symbol}:${target_freq}:${split_root}"
}
run_commodity_scale_save() {
    local target_freq=$1
    local start_date=$2
    local end_date=$3
    local symbol=$4
    local root_path=$5
    local contract=$6
    echo "scale_save:${symbol}:${contract}:${target_freq}:${start_date}:${end_date}:${root_path}"
}
run_commodity_dataset_split() {
    local summary_path=$1
    local target_freq=$2
    local start_date=$3
    local end_date=$4
    local symbol=$5
    echo "dataset_split:${symbol}:${target_freq}:${start_date}:${end_date}:${summary_path}"
}
run_commodity_maintenance_margin_dict() { echo "maintenance stdout"; }
```

- [x] **Step 2: Update expected step logs**

In the same test, replace `symbol_by_step` with this mapping:

```python
symbol_by_step = {
    "stitch_main_contract": "fu",
    "downscale_continuous_by_trading_day": "fu",
    "cross_section": "fu_fu2601",
    "merge": "fu_fu2601",
    "concat": "fu_fu2601",
    "time_feature": "fu_fu2601",
    "merge_clean": "fu_fu2601",
    "dataset_split": "fu",
    "feature_selection_train": "fu",
    "feature_selection_valid": "fu",
    "scale_save": "fu_fu2601",
    "maintenance_margin_dict": "fu",
}
```

After the existing `dataset_split_log` assertion, add these assertions:

```python
feature_train_log = (
    tmp_path
    / "log_futures"
    / "ticker_result"
    / "commodity"
    / "steps"
    / "fu_5min_2026-01-05_2026-01-07_feature_selection_train.log"
)
feature_valid_log = (
    tmp_path
    / "log_futures"
    / "ticker_result"
    / "commodity"
    / "steps"
    / "fu_5min_2026-01-05_2026-01-07_feature_selection_valid.log"
)
scale_log = (
    tmp_path
    / "log_futures"
    / "ticker_result"
    / "commodity"
    / "steps"
    / "fu_fu2601_5min_2026-01-05_2026-01-07_scale_save.log"
)
assert "feature_selection:train:fu:5min:" in feature_train_log.read_text(encoding="utf-8")
assert "feature_selection:valid:fu:5min:" in feature_valid_log.read_text(encoding="utf-8")
assert "scale_save:fu:fu2601:5min:2026-01-05:2026-01-07:" in scale_log.read_text(encoding="utf-8")
```

- [x] **Step 3: Replace the static ordering test**

Rename `test_commodity_full_process_shell_runs_scale_after_merge_clean_and_dataset_split_after_loop` to `test_commodity_full_process_shell_runs_scale_after_feature_selection_valid`, and replace its body with:

```python
def test_commodity_full_process_shell_runs_scale_after_feature_selection_valid():
    script = (
        REPO_ROOT
        / "data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh"
    )
    text = script.read_text(encoding="utf-8")

    assert '"dataset_split"' in text
    assert '"feature_selection_train"' in text
    assert '"feature_selection_valid"' in text
    assert '"feature_union"' not in text
    assert '"ic_candidate"' not in text
    assert '"ic_union_finalize"' not in text
    assert text.index('"merge_clean"') < text.index('"dataset_split"')
    assert text.index('"dataset_split"') < text.index('"feature_selection_train"')
    assert text.index('"feature_selection_train"') < text.index('"feature_selection_valid"')
    assert text.index('"feature_selection_valid"') < text.rindex('"scale_save"')
    assert text.rindex('"scale_save"') < text.index('"maintenance_margin_dict"')
```

- [x] **Step 4: Run the focused shell tests and confirm RED**

Run:

```bash
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest data_preprocess/tests/test_commodity_main_contract_cli.py::test_commodity_full_process_writes_step_logs_and_preserves_child_log_paths data_preprocess/tests/test_commodity_main_contract_cli.py::test_commodity_full_process_shell_runs_scale_after_feature_selection_valid -q'
```

Expected: FAIL because `fu_full_process.sh` still logs `scale_save` before `dataset_split` and does not define or call `feature_selection_train` / `feature_selection_valid`.

- [x] **Step 5: Commit the test changes**

Run:

```bash
git add data_preprocess/tests/test_commodity_main_contract_cli.py
git commit -m "test: cover commodity feature selection pipeline order"
```

Expected: Commit succeeds unless the working tree has unrelated staged changes; if unrelated staged changes exist, leave the commit for the final integration step.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 2: Add multi-contract feature selection tests

> **trace:** plan-ready.md → `### Task 2: Add multi-contract feature selection tests` | tasks.md → ``- [ ] 1.2 Add focused tests for a new multi-contract feature selection module covering train candidate output, valid candidate-restricted output, per-contract metric artifacts, aggregate `Mean` / `Std` / `Median` outputs, filtered contract `df.feather` outputs, manifest contents, and fail-fast behavior for missing input, empty candidate features, empty final features, and missing selected feature columns.``
> **sync:** tasks.md → ``- [ ] 1.2 Add focused tests for a new multi-contract feature selection module covering train candidate output, valid candidate-restricted output, per-contract metric artifacts, aggregate `Mean` / `Std` / `Median` outputs, filtered contract `df.feather` outputs, manifest contents, and fail-fast behavior for missing input, empty candidate features, empty final features, and missing selected feature columns.`` | plan-ready.md → `### Task 2: Add multi-contract feature selection tests`

**Files:**
- Create: `data_preprocess/tests/test_commodity_multi_contract_feature_selection.py`

- [x] **Step 1: Create focused tests for train, valid, metrics, and fail-fast**

Create `data_preprocess/tests/test_commodity_multi_contract_feature_selection.py` with this complete content:

```python
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from operator_futures.feature_selection.muti_contract.metrics import (
    aggregate_metric_frames,
    calculate_sharpe,
)
from operator_futures.feature_selection.muti_contract.pipeline import run_feature_selection


def _write_split_contract(root: Path, stage: str, contract: str, alpha, beta, gamma=None):
    stage_dir = (
        root
        / "PREPROCESS_DATASET"
        / "commodity-futures"
        / "SPLIT-TRAIN-VALID-TEST"
        / "5min"
        / "fu"
        / stage
    )
    stage_dir.mkdir(parents=True, exist_ok=True)
    values = {
        "timestamp": ["2026-01-01 09:00:00", "2026-01-01 09:05:00", "2026-01-01 09:10:00", "2026-01-01 09:15:00"],
        "trading_day": ["2026-01-01", "2026-01-01", "2026-01-01", "2026-01-01"],
        "contract": [contract, contract, contract, contract],
        "mark_price": [10.0, 11.0, 13.0, 16.0],
        "bid1_price": [9.9, 10.9, 12.9, 15.9],
        "ask1_price": [10.1, 11.1, 13.1, 16.1],
        "alpha": alpha,
        "beta": beta,
    }
    if gamma is not None:
        values["gamma"] = gamma
    frame = pl.DataFrame(values).with_columns(pl.col("timestamp").str.strptime(pl.Datetime))
    output = stage_dir / f"{contract}.feather"
    frame.write_ipc(output)
    return output


def test_calculate_sharpe_uses_single_feature_pseudo_returns():
    feature = np.array([1.0, 2.0, 3.0, 4.0])
    future_return = np.array([0.1, 0.2, -0.1, 0.3])

    result = calculate_sharpe(feature, future_return)

    z = (feature - feature.mean()) / feature.std(ddof=0)
    pseudo_returns = z * future_return
    expected = pseudo_returns.mean() / pseudo_returns.std(ddof=1)
    assert result == pytest.approx(expected)


def test_aggregate_metric_frames_writes_mean_std_median_columns():
    first = pl.DataFrame({"feature": ["alpha", "beta"], "IC": [0.5, 0.2], "Sharpe": [1.0, 0.5]})
    second = pl.DataFrame({"feature": ["alpha", "beta"], "IC": [0.7, 0.1], "Sharpe": [1.4, 0.4]})

    result = aggregate_metric_frames([first, second])

    alpha = result.filter(pl.col("feature") == "alpha").row(0, named=True)
    assert alpha["IC_Mean"] == pytest.approx(0.6)
    assert alpha["IC_Median"] == pytest.approx(0.6)
    assert alpha["IC_Std"] == pytest.approx(np.std([0.5, 0.7], ddof=1))
    assert "Sharpe_Mean" in result.columns
    assert "Sharpe_Std" in result.columns
    assert "Sharpe_Median" in result.columns


def test_train_stage_writes_candidates_metrics_filtered_outputs_and_manifest(tmp_path):
    _write_split_contract(tmp_path, "train", "fu2601", [1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0])
    _write_split_contract(tmp_path, "train", "fu2605", [2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0])

    manifest = run_feature_selection(
        root_path=tmp_path,
        split_path="PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST",
        save_path="PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION",
        symbol="fu",
        target_freq="5min",
        stage="train",
        orderbook_depth=5,
        min_abs_ic=0.01,
        max_correlation=0.99,
    )

    stage_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/train"
    candidates = np.load(stage_dir / "state_features_candidate.npy", allow_pickle=True).tolist()
    assert candidates
    assert (stage_dir / "per_contract" / "fu2601_metrics.csv").exists()
    assert (stage_dir / "per_contract" / "fu2605_metrics.csv").exists()
    assert (stage_dir / "aggregate_metrics.csv").exists()
    assert (stage_dir / "feature_selection_manifest.json").exists()
    assert (stage_dir / "fu2601" / "df.feather").exists()
    assert manifest["stage"] == "train"
    assert manifest["selected_feature_file"].endswith("state_features_candidate.npy")
    assert manifest["selected_feature_count"] == len(candidates)
    metrics = pl.read_csv(stage_dir / "aggregate_metrics.csv")
    assert {"IC_Mean", "IC_Std", "IC_Median", "Sharpe_Mean", "Sharpe_Std", "Sharpe_Median"}.issubset(metrics.columns)


def test_valid_stage_uses_train_candidates_and_writes_final_features(tmp_path):
    _write_split_contract(tmp_path, "valid", "fu2601", [1.0, 2.0, 3.0, 4.0], [4.0, 4.0, 4.0, 4.0], gamma=[9.0, 8.0, 7.0, 6.0])
    train_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/train"
    train_dir.mkdir(parents=True)
    np.save(train_dir / "state_features_candidate.npy", np.array(["alpha"]))

    manifest = run_feature_selection(
        root_path=tmp_path,
        split_path="PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST",
        save_path="PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION",
        symbol="fu",
        target_freq="5min",
        stage="valid",
        orderbook_depth=5,
        min_abs_ic=0.01,
        max_correlation=0.99,
    )

    stage_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/valid"
    selected = np.load(stage_dir / "state_features.npy", allow_pickle=True).tolist()
    assert selected == ["alpha"]
    filtered = pl.read_ipc(stage_dir / "fu2601" / "df.feather")
    assert "alpha" in filtered.columns
    assert "gamma" not in filtered.columns
    assert filtered.get_column("symbol").unique().to_list() == ["fu"]
    assert manifest["candidate_feature_file"].endswith("state_features_candidate.npy")
    assert manifest["selected_feature_file"].endswith("state_features.npy")


def test_feature_selection_fails_for_missing_split_input(tmp_path):
    with pytest.raises(FileNotFoundError, match="split input directory"):
        run_feature_selection(
            root_path=tmp_path,
            split_path="PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST",
            save_path="PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION",
            symbol="fu",
            target_freq="5min",
            stage="train",
            orderbook_depth=5,
        )


def test_valid_stage_fails_when_candidate_file_is_empty(tmp_path):
    _write_split_contract(tmp_path, "valid", "fu2601", [1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0])
    train_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/train"
    train_dir.mkdir(parents=True)
    np.save(train_dir / "state_features_candidate.npy", np.array([]))

    with pytest.raises(ValueError, match="candidate feature list is empty"):
        run_feature_selection(
            root_path=tmp_path,
            split_path="PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST",
            save_path="PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION",
            symbol="fu",
            target_freq="5min",
            stage="valid",
            orderbook_depth=5,
        )


def test_valid_stage_fails_when_candidate_column_is_missing(tmp_path):
    _write_split_contract(tmp_path, "valid", "fu2601", [1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0])
    train_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/train"
    train_dir.mkdir(parents=True)
    np.save(train_dir / "state_features_candidate.npy", np.array(["missing_alpha"]))

    with pytest.raises(ValueError, match="missing_alpha"):
        run_feature_selection(
            root_path=tmp_path,
            split_path="PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST",
            save_path="PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION",
            symbol="fu",
            target_freq="5min",
            stage="valid",
            orderbook_depth=5,
        )
```

- [x] **Step 2: Run the new tests and confirm RED**

Run:

```bash
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest data_preprocess/tests/test_commodity_multi_contract_feature_selection.py -q'
```

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'operator_futures.feature_selection.muti_contract'`.

- [x] **Step 3: Commit the test file**

Run:

```bash
git add data_preprocess/tests/test_commodity_multi_contract_feature_selection.py
git commit -m "test: cover commodity split feature selection"
```

Expected: Commit succeeds unless unrelated staged changes exist.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 3: Implement multi-contract feature selection module

> **trace:** plan-ready.md → `### Task 3: Implement multi-contract feature selection module` | tasks.md → ``- [ ] 1.3 Implement `data_preprocess/operator_futures/feature_selection/muti_contract/` with metric helpers for `Permutation Importance`, `CatBoost Importance`, `IC`, `RankIC`, `Sharpe`, aggregation helpers, ordered filters (`Hard Filter`, `Stability Filter`, `Composite Score`, `Correlation Filter`), manifest writing, and a CLI that supports `--stage train` and `--stage valid`.``
> **sync:** tasks.md → ``- [ ] 1.3 Implement `data_preprocess/operator_futures/feature_selection/muti_contract/` with metric helpers for `Permutation Importance`, `CatBoost Importance`, `IC`, `RankIC`, `Sharpe`, aggregation helpers, ordered filters (`Hard Filter`, `Stability Filter`, `Composite Score`, `Correlation Filter`), manifest writing, and a CLI that supports `--stage train` and `--stage valid`.`` | plan-ready.md → `### Task 3: Implement multi-contract feature selection module`

**Files:**
- Create: `data_preprocess/operator_futures/feature_selection/muti_contract/__init__.py`
- Create: `data_preprocess/operator_futures/feature_selection/muti_contract/metrics.py`
- Create: `data_preprocess/operator_futures/feature_selection/muti_contract/pipeline.py`
- Create: `data_preprocess/operator_futures/feature_selection/muti_contract/__main__.py`
- Test: `data_preprocess/tests/test_commodity_multi_contract_feature_selection.py`

- [x] **Step 1: Create the package init file**

Create `data_preprocess/operator_futures/feature_selection/muti_contract/__init__.py` with this content:

```python
from .pipeline import run_feature_selection

__all__ = ["run_feature_selection"]
```

- [x] **Step 2: Create metric helpers**

Create `data_preprocess/operator_futures/feature_selection/muti_contract/metrics.py` with this content:

```python
from __future__ import annotations

import numpy as np
import polars as pl


METRIC_COLUMNS = [
    "Permutation Importance",
    "CatBoost Importance",
    "IC",
    "RankIC",
    "Sharpe",
]


def _safe_corr(left, right) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    valid = ~(np.isnan(left) | np.isnan(right))
    left = left[valid]
    right = right[valid]
    if left.size < 2 or right.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _rank(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def calculate_future_return(df: pl.DataFrame, window: int = 1) -> np.ndarray:
    target = df.select((pl.col("mark_price").shift(-window) - pl.col("mark_price")).alias("target"))["target"]
    return target.slice(0, max(target.len() - window, 0)).to_numpy()


def calculate_sharpe(feature_values, future_return) -> float:
    feature_values = np.asarray(feature_values, dtype=float)
    future_return = np.asarray(future_return, dtype=float)
    size = min(feature_values.size, future_return.size)
    feature_values = feature_values[:size]
    future_return = future_return[:size]
    valid = ~(np.isnan(feature_values) | np.isnan(future_return))
    feature_values = feature_values[valid]
    future_return = future_return[valid]
    if feature_values.size < 2 or np.std(feature_values) == 0:
        return 0.0
    zscore = (feature_values - feature_values.mean()) / feature_values.std(ddof=0)
    pseudo_returns = zscore * future_return
    std = pseudo_returns.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(pseudo_returns.mean() / std)


def _permutation_importance(feature_values, future_return) -> float:
    baseline = abs(_safe_corr(feature_values, future_return))
    shuffled = np.asarray(feature_values, dtype=float).copy()
    if shuffled.size:
        shuffled = np.roll(shuffled, 1)
    return float(max(baseline - abs(_safe_corr(shuffled, future_return)), 0.0))


def _catboost_importance(df: pl.DataFrame, features: list[str], future_return: np.ndarray) -> dict[str, float]:
    if not features:
        return {}
    try:
        from catboost import CatBoostRegressor, Pool
    except Exception:
        return {feature: abs(_safe_corr(df[feature].to_numpy()[: future_return.size], future_return)) for feature in features}
    model_df = df.slice(0, future_return.size)
    x = model_df.select(features).to_numpy()
    y = np.asarray(future_return, dtype=float)
    model = CatBoostRegressor(
        iterations=20,
        learning_rate=0.1,
        depth=3,
        loss_function="RMSE",
        task_type="CPU",
        random_seed=42,
        verbose=False,
    )
    pool = Pool(x, y, feature_names=features)
    model.fit(pool, verbose=False)
    values = model.get_feature_importance(pool)
    return {feature: float(value) for feature, value in zip(features, values)}


def calculate_metric_frame(df: pl.DataFrame, features: list[str], *, window: int = 1) -> pl.DataFrame:
    future_return = calculate_future_return(df, window)
    if future_return.size == 0:
        raise ValueError("future return is empty; cannot calculate feature metrics")
    catboost_values = _catboost_importance(df, features, future_return)
    rows = []
    metric_df = df.slice(0, future_return.size)
    for feature in features:
        values = metric_df[feature].to_numpy()
        rows.append(
            {
                "feature": feature,
                "Permutation Importance": _permutation_importance(values, future_return),
                "CatBoost Importance": catboost_values.get(feature, 0.0),
                "IC": _safe_corr(values, future_return),
                "RankIC": _safe_corr(_rank(values), _rank(future_return)),
                "Sharpe": calculate_sharpe(values, future_return),
            }
        )
    return pl.DataFrame(rows)


def aggregate_metric_frames(frames: list[pl.DataFrame]) -> pl.DataFrame:
    if not frames:
        raise ValueError("cannot aggregate empty metric frame list")
    combined = pl.concat(frames, how="vertical")
    expressions = []
    for metric in METRIC_COLUMNS:
        expressions.extend(
            [
                pl.col(metric).mean().alias(f"{metric}_Mean"),
                pl.col(metric).std().fill_null(0.0).alias(f"{metric}_Std"),
                pl.col(metric).median().alias(f"{metric}_Median"),
            ]
        )
    return combined.group_by("feature", maintain_order=True).agg(expressions)
```

- [x] **Step 3: Create the pipeline implementation**

Create `data_preprocess/operator_futures/feature_selection/muti_contract/pipeline.py` with this content:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from operator_futures.commodity.schema import get_reward_execution_columns
from operator_futures.feature_selection.cor_util import select_feature
from operator_futures.feature_selection.muti_contract.metrics import (
    aggregate_metric_frames,
    calculate_metric_frame,
)


NON_STATE_COLUMNS = {"timestamp", "trading_day", "TradingDay", "symbol", "contract"}


def _stage_input_dir(root_path: Path, split_path: str, target_freq: str, symbol: str, stage: str) -> Path:
    return root_path / split_path / target_freq / symbol / stage


def _stage_output_dir(root_path: Path, save_path: str, target_freq: str, symbol: str, stage: str) -> Path:
    return root_path / save_path / target_freq / symbol / stage


def _load_contract_frames(input_dir: Path) -> dict[str, pl.DataFrame]:
    if not input_dir.exists():
        raise FileNotFoundError(f"split input directory does not exist: {input_dir}")
    paths = sorted(input_dir.glob("*.feather"))
    if not paths:
        raise FileNotFoundError(f"split input directory contains no contract feather files: {input_dir}")
    frames = {}
    for path in paths:
        contract = path.stem
        frames[contract] = pl.read_ipc(path)
    return frames


def _state_features(df: pl.DataFrame, *, orderbook_depth: int) -> list[str]:
    reward = set(get_reward_execution_columns(orderbook_depth))
    return [column for column in df.columns if column not in reward and column not in NON_STATE_COLUMNS]


def _load_candidates(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"candidate feature file does not exist: {path}")
    values = np.load(path, allow_pickle=True).tolist()
    values = [str(value) for value in values]
    if not values:
        raise ValueError(f"candidate feature list is empty: {path}")
    return values


def _ordered_filter_features(
    frames: dict[str, pl.DataFrame],
    aggregate: pl.DataFrame,
    feature_universe: list[str],
    *,
    min_abs_ic: float,
    max_metric_std: float,
    max_correlation: float,
) -> tuple[list[str], dict[str, list[str]]]:
    selected = aggregate.filter(pl.col("feature").is_in(feature_universe))
    hard = selected.filter(pl.col("IC_Mean").abs() >= min_abs_ic)["feature"].to_list()
    if not hard:
        raise ValueError("feature selection produced an empty list after Hard Filter")
    stability = selected.filter(pl.col("feature").is_in(hard)).filter(pl.col("IC_Std") <= max_metric_std)["feature"].to_list()
    if not stability:
        raise ValueError("feature selection produced an empty list after Stability Filter")
    scored = (
        selected.filter(pl.col("feature").is_in(stability))
        .with_columns(
            (
                pl.col("Permutation Importance_Mean").fill_null(0.0)
                + pl.col("CatBoost Importance_Mean").fill_null(0.0)
                + pl.col("IC_Mean").abs().fill_null(0.0)
                + pl.col("RankIC_Mean").abs().fill_null(0.0)
                + pl.col("Sharpe_Mean").abs().fill_null(0.0)
            ).alias("Composite Score")
        )
        .sort("Composite Score", descending=True)
    )
    composite = scored["feature"].to_list()
    if not composite:
        raise ValueError("feature selection produced an empty list after Composite Score")
    combined = pl.concat([frame.select(composite) for frame in frames.values()], how="vertical")
    correlation = select_feature(features=composite, df=combined, theshold=max_correlation)
    if not correlation:
        raise ValueError("feature selection produced an empty list after Correlation Filter")
    return correlation, {
        "Hard Filter": hard,
        "Stability Filter": stability,
        "Composite Score": composite,
        "Correlation Filter": correlation,
    }


def _write_filtered_outputs(
    frames: dict[str, pl.DataFrame],
    output_dir: Path,
    selected_features: list[str],
    *,
    symbol: str,
    orderbook_depth: int,
) -> list[dict[str, object]]:
    reward_columns = get_reward_execution_columns(orderbook_depth)
    outputs = []
    for contract, frame in frames.items():
        missing = [feature for feature in selected_features if feature not in frame.columns]
        if missing:
            raise ValueError(f"contract {contract} is missing selected feature columns: {missing}")
        reward_present = [column for column in reward_columns if column in frame.columns]
        columns = [*reward_present, *selected_features]
        filtered = frame.select(columns).with_columns(pl.lit(symbol).alias("symbol"))
        contract_dir = output_dir / contract
        contract_dir.mkdir(parents=True, exist_ok=True)
        output_path = contract_dir / "df.feather"
        filtered.write_ipc(output_path)
        outputs.append(
            {
                "contract": contract,
                "output_path": str(output_path),
                "output_row_count": filtered.height,
                "output_column_count": len(filtered.columns),
            }
        )
    return outputs


def run_feature_selection(
    *,
    root_path,
    split_path: str = "PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST",
    save_path: str = "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION",
    symbol: str,
    target_freq: str,
    stage: str,
    orderbook_depth: int = 5,
    min_abs_ic: float = 0.01,
    max_metric_std: float = 1.0,
    max_correlation: float = 0.7,
) -> dict[str, object]:
    if stage not in {"train", "valid"}:
        raise ValueError("stage must be 'train' or 'valid'")
    root_path = Path(root_path)
    input_dir = _stage_input_dir(root_path, split_path, target_freq, symbol, stage)
    output_dir = _stage_output_dir(root_path, save_path, target_freq, symbol, stage)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = _load_contract_frames(input_dir)

    if stage == "train":
        first_frame = next(iter(frames.values()))
        feature_universe = _state_features(first_frame, orderbook_depth=orderbook_depth)
        selected_file = output_dir / "state_features_candidate.npy"
        candidate_file = None
    else:
        candidate_file = _stage_output_dir(root_path, save_path, target_freq, symbol, "train") / "state_features_candidate.npy"
        feature_universe = _load_candidates(candidate_file)
        selected_file = output_dir / "state_features.npy"
    if not feature_universe:
        raise ValueError(f"{stage} feature universe is empty")

    per_contract_dir = output_dir / "per_contract"
    per_contract_dir.mkdir(parents=True, exist_ok=True)
    metric_frames = []
    per_contract = []
    for contract, frame in frames.items():
        missing = [feature for feature in feature_universe if feature not in frame.columns]
        if missing:
            raise ValueError(f"contract {contract} is missing required feature columns: {missing}")
        metrics = calculate_metric_frame(frame, feature_universe)
        metric_path = per_contract_dir / f"{contract}_metrics.csv"
        metrics.write_csv(metric_path)
        metric_frames.append(metrics)
        per_contract.append({"contract": contract, "input_path": str(input_dir / f"{contract}.feather"), "metric_path": str(metric_path)})

    aggregate = aggregate_metric_frames(metric_frames)
    aggregate_path = output_dir / "aggregate_metrics.csv"
    aggregate.write_csv(aggregate_path)
    selected_features, filter_results = _ordered_filter_features(
        frames,
        aggregate,
        feature_universe,
        min_abs_ic=min_abs_ic,
        max_metric_std=max_metric_std,
        max_correlation=max_correlation,
    )
    np.save(selected_file, np.array(selected_features))
    filtered_outputs = _write_filtered_outputs(
        frames,
        output_dir,
        selected_features,
        symbol=symbol,
        orderbook_depth=orderbook_depth,
    )
    manifest = {
        "symbol": symbol,
        "target_freq": target_freq,
        "stage": stage,
        "split_input_dir": str(input_dir),
        "candidate_feature_file": str(candidate_file) if candidate_file is not None else None,
        "selected_feature_file": str(selected_file),
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "aggregate_metrics_path": str(aggregate_path),
        "filter_results": filter_results,
        "contracts": per_contract,
        "filtered_outputs": filtered_outputs,
    }
    manifest_path = output_dir / "feature_selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=Path, default=Path("."))
    parser.add_argument("--split_path", type=str, default="PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST")
    parser.add_argument("--save_path", type=str, default="PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION")
    parser.add_argument("--symbol", "--symbols", dest="symbol", type=str, required=True)
    parser.add_argument("--target_freq", type=str, required=True)
    parser.add_argument("--stage", choices=["train", "valid"], required=True)
    parser.add_argument("--orderbook_depth", type=int, default=5)
    parser.add_argument("--min_abs_ic", type=float, default=0.01)
    parser.add_argument("--max_metric_std", type=float, default=1.0)
    parser.add_argument("--max_correlation", type=float, default=0.7)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_feature_selection(
        root_path=args.root_path,
        split_path=args.split_path,
        save_path=args.save_path,
        symbol=args.symbol,
        target_freq=args.target_freq,
        stage=args.stage,
        orderbook_depth=args.orderbook_depth,
        min_abs_ic=args.min_abs_ic,
        max_metric_std=args.max_metric_std,
        max_correlation=args.max_correlation,
    )


if __name__ == "__main__":
    main()
```

- [x] **Step 4: Create the module entrypoint**

Create `data_preprocess/operator_futures/feature_selection/muti_contract/__main__.py` with this content:

```python
from .pipeline import main


if __name__ == "__main__":
    main()
```

- [x] **Step 5: Run the feature selection tests**

Run:

```bash
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest data_preprocess/tests/test_commodity_multi_contract_feature_selection.py -q'
```

Expected: PASS.

- [x] **Step 6: Commit the module**

Run:

```bash
git add data_preprocess/operator_futures/feature_selection/muti_contract data_preprocess/tests/test_commodity_multi_contract_feature_selection.py
git commit -m "feat: add commodity split feature selection"
```

Expected: Commit succeeds unless unrelated staged changes exist.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 4: Update commodity full-process orchestration

> **trace:** plan-ready.md → `### Task 4: Update commodity full-process orchestration` | tasks.md → ``- [ ] 1.4 Update `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh` so dataset split reads `ALL_FEATURE`, feature selection runs after dataset split, scale-save runs after valid feature selection, and step logs include `feature_selection_train` and `feature_selection_valid`.``
> **sync:** tasks.md → ``- [ ] 1.4 Update `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh` so dataset split reads `ALL_FEATURE`, feature selection runs after dataset split, scale-save runs after valid feature selection, and step logs include `feature_selection_train` and `feature_selection_valid`.`` | plan-ready.md → `### Task 4: Update commodity full-process orchestration`

**Files:**
- Modify: `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`
- Test: `data_preprocess/tests/test_commodity_main_contract_cli.py`

- [x] **Step 1: Change dataset split to read ALL_FEATURE**

In `run_commodity_dataset_split`, change the `--input_root` argument to:

```bash
        --input_root "${root_path}/PREPROCESS_DATASET/commodity-futures/ALL_FEATURE" \
```

- [x] **Step 2: Add feature selection shell function**

Add this function after `run_commodity_dataset_split()`:

```bash
run_commodity_feature_selection() {
    local stage=$1
    local split_root=$2
    local target_freq=$3
    local symbol=$4
    local root_path=$5

    PYTHONPATH="${root_path}/data_preprocess${PYTHONPATH:+:${PYTHONPATH}}" python -u -m operator_futures.feature_selection.muti_contract \
        --root_path "${root_path}" \
        --split_path "PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST" \
        --save_path "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION" \
        --symbol "${symbol}" \
        --target_freq "${target_freq}" \
        --stage "${stage}" \
        --orderbook_depth 5
}
```

- [x] **Step 3: Change scale-save data path to FEATURE_SELECTION valid layout**

In `run_commodity_scale_save`, change the `--data_path` argument and add `--feature_selection_stage valid`:

```bash
        --data_path "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION" \
        --save_path "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/" \
        --feature_selection_stage valid \
```

Keep `--ic_choice ic` so old scale-save naming remains stable when the feature-selection-stage flag is absent.

- [x] **Step 4: Reorder the full process**

In `run_commodity_full_process`, remove the `scale_save` logged step from the first contract loop. After the post-loop `dataset_split` logged step, add:

```bash
    run_commodity_logged_step \
        "$log_dir" "$symbol" "$target_freq" "$start_date" "$end_date" \
        "feature_selection_train" \
        run_commodity_feature_selection "train" "${root_path}/PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/${target_freq}" "$target_freq" "$symbol" "$root_path"

    run_commodity_logged_step \
        "$log_dir" "$symbol" "$target_freq" "$start_date" "$end_date" \
        "feature_selection_valid" \
        run_commodity_feature_selection "valid" "${root_path}/PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/${target_freq}" "$target_freq" "$symbol" "$root_path"

    while IFS= read -r contract; do
        [ -n "$contract" ] || continue
        run_commodity_logged_step \
            "$log_dir" "${symbol}_${contract}" "$target_freq" "$start_date" "$end_date" \
            "scale_save" \
            run_commodity_scale_save "$target_freq" "$start_date" "$end_date" "$symbol" "$root_path" "$contract"
    done < <(run_commodity_summary_contracts "$summary_path")
```

The final `maintenance_margin_dict` logged step remains after this new second contract loop.

- [x] **Step 5: Run shell syntax and focused shell tests**

Run:

```bash
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest data_preprocess/tests/test_commodity_main_contract_cli.py::test_commodity_full_process_writes_step_logs_and_preserves_child_log_paths data_preprocess/tests/test_commodity_main_contract_cli.py::test_commodity_full_process_shell_runs_scale_after_feature_selection_valid -q'
```

Expected: both commands PASS.

- [x] **Step 6: Commit orchestration changes**

Run:

```bash
git add data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh data_preprocess/tests/test_commodity_main_contract_cli.py
git commit -m "feat: reorder commodity feature selection pipeline"
```

Expected: Commit succeeds unless unrelated staged changes exist.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 5: Add scale-save filtered input routing

> **trace:** plan-ready.md → `### Task 5: Add scale-save filtered input routing` | tasks.md → ``- [ ] 1.5 Update `data_preprocess/operator_futures/scale_describe_save/scale_save.py` routing so commodity full process can read filtered `FEATURE_SELECTION/{target_freq}/{symbol}/valid/{contract}/df.feather` and matching final `state_features.npy`, while preserving existing `IC_RESULT` behavior for old callers.``
> **sync:** tasks.md → ``- [ ] 1.5 Update `data_preprocess/operator_futures/scale_describe_save/scale_save.py` routing so commodity full process can read filtered `FEATURE_SELECTION/{target_freq}/{symbol}/valid/{contract}/df.feather` and matching final `state_features.npy`, while preserving existing `IC_RESULT` behavior for old callers.`` | plan-ready.md → `### Task 5: Add scale-save filtered input routing`

**Files:**
- Modify: `data_preprocess/operator_futures/scale_describe_save/scale_save.py`
- Modify: `data_preprocess/tests/test_feature_selection_polars.py`

- [x] **Step 1: Add a filtered-input CLI test**

In `data_preprocess/tests/test_feature_selection_polars.py`, add this test near the existing scale-save CLI tests:

```python
def test_scale_save_cli_reads_feature_selection_filtered_input(tmp_path):
    input_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/valid/fu2601"
    input_file = input_dir / "df.feather"
    _write_scale_fixture(input_file)
    np.save(input_dir.parent / "state_features.npy", np.array(["feature_a"]))

    result = subprocess.run(
        [
            sys.executable,
            "data_preprocess/operator_futures/scale_describe_save/scale_save.py",
            "--root_path",
            str(tmp_path),
            "--data_path",
            "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION",
            "--save_path",
            "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/",
            "--symbols",
            "fu",
            "--contract",
            "fu2601",
            "--target_freq",
            "5min",
            "--start_date",
            "2026-01-05",
            "--end_date",
            "2026-01-06",
            "--market_type",
            "commodity_futures",
            "--orderbook_depth",
            "5",
            "--ic_choice",
            "ic",
            "--feature_selection_stage",
            "valid",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        check=True,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    output_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/fu/fu2601/5min/2026-01-05-2026-01-06"
    assert (output_dir / "df.feather").exists()
    assert np.load(output_dir / "state_features.npy", allow_pickle=True).tolist() == ["feature_a"]
```

- [x] **Step 2: Run the new test and confirm RED**

Run:

```bash
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest data_preprocess/tests/test_feature_selection_polars.py::test_scale_save_cli_reads_feature_selection_filtered_input -q'
```

Expected: FAIL because `scale_save.py` does not accept `--feature_selection_stage` yet.

- [x] **Step 3: Add the parser argument**

In `data_preprocess/operator_futures/scale_describe_save/scale_save.py`, add this parser argument after `--orderbook_depth`:

```python
parser.add_argument(
    "--feature_selection_stage",
    type=str,
    default=None,
    choices=["train", "valid"],
    help="read filtered commodity FEATURE_SELECTION output for the selected stage",
)
```

- [x] **Step 4: Add input path routing helper**

Add this function before `main(args)`:

```python
def resolve_scale_input_paths(args, symbol_parts):
    if args.feature_selection_stage is None:
        input_dir = Path(args.data_path).joinpath(*symbol_parts, args.target_freq) / f"{args.start_date}-{args.end_date}"
        if args.ic_choice == "ic":
            df_name = "df"
            state_name = "state_features"
        elif args.ic_choice == "rank_ic":
            df_name = "df_rank"
            state_name = "state_features_rank"
        else:
            df_name = "df_catboost"
            state_name = "state_features_catboost"
        return input_dir / f"{df_name}.feather", input_dir / f"{state_name}.npy"

    if args.contract is None:
        raise ValueError("--feature_selection_stage requires --contract")
    stage_dir = Path(args.data_path).joinpath(args.target_freq, args.symbols, args.feature_selection_stage)
    return stage_dir / args.contract / "df.feather", stage_dir / "state_features.npy"
```

- [x] **Step 5: Use the helper in main**

In `main(args)`, replace the existing `df_name` / `state_name` branch and `input_dir` assignment with:

```python
    assert args.ic_choice in ["ic", "rank_ic", "catboost"]
    input_file, state_features_file = resolve_scale_input_paths(args, symbol_parts)
    output_dir = Path(args.save_path).joinpath(*symbol_parts, args.target_freq) / f"{args.start_date}-{args.end_date}"
    output_dir.mkdir(parents=True, exist_ok=True)
```

Then replace:

```python
    input_file = input_dir / f"{df_name}.feather"
    df = pl.read_ipc(input_file)
```

with:

```python
    df = pl.read_ipc(input_file)
```

Replace:

```python
    state_feature = np.load(input_dir / f"{state_name}.npy", allow_pickle=True).tolist()
```

with:

```python
    state_feature = np.load(state_features_file, allow_pickle=True).tolist()
```

In the logger call, replace `input_dir=%s` with `input_file=%s`, and pass `input_file` instead of `input_dir`.

- [x] **Step 6: Run scale-save routing tests**

Run:

```bash
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest data_preprocess/tests/test_feature_selection_polars.py::test_scale_save_cli_reads_feature_selection_filtered_input data_preprocess/tests/test_feature_selection_polars.py::test_scale_save_cli_writes_expected_files -q'
```

Expected: PASS. The new feature-selection layout and old `IC_RESULT` layout both work.

- [x] **Step 7: Commit scale-save routing changes**

Run:

```bash
git add data_preprocess/operator_futures/scale_describe_save/scale_save.py data_preprocess/tests/test_feature_selection_polars.py
git commit -m "feat: let scale-save read filtered feature selection output"
```

Expected: Commit succeeds unless unrelated staged changes exist.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 6: Update commodity pipeline documentation

> **trace:** plan-ready.md → `### Task 6: Update commodity pipeline documentation` | tasks.md → ``- [ ] 1.6 Update focused documentation for the commodity preprocessing pipeline to describe `dataset_split -> feature_selection(train) -> feature_selection(valid) -> scale_save -> maintenance_margin_dict` and the `FEATURE_SELECTION/{target_freq}` artifact layout.``
> **sync:** tasks.md → ``- [ ] 1.6 Update focused documentation for the commodity preprocessing pipeline to describe `dataset_split -> feature_selection(train) -> feature_selection(valid) -> scale_save -> maintenance_margin_dict` and the `FEATURE_SELECTION/{target_freq}` artifact layout.`` | plan-ready.md → `### Task 6: Update commodity pipeline documentation`

**Files:**
- Modify: `docs/datahandler/data_preparation_analysis.zh_cn.md`
- Modify: `docs/上海商品交易所/commodity_futures_preprocess.md`

- [x] **Step 1: Update data preparation analysis**

In `docs/datahandler/data_preparation_analysis.zh_cn.md`, replace the first paragraph under `## 商品期货多合约 FineFT 数据集` with:

```markdown
商品期货预处理主流程在所有合约完成 `ALL_FEATURE` 后，先运行第 9 阶段 `dataset_split`。该阶段读取 `main_contract_summary.json` 和合约级 `ALL_FEATURE/{symbol}/{contract}/{target_freq}/{start_date}-{end_date}.feather`，按 summary 中所有合约有效交易日的去重有序并集计算全局边界：
```

After the stage output path block, add:

````markdown
`dataset_split` 完成后，商品主流程进入 split 后特征选择：

```text
dataset_split -> feature_selection(train) -> feature_selection(valid) -> scale_save -> maintenance_margin_dict
```

`feature_selection(train)` 读取 `SPLIT-TRAIN-VALID-TEST/{target_freq}/{symbol}/train/*.feather`，计算每合约 `Permutation Importance`、`CatBoost Importance`、`IC`、`RankIC`、`Sharpe`，汇总 `Mean`、`Std`、`Median`，并写出候选特征：

```text
PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/{target_freq}/{symbol}/train/state_features_candidate.npy
```

`feature_selection(valid)` 读取 valid split，并且只使用 train 候选特征集合，写出最终特征和筛选后的合约级输入：

```text
PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/{target_freq}/{symbol}/valid/state_features.npy
PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/{target_freq}/{symbol}/valid/{contract}/df.feather
```

随后 `scale_save` 从 filtered valid feature selection 输出读取 `df.feather` 和最终 `state_features.npy`，最终仍写入 `PREPROCESS_DATASET/commodity-futures/SCALE_SAVE`。
````

- [x] **Step 2: Update commodity preprocess docs**

In `docs/上海商品交易所/commodity_futures_preprocess.md`, replace the paragraph beginning with `然后继续执行商品期货下采样` through the `FEATURE_UNION` path block with:

````markdown
然后继续执行商品期货下采样、cross-section、merge/concat、time feature、merge clean、dataset split、train feature selection、valid feature selection 和 scale/save。下游中间输出按合约增加一层目录；split 后特征选择统一写入：

```text
PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/train/state_features_candidate.npy
PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/valid/state_features.npy
PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/valid/fu2601/df.feather
```

最终训练入口数据仍写入：

```text
PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/fu/fu2601/5min/2026-01-01-2026-02-01/
```
````

- [x] **Step 3: Verify documentation text**

Run:

```bash
rg -n "FEATURE_SELECTION|feature_selection\(train\)|feature_selection\(valid\)|SPLIT-TRAIN-VALID-TEST|dataset_split -> feature_selection" docs/datahandler/data_preparation_analysis.zh_cn.md docs/上海商品交易所/commodity_futures_preprocess.md
```

Expected: Output includes the new pipeline order and `FEATURE_SELECTION` paths in both files.

- [x] **Step 4: Commit documentation changes**

Run:

```bash
git add docs/datahandler/data_preparation_analysis.zh_cn.md docs/上海商品交易所/commodity_futures_preprocess.md
git commit -m "docs: describe commodity feature selection pipeline"
```

Expected: Commit succeeds unless unrelated staged changes exist.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 7: Run OpenSpec validation

> **trace:** plan-ready.md → `### Task 7: Run OpenSpec validation` | tasks.md → ``- [ ] 2.1 Run strict OpenSpec validation for `adjust-commodity-feature-selection-pipeline`.``
> **sync:** tasks.md → ``- [ ] 2.1 Run strict OpenSpec validation for `adjust-commodity-feature-selection-pipeline`.`` | plan-ready.md → `### Task 7: Run OpenSpec validation`

**Files:**
- Verify: `openspec/changes/adjust-commodity-feature-selection-pipeline/`

- [x] **Step 1: Run strict OpenSpec validation**

Run:

```bash
openspec validate adjust-commodity-feature-selection-pipeline --strict
```

Expected: PASS with `Change 'adjust-commodity-feature-selection-pipeline' is valid`.

- [x] **Step 2: Inspect deltas if validation fails**

If validation fails, run:

```bash
openspec show adjust-commodity-feature-selection-pipeline --json --deltas-only
```

Expected: JSON output identifies the invalid delta. Fix the reported requirement/scenario formatting in `openspec/changes/adjust-commodity-feature-selection-pipeline/specs/**/spec.md`, then rerun Step 1.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 8: Run focused pytest validation

> **trace:** plan-ready.md → `### Task 8: Run focused pytest validation` | tasks.md → ``- [ ] 2.2 Run focused pytest commands with `conda activate finetf` for commodity full-process shell tests, multi-contract feature selection tests, and scale-save routing tests.``
> **sync:** tasks.md → ``- [ ] 2.2 Run focused pytest commands with `conda activate finetf` for commodity full-process shell tests, multi-contract feature selection tests, and scale-save routing tests.`` | plan-ready.md → `### Task 8: Run focused pytest validation`

**Files:**
- Verify: `data_preprocess/tests/test_commodity_main_contract_cli.py`
- Verify: `data_preprocess/tests/test_commodity_multi_contract_feature_selection.py`
- Verify: `data_preprocess/tests/test_feature_selection_polars.py`

- [x] **Step 1: Run full focused pytest set**

Run:

```bash
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest data_preprocess/tests/test_commodity_main_contract_cli.py data_preprocess/tests/test_commodity_multi_contract_feature_selection.py data_preprocess/tests/test_feature_selection_polars.py -q'
```

Expected: PASS. If a pre-existing unrelated test fails, rerun the named tests from Tasks 1, 2, and 5 and record the unrelated failure in the final build report.

- [x] **Step 2: Commit final checkbox-only plan synchronization if build uses it**

Run after all implementation tasks pass:

```bash
git add openspec/changes/adjust-commodity-feature-selection-pipeline/tasks.md openspec/changes/adjust-commodity-feature-selection-pipeline/plan-ready.md docs/superpowers/plans/2026-07-18-adjust-commodity-feature-selection-pipeline.md
git commit -m "chore: sync commodity feature selection plan status"
```

Expected: Commit succeeds if the build phase updates checkboxes in plan files.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 9: Run static syntax validation

> **trace:** plan-ready.md → `### Task 9: Run static syntax validation` | tasks.md → ``- [ ] 2.3 Run `bash -n` on changed shell scripts and `python -m py_compile` on changed Python modules with `conda activate finetf`.``
> **sync:** tasks.md → ``- [ ] 2.3 Run `bash -n` on changed shell scripts and `python -m py_compile` on changed Python modules with `conda activate finetf`.`` | plan-ready.md → `### Task 9: Run static syntax validation`

**Files:**
- Verify: `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`
- Verify: `data_preprocess/operator_futures/feature_selection/muti_contract/*.py`
- Verify: `data_preprocess/operator_futures/scale_describe_save/scale_save.py`

- [x] **Step 1: Run shell syntax validation**

Run:

```bash
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh
```

Expected: PASS with no output.

- [x] **Step 2: Run Python compile validation in finetf**

Run:

```bash
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && python -m py_compile data_preprocess/operator_futures/feature_selection/muti_contract/__init__.py data_preprocess/operator_futures/feature_selection/muti_contract/metrics.py data_preprocess/operator_futures/feature_selection/muti_contract/pipeline.py data_preprocess/operator_futures/feature_selection/muti_contract/__main__.py data_preprocess/operator_futures/scale_describe_save/scale_save.py'
```

Expected: PASS with no output.

- [x] **Step 3: Capture final diff summary**

Run:

```bash
git diff --stat
git diff -- openspec/changes/adjust-commodity-feature-selection-pipeline docs/superpowers/plans/2026-07-18-adjust-commodity-feature-selection-pipeline.md data_preprocess/tests/test_commodity_main_contract_cli.py data_preprocess/tests/test_commodity_multi_contract_feature_selection.py data_preprocess/tests/test_feature_selection_polars.py data_preprocess/operator_futures/feature_selection/muti_contract data_preprocess/operator_futures/scale_describe_save/scale_save.py data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh docs/datahandler/data_preparation_analysis.zh_cn.md docs/上海商品交易所/commodity_futures_preprocess.md
```

Expected: Diff only contains the OpenSpec artifacts, plan artifacts, focused tests, feature-selection module, full-process shell, scale-save routing, and focused documentation changes for this request.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 10: Amend metric and filter semantics in OpenSpec

> **trace:** plan-ready.md → `### Task 10: Amend metric and filter semantics in OpenSpec` | tasks.md → ``- [x] 1.7 Amend OpenSpec artifacts to document the implemented feature metric semantics and filter semantics: default `windows_list=[1,6,12]`, original-compatible IC/RankIC/CatBoost Importance, Sharpe and Permutation Importance formulas, Composite Score priority order, bottom 10% composite drop, and manifest fields.``
> **sync:** tasks.md → ``- [x] 1.7 Amend OpenSpec artifacts to document the implemented feature metric semantics and filter semantics: default `windows_list=[1,6,12]`, original-compatible IC/RankIC/CatBoost Importance, Sharpe and Permutation Importance formulas, Composite Score priority order, bottom 10% composite drop, and manifest fields.`` | plan-ready.md → `### Task 10: Amend metric and filter semantics in OpenSpec`

**Files:**
- Modify: `openspec/changes/adjust-commodity-feature-selection-pipeline/proposal.md`
- Modify: `openspec/changes/adjust-commodity-feature-selection-pipeline/design.md`
- Modify: `openspec/changes/adjust-commodity-feature-selection-pipeline/specs/commodity-futures-support/spec.md`
- Modify: `openspec/changes/adjust-commodity-feature-selection-pipeline/tasks.md`
- Modify: `openspec/changes/adjust-commodity-feature-selection-pipeline/plan-ready.md`
- Modify: `docs/superpowers/plans/2026-07-18-adjust-commodity-feature-selection-pipeline.md`

- [x] **Step 1: Add proposal amendment**

Append an amendment recording that the implemented metric and filter semantics are now part of the requirement scope.

- [x] **Step 2: Expand design metric/filter semantics**

Document target construction, default windows, IC, RankIC, CatBoost Importance, Sharpe, Permutation Importance, Composite Score priority ordering, bottom 10% composite drop, and manifest fields.

- [x] **Step 3: Expand commodity-futures-support spec scenarios**

Add OpenSpec scenarios covering the same metric and filter details as verifiable behavior.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 11: Re-run OpenSpec validation after amend

> **trace:** plan-ready.md → `### Task 11: Re-run OpenSpec validation after amend` | tasks.md → ``- [x] 2.4 Re-run strict OpenSpec validation after metric/filter semantics amend.``
> **sync:** tasks.md → ``- [x] 2.4 Re-run strict OpenSpec validation after metric/filter semantics amend.`` | plan-ready.md → `### Task 11: Re-run OpenSpec validation after amend`

**Files:**
- Verify: `openspec/changes/adjust-commodity-feature-selection-pipeline`

- [x] **Step 1: Run strict OpenSpec validation**

Run:

```bash
openspec validate adjust-commodity-feature-selection-pipeline --strict
```

Expected: PASS.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 12: Add revised train/valid feature-selection tests

> **trace:** plan-ready.md → `### Task 12: Add revised train/valid feature-selection tests` | tasks.md → ``- [ ] 1.8 Add focused tests for the revised train/valid feature-selection semantics: train writes final `state_features.npy`, no longer writes `state_features_candidate.npy`, Hard Filter uses `RankIC_Mean` instead of `IC_Mean`, and valid writes metrics/manifest only without running filters or producing a downstream feature list.``
> **sync:** tasks.md → ``- [ ] 1.8 Add focused tests for the revised train/valid feature-selection semantics: train writes final `state_features.npy`, no longer writes `state_features_candidate.npy`, Hard Filter uses `RankIC_Mean` instead of `IC_Mean`, and valid writes metrics/manifest only without running filters or producing a downstream feature list.`` | plan-ready.md → `### Task 12: Add revised train/valid feature-selection tests`

**Files:**
- Modify: `data_preprocess/tests/test_commodity_multi_contract_feature_selection.py`

- [x] **Step 1: Update train selected-file assertions**

Change the train stage test so it expects:
- `FEATURE_SELECTION/5min/fu/train/state_features.npy` exists and contains the selected features.
- `FEATURE_SELECTION/5min/fu/train/state_features_candidate.npy` does not exist.
- train manifest `selected_feature_file` ends with `train/state_features.npy`.

- [x] **Step 2: Add a Hard Filter RankIC test**

Add a focused `_ordered_filter_features` test with at least two features where one feature has high `IC_Mean` but low `RankIC_Mean`, and another has sufficient `RankIC_Mean`. Assert the high-IC/low-RankIC feature is rejected by `Hard Filter`.

- [x] **Step 3: Update valid stage assertions**

Change the valid stage test so it creates `train/state_features.npy`, runs `stage="valid"`, and asserts:
- per-contract metrics, aggregate metrics, and `feature_selection_manifest.json` are written.
- `valid/state_features.npy` is not written.
- valid manifest references the train feature list as the evaluated feature source.
- valid manifest has no authoritative `filter_results` / `selected_feature_file` for downstream scale-save.

- [x] **Step 4: Confirm tests fail before implementation**

Run:

```bash
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest data_preprocess/tests/test_commodity_multi_contract_feature_selection.py -q'
```

Expected: FAIL until the pipeline is updated.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 13: Update multi-contract feature-selection pipeline

> **trace:** plan-ready.md → `### Task 13: Update multi-contract feature-selection pipeline` | tasks.md → ``- [ ] 1.9 Update `operator_futures.feature_selection.muti_contract.pipeline` so `train` is the only filtering stage, `train/state_features.npy` is the canonical selected feature file, `valid` loads that train file for evaluation/reporting only, and valid manifest fields cannot be mistaken for selected downstream features.``
> **sync:** tasks.md → ``- [ ] 1.9 Update `operator_futures.feature_selection.muti_contract.pipeline` so `train` is the only filtering stage, `train/state_features.npy` is the canonical selected feature file, `valid` loads that train file for evaluation/reporting only, and valid manifest fields cannot be mistaken for selected downstream features.`` | plan-ready.md → `### Task 13: Update multi-contract feature-selection pipeline`

**Files:**
- Modify: `data_preprocess/operator_futures/feature_selection/muti_contract/pipeline.py`

- [x] **Step 1: Make train write the canonical selected file**

In `run_feature_selection`, change the train selected output path from `state_features_candidate.npy` to `state_features.npy`.

- [x] **Step 2: Make valid load the train selected file**

In the valid branch, load candidates from `FEATURE_SELECTION/{target_freq}/{symbol}/train/state_features.npy`.

- [x] **Step 3: Skip filters and filtered outputs for valid**

After writing valid per-contract metrics and `aggregate_metrics.csv`, write a valid manifest/report and return. Do not call `_ordered_filter_features`, do not save `valid/state_features.npy`, and do not call `_write_filtered_outputs`.

- [x] **Step 4: Change Hard Filter to RankIC**

In `_ordered_filter_features`, change the first hard filter from `abs(IC_Mean) >= min_abs_ic` to `abs(RankIC_Mean) >= min_abs_ic`. Keep the CLI argument name unless a broader rename is explicitly requested.

- [x] **Step 5: Run the focused feature-selection tests**

Run:

```bash
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest data_preprocess/tests/test_commodity_multi_contract_feature_selection.py -q'
```

Expected: PASS.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 14: Update full-process scale-save handoff

> **trace:** plan-ready.md → `### Task 14: Update full-process scale-save handoff` | tasks.md → ``- [ ] 1.10 Update `fu_full_process.sh` orchestration so `feature_selection_valid` remains after train as an evaluation/report step, and subsequent `scale_save` uses the train-produced `state_features.npy` instead of any valid-produced feature list.``
> **sync:** tasks.md → ``- [ ] 1.10 Update `fu_full_process.sh` orchestration so `feature_selection_valid` remains after train as an evaluation/report step, and subsequent `scale_save` uses the train-produced `state_features.npy` instead of any valid-produced feature list.`` | plan-ready.md → `### Task 14: Update full-process scale-save handoff`

**Files:**
- Modify: `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`
- Modify: `data_preprocess/tests/test_commodity_main_contract_cli.py`

- [x] **Step 1: Update shell assertions for train-list scale-save**

Keep the order assertion `dataset_split -> feature_selection_train -> feature_selection_valid -> scale_save -> maintenance_margin_dict`. Add an assertion that the scale-save call no longer passes `--feature_selection_stage valid` as the source of the selected feature list, and does pass the new split-stage/train-feature-list routing arguments introduced in Task 15.

- [x] **Step 2: Update `run_commodity_scale_save` arguments**

Adjust `run_commodity_scale_save` so it points `scale_save.py` at:
- split-stage data root: `PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST`
- train feature-selection root/list: `PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/{target_freq}/{symbol}/train/state_features.npy`

- [x] **Step 3: Run shell-focused validation**

Run:

```bash
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest data_preprocess/tests/test_commodity_main_contract_cli.py::test_commodity_full_process_writes_step_logs_and_preserves_child_log_paths data_preprocess/tests/test_commodity_main_contract_cli.py::test_commodity_full_process_shell_runs_scale_after_feature_selection_valid -q'
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh
```

Expected: PASS after Task 15 routing exists.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 15: Enhance scale-save for split-stage inputs

> **trace:** plan-ready.md → `### Task 15: Enhance scale-save for split-stage inputs` | tasks.md → ``- [ ] 1.11 Enhance `scale_save.py` and focused tests so commodity scale-save can read split-stage inputs, apply `FEATURE_SELECTION/{target_freq}/{symbol}/train/state_features.npy`, skip contract-stage inputs that do not exist, fail when a requested contract has no split-stage input at all, and preserve existing non-feature-selection `IC_RESULT` behavior.``
> **sync:** tasks.md → ``- [ ] 1.11 Enhance `scale_save.py` and focused tests so commodity scale-save can read split-stage inputs, apply `FEATURE_SELECTION/{target_freq}/{symbol}/train/state_features.npy`, skip contract-stage inputs that do not exist, fail when a requested contract has no split-stage input at all, and preserve existing non-feature-selection `IC_RESULT` behavior.`` | plan-ready.md → `### Task 15: Enhance scale-save for split-stage inputs`

**Files:**
- Modify: `data_preprocess/operator_futures/scale_describe_save/scale_save.py`
- Modify: `data_preprocess/tests/test_feature_selection_polars.py`

- [x] **Step 1: Add split-stage scale-save tests**

Add tests that create `SPLIT-TRAIN-VALID-TEST/5min/fu/train/fu2601.feather`, omit `valid/fu2601.feather`, write `FEATURE_SELECTION/5min/fu/train/state_features.npy`, run the CLI, and assert:
- train output is written under `SCALE_SAVE/fu/fu2601/5min/train/{start_date}-{end_date}/`.
- the missing valid stage is skipped without nonzero exit.
- output `state_features.npy` equals the train list.

Add fail-fast tests for:
- requested contract missing from all split stages.
- existing split-stage input missing a selected feature column.
- train `state_features.npy` missing or empty.

- [x] **Step 2: Add split-stage CLI/routing arguments**

Add narrowly scoped arguments for commodity split-stage routing. Keep old `--feature_selection_stage` / `IC_RESULT` behavior unless tests prove it must be replaced.

- [x] **Step 3: Implement stage iteration**

For split-stage routing, check `train`, `valid`, and `test` stage files for the requested contract. Process existing files and skip missing files with an informative log containing contract and stage. Fail only if none exist.

- [x] **Step 4: Apply train feature list to every existing stage**

Load `FEATURE_SELECTION/{target_freq}/{symbol}/train/state_features.npy` once and select those state columns from every existing split-stage input. Preserve reward/execution column behavior and scaling algorithm.

- [x] **Step 5: Run scale-save focused tests**

Run:

```bash
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest data_preprocess/tests/test_feature_selection_polars.py -q'
```

Expected: PASS.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 16: Validate revised amend implementation

> **trace:** plan-ready.md → `### Task 16: Validate revised amend implementation` | tasks.md → ``- [ ] 2.5 Re-run strict OpenSpec validation and the revised focused pytest/static checks after the 2026-07-19 train-list/valid-report/scale-save amend is implemented.``
> **sync:** tasks.md → ``- [ ] 2.5 Re-run strict OpenSpec validation and the revised focused pytest/static checks after the 2026-07-19 train-list/valid-report/scale-save amend is implemented.`` | plan-ready.md → `### Task 16: Validate revised amend implementation`

**Files:**
- Verify: `openspec/changes/adjust-commodity-feature-selection-pipeline`
- Verify: changed tests, shell, and Python modules

- [x] **Step 1: Run OpenSpec validation**

Run:

```bash
openspec validate adjust-commodity-feature-selection-pipeline --strict
```

Expected: PASS.

- [x] **Step 2: Run focused pytest**

Run:

```bash
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest data_preprocess/tests/test_commodity_multi_contract_feature_selection.py data_preprocess/tests/test_commodity_main_contract_cli.py data_preprocess/tests/test_feature_selection_polars.py -q'
```

Expected: PASS.

- [x] **Step 3: Run static validation**

Run:

```bash
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && bash -n data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh && python -m py_compile data_preprocess/operator_futures/feature_selection/muti_contract/*.py data_preprocess/operator_futures/scale_describe_save/scale_save.py'
```

Expected: PASS.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 17: Add multi-contract scale-save csv output tests

> **trace:** plan-ready.md → `### Task 17: Add multi-contract scale-save csv output tests` | tasks.md → ``- [ ] 1.12 Add focused tests for `muti_contract_scale_save.py` as the commodity split-stage scale-save source of truth: scan existing `SPLIT-TRAIN-VALID-TEST/{target_freq}/{symbol}/{stage}/*.feather`, write `SCALE_SAVE/{symbol}/{target_freq}/{stage}/{contract}.feather`, write same-basename `.csv` beside each feather, and preserve selected-feature-only output.``
> **sync:** tasks.md → ``- [ ] 1.12 Add focused tests for `muti_contract_scale_save.py` as the commodity split-stage scale-save source of truth: scan existing `SPLIT-TRAIN-VALID-TEST/{target_freq}/{symbol}/{stage}/*.feather`, write `SCALE_SAVE/{symbol}/{target_freq}/{stage}/{contract}.feather`, write same-basename `.csv` beside each feather, and preserve selected-feature-only output.`` | plan-ready.md → `### Task 17: Add multi-contract scale-save csv output tests`

**Files:**
- Modify: `data_preprocess/tests/test_feature_selection_polars.py`

- [x] **Step 1: Add csv assertions to the existing multi-contract scale-save test**

In `data_preprocess/tests/test_feature_selection_polars.py`, inside `test_multi_contract_scale_save_cli_scans_all_split_stage_contracts`, replace the final assertion block:

```python
        assert output_file.exists()
        assert not old_output_file.exists()
        assert "feature_a" in pl.read_ipc(output_file).columns
```

with:

```python
        output_csv = output_file.with_suffix(".csv")
        assert output_file.exists()
        assert output_csv.exists()
        assert not old_output_file.exists()
        feather = pl.read_ipc(output_file)
        csv = pl.read_csv(output_csv)
        assert feather.shape == csv.shape
        assert "feature_a" in feather.columns
        assert "feature_a" in csv.columns
        assert "timestamp" in feather.columns
        assert "timestamp" in csv.columns
```

- [x] **Step 2: Run the focused test and confirm RED**

Run:

```bash
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest data_preprocess/tests/test_feature_selection_polars.py::test_multi_contract_scale_save_cli_scans_all_split_stage_contracts -q'
```

Expected: FAIL because `muti_contract_scale_save.py` writes the feather output but does not write `output_file.with_suffix(".csv")` yet.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 18: Write csv beside each muti_contract_scale_save feather

> **trace:** plan-ready.md → `### Task 18: Write csv beside each muti_contract_scale_save feather` | tasks.md → ``- [ ] 1.13 Update `data_preprocess/operator_futures/scale_describe_save/muti_contract_scale_save.py` so every successful feather output is accompanied by a same-directory, same-basename csv debug output without changing the scaling algorithm or selected feature list semantics.``
> **sync:** tasks.md → ``- [ ] 1.13 Update `data_preprocess/operator_futures/scale_describe_save/muti_contract_scale_save.py` so every successful feather output is accompanied by a same-directory, same-basename csv debug output without changing the scaling algorithm or selected feature list semantics.`` | plan-ready.md → `### Task 18: Write csv beside each muti_contract_scale_save feather`

**Files:**
- Modify: `data_preprocess/operator_futures/scale_describe_save/muti_contract_scale_save.py`
- Test: `data_preprocess/tests/test_feature_selection_polars.py`

- [x] **Step 1: Write the csv after the feather output**

In `scale_one_input`, replace:

```python
    out.write_ipc(output_file)
    logger.info(
        "Wrote split-stage scale-save output: output_file=%s rows=%d columns=%d",
        output_file,
        out.height,
        len(out.columns),
    )
```

with:

```python
    out.write_ipc(output_file)
    csv_output_file = output_file.with_suffix(".csv")
    out.write_csv(csv_output_file)
    logger.info(
        "Wrote split-stage scale-save output: output_file=%s csv_output_file=%s rows=%d columns=%d",
        output_file,
        csv_output_file,
        out.height,
        len(out.columns),
    )
```

- [x] **Step 2: Run the focused test and confirm GREEN**

Run:

```bash
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest data_preprocess/tests/test_feature_selection_polars.py::test_multi_contract_scale_save_cli_scans_all_split_stage_contracts -q'
```

Expected: PASS.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 19: Validate csv amend implementation

> **trace:** plan-ready.md → `### Task 19: Validate csv amend implementation` | tasks.md → ``- [ ] 2.6 Re-run strict OpenSpec validation and focused `muti_contract_scale_save.py` pytest/static checks after adding csv debug outputs.``
> **sync:** tasks.md → ``- [ ] 2.6 Re-run strict OpenSpec validation and focused `muti_contract_scale_save.py` pytest/static checks after adding csv debug outputs.`` | plan-ready.md → `### Task 19: Validate csv amend implementation`

**Files:**
- Verify: `openspec/changes/adjust-commodity-feature-selection-pipeline`
- Verify: `data_preprocess/tests/test_feature_selection_polars.py`
- Verify: `data_preprocess/operator_futures/scale_describe_save/muti_contract_scale_save.py`

- [x] **Step 1: Run OpenSpec validation**

Run:

```bash
openspec validate adjust-commodity-feature-selection-pipeline --strict
```

Expected: PASS.

- [x] **Step 2: Run focused pytest**

Run:

```bash
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest data_preprocess/tests/test_feature_selection_polars.py::test_multi_contract_scale_save_cli_scans_all_split_stage_contracts -q'
```

Expected: PASS.

- [x] **Step 3: Run Python compile**

Run:

```bash
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && python -m py_compile data_preprocess/operator_futures/scale_describe_save/muti_contract_scale_save.py'
```

Expected: PASS.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

## Self-Review

Spec coverage: Tasks 1 and 4 cover the full-process ordering, step logs, dataset split input, and scale-save timing. Tasks 2 and 3 cover the original train/valid feature selection implementation. Task 5 covers the original feature-selection input routing for scale-save while preserving old `IC_RESULT` behavior. Task 6 covers documentation. Task 10 covers the amended metric/filter semantics: default windows, IC, RankIC, CatBoost Importance, Sharpe, Permutation Importance, Composite Score priority, bottom 10% drop, and manifest fields. Tasks 12 through 15 cover the 2026-07-19 semantic change: train-only final `state_features.npy`, valid reporting only, RankIC hard filter, and split-stage scale-save with missing contract-stage support. Tasks 17 and 18 cover the `muti_contract_scale_save.py` source-of-truth output layout and same-basename csv debug output. Tasks 7 through 9, 11, 16, and 19 cover OpenSpec, pytest, shell syntax, Python compile, and amend validation.

Placeholder scan: No placeholder markers are intentionally present. All code-changing steps include concrete code blocks or exact replacements, and all validation steps include exact commands and expected results.

Type consistency: The plan consistently uses `run_feature_selection`, `calculate_metric_frame`, `aggregate_metric_frames`, `calculate_sharpe`, `FEATURE_SELECTION/{target_freq}/{symbol}/{stage}`, `SPLIT-TRAIN-VALID-TEST/{target_freq}/{symbol}/{stage}/{contract}.feather`, and the package path `operator_futures.feature_selection.muti_contract`.
