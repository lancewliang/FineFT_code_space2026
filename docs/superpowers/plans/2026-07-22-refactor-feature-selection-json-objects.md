# Refactor Feature Selection JSON Objects Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `data_preprocess/operator_futures/feature_selection` JSON generation so manifest and score JSON contracts are represented by dataclass objects internally while preserving existing output files.

**Architecture:** Add a focused `manifests.py` module that owns JSON contract dataclasses, `to_dict()` serialization, JSON writing, and result return types. Existing feature selection scripts keep their calculation responsibilities and only convert to objects at JSON/write and public return boundaries.

**Tech Stack:** Python standard-library `dataclasses`, `pathlib`, `json`; existing NumPy, Polars, Pytest, and OpenSpec.

**Traceability (sddflow):**
- plan-ready: `openspec/changes/refactor-feature-selection-json-objects/plan-ready.md`
- tasks: `openspec/changes/refactor-feature-selection-json-objects/tasks.md`
- plan: `docs/superpowers/plans/2026-07-22-refactor-feature-selection-json-objects.md`

---

### Task 1: Add focused tests for feature selection JSON dataclass objects

> **trace:** plan-ready.md -> `### Task 1: Add focused tests for feature selection JSON dataclass objects` | tasks.md -> `- [ ] 1.1 Add focused tests for feature selection JSON dataclass objects covering object return types, object attribute access, and `to_dict()` equality with written JSON files.`
> **sync:** tasks.md -> `- [ ] 1.1 Add focused tests for feature selection JSON dataclass objects covering object return types, object attribute access, and `to_dict()` equality with written JSON files.` | plan-ready.md -> `### Task 1: Add focused tests for feature selection JSON dataclass objects`

**Files:**
- Modify: `data_preprocess/tests/test_commodity_multi_contract_feature_selection.py`
- Modify: `data_preprocess/tests/test_commodity_feature_pipeline.py`
- Modify: `data_preprocess/tests/test_feature_selection_polars.py`

- [x] **Step 1: Add imports for manifest/result classes in multi-contract tests**

Add these imports near the existing feature selection imports in `data_preprocess/tests/test_commodity_multi_contract_feature_selection.py`:

```python
from operator_futures.feature_selection.manifests import (
    FeatureSelectionResult,
)
```

- [x] **Step 2: Update the train-stage multi-contract test to assert object return and JSON compatibility**

In `test_train_stage_writes_final_features_metrics_filtered_outputs_and_manifest`, replace dict-style manifest assertions with:

```python
    manifest_path = stage_dir / "feature_selection_manifest.json"
    persisted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert isinstance(manifest, FeatureSelectionResult)
    assert manifest.output_dir == stage_dir
    assert manifest.manifest.stage == "train"
    assert manifest.manifest.selected_feature_file.endswith("train/state_features.npy")
    assert manifest.manifest.selected_feature_count == len(selected_features)
    assert manifest.manifest.windows_list == [1, 6, 12]
    assert persisted_manifest == manifest.manifest.to_dict()
```

Keep the existing artifact assertions around this block.

- [x] **Step 3: Update the valid-stage multi-contract test to assert object return and JSON compatibility**

In `test_valid_stage_evaluates_train_features_without_writing_downstream_features`, replace dict-style manifest assertions with:

```python
    assert isinstance(manifest, FeatureSelectionResult)
    assert manifest.output_dir == stage_dir
    assert manifest.manifest.evaluated_feature_file.endswith("train/state_features.npy")
    assert manifest.manifest.report_only is True
    assert manifest.manifest.evaluated_feature_count == 1
    assert manifest.manifest.evaluated_features == ["alpha"]

    persisted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted_manifest == manifest.manifest.to_dict()
    assert persisted_manifest["report_only"] is True
    assert persisted_manifest["evaluated_feature_count"] == 1
    assert persisted_manifest["evaluated_features"] == ["alpha"]
    assert "filter_results" not in persisted_manifest
    assert "selected_feature_file" not in persisted_manifest
    assert "filtered_outputs" not in persisted_manifest
```

- [x] **Step 4: Add imports for feature union result tests**

Add this import near the existing `write_contract_feature_union` import in `data_preprocess/tests/test_commodity_feature_pipeline.py`:

```python
from operator_futures.feature_selection.manifests import FeatureUnionResult
```

- [x] **Step 5: Update feature union tests to use `result.output_dir` and `result.manifest`**

In `test_write_contract_feature_union_writes_symbol_level_manifest`, change:

```python
    output_dir = write_contract_feature_union(
```

to:

```python
    result = write_contract_feature_union(
```

Then add:

```python
    assert isinstance(result, FeatureUnionResult)
    output_dir = result.output_dir
```

After loading JSON, add:

```python
    assert manifest == result.manifest.to_dict()
    assert result.manifest.contracts == ["fu2601", "fu2605"]
    assert result.manifest.state_features == ["alpha", "beta", "gamma"]
    assert result.manifest.state_feature_count == 3
```

In `test_write_contract_feature_union_finalizes_ic_result_from_candidates`, assign the call result to `result`, assert `isinstance(result, FeatureUnionResult)`, and set `output_dir = result.output_dir` before existing output assertions. Then add:

```python
    assert manifest == result.manifest.to_dict()
    assert result.manifest.per_contract_output_shapes["fu2601"].rows == 2
    assert result.manifest.per_contract_output_shapes["fu2605"].columns == 12
```

- [x] **Step 6: Add imports and tests for IC / Rank IC result objects**

Add these imports near the top of `data_preprocess/tests/test_feature_selection_polars.py`:

```python
import json

from operator_futures.feature_selection.manifests import (
    FeatureScoreWindow,
    IcCorrelationResult,
    RankIcCorrelationResult,
)
```

Add this helper after `_ic_args`:

```python
def _rank_ic_args(tmp_path):
    return SimpleNamespace(
        root_path=str(tmp_path),
        data_path="PREPROCESS_DATASET/commodity-futures/ALL_FEATURE/",
        save_path="PREPROCESS_DATASET/commodity-futures/IC_RESULT/",
        symbols="fu",
        target_freq="5min",
        start_date="2026-01-05",
        end_date="2026-01-06",
        ic_theshold=0.01,
        cor_theshold=0.7,
        windows_list=[1],
        market_type="commodity_futures",
        orderbook_depth=5,
    )
```

Add this test after `test_ic_correlation_cli_writes_expected_files`:

```python
def test_ic_correlation_returns_result_object_and_score_window_json(tmp_path):
    from operator_futures.feature_selection import ic_correlation

    input_file = (
        tmp_path
        / "PREPROCESS_DATASET/commodity-futures/ALL_FEATURE/fu/5min"
        / "2026-01-05-2026-01-06.feather"
    )
    _write_ic_fixture(input_file)

    result = ic_correlation.main(_ic_args(tmp_path))

    output_dir = (
        tmp_path
        / "PREPROCESS_DATASET/commodity-futures/IC_RESULT/fu/5min/2026-01-05-2026-01-06"
    )
    score_payload = json.loads((output_dir / "ic_window_1.json").read_text(encoding="utf-8"))
    assert isinstance(result, IcCorrelationResult)
    assert result.output_dir == output_dir
    assert result.frame.shape == pl.read_ipc(output_dir / "df.feather").shape
    assert result.selected_features == np.load(output_dir / "state_features.npy", allow_pickle=True).tolist()
    assert len(result.score_windows) == 1
    assert isinstance(result.score_windows[0], FeatureScoreWindow)
    assert score_payload == result.score_windows[0].to_dict()
    assert "feature_a" in score_payload
```

Add this test near the IC object test:

```python
def test_rank_ic_correlation_returns_result_object_and_score_window_json(tmp_path):
    from operator_futures.feature_selection import rank_ic_correlation

    input_file = (
        tmp_path
        / "PREPROCESS_DATASET/commodity-futures/ALL_FEATURE/fu/5min"
        / "2026-01-05-2026-01-06.feather"
    )
    _write_ic_fixture(input_file)

    result = rank_ic_correlation.main(_rank_ic_args(tmp_path))

    output_dir = (
        tmp_path
        / "PREPROCESS_DATASET/commodity-futures/IC_RESULT/fu/5min/2026-01-05-2026-01-06"
    )
    score_payload = json.loads((output_dir / "rank_ic_window_1.json").read_text(encoding="utf-8"))
    assert isinstance(result, RankIcCorrelationResult)
    assert result.output_dir == output_dir
    assert result.frame.shape == pl.read_ipc(output_dir / "df_rank.feather").shape
    assert result.selected_features == np.load(output_dir / "state_features_rank.npy", allow_pickle=True).tolist()
    assert len(result.score_windows) == 1
    assert isinstance(result.score_windows[0], FeatureScoreWindow)
    assert score_payload == result.score_windows[0].to_dict()
    assert "feature_a" in score_payload
```

- [x] **Step 7: Run focused tests to confirm the object expectations fail before implementation**

Run:

```bash
conda activate finetf && pytest data_preprocess/tests/test_commodity_multi_contract_feature_selection.py data_preprocess/tests/test_commodity_feature_pipeline.py data_preprocess/tests/test_feature_selection_polars.py -q
```

Expected: FAIL because `operator_futures.feature_selection.manifests` does not exist yet or existing functions return old dict/DataFrame values.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 2: Add feature selection manifest dataclasses

> **trace:** plan-ready.md -> `### Task 2: Add feature selection manifest dataclasses` | tasks.md -> `- [ ] 1.2 Add `data_preprocess/operator_futures/feature_selection/manifests.py` with dataclass models for feature selection manifests, feature union manifests, feature score windows, and result return objects.`
> **sync:** tasks.md -> `- [ ] 1.2 Add `data_preprocess/operator_futures/feature_selection/manifests.py` with dataclass models for feature selection manifests, feature union manifests, feature score windows, and result return objects.` | plan-ready.md -> `### Task 2: Add feature selection manifest dataclasses`

**Files:**
- Create: `data_preprocess/operator_futures/feature_selection/manifests.py`

- [x] **Step 1: Create the dataclass module**

Create `data_preprocess/operator_futures/feature_selection/manifests.py` with:

```python
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl


def _json_safe_number(value: Any) -> int | float | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    return float(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


@dataclass
class FeatureSelectionContractRecord:
    contract: str
    input_path: str
    metric_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "input_path": self.input_path,
            "metric_path": self.metric_path,
        }


@dataclass
class FilteredOutputRecord:
    contract: str
    output_path: str
    output_row_count: int
    output_column_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "output_path": self.output_path,
            "output_row_count": self.output_row_count,
            "output_column_count": self.output_column_count,
        }


@dataclass
class ContractOutputShape:
    rows: int
    columns: int

    def to_dict(self) -> dict[str, int]:
        return {"rows": self.rows, "columns": self.columns}


@dataclass
class FeatureSelectionManifest:
    symbol: str
    target_freq: str
    stage: str
    split_input_dir: str
    windows_list: list[int]
    aggregate_metrics_path: str
    contracts: list[FeatureSelectionContractRecord] = field(default_factory=list)
    selected_feature_file: str | None = None
    selected_feature_count: int | None = None
    selected_features: list[str] | None = None
    composite_drop_ratio: float | None = None
    filter_results: dict[str, list[str]] | None = None
    filtered_outputs: list[FilteredOutputRecord] | None = None
    evaluated_feature_file: str | None = None
    evaluated_feature_count: int | None = None
    evaluated_features: list[str] | None = None
    report_only: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "symbol": self.symbol,
            "target_freq": self.target_freq,
            "stage": self.stage,
            "split_input_dir": self.split_input_dir,
        }
        if self.selected_feature_file is not None:
            payload["selected_feature_file"] = self.selected_feature_file
            payload["selected_feature_count"] = self.selected_feature_count
            payload["selected_features"] = list(self.selected_features or [])
        if self.evaluated_feature_file is not None:
            payload["evaluated_feature_file"] = self.evaluated_feature_file
            payload["evaluated_feature_count"] = self.evaluated_feature_count
            payload["evaluated_features"] = list(self.evaluated_features or [])
        payload["windows_list"] = list(self.windows_list)
        if self.composite_drop_ratio is not None:
            payload["composite_drop_ratio"] = self.composite_drop_ratio
        payload["aggregate_metrics_path"] = self.aggregate_metrics_path
        if self.filter_results is not None:
            payload["filter_results"] = {
                key: list(values) for key, values in self.filter_results.items()
            }
        payload["contracts"] = [contract.to_dict() for contract in self.contracts]
        if self.filtered_outputs is not None:
            payload["filtered_outputs"] = [
                output.to_dict() for output in self.filtered_outputs
            ]
        if self.report_only is not None:
            payload["report_only"] = self.report_only
        return payload

    def write_json(self, path: Path) -> None:
        _write_json(path, self.to_dict())


@dataclass
class FeatureUnionManifest:
    symbol: str
    target_freq: str
    start_date: str
    end_date: str
    summary_path: str
    contracts: list[str]
    contract_state_feature_paths: dict[str, str]
    per_contract_feature_counts: dict[str, int]
    state_feature_count: int
    state_features: list[str]
    candidate_source_path: str | None
    all_feature_path: str
    ic_result_path: str
    finalize_filtered_df: bool
    per_contract_output_paths: dict[str, str] = field(default_factory=dict)
    per_contract_output_shapes: dict[str, ContractOutputShape] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "target_freq": self.target_freq,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "summary_path": self.summary_path,
            "contracts": list(self.contracts),
            "contract_state_feature_paths": dict(self.contract_state_feature_paths),
            "per_contract_feature_counts": dict(self.per_contract_feature_counts),
            "state_feature_count": self.state_feature_count,
            "state_features": list(self.state_features),
            "candidate_source_path": self.candidate_source_path,
            "all_feature_path": self.all_feature_path,
            "ic_result_path": self.ic_result_path,
            "finalize_filtered_df": self.finalize_filtered_df,
            "per_contract_output_paths": dict(self.per_contract_output_paths),
            "per_contract_output_shapes": {
                contract: shape.to_dict()
                for contract, shape in self.per_contract_output_shapes.items()
            },
        }

    def write_json(self, path: Path) -> None:
        _write_json(path, self.to_dict())


@dataclass
class FeatureScoreWindow:
    window_length: int
    scores: dict[str, float]

    def to_dict(self) -> dict[str, float]:
        return {
            str(feature): float(score)
            for feature, score in self.scores.items()
        }

    def write_json(self, path: Path) -> None:
        _write_json(path, self.to_dict())


@dataclass
class FeatureSelectionResult:
    output_dir: Path
    manifest: FeatureSelectionManifest


@dataclass
class FeatureUnionResult:
    output_dir: Path
    manifest: FeatureUnionManifest


@dataclass
class IcCorrelationResult:
    frame: pl.DataFrame
    output_dir: Path
    selected_features: list[str]
    score_windows: list[FeatureScoreWindow]


@dataclass
class RankIcCorrelationResult:
    frame: pl.DataFrame
    output_dir: Path
    selected_features: list[str]
    score_windows: list[FeatureScoreWindow]
```

- [x] **Step 2: Compile the dataclass module**

Run:

```bash
conda activate finetf && python -m py_compile data_preprocess/operator_futures/feature_selection/manifests.py
```

Expected: command exits 0.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 3: Refactor multi-contract feature selection manifest boundary

> **trace:** plan-ready.md -> `### Task 3: Refactor multi-contract feature selection manifest boundary` | tasks.md -> `- [ ] 1.3 Refactor `data_preprocess/operator_futures/feature_selection/muti_contract/pipeline.py` to use `FeatureSelectionManifest` and return `FeatureSelectionResult` while preserving `feature_selection_manifest.json`.`
> **sync:** tasks.md -> `- [ ] 1.3 Refactor `data_preprocess/operator_futures/feature_selection/muti_contract/pipeline.py` to use `FeatureSelectionManifest` and return `FeatureSelectionResult` while preserving `feature_selection_manifest.json`.` | plan-ready.md -> `### Task 3: Refactor multi-contract feature selection manifest boundary`

**Files:**
- Modify: `data_preprocess/operator_futures/feature_selection/muti_contract/pipeline.py`
- Test: `data_preprocess/tests/test_commodity_multi_contract_feature_selection.py`

- [x] **Step 1: Import manifest classes in the pipeline**

Add to `data_preprocess/operator_futures/feature_selection/muti_contract/pipeline.py`:

```python
from operator_futures.feature_selection.manifests import (
    FeatureSelectionContractRecord,
    FeatureSelectionManifest,
    FeatureSelectionResult,
    FilteredOutputRecord,
)
```

- [x] **Step 2: Return typed filtered output records**

Change `_write_filtered_outputs()` return annotation and append call:

```python
) -> list[FilteredOutputRecord]:
```

Replace the dict append with:

```python
        outputs.append(
            FilteredOutputRecord(
                contract=contract,
                output_path=str(output_path),
                output_row_count=filtered.height,
                output_column_count=len(filtered.columns),
            )
        )
```

- [x] **Step 3: Type the public return and per-contract records**

Change `run_feature_selection()` return annotation:

```python
) -> FeatureSelectionResult:
```

Change the existing `per_contract.append` call that appends a dict with `contract`, `input_path`, and `metric_path` keys to:

```python
        per_contract.append(
            FeatureSelectionContractRecord(
                contract=contract,
                input_path=str(input_dir / f"{contract}.feather"),
                metric_path=str(metric_path),
            )
        )
```

- [x] **Step 4: Build the valid manifest object**

Replace the valid-stage manifest dict/write/return block with:

```python
        manifest = FeatureSelectionManifest(
            symbol=symbol,
            target_freq=target_freq,
            stage=stage,
            split_input_dir=str(input_dir),
            evaluated_feature_file=str(train_feature_file),
            evaluated_feature_count=len(feature_universe),
            evaluated_features=feature_universe,
            windows_list=windows_list,
            aggregate_metrics_path=str(aggregate_path),
            contracts=per_contract,
            report_only=True,
        )
        manifest_path = output_dir / "feature_selection_manifest.json"
        manifest.write_json(manifest_path)
        return FeatureSelectionResult(output_dir=output_dir, manifest=manifest)
```

- [x] **Step 5: Build the train manifest object**

Replace the train-stage manifest dict/write/return block with:

```python
    manifest = FeatureSelectionManifest(
        symbol=symbol,
        target_freq=target_freq,
        stage=stage,
        split_input_dir=str(input_dir),
        selected_feature_file=str(selected_file),
        selected_feature_count=len(selected_features),
        selected_features=selected_features,
        windows_list=windows_list,
        composite_drop_ratio=composite_drop_ratio,
        aggregate_metrics_path=str(aggregate_path),
        filter_results=filter_results,
        contracts=per_contract,
        filtered_outputs=filtered_outputs,
    )
    manifest_path = output_dir / "feature_selection_manifest.json"
    manifest.write_json(manifest_path)
    return FeatureSelectionResult(output_dir=output_dir, manifest=manifest)
```

- [x] **Step 6: Run the multi-contract focused tests**

Run:

```bash
conda activate finetf && pytest data_preprocess/tests/test_commodity_multi_contract_feature_selection.py -q
```

Expected: PASS.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 4: Refactor contract feature union manifest boundary

> **trace:** plan-ready.md -> `### Task 4: Refactor contract feature union manifest boundary` | tasks.md -> `- [ ] 1.4 Refactor `data_preprocess/operator_futures/feature_selection/contract_feature_union.py` to use `FeatureUnionManifest` and return `FeatureUnionResult` while preserving `feature_union_manifest.json`.`
> **sync:** tasks.md -> `- [ ] 1.4 Refactor `data_preprocess/operator_futures/feature_selection/contract_feature_union.py` to use `FeatureUnionManifest` and return `FeatureUnionResult` while preserving `feature_union_manifest.json`.` | plan-ready.md -> `### Task 4: Refactor contract feature union manifest boundary`

**Files:**
- Modify: `data_preprocess/operator_futures/feature_selection/contract_feature_union.py`
- Test: `data_preprocess/tests/test_commodity_feature_pipeline.py`

- [x] **Step 1: Import feature union dataclasses**

Add to `contract_feature_union.py`:

```python
from operator_futures.feature_selection.manifests import (
    ContractOutputShape,
    FeatureUnionManifest,
    FeatureUnionResult,
)
```

- [x] **Step 2: Change typed output shape storage**

Change the local variable annotation:

```python
    per_contract_output_shapes: dict[str, ContractOutputShape] = {}
```

Replace the shape assignment with:

```python
            per_contract_output_shapes[contract.contract] = ContractOutputShape(
                rows=out.height,
                columns=len(out.columns),
            )
```

- [x] **Step 3: Return a `FeatureUnionResult`**

Change the function return annotation:

```python
) -> FeatureUnionResult:
```

Replace manifest dict/write/return at the end with:

```python
    manifest = FeatureUnionManifest(
        symbol=symbol,
        target_freq=target_freq,
        start_date=start_date,
        end_date=end_date,
        summary_path=str(summary_path),
        contracts=list(contract_features),
        contract_state_feature_paths=contract_feature_paths,
        per_contract_feature_counts={
            contract: len(features) for contract, features in contract_features.items()
        },
        state_feature_count=len(union),
        state_features=union,
        candidate_source_path=candidate_path,
        all_feature_path=all_feature_path,
        ic_result_path=ic_result_path,
        finalize_filtered_df=finalize_filtered_df,
        per_contract_output_paths=per_contract_outputs,
        per_contract_output_shapes=per_contract_output_shapes,
    )
    manifest.write_json(output_dir / "feature_union_manifest.json")
    logger.info(
        "Wrote contract feature union: symbol=%s contracts=%d state_features=%d output_dir=%s",
        symbol,
        len(contract_features),
        len(union),
        output_dir,
    )
    return FeatureUnionResult(output_dir=output_dir, manifest=manifest)
```

- [x] **Step 4: Run the feature pipeline focused tests**

Run:

```bash
conda activate finetf && pytest data_preprocess/tests/test_commodity_feature_pipeline.py -q
```

Expected: PASS.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 5: Refactor IC and Rank IC JSON score boundaries

> **trace:** plan-ready.md -> `### Task 5: Refactor IC and Rank IC JSON score boundaries` | tasks.md -> `- [ ] 1.5 Refactor `data_preprocess/operator_futures/feature_selection/ic_correlation.py` and `rank_ic_correlation.py` to use `FeatureScoreWindow` and return result objects while preserving window score JSON.`
> **sync:** tasks.md -> `- [ ] 1.5 Refactor `data_preprocess/operator_futures/feature_selection/ic_correlation.py` and `rank_ic_correlation.py` to use `FeatureScoreWindow` and return result objects while preserving window score JSON.` | plan-ready.md -> `### Task 5: Refactor IC and Rank IC JSON score boundaries`

**Files:**
- Modify: `data_preprocess/operator_futures/feature_selection/ic_correlation.py`
- Modify: `data_preprocess/operator_futures/feature_selection/rank_ic_correlation.py`
- Test: `data_preprocess/tests/test_feature_selection_polars.py`

- [x] **Step 1: Import IC result dataclasses**

Add to `ic_correlation.py`:

```python
from operator_futures.feature_selection.manifests import (
    FeatureScoreWindow,
    IcCorrelationResult,
)
```

- [x] **Step 2: Use `FeatureScoreWindow` in IC correlation**

Inside `ic_correlation.main(args)`, before the window loop add:

```python
    score_windows: list[FeatureScoreWindow] = []
```

Replace:

```python
        cor = {feature: result for feature, result in sorted_pairs}
        with open(output_dir / "ic_window_{}.json".format(window_length), "w") as f:
            json.dump(cor, f)
```

with:

```python
        score_window = FeatureScoreWindow(
            window_length=window_length,
            scores={feature: result for feature, result in sorted_pairs},
        )
        score_windows.append(score_window)
        cor = score_window.to_dict()
        score_window.write_json(output_dir / "ic_window_{}.json".format(window_length))
```

Replace the final return:

```python
    return out
```

with:

```python
    return IcCorrelationResult(
        frame=out,
        output_dir=output_dir,
        selected_features=state_feature,
        score_windows=score_windows,
    )
```

- [x] **Step 3: Import Rank IC result dataclasses**

Add to `rank_ic_correlation.py`:

```python
from operator_futures.feature_selection.manifests import (
    FeatureScoreWindow,
    RankIcCorrelationResult,
)
```

- [x] **Step 4: Ensure Rank IC parser defines commodity options already read by `main`**

Add parser arguments after `windows_list` in `rank_ic_correlation.py`:

```python
parser.add_argument(
    "--market_type",
    type=str,
    default="crypto_futures",
    choices=["crypto_futures", "commodity_futures"],
    help="the market type of the preprocessed data",
)
parser.add_argument(
    "--orderbook_depth",
    type=int,
    default=25,
    help="the available orderbook depth",
)
```

- [x] **Step 5: Use `FeatureScoreWindow` in Rank IC correlation**

Inside `rank_ic_correlation.main(args)`, before the window loop add:

```python
    score_windows: list[FeatureScoreWindow] = []
```

Replace:

```python
        cor = {feature: result for feature, result in sorted_pairs}
        with open(output_dir / f"rank_ic_window_{window_length}.json", "w") as f:
            json.dump(cor, f)
```

with:

```python
        score_window = FeatureScoreWindow(
            window_length=window_length,
            scores={feature: result for feature, result in sorted_pairs},
        )
        score_windows.append(score_window)
        cor = score_window.to_dict()
        score_window.write_json(output_dir / f"rank_ic_window_{window_length}.json")
```

Replace the final return:

```python
    return out
```

with:

```python
    return RankIcCorrelationResult(
        frame=out,
        output_dir=output_dir,
        selected_features=selected_feature_names,
        score_windows=score_windows,
    )
```

- [x] **Step 6: Run feature selection Polars focused tests**

Run:

```bash
conda activate finetf && pytest data_preprocess/tests/test_feature_selection_polars.py -q
```

Expected: PASS.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 6: Run focused verification

> **trace:** plan-ready.md -> `### Task 6: Run focused verification` | tasks.md -> `- [ ] 1.6 Run focused verification for feature selection tests, Python compilation, and OpenSpec validation.`
> **sync:** tasks.md -> `- [ ] 1.6 Run focused verification for feature selection tests, Python compilation, and OpenSpec validation.` | plan-ready.md -> `### Task 6: Run focused verification`

**Files:**
- Modify: `openspec/changes/refactor-feature-selection-json-objects/tasks.md`
- Modify: `openspec/changes/refactor-feature-selection-json-objects/plan-ready.md`
- Modify: `docs/superpowers/plans/2026-07-22-refactor-feature-selection-json-objects.md`

- [x] **Step 1: Run all focused tests named in the OpenSpec proposal**

Run:

```bash
conda activate finetf && pytest data_preprocess/tests/test_commodity_multi_contract_feature_selection.py data_preprocess/tests/test_commodity_feature_pipeline.py data_preprocess/tests/test_feature_selection_polars.py
```

Expected: PASS.

- [x] **Step 2: Compile changed runtime modules**

Run:

```bash
conda activate finetf && python -m py_compile data_preprocess/operator_futures/feature_selection/manifests.py data_preprocess/operator_futures/feature_selection/muti_contract/pipeline.py data_preprocess/operator_futures/feature_selection/contract_feature_union.py data_preprocess/operator_futures/feature_selection/ic_correlation.py data_preprocess/operator_futures/feature_selection/rank_ic_correlation.py
```

Expected: command exits 0.

- [x] **Step 3: Validate the OpenSpec change**

Run:

```bash
openspec validate refactor-feature-selection-json-objects --strict
```

Expected: `Change 'refactor-feature-selection-json-objects' is valid`.

- [x] **Step 4: Review the final diff scope**

Run:

```bash
git diff -- data_preprocess/operator_futures/feature_selection data_preprocess/tests openspec/changes/refactor-feature-selection-json-objects docs/superpowers/plans/2026-07-22-refactor-feature-selection-json-objects.md
```

Expected: diff is limited to the dataclass JSON object refactor, focused tests, and sddflow tracking docs.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）
