# Add Commodity Contract Dataset Manifest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a manifest-driven FineFT commodity datahandler path that creates non-overlapping 5:3:2 train/valid/test datasets from contract-scoped commodity futures feature files.

**Architecture:** Add a focused `FineFT/datahandler/commodity_contract_dataset.py` CLI that reads `main_contract_summary.json`, computes global split boundaries, writes a dataset manifest, emits contract-scoped stage files, creates train slices, and creates valid dynamic slices. Keep the old single-file `preprocess_data.py` path intact, while updating the existing commodity shell entrypoints to call the new CLI.

**Tech Stack:** Python 3.10, pandas, numpy, pyarrow/feather, pytest, bash, OpenSpec.

**Traceability (sddflow):**
- plan-ready: `openspec/changes/add-commodity-contract-dataset-manifest/plan-ready.md`
- tasks: `openspec/changes/add-commodity-contract-dataset-manifest/tasks.md`
- plan: `docs/superpowers/plans/2026-07-13-add-commodity-contract-dataset-manifest.md`

---

### Task 1: Commodity split boundary tests

> **trace:** plan-ready.md → `### Task 1: Commodity split boundary tests` | tasks.md → `- [ ] 1.1 Add focused tests for commodity split boundary calculation from multi-contract summary trading days.`
> **sync:** tasks.md → `- [ ] 1.1 Add focused tests for commodity split boundary calculation from multi-contract summary trading days.` | plan-ready.md → `### Task 1: Commodity split boundary tests`

**Files:**
- Create: `FineFT/tests/datahandler/test_commodity_contract_dataset.py`
- Implement in Task 3: `FineFT/datahandler/commodity_contract_dataset.py`

- [x] **Step 1: Write the failing split-boundary test**

Add this test file:

```python
from pathlib import Path

import pytest

from FineFT.datahandler.commodity_contract_dataset import (
    calculate_split_boundaries,
)


def _contract(name, dates):
    return {
        "contract": name,
        "trading_days": [
            {
                "trading_day": date.replace("-", ""),
                "date": date,
                "source_file": f"/raw/{name}/{date}.csv",
                "daily_volume": 100,
            }
            for date in dates
        ],
    }


def test_calculate_split_boundaries_uses_union_trading_days_5_3_2():
    summary = {
        "symbol": "fu",
        "contracts": [
            _contract("fu2601", ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06"]),
            _contract("fu2605", ["2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-10"]),
        ],
    }

    boundaries = calculate_split_boundaries(summary, train_ratio=5, valid_ratio=3, test_ratio=2)

    assert boundaries == {
        "start": "2026-01-01",
        "a": "2026-01-06",
        "b": "2026-01-09",
        "c": "2026-01-11",
    }


def test_calculate_split_boundaries_requires_non_empty_sets():
    summary = {"symbol": "fu", "contracts": [_contract("fu2601", ["2026-01-01", "2026-01-02"])]}

    with pytest.raises(ValueError, match="start < a < b < c"):
        calculate_split_boundaries(summary, train_ratio=5, valid_ratio=3, test_ratio=2)
```

- [x] **Step 2: Run test to verify it fails**

Run: `conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py -q`

Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `FineFT.datahandler.commodity_contract_dataset`.

- [x] **Step 3: Commit the failing test**

```bash
git add FineFT/tests/datahandler/test_commodity_contract_dataset.py
git commit -m "test: add commodity contract split boundary tests"
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 2: Manifest builder tests

> **trace:** plan-ready.md → `### Task 2: Manifest builder tests` | tasks.md → `- [ ] 1.2 Add manifest-builder tests for 5:3:2 non-overlapping set assignment, contract/date intersections, input paths, stage output paths, and slice plans.`
> **sync:** tasks.md → `- [ ] 1.2 Add manifest-builder tests for 5:3:2 non-overlapping set assignment, contract/date intersections, input paths, stage output paths, and slice plans.` | plan-ready.md → `### Task 2: Manifest builder tests`

**Files:**
- Modify: `FineFT/tests/datahandler/test_commodity_contract_dataset.py`
- Implement in Task 3: `FineFT/datahandler/commodity_contract_dataset.py`

- [x] **Step 1: Add manifest-builder test**

Append:

```python
from FineFT.datahandler.commodity_contract_dataset import build_dataset_manifest


def test_build_dataset_manifest_records_contract_intersections_and_slice_plan(tmp_path):
    summary = {
        "symbol": "fu",
        "contracts": [
            _contract("fu2601", ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06"]),
            _contract("fu2605", ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-10"]),
        ],
    }
    boundaries = {"start": "2026-01-01", "a": "2026-01-06", "b": "2026-01-09", "c": "2026-01-11"}

    manifest = build_dataset_manifest(
        summary=summary,
        boundaries=boundaries,
        symbol="fu",
        target_freq="5min",
        start_date="2026-01-01",
        end_date="2026-04-01",
        input_root=tmp_path / "SCALE_SAVE",
        feature_union_path=tmp_path / "FEATURE_UNION" / "state_features.npy",
        output_root=tmp_path / "dataset",
        chunk_length=2,
        early_stop=1,
    )

    assert manifest["split_ratio"] == {"train": 5, "valid": 3, "test": 2}
    assert manifest["boundaries"] == boundaries
    train_contracts = {item["contract"]: item for item in manifest["sets"]["train"]["contracts"]}
    valid_contracts = {item["contract"]: item for item in manifest["sets"]["valid"]["contracts"]}
    test_contracts = {item["contract"]: item for item in manifest["sets"]["test"]["contracts"]}
    assert train_contracts["fu2601"]["trading_days"] == ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]
    assert train_contracts["fu2605"]["trading_days"] == ["2026-01-05"]
    assert valid_contracts["fu2601"]["trading_days"] == ["2026-01-06"]
    assert valid_contracts["fu2605"]["trading_days"] == ["2026-01-06", "2026-01-07", "2026-01-08"]
    assert test_contracts["fu2605"]["trading_days"] == ["2026-01-09", "2026-01-10"]
    assert train_contracts["fu2601"]["output_path"].endswith("dataset/fu/train/df_fu2601.feather")
    assert train_contracts["fu2601"]["slice_outputs"][0]["path"].endswith("dataset/fu/train/slice/df_0.feather")
```

- [x] **Step 2: Run test to verify it fails**

Run: `conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py -q`

Expected: FAIL because `build_dataset_manifest` is not implemented.

- [x] **Step 3: Commit the failing manifest test**

```bash
git add FineFT/tests/datahandler/test_commodity_contract_dataset.py
git commit -m "test: add commodity contract manifest tests"
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 3: Boundary and manifest implementation

> **trace:** plan-ready.md → `### Task 3: Boundary and manifest implementation` | tasks.md → `- [ ] 1.3 Implement `FineFT/datahandler/commodity_contract_dataset.py` boundary calculation and manifest generation.`
> **sync:** tasks.md → `- [ ] 1.3 Implement `FineFT/datahandler/commodity_contract_dataset.py` boundary calculation and manifest generation.` | plan-ready.md → `### Task 3: Boundary and manifest implementation`

**Files:**
- Create: `FineFT/datahandler/commodity_contract_dataset.py`
- Modify: `FineFT/tests/datahandler/test_commodity_contract_dataset.py`

- [x] **Step 1: Add minimal boundary and manifest implementation**

Create `FineFT/datahandler/commodity_contract_dataset.py`:

```python
import argparse
import json
import math
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_date(value):
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def _date_str(value):
    return value.isoformat()


def _collect_union_dates(summary):
    dates = set()
    for contract in summary.get("contracts", []):
        for day in contract.get("trading_days", []):
            dates.add(str(day["date"]))
    return sorted(dates)


def calculate_split_boundaries(summary, train_ratio=5, valid_ratio=3, test_ratio=2):
    dates = _collect_union_dates(summary)
    total = len(dates)
    ratio_total = train_ratio + valid_ratio + test_ratio
    train_count = int(math.floor(total * train_ratio / ratio_total))
    valid_count = int(math.floor(total * valid_ratio / ratio_total))
    test_count = total - train_count - valid_count
    if train_count <= 0 or valid_count <= 0 or test_count <= 0:
        raise ValueError("cannot satisfy start < a < b < c with non-empty train/valid/test sets")
    start = dates[0]
    a = dates[train_count]
    b = dates[train_count + valid_count]
    c = _date_str(_parse_date(dates[-1]) + timedelta(days=1))
    if not (_parse_date(start) < _parse_date(a) < _parse_date(b) < _parse_date(c)):
        raise ValueError("cannot satisfy start < a < b < c with computed boundaries")
    return {"start": start, "a": a, "b": b, "c": c}


def _in_range(date, start, end):
    parsed = _parse_date(date)
    return _parse_date(start) <= parsed < _parse_date(end)


def _input_path(input_root, symbol, contract, target_freq, start_date, end_date):
    return Path(input_root) / symbol / contract / target_freq / f"{start_date}-{end_date}" / "df.feather"


def _stage_output(output_root, symbol, set_name, contract):
    return Path(output_root) / symbol / set_name / f"df_{contract}.feather"


def _build_slice_plan(contract_days, output_root, symbol, contract, start_index, chunk_length, early_stop):
    outputs = []
    if not contract_days:
        return outputs, start_index
    row_start = 0
    index = start_index
    while row_start + chunk_length <= len(contract_days):
        row_end = min(row_start + chunk_length + early_stop, len(contract_days))
        outputs.append(
            {
                "index": index,
                "contract": contract,
                "path": str(Path(output_root) / symbol / "train" / "slice" / f"df_{index}.feather"),
                "source_output": str(_stage_output(output_root, symbol, "train", contract)),
                "trading_days": contract_days[row_start:row_end],
                "row_start": row_start,
                "row_end": row_end,
            }
        )
        index += 1
        row_start += chunk_length
    return outputs, index


def build_dataset_manifest(
    summary,
    boundaries,
    symbol,
    target_freq,
    start_date,
    end_date,
    input_root,
    feature_union_path,
    output_root,
    chunk_length,
    early_stop,
):
    ranges = {
        "train": (boundaries["start"], boundaries["a"]),
        "valid": (boundaries["a"], boundaries["b"]),
        "test": (boundaries["b"], boundaries["c"]),
    }
    manifest = {
        "symbol": symbol,
        "target_freq": target_freq,
        "split_ratio": {"train": 5, "valid": 3, "test": 2},
        "boundaries": boundaries,
        "state_features_path": str(Path(output_root) / symbol / "state_features.npy"),
        "feature_union_path": str(feature_union_path),
        "sets": {},
    }
    next_slice = 0
    for set_name, (range_start, range_end) in ranges.items():
        set_contracts = []
        skipped = []
        for contract in summary.get("contracts", []):
            contract_name = contract["contract"]
            days = [str(day["date"]) for day in contract.get("trading_days", []) if _in_range(day["date"], range_start, range_end)]
            if not days:
                skipped.append({"contract": contract_name, "reason": f"no trading days in {set_name} range"})
                continue
            record = {
                "contract": contract_name,
                "range": [range_start, range_end],
                "trading_days": days,
                "input_path": str(_input_path(input_root, symbol, contract_name, target_freq, start_date, end_date)),
                "output_path": str(_stage_output(output_root, symbol, set_name, contract_name)),
            }
            if set_name == "train":
                slices, next_slice = _build_slice_plan(days, output_root, symbol, contract_name, next_slice, chunk_length, early_stop)
                record["slice_outputs"] = slices
            set_contracts.append(record)
        manifest["sets"][set_name] = {
            "range": [range_start, range_end],
            "contracts": set_contracts,
            "skipped_contracts": skipped,
        }
    return manifest
```

- [x] **Step 2: Run boundary and manifest tests**

Run: `conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py -q`

Expected: PASS for Task 1 and Task 2 tests.

- [x] **Step 3: Commit implementation**

```bash
git add FineFT/datahandler/commodity_contract_dataset.py FineFT/tests/datahandler/test_commodity_contract_dataset.py
git commit -m "feat: add commodity contract dataset manifest"
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 4: Contract stage dataset writer

> **trace:** plan-ready.md → `### Task 4: Contract stage dataset writer` | tasks.md → `- [ ] 1.4 Add stage dataset writer tests and implementation for `train/df_<contract>.feather`, `valid/df_<contract>.feather`, `test/df_<contract>.feather`, state feature copy, and absence of legacy `train.feather`/`valid.feather`/`test.feather`.`
> **sync:** tasks.md → `- [ ] 1.4 Add stage dataset writer tests and implementation for `train/df_<contract>.feather`, `valid/df_<contract>.feather`, `test/df_<contract>.feather`, state feature copy, and absence of legacy `train.feather`/`valid.feather`/`test.feather`.` | plan-ready.md → `### Task 4: Contract stage dataset writer`

**Files:**
- Modify: `FineFT/datahandler/commodity_contract_dataset.py`
- Modify: `FineFT/datahandler/slice_model.py`
- Modify: `FineFT/tests/datahandler/test_commodity_contract_dataset.py`

- [x] **Step 1: Add stage writer test**

Append:

```python
import numpy as np
import pandas as pd

from FineFT.datahandler.commodity_contract_dataset import write_stage_datasets


def test_write_stage_datasets_filters_contract_files_and_omits_legacy_files(tmp_path):
    input_file = tmp_path / "SCALE_SAVE" / "fu" / "fu2601" / "5min" / "2026-01-01-2026-04-01" / "df.feather"
    input_file.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-06", "2026-01-09"]),
            "symbol": ["fu2601"] * 4,
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "bid1_price": [10.0, 11.0, 12.0, 13.0],
        }
    ).to_feather(input_file)
    feature_union = tmp_path / "FEATURE_UNION" / "state_features.npy"
    feature_union.parent.mkdir(parents=True)
    np.save(feature_union, np.array(["feature_a"]))
    manifest = {
        "symbol": "fu",
        "state_features_path": str(tmp_path / "dataset" / "fu" / "state_features.npy"),
        "feature_union_path": str(feature_union),
        "sets": {
            "train": {"contracts": [{"contract": "fu2601", "trading_days": ["2026-01-01", "2026-01-02"], "input_path": str(input_file), "output_path": str(tmp_path / "dataset" / "fu" / "train" / "df_fu2601.feather")}], "skipped_contracts": []},
            "valid": {"contracts": [{"contract": "fu2601", "trading_days": ["2026-01-06"], "input_path": str(input_file), "output_path": str(tmp_path / "dataset" / "fu" / "valid" / "df_fu2601.feather")}], "skipped_contracts": []},
            "test": {"contracts": [{"contract": "fu2601", "trading_days": ["2026-01-09"], "input_path": str(input_file), "output_path": str(tmp_path / "dataset" / "fu" / "test" / "df_fu2601.feather")}], "skipped_contracts": []},
        },
    }

    write_stage_datasets(manifest)

    assert pd.read_feather(tmp_path / "dataset" / "fu" / "train" / "df_fu2601.feather")["feature_a"].tolist() == [1.0, 2.0]
    assert pd.read_feather(tmp_path / "dataset" / "fu" / "valid" / "df_fu2601.feather")["feature_a"].tolist() == [3.0]
    assert pd.read_feather(tmp_path / "dataset" / "fu" / "test" / "df_fu2601.feather")["feature_a"].tolist() == [4.0]
    assert np.load(tmp_path / "dataset" / "fu" / "state_features.npy", allow_pickle=True).tolist() == ["feature_a"]
    assert not (tmp_path / "dataset" / "fu" / "train.feather").exists()
    assert not (tmp_path / "dataset" / "fu" / "valid.feather").exists()
    assert not (tmp_path / "dataset" / "fu" / "test.feather").exists()
```

- [x] **Step 2: Implement stage writer**

Append to `commodity_contract_dataset.py`:

```python
def _date_series(df):
    if "trading_day" in df.columns:
        return pd.to_datetime(df["trading_day"].astype(str)).dt.strftime("%Y-%m-%d")
    if "TradingDay" in df.columns:
        return pd.to_datetime(df["TradingDay"].astype(str)).dt.strftime("%Y-%m-%d")
    if "timestamp" not in df.columns:
        raise ValueError("commodity dataset input missing timestamp column")
    return pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d")


def _filter_days(df, trading_days):
    day_set = set(trading_days)
    filtered = df.loc[_date_series(df).isin(day_set)].copy()
    if filtered.empty:
        raise ValueError(f"planned trading days produced empty output: {sorted(day_set)}")
    if "timestamp" in filtered.columns:
        filtered = filtered.sort_values("timestamp")
    return filtered.reset_index(drop=True)


def write_stage_datasets(manifest):
    feature_union_path = Path(manifest["feature_union_path"])
    if not feature_union_path.exists():
        raise FileNotFoundError(f"Missing feature union state_features.npy: {feature_union_path}")
    state_features_path = Path(manifest["state_features_path"])
    state_features_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(feature_union_path, state_features_path)
    for set_info in manifest["sets"].values():
        for contract in set_info["contracts"]:
            input_path = Path(contract["input_path"])
            if not input_path.exists():
                raise FileNotFoundError(f"Missing df.feather for contract {contract['contract']}: {input_path}")
            df = pd.read_feather(input_path)
            output_df = _filter_days(df, contract["trading_days"])
            output_path = Path(contract["output_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_df.to_feather(output_path)
```

- [x] **Step 3: Run stage writer tests**

Run: `conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py -q`

Expected: PASS for boundary, manifest, and stage writer tests.

- [x] **Step 4: Commit stage writer**

```bash
git add FineFT/datahandler/commodity_contract_dataset.py FineFT/tests/datahandler/test_commodity_contract_dataset.py
git commit -m "feat: write commodity contract stage datasets"
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 5: Train slice writer

> **trace:** plan-ready.md → `### Task 5: Train slice writer` | tasks.md → `- [ ] 1.5 Add train slice writer tests and implementation for continuous `train/slice/df_*.feather` numbering, no-cross-contract slices, and `early_stop` clipping inside train.`
> **sync:** tasks.md → `- [ ] 1.5 Add train slice writer tests and implementation for continuous `train/slice/df_*.feather` numbering, no-cross-contract slices, and `early_stop` clipping inside train.` | plan-ready.md → `### Task 5: Train slice writer`

**Files:**
- Modify: `FineFT/datahandler/commodity_contract_dataset.py`
- Modify: `FineFT/datahandler/slice_model.py`
- Modify: `FineFT/tests/datahandler/test_commodity_contract_dataset.py`

- [x] **Step 1: Add train slice writer test**

Append:

```python
from FineFT.datahandler.commodity_contract_dataset import write_train_slices


def test_write_train_slices_uses_contiguous_indices_and_single_contract_files(tmp_path):
    train_dir = tmp_path / "dataset" / "fu" / "train"
    train_dir.mkdir(parents=True)
    pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=5, freq="D"), "symbol": ["fu2601"] * 5, "feature_a": range(5)}).to_feather(train_dir / "df_fu2601.feather")
    pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=3, freq="D"), "symbol": ["fu2605"] * 3, "feature_a": range(10, 13)}).to_feather(train_dir / "df_fu2605.feather")
    manifest = {
        "sets": {
            "train": {
                "contracts": [
                    {"contract": "fu2601", "output_path": str(train_dir / "df_fu2601.feather"), "slice_outputs": [{"path": str(train_dir / "slice" / "df_0.feather"), "row_start": 0, "row_end": 3}, {"path": str(train_dir / "slice" / "df_1.feather"), "row_start": 2, "row_end": 5}]},
                    {"contract": "fu2605", "output_path": str(train_dir / "df_fu2605.feather"), "slice_outputs": [{"path": str(train_dir / "slice" / "df_2.feather"), "row_start": 0, "row_end": 3}]},
                ]
            }
        }
    }

    write_train_slices(manifest)

    slice_paths = sorted((train_dir / "slice").glob("df_*.feather"))
    assert [path.name for path in slice_paths] == ["df_0.feather", "df_1.feather", "df_2.feather"]
    assert pd.read_feather(slice_paths[0])["symbol"].unique().tolist() == ["fu2601"]
    assert pd.read_feather(slice_paths[2])["symbol"].unique().tolist() == ["fu2605"]
```

- [x] **Step 2: Implement train slice writer**

Append:

```python
def write_train_slices(manifest):
    expected_index = 0
    for contract in manifest["sets"]["train"]["contracts"]:
        df = pd.read_feather(contract["output_path"])
        for slice_info in contract.get("slice_outputs", []):
            if int(slice_info["index"]) != expected_index and "index" in slice_info:
                raise ValueError("train slice indices must be continuous")
            row_start = int(slice_info["row_start"])
            row_end = int(slice_info["row_end"])
            sliced = df.iloc[row_start:row_end].reset_index(drop=True)
            if sliced.empty:
                continue
            output_path = Path(slice_info["path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sliced.to_feather(output_path)
            expected_index += 1
```

If Task 5 test omits `index`, adjust the continuity check:

```python
if "index" in slice_info and int(slice_info["index"]) != expected_index:
    raise ValueError("train slice indices must be continuous")
```

- [x] **Step 3: Run train slice tests**

Run: `conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py -q`

Expected: PASS including contiguous `train/slice` outputs.

- [x] **Step 4: Commit train slice writer**

```bash
git add FineFT/datahandler/commodity_contract_dataset.py FineFT/datahandler/slice_model.py FineFT/tests/datahandler/test_commodity_contract_dataset.py
git commit -m "feat: write commodity train slices"
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 6: Valid dynamic slice writer

> **trace:** plan-ready.md → `### Task 6: Valid dynamic slice writer` | tasks.md → `- [ ] 1.6 Add valid dynamic slicing tests and implementation so valid slices are produced per contract under `valid/label_*/` without cross-contract concatenation.`
> **sync:** tasks.md → `- [ ] 1.6 Add valid dynamic slicing tests and implementation so valid slices are produced per contract under `valid/label_*/` without cross-contract concatenation.` | plan-ready.md → `### Task 6: Valid dynamic slice writer`

**Files:**
- Modify: `FineFT/datahandler/commodity_contract_dataset.py`
- Modify: `FineFT/tests/datahandler/test_commodity_contract_dataset.py`

- [x] **Step 1: Add valid dynamic slice test with injected labeler**

Append:

```python
from FineFT.datahandler.commodity_contract_dataset import write_valid_dynamic_slices


def test_write_valid_dynamic_slices_runs_per_contract_without_cross_contract_output(tmp_path):
    valid_dir = tmp_path / "dataset" / "fu" / "valid"
    valid_dir.mkdir(parents=True)
    pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=3, freq="D"), "symbol": ["fu2601"] * 3, "feature_a": [1, 2, 3]}).to_feather(valid_dir / "df_fu2601.feather")
    pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=3, freq="D"), "symbol": ["fu2605"] * 3, "feature_a": [4, 5, 6]}).to_feather(valid_dir / "df_fu2605.feather")
    manifest = {"sets": {"valid": {"contracts": [{"contract": "fu2601", "output_path": str(valid_dir / "df_fu2601.feather")}, {"contract": "fu2605", "output_path": str(valid_dir / "df_fu2605.feather")}]}}}

    def labeler(df, contract):
        return [(0, df.iloc[:2].reset_index(drop=True)), (1, df.iloc[2:].reset_index(drop=True))]

    write_valid_dynamic_slices(manifest, dynamic_number=2, labeler=labeler)

    label0_files = sorted((valid_dir / "label_0").glob("df_*.feather"))
    label1_files = sorted((valid_dir / "label_1").glob("df_*.feather"))
    assert len(label0_files) == 2
    assert len(label1_files) == 2
    for path in label0_files + label1_files:
        assert len(pd.read_feather(path)["symbol"].unique()) == 1
```

- [x] **Step 2: Add a reusable slice-model entrypoint**

Add this helper to `FineFT/datahandler/slice_model.py` near `Linear_Market_Dynamics_Model`:

```python
def run_slice_model_to_directory(args, output_dir):
    model = Linear_Market_Dynamics_Model(args)
    raw_data = pd.read_feather(model.data_path)
    raw_data = model.prepare_raw_data(raw_data)
    process_data_path = os.path.join(output_dir, "valid_processed.feather")
    raw_data.to_feather(process_data_path)
    model.data_path = process_data_path

    worker = util.Worker(
        model.data_path,
        "slice_and_merge",
        filter_strength=model.filter_strength,
        key_indicator=model.key_indicator,
        timestamp=model.timestamp,
        tic=model.tic,
        labeling_method=model.labeling_method,
        min_length_limit=model.min_length_limit,
        merging_threshold=model.merging_threshold,
        merging_metric=model.merging_metric,
        merging_dynamic_constraint=model.merging_dynamic_constraint,
    )
    worker.fit(model.dynamic_number, model.max_length_expectation, model.min_length_limit)
    worker.label(output_dir)
    labeled_data = pd.concat([v for v in worker.data_dict.values()], axis=0)
    data = pd.read_feather(model.data_path)
    merge_keys = [model.timestamp, model.tic, model.key_indicator]
    merged_data = data.merge(
        labeled_data, how="left", on=merge_keys, suffixes=("", "_DROP")
    ).filter(regex="^(?!.*_DROP)")
    return merged_data
```

This helper reuses the same `Worker` path as `Linear_Market_Dynamics_Model.run()` but returns the labeled DataFrame so the commodity writer controls the final `valid/label_*` output paths.

- [x] **Step 3: Implement valid dynamic slice writer with injection point**

Append:

```python
def _default_labeler(df, contract):
    raise RuntimeError("default commodity valid labeler must be provided by build_valid_labeler")


def build_valid_labeler(dynamic_number, timestamp="timestamp"):
    from types import SimpleNamespace
    from tempfile import TemporaryDirectory
    from FineFT.datahandler.slice_model import run_slice_model_to_directory

    def labeler(df, contract):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / f"{contract}.feather"
            df.reset_index(drop=True).to_feather(tmp_path)
            args = SimpleNamespace(
                data_path=str(tmp_path),
                filter_strength=1,
                dynamic_number=dynamic_number,
                max_length_expectation=864,
                key_indicator="mark_price",
                timestamp=timestamp,
                tic="symbol",
                labeling_method="slope",
                min_length_limit=288,
                merging_metric="DTW_distance",
                merging_threshold=0.0003,
                merging_dynamic_constraint=1,
            )
            labeled = run_slice_model_to_directory(args, tmp_dir)
            segments = []
            previous_label = int(labeled.label.iloc[0])
            previous_start = 0
            for index in range(len(labeled)):
                current_label = int(labeled.label.iloc[index])
                if current_label != previous_label:
                    segments.append((previous_label, labeled.iloc[previous_start:index].reset_index(drop=True)))
                    previous_start = index
                    previous_label = current_label
            segments.append((previous_label, labeled.iloc[previous_start:].reset_index(drop=True)))
            return segments
    return labeler


def write_valid_dynamic_slices(manifest, dynamic_number=5, labeler=None):
    labeler = labeler or build_valid_labeler(dynamic_number)
    valid_contracts = manifest["sets"]["valid"]["contracts"]
    valid_root = Path(valid_contracts[0]["output_path"]).parent if valid_contracts else None
    if valid_root is None:
        return
    counters = {label: 0 for label in range(dynamic_number)}
    for label in range(dynamic_number):
        (valid_root / f"label_{label}").mkdir(parents=True, exist_ok=True)
    for contract in valid_contracts:
        df = pd.read_feather(contract["output_path"])
        for label, segment in labeler(df, contract["contract"]):
            label = int(label)
            if label < 0 or label >= dynamic_number:
                raise ValueError(f"invalid dynamic label {label} for contract {contract['contract']}")
            if segment.empty:
                continue
            output_path = valid_root / f"label_{label}" / f"df_{counters[label]}.feather"
            segment.reset_index(drop=True).to_feather(output_path)
            counters[label] += 1
```

The injected `labeler` keeps unit tests deterministic. The default production labeler is wired to `slice_model.py` in this task, so the commodity CLI uses the same dynamic labeling path as the existing valid split flow.

- [x] **Step 4: Run valid dynamic slice tests**

Run: `conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py -q`

Expected: PASS with per-contract label outputs.

- [x] **Step 5: Commit valid dynamic writer**

```bash
git add FineFT/datahandler/commodity_contract_dataset.py FineFT/tests/datahandler/test_commodity_contract_dataset.py
git commit -m "feat: write commodity valid dynamic slices"
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 7: VAE data creation compatibility

> **trace:** plan-ready.md → `### Task 7: VAE data creation compatibility` | tasks.md → `- [ ] 1.7 Update `vae_data_creation.py` tests and implementation so commodity VAE generation reads `valid/label_*/*.feather` and `test/df_<contract>.feather` without requiring `test.feather`.`
> **sync:** tasks.md → `- [ ] 1.7 Update `vae_data_creation.py` tests and implementation so commodity VAE generation reads `valid/label_*/*.feather` and `test/df_<contract>.feather` without requiring `test.feather`.` | plan-ready.md → `### Task 7: VAE data creation compatibility`

**Files:**
- Modify: `FineFT/datahandler/vae_data_creation.py`
- Modify: `FineFT/tests/datahandler/test_vae_data_creation.py`

- [x] **Step 1: Add VAE compatibility test**

Append to `FineFT/tests/datahandler/test_vae_data_creation.py`:

```python
def test_make_data_reads_contract_scoped_test_directory(tmp_path):
    dataset_path = tmp_path / "dataset" / "fu"
    (dataset_path / "valid" / "label_0").mkdir(parents=True)
    (dataset_path / "test").mkdir()
    np.save(dataset_path / "state_features.npy", np.array(["feature_a", "feature_b"]))
    pd.DataFrame({"feature_a": [1.0], "feature_b": [2.0]}).to_feather(dataset_path / "valid" / "label_0" / "df_0.feather")
    pd.DataFrame({"feature_a": [3.0], "feature_b": [4.0]}).to_feather(dataset_path / "test" / "df_fu2601.feather")
    pd.DataFrame({"feature_a": [5.0], "feature_b": [6.0]}).to_feather(dataset_path / "test" / "df_fu2605.feather")

    make_data(SimpleNamespace(base_path=str(tmp_path / "dataset"), dataset_name="fu", save_path=str(tmp_path / "dataset")))

    np.testing.assert_array_equal(
        np.load(dataset_path / "VAE_data" / "test.npy"),
        np.array([[3.0, 4.0], [5.0, 6.0]]),
    )
```

- [x] **Step 2: Implement contract-scoped test fallback**

Modify `make_data` in `FineFT/datahandler/vae_data_creation.py`:

```python
    test_path = os.path.join(args.base_path, args.dataset_name, "test.feather")
    if os.path.exists(test_path):
        test_frames = [pd.read_feather(test_path)]
    else:
        test_dir = os.path.join(args.base_path, args.dataset_name, "test")
        test_frames = [
            pd.read_feather(os.path.join(test_dir, file_name))
            for file_name in sorted(os.listdir(test_dir))
            if file_name.endswith(".feather")
        ]
        if not test_frames:
            raise FileNotFoundError(
                f"missing test.feather and no test/df_<contract>.feather files under {test_dir}"
            )
    test_data = np.concatenate(
        [df[state_features].values for df in test_frames],
        axis=0,
    )
    np.save(os.path.join(save_path, "test.npy"), test_data)
```

Replace the existing bottom block that reads a single `test.feather`.

- [x] **Step 3: Run VAE tests**

Run: `conda activate finetf && pytest FineFT/tests/datahandler/test_vae_data_creation.py -q`

Expected: PASS for existing empty-label behavior and new contract-scoped test fallback.

- [x] **Step 4: Commit VAE compatibility**

```bash
git add FineFT/datahandler/vae_data_creation.py FineFT/tests/datahandler/test_vae_data_creation.py
git commit -m "feat: support contract-scoped VAE test data"
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 8: Commodity data handler scripts

> **trace:** plan-ready.md → `### Task 8: Commodity data handler scripts` | tasks.md → `- [ ] 1.8 Update `commodity_data_handler_fu.sh` and `commodity_data_handler_al.sh` to call the new multi-contract dataset tool and stop calling legacy single-file preprocess/slice commands.`
> **sync:** tasks.md → `- [ ] 1.8 Update `commodity_data_handler_fu.sh` and `commodity_data_handler_al.sh` to call the new multi-contract dataset tool and stop calling legacy single-file preprocess/slice commands.` | plan-ready.md → `### Task 8: Commodity data handler scripts`

**Files:**
- Modify: `FineFT/script/data/commodity_data_handler_fu.sh`
- Modify: `FineFT/script/data/commodity_data_handler_al.sh`
- Modify: `FineFT/tests/datahandler/test_commodity_contract_dataset.py`

- [x] **Step 1: Add script text assertions**

Append:

```python
def test_commodity_data_handler_scripts_use_contract_dataset_tool():
    root = Path(__file__).resolve().parents[3]
    for script_name, symbol in [
        ("commodity_data_handler_fu.sh", "fu"),
        ("commodity_data_handler_al.sh", "al"),
    ]:
        text = (root / "FineFT" / "script" / "data" / script_name).read_text()
        assert "commodity_contract_dataset.py" in text
        assert f"--symbol {symbol}" in text or f'--symbol "${{SYMBOL}}"' in text
        assert "preprocess_data.py --trading_pair" not in text
        assert "slice_model.py --data_path dataset/" not in text
```

- [x] **Step 2: Update `commodity_data_handler_fu.sh`**

Replace content with:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOTPATH=${ROOTPATH:-$(pwd)}
SYMBOL=${SYMBOL:-fu}
TARGET_FREQ=${TARGET_FREQ:-5min}
START_DATE=${START_DATE:-2026-01-01}
END_DATE=${END_DATE:-2026-04-01}
CHUNK_LENGTH=${CHUNK_LENGTH:-3200}
EARLY_STOP=${EARLY_STOP:-320}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate finetf
cd "${ROOTPATH}"

python FineFT/datahandler/commodity_contract_dataset.py \
  --summary_path "PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/${SYMBOL}/main_contract_summary.json" \
  --input_root "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE" \
  --feature_union_path "PREPROCESS_DATASET/commodity-futures/FEATURE_UNION/${SYMBOL}/${TARGET_FREQ}/${START_DATE}-${END_DATE}/state_features.npy" \
  --output_root "dataset/${TARGET_FREQ}" \
  --symbol "${SYMBOL}" \
  --target_freq "${TARGET_FREQ}" \
  --start_date "${START_DATE}" \
  --end_date "${END_DATE}" \
  --train_ratio 5 \
  --valid_ratio 3 \
  --test_ratio 2 \
  --chunk_length "${CHUNK_LENGTH}" \
  --early_stop "${EARLY_STOP}"

python FineFT/datahandler/vae_data_creation.py \
  --base_path "dataset/${TARGET_FREQ}" \
  --dataset_name "${SYMBOL}" \
  --save_path "dataset/${TARGET_FREQ}"
```

- [x] **Step 3: Update `commodity_data_handler_al.sh`**

Use the same script body with `SYMBOL=${SYMBOL:-al}`.

- [x] **Step 4: Run script assertion test**

Run: `conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py::test_commodity_data_handler_scripts_use_contract_dataset_tool -q`

Expected: PASS.

- [x] **Step 5: Commit script updates**

```bash
git add FineFT/script/data/commodity_data_handler_fu.sh FineFT/script/data/commodity_data_handler_al.sh FineFT/tests/datahandler/test_commodity_contract_dataset.py
git commit -m "feat: route commodity data handlers through contract datasets"
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 9: Documentation and final validation

> **trace:** plan-ready.md → `### Task 9: Documentation and final validation` | tasks.md → `- [ ] 1.9 Update or add datahandler documentation covering the commodity multi-contract dataset manifest workflow.`
> **sync:** tasks.md → `- [ ] 1.9 Update or add datahandler documentation covering the commodity multi-contract dataset manifest workflow.` | plan-ready.md → `### Task 9: Documentation and final validation`

**Files:**
- Modify: `docs/datahandler/data_preparation_analysis.zh_cn.md`
- Verify: `openspec/changes/add-commodity-contract-dataset-manifest/*`

- [x] **Step 1: Add commodity workflow docs section**

Append to `docs/datahandler/data_preparation_analysis.zh_cn.md`:

```markdown
## 商品期货多合约 FineFT 数据集

商品期货数据不再通过单个 `df.feather` 直接生成 `train.feather`、`valid.feather` 和 `test.feather`。商品入口读取 `main_contract_summary.json`、合约级 `SCALE_SAVE/{symbol}/{contract}/{target_freq}/{start_date}-{end_date}/df.feather` 和品种级 `FEATURE_UNION/{symbol}/{target_freq}/{start_date}-{end_date}/state_features.npy`。

数据集生成先基于 summary 中所有合约有效交易日的去重有序并集，按 `train:valid:test = 5:3:2` 计算全局边界：

```text
train: [start, a)
valid: [a, b)
test:  [b, c)
```

然后每个合约用自身有效交易日与全局区间求交，写入 `dataset/{target_freq}/{symbol}/dataset_manifest.json`。阶段数据集按合约落盘：

```text
dataset/{target_freq}/{symbol}/train/df_<contract>.feather
dataset/{target_freq}/{symbol}/valid/df_<contract>.feather
dataset/{target_freq}/{symbol}/test/df_<contract>.feather
```

训练实际读取 `train/slice/df_*.feather`。这些 slice 连续编号，且单个 slice 不跨合约、不跨 train 日期边界。验证动态切片逐合约写入 `valid/label_*/df_*.feather`，不会把多个合约拼接后再切片。
```

- [x] **Step 2: Run OpenSpec validation**

Run: `openspec validate add-commodity-contract-dataset-manifest --strict`

Expected: `Change 'add-commodity-contract-dataset-manifest' is valid`

- [x] **Step 3: Run focused tests**

Run: `conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py FineFT/tests/datahandler/test_vae_data_creation.py -q`

Expected: PASS.

- [x] **Step 4: Commit docs and validation-ready state**

```bash
git add docs/datahandler/data_preparation_analysis.zh_cn.md openspec/changes/add-commodity-contract-dataset-manifest docs/superpowers/plans/2026-07-13-add-commodity-contract-dataset-manifest.md
git commit -m "docs: plan commodity contract dataset manifest"
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

## Self-Review

- Spec coverage: Tasks cover boundary calculation, manifest, stage files, train slices, valid dynamic slices, VAE compatibility, script entrypoints, documentation, OpenSpec validation, and focused tests.
- Placeholder scan: No placeholder markers remain. The plan includes exact file paths, concrete test snippets, implementation snippets, commands, and expected results.
- Type consistency: Functions introduced in tests are implemented with matching names: `calculate_split_boundaries`, `build_dataset_manifest`, `write_stage_datasets`, `write_train_slices`, and `write_valid_dynamic_slices`.

## Amendment: 2026-07-13 Shell-Orchestrated Valid Slicing

### Task 10: Remove slice model coupling from dataset tool

> **trace:** plan-ready.md → `### Task 10: Remove slice model coupling from dataset tool` | tasks.md → `- [ ] 1.10 Remove any `slice_model.py` import/call and valid label output responsibility from `commodity_contract_dataset.py`.`
> **sync:** tasks.md → `- [ ] 1.10 Remove any `slice_model.py` import/call and valid label output responsibility from `commodity_contract_dataset.py`.` | plan-ready.md → `### Task 10: Remove slice model coupling from dataset tool`

**Files:**
- Modify: `FineFT/datahandler/commodity_contract_dataset.py`
- Modify: `FineFT/datahandler/slice_model.py`
- Modify: `FineFT/tests/datahandler/test_commodity_contract_dataset.py`

- [x] **Step 1: Add failing tests for no slice-model coupling**

Add tests that read `FineFT/datahandler/commodity_contract_dataset.py` as text and assert it does not contain `slice_model`, `build_valid_labeler`, or `write_valid_dynamic_slices`. Add a generation test that runs `run_dataset_generation(..., write_valid_slices=False)` or the updated default and asserts no `valid/label_0` directory is created.

Run: `conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py -q`
Expected: FAIL while current implementation still contains `slice_model` coupling.

- [x] **Step 2: Remove production coupling**

Remove `build_valid_labeler`, `write_valid_dynamic_slices`, and the `slice_model.py` helper that was added for commodity orchestration. Update `run_dataset_generation` so it never calls valid dynamic slicing and does not expose `write_valid_slices`.

Run: `conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py -q`
Expected: PASS for dataset-generation tests and no valid label output from `commodity_contract_dataset.py`.

- [x] **Step 3: Commit dataset-tool decoupling**

```bash
git add FineFT/datahandler/commodity_contract_dataset.py FineFT/datahandler/slice_model.py FineFT/tests/datahandler/test_commodity_contract_dataset.py
git commit -m "refactor: decouple commodity dataset from slice model"
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 11: Shell orchestrates per-contract slice_model

> **trace:** plan-ready.md → `### Task 11: Shell orchestrates per-contract slice_model` | tasks.md → `- [ ] 1.11 Update `commodity_data_handler_fu.sh` and `commodity_data_handler_al.sh` to call `slice_model.py` independently for each `valid/df_<contract>.feather` after dataset generation.`
> **sync:** tasks.md → `- [ ] 1.11 Update `commodity_data_handler_fu.sh` and `commodity_data_handler_al.sh` to call `slice_model.py` independently for each `valid/df_<contract>.feather` after dataset generation.` | plan-ready.md → `### Task 11: Shell orchestrates per-contract slice_model`

**Files:**
- Modify: `FineFT/script/data/commodity_data_handler_fu.sh`
- Modify: `FineFT/script/data/commodity_data_handler_al.sh`
- Modify: `FineFT/tests/datahandler/test_commodity_contract_dataset.py`

- [x] **Step 1: Add failing shell orchestration assertions**

Update script tests to require a loop over `dataset/${TARGET_FREQ}/${SYMBOL}/valid/df_*.feather`, require `FineFT/datahandler/slice_model.py --data_path "${valid_file}" --timestamp timestamp`, and continue rejecting `dataset/<symbol>/valid.feather`.

Run: `conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py::test_commodity_data_handler_scripts_use_contract_dataset_tool -q`
Expected: FAIL until the shell scripts include the per-contract slice loop.

- [x] **Step 2: Update both commodity shell scripts**

After `commodity_contract_dataset.py` completes, add:

```bash
for valid_file in "dataset/${TARGET_FREQ}/${SYMBOL}/valid"/df_*.feather; do
  [ -e "${valid_file}" ] || continue
  python FineFT/datahandler/slice_model.py --data_path "${valid_file}" --timestamp timestamp
done
```

Keep the VAE generation call after this loop.

Run: `conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py::test_commodity_data_handler_scripts_use_contract_dataset_tool -q`
Expected: PASS.

- [x] **Step 3: Commit shell orchestration**

```bash
git add -f FineFT/script/data/commodity_data_handler_fu.sh FineFT/script/data/commodity_data_handler_al.sh
git add FineFT/tests/datahandler/test_commodity_contract_dataset.py
git commit -m "feat: orchestrate commodity valid slicing in shell"
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 12: Update docs and validate amend

> **trace:** plan-ready.md → `### Task 12: Update docs and validate amend` | tasks.md → `- [ ] 1.12 Update tests and docs for shell-orchestrated valid slicing.`
> **sync:** tasks.md → `- [ ] 1.12 Update tests and docs for shell-orchestrated valid slicing.` | plan-ready.md → `### Task 12: Update docs and validate amend`

**Files:**
- Modify: `docs/datahandler/data_preparation_analysis.zh_cn.md`
- Verify: `openspec/changes/add-commodity-contract-dataset-manifest/*`

- [x] **Step 1: Update docs**

Revise the commodity multi-contract section to say the dataset tool only creates stage files and train slices. State that `commodity_data_handler_*.sh` then calls `slice_model.py` independently for each `valid/df_<contract>.feather` before `vae_data_creation.py`.

- [x] **Step 2: Run validation**

Run: `openspec validate add-commodity-contract-dataset-manifest --strict`
Expected: `Change 'add-commodity-contract-dataset-manifest' is valid`

Run: `conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py FineFT/tests/datahandler/test_vae_data_creation.py -q`
Expected: PASS.

- [x] **Step 3: Commit docs and validation state**

```bash
git add docs/datahandler/data_preparation_analysis.zh_cn.md openspec/changes/add-commodity-contract-dataset-manifest docs/superpowers/plans/2026-07-13-add-commodity-contract-dataset-manifest.md
git commit -m "docs: amend commodity valid slicing orchestration"
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

## Amendment: 2026-07-13 Manifest Row Counts

### Task 13: Manifest output row counts

> **trace:** plan-ready.md → `### Task 13: Manifest output row counts` | tasks.md → `- [ ] 1.13 Add manifest row counts for each contract `output_path` and each split's `contracts_total_count`.`
> **sync:** tasks.md → `- [ ] 1.13 Add manifest row counts for each contract `output_path` and each split's `contracts_total_count`.` | plan-ready.md → `### Task 13: Manifest output row counts`

**Files:**
- Modify: `FineFT/tests/datahandler/test_commodity_contract_dataset.py`
- Modify: `FineFT/datahandler/commodity_contract_dataset.py`
- Modify: `docs/datahandler/data_preparation_analysis.zh_cn.md`

- [x] **Step 1: Add failing manifest row-count test**

Extend the dataset generation test to load `dataset_manifest.json` after generation and assert:
- each contract record containing `output_path` also has `output_row_count`
- `output_row_count` equals the row count of the corresponding feather file
- `sets.train.contracts_total_count`, `sets.valid.contracts_total_count`, and `sets.test.contracts_total_count` equal the sums of their contract row counts

Run: `conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py -q`
Expected: FAIL while manifest lacks the new row count fields.

- [x] **Step 2: Populate counts from written stage files**

Update `write_stage_datasets(manifest)` so it records `output_row_count = len(output_df)` on each contract record and `contracts_total_count` on each split after writing stage outputs. Write `dataset_manifest.json` after those counts exist in `run_dataset_generation`.

Run: `conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py -q`
Expected: PASS.

- [x] **Step 3: Update docs and validate**

Document `output_row_count` and `contracts_total_count`, then run:
- `openspec validate add-commodity-contract-dataset-manifest --strict`
- `conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py FineFT/tests/datahandler/test_vae_data_creation.py -q`

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

## Amendment: 2026-07-13 Train Short Slices

### Task 14: Train short slices and slice row counts

> **trace:** plan-ready.md → `### Task 14: Train short slices and slice row counts` | tasks.md → `- [ ] 1.14 Keep short train slices below `chunk_length` and record slice `output_row_count` in manifest.`
> **sync:** tasks.md → `- [ ] 1.14 Keep short train slices below `chunk_length` and record slice `output_row_count` in manifest.` | plan-ready.md → `### Task 14: Train short slices and slice row counts`

**Files:**
- Modify: `FineFT/tests/datahandler/test_commodity_contract_dataset.py`
- Modify: `FineFT/datahandler/commodity_contract_dataset.py`
- Modify: `docs/datahandler/data_preparation_analysis.zh_cn.md`

- [x] **Step 1: Add failing short-slice test**

Update the dataset-generation test so train data contains more rows than one chunk plus a short tail. Assert that `train/slice/df_*.feather` includes the tail slice and that every `slice_outputs[]` item has `output_row_count` matching its feather file row count.

Run: `conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py::test_run_dataset_generation_writes_manifest_stage_files_and_train_slices -q`
Expected: FAIL while short tail slices are dropped and slice row counts are absent.

- [x] **Step 2: Keep short slices and record counts**

Update train slice planning so any positive remaining rows below `chunk_length` produce a final short slice. Update `write_train_slices()` to record `output_row_count = len(sliced)` on each slice record after writing.

Run: `conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py -q`
Expected: PASS.

- [x] **Step 3: Update docs and validate**

Document short train slices and slice-level row counts, then run:
- `openspec validate add-commodity-contract-dataset-manifest --strict`
- `conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py FineFT/tests/datahandler/test_vae_data_creation.py -q`

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

## Amendment: 2026-07-13 Contract-Scoped Valid Dynamic Slices

### Task 15: Contract-scoped valid dynamic slices

> **trace:** plan-ready.md → `### Task 15: Contract-scoped valid dynamic slices` | tasks.md → `- [ ] 1.15 Write valid processed and label slices under contract-scoped directories and update VAE reading.`
> **sync:** tasks.md → `- [ ] 1.15 Write valid processed and label slices under contract-scoped directories and update VAE reading.` | plan-ready.md → `### Task 15: Contract-scoped valid dynamic slices`

**Files:**
- Modify: `FineFT/datahandler/slice_model.py`
- Modify: `FineFT/datahandler/vae_data_creation.py`
- Modify: `FineFT/tests/datahandler/test_slice_model.py`
- Modify: `FineFT/tests/datahandler/test_vae_data_creation.py`
- Modify: `docs/datahandler/data_preparation_analysis.zh_cn.md`

- [x] **Step 1: Add failing output-structure tests**

Add tests proving `slice_model.py` writes `valid/processed/valid_processed_<contract>.feather` and `valid/<contract>/label_*/df_*.feather`, and proving VAE reads `valid/<contract>/label_*/df_*.feather` by aggregating same-label files across contracts.

- [x] **Step 2: Implement contract-scoped valid output**

Update `slice_model.py` path derivation and directory creation. Keep each contract's label numbering local under its own contract directory, and use `exist_ok=True` so multiple contracts can be processed sequentially.

- [x] **Step 3: Update VAE reader and docs**

Update `vae_data_creation.py` to recursively collect both old `valid/label_*/*.feather` and new `valid/<contract>/label_*/*.feather` layouts, then document the new commodity layout.

- [x] **Step 4: Validate**

Run:
- `openspec validate add-commodity-contract-dataset-manifest --strict`
- `conda activate finetf && pytest FineFT/tests/datahandler/test_slice_model.py FineFT/tests/datahandler/test_vae_data_creation.py FineFT/tests/datahandler/test_commodity_contract_dataset.py -q`

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

## Amendment: 2026-07-13 Robust Small-Segment Slope Labels

### Task 16: Robust small-segment slope labels

> **trace:** plan-ready.md → `### Task 16: Robust small-segment slope labels` | tasks.md → `- [x] 1.16 Make slope dynamic labeling robust when valid segment counts are small.`
> **sync:** tasks.md → `- [x] 1.16 Make slope dynamic labeling robust when valid segment counts are small.` | plan-ready.md → `### Task 16: Robust small-segment slope labels`

**Files:**
- Modify: `FineFT/datahandler/label_util.py`
- Modify: `FineFT/tests/datahandler/test_slice_model.py`

- [x] **Step 1: Add failing small-segment regression test**

Add a test for `Dynamic_labeler(labeling_method="slope")` with only two normalized coefficients and verify it does not raise `IndexError`.

- [x] **Step 2: Implement safe threshold calculation**

Use full min/max when segment count is small; keep the existing trimmed threshold behavior for larger segment counts.

- [x] **Step 3: Validate**

Run:
- `conda activate finetf && pytest FineFT/tests/datahandler/test_slice_model.py -q`
- `conda activate finetf && pytest FineFT/tests/datahandler/test_vae_data_creation.py FineFT/tests/datahandler/test_commodity_contract_dataset.py -q`

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

## Amendment: 2026-07-13 Valid Slice Manifest

### Task 17: Valid slice manifest

> **trace:** plan-ready.md → `### Task 17: Valid slice manifest` | tasks.md → `- [x] 1.17 Generate `valid/slice_manifest.json` with contract and label row counts.`
> **sync:** tasks.md → `- [x] 1.17 Generate `valid/slice_manifest.json` with contract and label row counts.` | plan-ready.md → `### Task 17: Valid slice manifest`

**Files:**
- Modify: `FineFT/datahandler/slice_model.py`
- Modify: `FineFT/tests/datahandler/test_slice_model.py`
- Modify: `docs/datahandler/data_preparation_analysis.zh_cn.md`

- [x] **Step 1: Add failing slice manifest test**

Extend the contract-scoped slice-model test to assert `valid/slice_manifest.json` records non-empty labels under both `contracts.<contract>.labels` and global `labels`, including file paths, file row counts, file counts and total row counts.

- [x] **Step 2: Implement manifest writer**

Update `slice_model.py` to collect row counts while writing label segment files, replace the current contract's manifest entry on rerun, rebuild global label summaries, and skip labels with no files.

- [x] **Step 3: Validate**

Run:
- `openspec validate add-commodity-contract-dataset-manifest --strict`
- `conda activate finetf && pytest FineFT/tests/datahandler/test_slice_model.py FineFT/tests/datahandler/test_vae_data_creation.py FineFT/tests/datahandler/test_commodity_contract_dataset.py -q`

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

## Amendment: 2026-07-13 Contract-Scoped VAE Test Arrays

### Task 18: Contract-scoped VAE test arrays

> **trace:** plan-ready.md → `### Task 18: Contract-scoped VAE test arrays` | tasks.md → `- [x] 1.18 Write multi-contract VAE test arrays as `VAE_data/test/test_<contract>.npy` instead of one merged `test.npy`.`
> **sync:** tasks.md → `- [x] 1.18 Write multi-contract VAE test arrays as `VAE_data/test/test_<contract>.npy` instead of one merged `test.npy`.` | plan-ready.md → `### Task 18: Contract-scoped VAE test arrays`

**Files:**
- Modify: `FineFT/datahandler/vae_data_creation.py`
- Modify: `FineFT/tests/datahandler/test_vae_data_creation.py`
- Modify: `docs/datahandler/data_preparation_analysis.zh_cn.md`

- [x] **Step 1: Add failing multi-contract test output test**

Assert that `test/df_<contract>.feather` produces `VAE_data/test/test_<contract>.npy` files and does not produce merged `VAE_data/test.npy`.

- [x] **Step 2: Implement contract-scoped test output**

Keep old `test.feather -> VAE_data/test.npy` compatibility; for `test/df_<contract>.feather`, create `VAE_data/test/` and write one `.npy` per contract.

- [x] **Step 3: Validate**

Run:
- `openspec validate add-commodity-contract-dataset-manifest --strict`
- `conda activate finetf && pytest FineFT/tests/datahandler/test_vae_data_creation.py FineFT/tests/datahandler/test_commodity_contract_dataset.py FineFT/tests/datahandler/test_slice_model.py -q`

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

## Amendment: 2026-07-13 Contract-Scoped VAE Valid Arrays

### Task 19: Contract-scoped VAE valid arrays

> **trace:** plan-ready.md → `### Task 19: Contract-scoped VAE valid arrays` | tasks.md → `- [x] 1.19 Write multi-contract VAE valid arrays as `VAE_data/<contract>/label_*.npy` instead of cross-contract `label_*.npy`.`
> **sync:** tasks.md → `- [x] 1.19 Write multi-contract VAE valid arrays as `VAE_data/<contract>/label_*.npy` instead of cross-contract `label_*.npy`.` | plan-ready.md → `### Task 19: Contract-scoped VAE valid arrays`

**Files:**
- Modify: `FineFT/datahandler/vae_data_creation.py`
- Modify: `FineFT/tests/datahandler/test_vae_data_creation.py`
- Modify: `docs/datahandler/data_preparation_analysis.zh_cn.md`

- [x] **Step 1: Add failing multi-contract valid output test**

Assert that `valid/<contract>/label_*/df_*.feather` produces `VAE_data/<contract>/label_*.npy` files and does not produce cross-contract `VAE_data/label_*.npy`.

- [x] **Step 2: Implement contract-scoped valid output**

Keep old `valid/label_*/*.feather -> VAE_data/label_*.npy` compatibility; for `valid/<contract>/label_*/*.feather`, create `VAE_data/<contract>/` and write one label `.npy` per contract.

- [x] **Step 3: Validate**

Run:
- `openspec validate add-commodity-contract-dataset-manifest --strict`
- `conda activate finetf && pytest FineFT/tests/datahandler/test_vae_data_creation.py FineFT/tests/datahandler/test_commodity_contract_dataset.py FineFT/tests/datahandler/test_slice_model.py -q`

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）
