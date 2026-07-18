# Adjust Commodity Dataset Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adjust commodity full-process orchestration and add a ninth dataset split stage that writes both contract-level and merged train/valid/test feather files.

**Architecture:** Keep shell scripts responsible for orchestration and put dataset split data processing in `data_preprocess/operator_futures/dataset_split/dataset_split.py`. Reuse the existing summary-based split algorithm concept: global union trading-day boundaries first, then per-contract intersections, then per-stage writes and vertical merges.

**Tech Stack:** Bash, Python 3, Polars, pytest, OpenSpec, conda environment `finetf`.

**Traceability (sddflow):**
- plan-ready: `openspec/changes/adjust-commodity-dataset-split/plan-ready.md`
- tasks: `openspec/changes/adjust-commodity-dataset-split/tasks.md`
- plan: `docs/superpowers/plans/2026-07-18-adjust-commodity-dataset-split.md`

---

### Task 1: Add dataset split tests

> **trace:** plan-ready.md -> `### Task 1: Add dataset split tests` | tasks.md -> ``- [ ] 1.1 Add focused tests for `operator_futures.dataset_split.dataset_split` covering boundary calculation, contract/date intersections, skipped sets, all-column preservation, merged outputs, manifest row counts, and fail-fast cases.``
> **sync:** tasks.md -> ``- [ ] 1.1 Add focused tests for `operator_futures.dataset_split.dataset_split` covering boundary calculation, contract/date intersections, skipped sets, all-column preservation, merged outputs, manifest row counts, and fail-fast cases.`` | plan-ready.md -> `### Task 1: Add dataset split tests`

**Files:**
- Create: `data_preprocess/tests/test_commodity_dataset_split.py`

- [x] **Step 1: Create focused failing tests**

Create `data_preprocess/tests/test_commodity_dataset_split.py` with this complete content:

```python
import json
from pathlib import Path

import polars as pl
import pytest

from operator_futures.dataset_split.dataset_split import (
    calculate_split_boundaries,
    run_dataset_split,
)


def _contract(name, dates):
    return {
        "contract": name,
        "trading_days": [
            {
                "trading_day": date.replace("-", ""),
                "date": date,
                "source_file": f"/raw/{name}/{date}.csv",
                "daily_volume": 100.0,
            }
            for date in dates
        ],
    }


def _write_summary(path: Path, contracts):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"symbol": "fu", "contracts": contracts}, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_scale_file(root: Path, contract: str, dates):
    output = (
        root
        / "PREPROCESS_DATASET"
        / "commodity-futures"
        / "SCALE_SAVE"
        / "fu"
        / contract
        / "5min"
        / "2026-01-01-2026-04-01"
        / "df.feather"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(
        {
            "timestamp": [
                f"{date} 09:{index:02d}:00"
                for index, date in enumerate(dates)
            ],
            "trading_day": dates,
            "symbol": [contract] * len(dates),
            "feature_a": list(range(len(dates))),
            "feature_b": [float(value) + 0.5 for value in range(len(dates))],
        }
    ).with_columns(pl.col("timestamp").str.strptime(pl.Datetime))
    frame.write_ipc(output)
    return output


def test_calculate_split_boundaries_uses_union_trading_days_5_3_2():
    summary = {
        "contracts": [
            _contract(
                "fu2601",
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-04",
                    "2026-01-05",
                    "2026-01-06",
                ],
            ),
            _contract(
                "fu2605",
                [
                    "2026-01-03",
                    "2026-01-04",
                    "2026-01-05",
                    "2026-01-06",
                    "2026-01-07",
                    "2026-01-08",
                    "2026-01-09",
                    "2026-01-10",
                ],
            ),
        ]
    }

    assert calculate_split_boundaries(summary) == {
        "start": "2026-01-01",
        "a": "2026-01-06",
        "b": "2026-01-09",
        "c": "2026-01-11",
    }


def test_run_dataset_split_writes_contract_and_merged_outputs_with_all_columns(tmp_path):
    summary_path = (
        tmp_path
        / "PREPROCESS_DATASET"
        / "commodity-futures"
        / "CONTINUOUS_RAW"
        / "fu"
        / "main_contract_summary.json"
    )
    _write_summary(
        summary_path,
        [
            _contract(
                "fu2601",
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-04",
                    "2026-01-05",
                    "2026-01-06",
                ],
            ),
            _contract(
                "fu2605",
                [
                    "2026-01-05",
                    "2026-01-06",
                    "2026-01-07",
                    "2026-01-08",
                    "2026-01-09",
                    "2026-01-10",
                ],
            ),
        ],
    )
    _write_scale_file(
        tmp_path,
        "fu2601",
        [
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
            "2026-01-04",
            "2026-01-05",
            "2026-01-06",
        ],
    )
    _write_scale_file(
        tmp_path,
        "fu2605",
        [
            "2026-01-05",
            "2026-01-06",
            "2026-01-07",
            "2026-01-08",
            "2026-01-09",
            "2026-01-10",
        ],
    )

    manifest = run_dataset_split(
        summary_path=summary_path,
        input_root=tmp_path / "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE",
        output_root=tmp_path / "dataset/5min",
        symbol="fu",
        target_freq="5min",
        start_date="2026-01-01",
        end_date="2026-04-01",
    )

    dataset_root = tmp_path / "dataset" / "5min" / "fu"
    assert (dataset_root / "dataset_split_manifest.json").exists()
    assert (dataset_root / "train" / "fu2601.feather").exists()
    assert (dataset_root / "train" / "fu2605.feather").exists()
    assert (dataset_root / "valid" / "fu2601.feather").exists()
    assert (dataset_root / "valid" / "fu2605.feather").exists()
    assert (dataset_root / "test" / "fu2605.feather").exists()
    assert (dataset_root / "train.feather").exists()
    assert (dataset_root / "valid.feather").exists()
    assert (dataset_root / "test.feather").exists()

    train = pl.read_ipc(dataset_root / "train.feather")
    valid = pl.read_ipc(dataset_root / "valid.feather")
    test = pl.read_ipc(dataset_root / "test.feather")
    assert train.columns == ["timestamp", "trading_day", "symbol", "feature_a", "feature_b"]
    assert valid.columns == train.columns
    assert test.columns == train.columns
    assert train.height == 6
    assert valid.height == 4
    assert test.height == 2
    assert set(train.get_column("symbol").to_list()) == {"fu2601", "fu2605"}
    assert test.get_column("symbol").to_list() == ["fu2605", "fu2605"]

    assert manifest["sets"]["train"]["contracts_total_count"] == 6
    assert manifest["sets"]["valid"]["contracts_total_count"] == 4
    assert manifest["sets"]["test"]["contracts_total_count"] == 2
    assert manifest["sets"]["test"]["skipped_contracts"] == [
        {"contract": "fu2601", "reason": "no trading days in test range"}
    ]
    assert manifest["sets"]["train"]["merged_output_path"].endswith(
        "dataset/5min/fu/train.feather"
    )


def test_run_dataset_split_fails_when_planned_input_file_is_missing(tmp_path):
    summary_path = tmp_path / "summary.json"
    _write_summary(summary_path, [_contract("fu2601", [f"2026-01-{day:02d}" for day in range(1, 11)])])

    with pytest.raises(FileNotFoundError, match="fu2601"):
        run_dataset_split(
            summary_path=summary_path,
            input_root=tmp_path / "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE",
            output_root=tmp_path / "dataset/5min",
            symbol="fu",
            target_freq="5min",
            start_date="2026-01-01",
            end_date="2026-04-01",
        )


def test_run_dataset_split_requires_non_empty_train_valid_test_sets(tmp_path):
    summary_path = tmp_path / "summary.json"
    _write_summary(summary_path, [_contract("fu2601", ["2026-01-01", "2026-01-02"])])

    with pytest.raises(ValueError, match="start < a < b < c"):
        run_dataset_split(
            summary_path=summary_path,
            input_root=tmp_path / "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE",
            output_root=tmp_path / "dataset/5min",
            symbol="fu",
            target_freq="5min",
            start_date="2026-01-01",
            end_date="2026-04-01",
        )
```

- [x] **Step 2: Run tests and verify they fail for the expected reason**

Run:

```bash
conda activate finetf && pytest data_preprocess/tests/test_commodity_dataset_split.py -q
```

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'operator_futures.dataset_split'`.

- [x] **Step 3: Commit the failing tests**

```bash
git add data_preprocess/tests/test_commodity_dataset_split.py
git commit -m "test: add commodity dataset split coverage"
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 2: Implement dataset split operator

> **trace:** plan-ready.md -> `### Task 2: Implement dataset split operator` | tasks.md -> ``- [ ] 1.2 Implement `data_preprocess/operator_futures/dataset_split/dataset_split.py` with CLI arguments, contract-level stage writing, top-level vertical concatenation, and `dataset_split_manifest.json`.``
> **sync:** tasks.md -> ``- [ ] 1.2 Implement `data_preprocess/operator_futures/dataset_split/dataset_split.py` with CLI arguments, contract-level stage writing, top-level vertical concatenation, and `dataset_split_manifest.json`.`` | plan-ready.md -> `### Task 2: Implement dataset split operator`

**Files:**
- Create: `data_preprocess/operator_futures/dataset_split/__init__.py`
- Create: `data_preprocess/operator_futures/dataset_split/dataset_split.py`
- Test: `data_preprocess/tests/test_commodity_dataset_split.py`

- [x] **Step 1: Create the package init file**

Create `data_preprocess/operator_futures/dataset_split/__init__.py` with this content:

```python
"""Dataset split operators for operator-futures preprocessing."""
```

- [x] **Step 2: Implement the dataset split module**

Create `data_preprocess/operator_futures/dataset_split/dataset_split.py` with this complete content:

```python
import argparse
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl


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
        raise ValueError(
            "cannot satisfy start < a < b < c with non-empty train/valid/test sets"
        )

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
    return (
        Path(input_root)
        / symbol
        / contract
        / target_freq
        / f"{start_date}-{end_date}"
        / "df.feather"
    )


def _contract_output_path(output_root, symbol, set_name, contract):
    return Path(output_root) / symbol / set_name / f"{contract}.feather"


def _merged_output_path(output_root, symbol, set_name):
    return Path(output_root) / symbol / f"{set_name}.feather"


def _date_expr(df):
    if "trading_day" in df.columns:
        return pl.col("trading_day").cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False)
    if "TradingDay" in df.columns:
        return pl.col("TradingDay").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False)
    if "timestamp" not in df.columns:
        raise ValueError("dataset split input missing timestamp column")
    return pl.col("timestamp").cast(pl.Datetime).dt.date()


def _filter_days(df, trading_days, *, contract, set_name, input_path):
    day_set = set(trading_days)
    filtered = (
        df.with_columns(_date_expr(df).alias("__split_date"))
        .filter(pl.col("__split_date").cast(pl.Utf8).is_in(day_set))
        .drop("__split_date")
    )
    if filtered.is_empty():
        raise ValueError(
            f"planned trading days produced empty output: contract={contract} set={set_name} input={input_path}"
        )
    if "timestamp" in filtered.columns:
        filtered = filtered.sort("timestamp")
    return filtered


def build_manifest(
    summary,
    boundaries,
    *,
    input_root,
    output_root,
    symbol,
    target_freq,
    start_date,
    end_date,
    train_ratio,
    valid_ratio,
    test_ratio,
):
    ranges = {
        "train": (boundaries["start"], boundaries["a"]),
        "valid": (boundaries["a"], boundaries["b"]),
        "test": (boundaries["b"], boundaries["c"]),
    }
    manifest = {
        "symbol": symbol,
        "target_freq": target_freq,
        "split_ratio": {"train": train_ratio, "valid": valid_ratio, "test": test_ratio},
        "boundaries": boundaries,
        "sets": {},
    }
    for set_name, (range_start, range_end) in ranges.items():
        contracts = []
        skipped = []
        for item in summary.get("contracts", []):
            contract = item["contract"]
            days = [
                str(day["date"])
                for day in item.get("trading_days", [])
                if _in_range(day["date"], range_start, range_end)
            ]
            if not days:
                skipped.append(
                    {
                        "contract": contract,
                        "reason": f"no trading days in {set_name} range",
                    }
                )
                continue
            contracts.append(
                {
                    "contract": contract,
                    "range": [range_start, range_end],
                    "trading_days": days,
                    "input_path": str(
                        _input_path(input_root, symbol, contract, target_freq, start_date, end_date)
                    ),
                    "output_path": str(_contract_output_path(output_root, symbol, set_name, contract)),
                }
            )
        manifest["sets"][set_name] = {
            "range": [range_start, range_end],
            "merged_output_path": str(_merged_output_path(output_root, symbol, set_name)),
            "contracts": contracts,
            "skipped_contracts": skipped,
        }
    return manifest


def write_split_outputs(manifest):
    for set_name, set_info in manifest["sets"].items():
        frames = []
        total_count = 0
        for contract_info in set_info["contracts"]:
            input_path = Path(contract_info["input_path"])
            if not input_path.exists():
                raise FileNotFoundError(
                    f"Missing df.feather for contract {contract_info['contract']}: {input_path}"
                )
            df = pl.read_ipc(input_path)
            output_df = _filter_days(
                df,
                contract_info["trading_days"],
                contract=contract_info["contract"],
                set_name=set_name,
                input_path=input_path,
            )
            output_path = Path(contract_info["output_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_df.write_ipc(output_path)
            contract_info["output_row_count"] = output_df.height
            total_count += output_df.height
            frames.append(output_df)

        if not frames:
            raise ValueError(f"cannot write {set_name}.feather without contract outputs")
        merged = pl.concat(frames, how="vertical")
        merged_output_path = Path(set_info["merged_output_path"])
        merged_output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.write_ipc(merged_output_path)
        set_info["contracts_total_count"] = total_count
        set_info["merged_output_row_count"] = merged.height


def run_dataset_split(
    *,
    summary_path,
    input_root,
    output_root,
    symbol,
    target_freq,
    start_date,
    end_date,
    train_ratio=5,
    valid_ratio=3,
    test_ratio=2,
):
    summary_path = Path(summary_path)
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary_path: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if not summary.get("contracts"):
        raise ValueError("main contract summary contains no contracts")

    boundaries = calculate_split_boundaries(
        summary,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
    )
    manifest = build_manifest(
        summary,
        boundaries,
        input_root=input_root,
        output_root=output_root,
        symbol=symbol,
        target_freq=target_freq,
        start_date=start_date,
        end_date=end_date,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
    )
    write_split_outputs(manifest)
    dataset_root = Path(output_root) / symbol
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "dataset_split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_path", type=Path, required=True)
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--target_freq", type=str, required=True)
    parser.add_argument("--start_date", type=str, required=True)
    parser.add_argument("--end_date", type=str, required=True)
    parser.add_argument("--train_ratio", type=int, default=5)
    parser.add_argument("--valid_ratio", type=int, default=3)
    parser.add_argument("--test_ratio", type=int, default=2)
    return parser


def main(args=None):
    parsed = build_parser().parse_args(args)
    run_dataset_split(
        summary_path=parsed.summary_path,
        input_root=parsed.input_root,
        output_root=parsed.output_root,
        symbol=parsed.symbol,
        target_freq=parsed.target_freq,
        start_date=parsed.start_date,
        end_date=parsed.end_date,
        train_ratio=parsed.train_ratio,
        valid_ratio=parsed.valid_ratio,
        test_ratio=parsed.test_ratio,
    )


if __name__ == "__main__":
    main()
```

- [x] **Step 3: Run dataset split tests**

Run:

```bash
conda activate finetf && pytest data_preprocess/tests/test_commodity_dataset_split.py -q
```

Expected: PASS, with `4 passed`.

- [x] **Step 4: Commit the dataset split operator**

```bash
git add data_preprocess/operator_futures/dataset_split data_preprocess/tests/test_commodity_dataset_split.py
git commit -m "feat: add commodity dataset split operator"
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 3: Add ninth-stage shell wrapper

> **trace:** plan-ready.md -> `### Task 3: Add ninth-stage shell wrapper` | tasks.md -> ``- [ ] 1.3 Add `data_preprocess/script_preprocess/future_upgraded/9_dataset_split/dataset_split.sh` that activates `finetf` and calls `operator_futures.dataset_split.dataset_split`.``
> **sync:** tasks.md -> ``- [ ] 1.3 Add `data_preprocess/script_preprocess/future_upgraded/9_dataset_split/dataset_split.sh` that activates `finetf` and calls `operator_futures.dataset_split.dataset_split`.`` | plan-ready.md -> `### Task 3: Add ninth-stage shell wrapper`

**Files:**
- Create: `data_preprocess/script_preprocess/future_upgraded/9_dataset_split/dataset_split.sh`

- [x] **Step 1: Create the shell wrapper**

Create `data_preprocess/script_preprocess/future_upgraded/9_dataset_split/dataset_split.sh` with this content:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOTPATH=${ROOTPATH:-$(pwd)}
SYMBOL=${SYMBOL:-fu}
TARGET_FREQ=${TARGET_FREQ:-5min}
START_DATE=${START_DATE:-2023-01-01}
END_DATE=${END_DATE:-2026-03-01}
TRAIN_RATIO=${TRAIN_RATIO:-5}
VALID_RATIO=${VALID_RATIO:-3}
TEST_RATIO=${TEST_RATIO:-2}
OUTPUT_ROOT=${OUTPUT_ROOT:-dataset/${TARGET_FREQ}}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate finetf
cd "${ROOTPATH}"

PYTHONPATH="${ROOTPATH}/data_preprocess${PYTHONPATH:+:${PYTHONPATH}}" \
python -m operator_futures.dataset_split.dataset_split \
  --summary_path "PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/${SYMBOL}/main_contract_summary.json" \
  --input_root "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE" \
  --output_root "${OUTPUT_ROOT}" \
  --symbol "${SYMBOL}" \
  --target_freq "${TARGET_FREQ}" \
  --start_date "${START_DATE}" \
  --end_date "${END_DATE}" \
  --train_ratio "${TRAIN_RATIO}" \
  --valid_ratio "${VALID_RATIO}" \
  --test_ratio "${TEST_RATIO}"
```

- [x] **Step 2: Make the shell wrapper executable**

Run:

```bash
chmod +x data_preprocess/script_preprocess/future_upgraded/9_dataset_split/dataset_split.sh
```

Expected: command exits with status 0.

- [x] **Step 3: Run shell syntax validation**

Run:

```bash
bash -n data_preprocess/script_preprocess/future_upgraded/9_dataset_split/dataset_split.sh
```

Expected: command exits with status 0 and prints no output.

- [x] **Step 4: Commit the shell wrapper**

```bash
git add data_preprocess/script_preprocess/future_upgraded/9_dataset_split/dataset_split.sh
git commit -m "feat: add commodity dataset split stage wrapper"
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 4: Update full process orchestration tests

> **trace:** plan-ready.md -> `### Task 4: Update full process orchestration tests` | tasks.md -> ``- [ ] 1.4 Update `fu_full_process.sh` tests to reject old `ic_candidate` / `ic_union_finalize` functions and steps, require `scale_save` inside the contract loop after `merge_clean`, and require one post-loop `dataset_split`.``
> **sync:** tasks.md -> ``- [ ] 1.4 Update `fu_full_process.sh` tests to reject old `ic_candidate` / `ic_union_finalize` functions and steps, require `scale_save` inside the contract loop after `merge_clean`, and require one post-loop `dataset_split`.`` | plan-ready.md -> `### Task 4: Update full process orchestration tests`

**Files:**
- Modify: `data_preprocess/tests/test_commodity_main_contract_cli.py`

- [x] **Step 1: Update expected functions test**

In `test_commodity_full_process_shell_exposes_expected_functions`, replace the old `run_commodity_ic_union_finalize` assertion with these assertions:

```python
    assert "run_commodity_ic_candidate" not in text
    assert "run_commodity_ic_union_finalize" not in text
    assert '"ic_candidate"' not in text
    assert '"ic_union_finalize"' not in text
    assert "run_commodity_dataset_split" in text
    assert "operator_futures.dataset_split.dataset_split" in text
```

- [x] **Step 2: Update step-log stub test**

In `test_commodity_full_process_writes_step_logs_and_preserves_child_log_paths`, remove the stub definitions for `run_commodity_ic_candidate` and `run_commodity_ic_union_finalize`, then add this stub after `run_commodity_scale_save()`:

```bash
run_commodity_dataset_split() {
    local summary_path=$1
    local target_freq=$2
    local start_date=$3
    local end_date=$4
    local symbol=$5
    echo "dataset_split:${symbol}:${target_freq}:${start_date}:${end_date}:${summary_path}"
}
```

In the same test, replace the `symbol_by_step` dictionary with:

```python
    symbol_by_step = {
        "stitch_main_contract": "fu",
        "downscale_continuous_by_trading_day": "fu",
        "cross_section": "fu_fu2601",
        "merge": "fu_fu2601",
        "concat": "fu_fu2601",
        "time_feature": "fu_fu2601",
        "merge_clean": "fu_fu2601",
        "scale_save": "fu_fu2601",
        "dataset_split": "fu",
        "maintenance_margin_dict": "fu",
    }
```

Replace the old `union_log` assertion with:

```python
    dataset_split_log = (
        tmp_path
        / "log_futures"
        / "ticker_result"
        / "commodity"
        / "steps"
        / "fu_5min_2026-01-05_2026-01-07_dataset_split.log"
    )
    assert "dataset_split:fu:5min:2026-01-05:2026-01-07:" in (
        dataset_split_log.read_text(encoding="utf-8")
    )
```

- [x] **Step 3: Replace old static ordering test**

Rename `test_commodity_full_process_shell_runs_scale_after_ic_union_finalize` to `test_commodity_full_process_shell_runs_scale_after_merge_clean_and_dataset_split_after_loop`, and replace its body with:

```python
def test_commodity_full_process_shell_runs_scale_after_merge_clean_and_dataset_split_after_loop():
    script = (
        REPO_ROOT
        / "data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh"
    )
    text = script.read_text(encoding="utf-8")

    assert "run_commodity_ic_candidate()" not in text
    assert "run_commodity_ic_union_finalize()" not in text
    assert '"ic_candidate"' not in text
    assert '"ic_union_finalize"' not in text
    assert '"dataset_split"' in text
    assert text.index('"merge_clean"') < text.index('"scale_save"')
    assert text.rindex('"scale_save"') < text.index('"dataset_split"')
    assert text.index('"dataset_split"') < text.index('"maintenance_margin_dict"')
```

- [x] **Step 4: Run focused shell tests and verify they fail before implementation**

Run:

```bash
conda activate finetf && pytest \
  data_preprocess/tests/test_commodity_main_contract_cli.py::test_commodity_full_process_shell_exposes_expected_functions \
  data_preprocess/tests/test_commodity_main_contract_cli.py::test_commodity_full_process_writes_step_logs_and_preserves_child_log_paths \
  data_preprocess/tests/test_commodity_main_contract_cli.py::test_commodity_full_process_shell_runs_scale_after_merge_clean_and_dataset_split_after_loop \
  -q
```

Expected: FAIL because `fu_full_process.sh` still defines and runs `ic_candidate` / `ic_union_finalize` and does not define `run_commodity_dataset_split`.

- [x] **Step 5: Commit the failing orchestration tests**

```bash
git add data_preprocess/tests/test_commodity_main_contract_cli.py
git commit -m "test: update commodity full process orchestration expectations"
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 5: Update fu full process orchestration

> **trace:** plan-ready.md -> `### Task 5: Update fu full process orchestration` | tasks.md -> ``- [ ] 1.5 Update `fu_full_process.sh` to remove old IC candidate/union functions and steps, run `scale_save` after each contract `merge_clean`, and run `dataset_split` once before `maintenance_margin_dict`.``
> **sync:** tasks.md -> ``- [ ] 1.5 Update `fu_full_process.sh` to remove old IC candidate/union functions and steps, run `scale_save` after each contract `merge_clean`, and run `dataset_split` once before `maintenance_margin_dict`.`` | plan-ready.md -> `### Task 5: Update fu full process orchestration`

**Files:**
- Modify: `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`
- Test: `data_preprocess/tests/test_commodity_main_contract_cli.py`

- [x] **Step 1: Delete old IC candidate and union functions**

In `fu_full_process.sh`, remove the complete `run_commodity_ic_candidate()` function and the complete `run_commodity_ic_union_finalize()` function. The next function after `run_commodity_merge_and_clean()` should be `run_commodity_maintenance_margin_dict()` until Step 2 adds `run_commodity_dataset_split()`.

- [x] **Step 2: Add the dataset split function**

Add this function before `run_commodity_maintenance_margin_dict()`:

```bash
run_commodity_dataset_split() {
    local summary_path=$1
    local target_freq=$2
    local start_date=$3
    local end_date=$4
    local symbol=$5
    local root_path=$6

    PYTHONPATH="${root_path}/data_preprocess${PYTHONPATH:+:${PYTHONPATH}}" python -u -m operator_futures.dataset_split.dataset_split \
        --summary_path "${summary_path}" \
        --input_root "${root_path}/PREPROCESS_DATASET/commodity-futures/SCALE_SAVE" \
        --output_root "${root_path}/dataset/${target_freq}" \
        --symbol "${symbol}" \
        --target_freq "${target_freq}" \
        --start_date "${start_date}" \
        --end_date "${end_date}" \
        --train_ratio 5 \
        --valid_ratio 3 \
        --test_ratio 2
}
```

- [x] **Step 3: Move scale save into the contract loop**

Inside `run_commodity_full_process()`, immediately after the `merge_clean` logged step in the first contract loop, add:

```bash
        run_commodity_logged_step \
            "$log_dir" "${symbol}_${contract}" "$target_freq" "$start_date" "$end_date" \
            "scale_save" \
            run_commodity_scale_save "$target_freq" "$start_date" "$end_date" "$symbol" "$root_path" "$contract"
```

- [x] **Step 4: Remove old post-loop IC candidate, IC union, and second scale loop**

Delete the post-loop block that starts with the comment `#   split train and test`, delete the `ic_candidate` logged step, delete the `ic_union_finalize` logged step, and delete the second `while IFS= read -r contract` loop that only runs `scale_save`.

- [x] **Step 5: Add dataset split after the contract loop**

After the first contract loop, before `maintenance_margin_dict`, add:

```bash
    run_commodity_logged_step \
        "$log_dir" "$symbol" "$target_freq" "$start_date" "$end_date" \
        "dataset_split" \
        run_commodity_dataset_split "$summary_path" "$target_freq" "$start_date" "$end_date" "$symbol" "$root_path"
```

- [x] **Step 6: Run shell syntax validation**

Run:

```bash
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh
```

Expected: command exits with status 0 and prints no output.

- [x] **Step 7: Run focused orchestration tests**

Run:

```bash
conda activate finetf && pytest \
  data_preprocess/tests/test_commodity_main_contract_cli.py::test_commodity_full_process_shell_exposes_expected_functions \
  data_preprocess/tests/test_commodity_main_contract_cli.py::test_commodity_full_process_writes_step_logs_and_preserves_child_log_paths \
  data_preprocess/tests/test_commodity_main_contract_cli.py::test_commodity_full_process_shell_runs_scale_after_merge_clean_and_dataset_split_after_loop \
  -q
```

Expected: PASS.

- [x] **Step 8: Commit full process orchestration changes**

```bash
git add data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh data_preprocess/tests/test_commodity_main_contract_cli.py
git commit -m "feat: adjust commodity full process dataset split flow"
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 6: Update focused documentation

> **trace:** plan-ready.md -> `### Task 6: Update focused documentation` | tasks.md -> ``- [ ] 1.6 Update focused documentation to reflect the ninth dataset split stage and merged `train.feather`、`valid.feather`、`test.feather` outputs.``
> **sync:** tasks.md -> ``- [ ] 1.6 Update focused documentation to reflect the ninth dataset split stage and merged `train.feather`、`valid.feather`、`test.feather` outputs.`` | plan-ready.md -> `### Task 6: Update focused documentation`

**Files:**
- Modify: `docs/datahandler/data_preparation_analysis.zh_cn.md`

- [x] **Step 1: Replace the commodity dataset stage description**

In `docs/datahandler/data_preparation_analysis.zh_cn.md`, update the commodity multi-contract section so it states:

````markdown
`future_upgraded/9_dataset_split` 是商品 full process 的第 9 阶段。该阶段在所有 summary 合约完成 `scale_save` 后运行，读取：

```text
PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/{symbol}/main_contract_summary.json
PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/{symbol}/{contract}/{target_freq}/{start_date}-{end_date}/df.feather
```

该阶段复用商品多合约数据集的时间边界算法：先对 summary 中所有合约有效交易日取去重升序并集，按 `train:valid:test = 5:3:2` 计算全局边界，再让每个合约用自身交易日与全局区间求交。该阶段不读取 `state_features.npy`，不筛选特征列，输出保留输入 feather 的所有列。

输出包含合约级阶段文件：

```text
dataset/{target_freq}/{symbol}/train/{contract}.feather
dataset/{target_freq}/{symbol}/valid/{contract}.feather
dataset/{target_freq}/{symbol}/test/{contract}.feather
```

合约级目录会保留。同时该阶段会分别纵向合并所有合约级阶段文件，写出：

```text
dataset/{target_freq}/{symbol}/train.feather
dataset/{target_freq}/{symbol}/valid.feather
dataset/{target_freq}/{symbol}/test.feather
dataset/{target_freq}/{symbol}/dataset_split_manifest.json
```
````

- [x] **Step 2: Remove stale contradiction**

In the same file, remove or rewrite any sentence that says commodity dataset generation does not produce `train.feather`, `valid.feather`, or `test.feather`. The remaining text must say those top-level files are generated by `9_dataset_split` as vertical concatenations while contract-level directories are retained.

- [x] **Step 3: Verify documentation text**

Run:

```bash
rg -n "9_dataset_split|dataset_split|train\\.feather|valid\\.feather|test\\.feather" docs/datahandler/data_preparation_analysis.zh_cn.md
```

Expected: output contains `9_dataset_split`, `dataset_split_manifest.json`, and top-level `train.feather`, `valid.feather`, `test.feather` descriptions.

- [x] **Step 4: Commit documentation changes**

```bash
git add docs/datahandler/data_preparation_analysis.zh_cn.md
git commit -m "docs: describe commodity dataset split stage"
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 7: Run validation

> **trace:** plan-ready.md -> `### Task 7: Run validation` | tasks.md -> ``- [ ] 2.1 Run strict OpenSpec validation, focused pytest commands with `conda activate finetf`, and `bash -n` on changed shell scripts.``
> **sync:** tasks.md -> ``- [ ] 2.1 Run strict OpenSpec validation, focused pytest commands with `conda activate finetf`, and `bash -n` on changed shell scripts.`` | plan-ready.md -> `### Task 7: Run validation`

**Files:**
- Verify: `openspec/changes/adjust-commodity-dataset-split/`
- Verify: `data_preprocess/tests/test_commodity_dataset_split.py`
- Verify: `data_preprocess/tests/test_commodity_main_contract_cli.py`
- Verify: `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`
- Verify: `data_preprocess/script_preprocess/future_upgraded/9_dataset_split/dataset_split.sh`

- [x] **Step 1: Run OpenSpec strict validation**

Run:

```bash
openspec validate adjust-commodity-dataset-split --strict
```

Expected: `Change 'adjust-commodity-dataset-split' is valid`.

- [x] **Step 2: Run focused pytest validation**

Run:

```bash
conda activate finetf && pytest \
  data_preprocess/tests/test_commodity_dataset_split.py \
  data_preprocess/tests/test_commodity_main_contract_cli.py \
  -q
```

Expected: all selected tests pass.

- [x] **Step 3: Run shell syntax validation**

Run:

```bash
bash -n \
  data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh \
  data_preprocess/script_preprocess/future_upgraded/9_dataset_split/dataset_split.sh
```

Expected: command exits with status 0 and prints no output.

- [x] **Step 4: Commit final validation doc-state updates**

If only checkbox state changed during implementation, commit the plan-state updates:

```bash
git add openspec/changes/adjust-commodity-dataset-split/tasks.md openspec/changes/adjust-commodity-dataset-split/plan-ready.md docs/superpowers/plans/2026-07-18-adjust-commodity-dataset-split.md
git commit -m "chore: mark commodity dataset split plan progress"
```

Expected: commit succeeds when there are checkbox changes; if there are no checkbox changes, `git status --short` shows no staged files for these paths.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）
