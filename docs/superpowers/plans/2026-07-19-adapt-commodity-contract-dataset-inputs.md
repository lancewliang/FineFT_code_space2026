# Adapt Commodity Contract Dataset Inputs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adapt `commodity_contract_dataset.py` to assemble FineFT commodity datasets from `dataset_split_manifest.json` and stage-split `SCALE_SAVE` files.

**Architecture:** Keep `commodity_contract_dataset.py` as the single FineFT dataset assembly entrypoint. It reads split metadata from `dataset_split_manifest.json`, resolves real inputs from `SCALE_SAVE/{symbol}/{target_freq}/{stage}/{contract}.feather`, copies contract files into `dataset/{target_freq}/{symbol}/{stage}/{contract}.feather`, copies selected state features, and generates train slices. Valid label generation remains in the commodity data handler shell via `slice_model.py`.

**Tech Stack:** Python, pandas feather IO, NumPy, pytest, bash scripts, OpenSpec.

**Traceability (sddflow):**
- plan-ready: `openspec/changes/adapt-commodity-contract-dataset-inputs/plan-ready.md`
- tasks: `openspec/changes/adapt-commodity-contract-dataset-inputs/tasks.md`
- plan: `docs/superpowers/plans/2026-07-19-adapt-commodity-contract-dataset-inputs.md`

---

### Task 1: Update commodity contract dataset tests for new input contract

> **trace:** plan-ready.md → `### Task 1: Update commodity contract dataset tests for new input contract` | tasks.md → ``- [ ] 1.1 Update `FineFT/tests/datahandler/test_commodity_contract_dataset.py` for the new manifest-driven input contract, stage file naming, state feature path, train slices, and failure cases.``
> **sync:** tasks.md → ``- [ ] 1.1 Update `FineFT/tests/datahandler/test_commodity_contract_dataset.py` for the new manifest-driven input contract, stage file naming, state feature path, train slices, and failure cases.`` | plan-ready.md → `### Task 1: Update commodity contract dataset tests for new input contract`

**Files:**
- Modify: `FineFT/tests/datahandler/test_commodity_contract_dataset.py`

- [x] **Step 1: Replace old summary-oriented imports**

Update the import block so tests use the new manifest loader and stop importing boundary helpers:

```python
from FineFT.datahandler.commodity_contract_dataset import (
    build_dataset_manifest,
    load_dataset_split_manifest,
    run_dataset_generation,
    write_stage_datasets,
    write_train_slices,
)
```

- [x] **Step 2: Add split manifest and stage file helpers**

Add these helpers near the existing `_contract` helper:

```python
def _dataset_split_manifest(symbol="fu", target_freq="10min"):
    return {
        "symbol": symbol,
        "target_freq": target_freq,
        "sets": {
            "train": {
                "range": ["2026-01-01", "2026-01-06"],
                "contracts": [
                    {
                        "contract": "fu2508",
                        "trading_days": ["2026-01-01", "2026-01-02"],
                        "output_row_count": 4,
                    },
                    {
                        "contract": "fu2509",
                        "trading_days": ["2026-01-03"],
                        "output_row_count": 2,
                    },
                ],
                "skipped_contracts": [],
            },
            "valid": {
                "range": ["2026-01-06", "2026-01-09"],
                "contracts": [
                    {
                        "contract": "fu2508",
                        "trading_days": ["2026-01-06"],
                        "output_row_count": 2,
                    }
                ],
                "skipped_contracts": [
                    {"contract": "fu2509", "reason": "no trading days in valid range"}
                ],
            },
            "test": {
                "range": ["2026-01-09", "2026-01-11"],
                "contracts": [
                    {
                        "contract": "fu2509",
                        "trading_days": ["2026-01-09"],
                        "output_row_count": 2,
                    }
                ],
                "skipped_contracts": [
                    {"contract": "fu2508", "reason": "no trading days in test range"}
                ],
            },
        },
    }


def _write_dataset_split_manifest(path, manifest=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = manifest or _dataset_split_manifest()
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def _write_scale_save_file(root, stage, contract, rows=2):
    output = root / "SCALE_SAVE" / "fu" / "10min" / stage / f"{contract}.feather"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="10min"),
            "symbol": [contract] * rows,
            "feature_a": list(range(rows)),
            "bid1_price": list(range(100, 100 + rows)),
            "mark_price": list(range(200, 200 + rows)),
        }
    ).to_feather(output)
    return output
```

- [x] **Step 3: Replace boundary tests with manifest loader tests**

Remove `test_calculate_split_boundaries_uses_union_trading_days_5_3_2` and `test_calculate_split_boundaries_requires_non_empty_sets`. Add:

```python
def test_load_dataset_split_manifest_validates_symbol_and_target_freq(tmp_path):
    manifest_path = _write_dataset_split_manifest(tmp_path / "dataset_split_manifest.json")

    manifest = load_dataset_split_manifest(
        manifest_path,
        symbol="fu",
        target_freq="10min",
    )

    assert manifest["symbol"] == "fu"
    assert manifest["target_freq"] == "10min"
    assert [item["contract"] for item in manifest["sets"]["train"]["contracts"]] == [
        "fu2508",
        "fu2509",
    ]


def test_load_dataset_split_manifest_fails_on_symbol_mismatch(tmp_path):
    manifest_path = _write_dataset_split_manifest(tmp_path / "dataset_split_manifest.json")

    with pytest.raises(ValueError, match="symbol"):
        load_dataset_split_manifest(
            manifest_path,
            symbol="al",
            target_freq="10min",
        )
```

- [x] **Step 4: Replace build manifest test**

Replace `test_build_dataset_manifest_records_contract_intersections_and_slice_plan` with:

```python
def test_build_dataset_manifest_uses_split_manifest_and_stage_scale_save_paths(tmp_path):
    split_manifest = _dataset_split_manifest()

    manifest = build_dataset_manifest(
        split_manifest=split_manifest,
        dataset_split_manifest_path=tmp_path / "dataset_split_manifest.json",
        input_root=tmp_path / "SCALE_SAVE",
        state_features_path=tmp_path / "FEATURE_SELECTION" / "state_features.npy",
        output_root=tmp_path / "dataset",
        symbol="fu",
        target_freq="10min",
        chunk_length=2,
        early_stop=1,
    )

    assert manifest["dataset_split_manifest_path"].endswith("dataset_split_manifest.json")
    assert manifest["state_features_path"].endswith("dataset/fu/state_features.npy")
    train_contracts = {
        item["contract"]: item for item in manifest["sets"]["train"]["contracts"]
    }
    assert train_contracts["fu2508"]["input_path"].endswith(
        "SCALE_SAVE/fu/10min/train/fu2508.feather"
    )
    assert train_contracts["fu2508"]["output_path"].endswith(
        "dataset/fu/train/fu2508.feather"
    )
    assert train_contracts["fu2508"]["slice_outputs"][0]["path"].endswith(
        "dataset/fu/train/slice/df_0.feather"
    )
    assert manifest["sets"]["valid"]["skipped_contracts"] == [
        {"contract": "fu2509", "reason": "no trading days in valid range"}
    ]
```

- [x] **Step 5: Add stage copy and fail-fast tests**

Replace `test_write_stage_datasets_filters_contract_files_and_omits_legacy_files` with:

```python
def test_write_stage_datasets_copies_stage_files_and_state_features(tmp_path):
    train_file = _write_scale_save_file(tmp_path, "train", "fu2508", rows=3)
    valid_file = _write_scale_save_file(tmp_path, "valid", "fu2508", rows=2)
    test_file = _write_scale_save_file(tmp_path, "test", "fu2509", rows=2)
    state_features = tmp_path / "FEATURE_SELECTION" / "10min" / "fu" / "train" / "state_features.npy"
    state_features.parent.mkdir(parents=True)
    np.save(state_features, np.array(["feature_a"]))
    manifest = {
        "symbol": "fu",
        "target_freq": "10min",
        "state_features_source_path": str(state_features),
        "state_features_path": str(tmp_path / "dataset" / "fu" / "state_features.npy"),
        "sets": {
            "train": {
                "contracts": [
                    {
                        "contract": "fu2508",
                        "input_path": str(train_file),
                        "output_path": str(tmp_path / "dataset" / "fu" / "train" / "fu2508.feather"),
                    }
                ],
                "skipped_contracts": [],
            },
            "valid": {
                "contracts": [
                    {
                        "contract": "fu2508",
                        "input_path": str(valid_file),
                        "output_path": str(tmp_path / "dataset" / "fu" / "valid" / "fu2508.feather"),
                    }
                ],
                "skipped_contracts": [],
            },
            "test": {
                "contracts": [
                    {
                        "contract": "fu2509",
                        "input_path": str(test_file),
                        "output_path": str(tmp_path / "dataset" / "fu" / "test" / "fu2509.feather"),
                    }
                ],
                "skipped_contracts": [],
            },
        },
    }

    write_stage_datasets(manifest)

    assert (tmp_path / "dataset" / "fu" / "train" / "fu2508.feather").exists()
    assert pd.read_feather(tmp_path / "dataset" / "fu" / "train" / "fu2508.feather")[
        "feature_a"
    ].tolist() == [0, 1, 2]
    assert np.load(tmp_path / "dataset" / "fu" / "state_features.npy", allow_pickle=True).tolist() == [
        "feature_a"
    ]
    assert manifest["sets"]["train"]["contracts"][0]["output_row_count"] == 3
    assert manifest["sets"]["valid"]["contracts_total_count"] == 2
    assert manifest["sets"]["test"]["contracts_total_count"] == 2
    assert not (tmp_path / "dataset" / "fu" / "train.feather").exists()


def test_write_stage_datasets_fails_when_state_features_missing(tmp_path):
    train_file = _write_scale_save_file(tmp_path, "train", "fu2508", rows=2)
    manifest = {
        "state_features_source_path": str(tmp_path / "missing" / "state_features.npy"),
        "state_features_path": str(tmp_path / "dataset" / "fu" / "state_features.npy"),
        "sets": {
            "train": {
                "contracts": [
                    {
                        "contract": "fu2508",
                        "input_path": str(train_file),
                        "output_path": str(tmp_path / "dataset" / "fu" / "train" / "fu2508.feather"),
                    }
                ],
                "skipped_contracts": [],
            }
        },
    }

    with pytest.raises(FileNotFoundError, match="state_features"):
        write_stage_datasets(manifest)


def test_write_stage_datasets_fails_when_scale_save_file_missing(tmp_path):
    state_features = tmp_path / "FEATURE_SELECTION" / "state_features.npy"
    state_features.parent.mkdir(parents=True)
    np.save(state_features, np.array(["feature_a"]))
    manifest = {
        "state_features_source_path": str(state_features),
        "state_features_path": str(tmp_path / "dataset" / "fu" / "state_features.npy"),
        "sets": {
            "train": {
                "contracts": [
                    {
                        "contract": "fu2508",
                        "input_path": str(tmp_path / "SCALE_SAVE" / "fu" / "10min" / "train" / "fu2508.feather"),
                        "output_path": str(tmp_path / "dataset" / "fu" / "train" / "fu2508.feather"),
                    }
                ],
                "skipped_contracts": [],
            }
        },
    }

    with pytest.raises(FileNotFoundError, match="fu2508"):
        write_stage_datasets(manifest)
```

- [x] **Step 6: Run tests and verify they fail**

Run: `conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py -q`

Expected: FAIL with import errors for `load_dataset_split_manifest` or `TypeError` for `build_dataset_manifest` because implementation still uses the old summary-driven signature.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 2: Refactor commodity_contract_dataset main path

> **trace:** plan-ready.md → `### Task 2: Refactor commodity_contract_dataset main path` | tasks.md → ``- [ ] 1.2 Refactor `FineFT/datahandler/commodity_contract_dataset.py` to read `dataset_split_manifest.json`, build a FineFT manifest from stage/contract metadata, copy staged SCALE_SAVE files, copy `--state_features_path`, and remove internal split-boundary filtering from the main path.``
> **sync:** tasks.md → ``- [ ] 1.2 Refactor `FineFT/datahandler/commodity_contract_dataset.py` to read `dataset_split_manifest.json`, build a FineFT manifest from stage/contract metadata, copy staged SCALE_SAVE files, copy `--state_features_path`, and remove internal split-boundary filtering from the main path.`` | plan-ready.md → `### Task 2: Refactor commodity_contract_dataset main path`

**Files:**
- Modify: `FineFT/datahandler/commodity_contract_dataset.py`

- [x] **Step 1: Replace split-boundary helpers with manifest path helpers**

Remove unused imports `math`, `datetime`, and `timedelta`. Add `import numpy as np`, then add these helpers after imports:

```python
STAGES = ("train", "valid", "test")


def _stage_input_path(input_root, symbol, target_freq, stage, contract):
    return Path(input_root) / symbol / target_freq / stage / f"{contract}.feather"


def _stage_output(output_root, symbol, set_name, contract):
    return Path(output_root) / symbol / set_name / f"{contract}.feather"
```

- [x] **Step 2: Add manifest loader**

Add:

```python
def load_dataset_split_manifest(path, symbol, target_freq):
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing dataset split manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if manifest.get("symbol") != symbol:
        raise ValueError(
            f"dataset split manifest symbol mismatch: expected={symbol} actual={manifest.get('symbol')}"
        )
    if manifest.get("target_freq") != target_freq:
        raise ValueError(
            "dataset split manifest target_freq mismatch: "
            f"expected={target_freq} actual={manifest.get('target_freq')}"
        )
    sets = manifest.get("sets")
    if not isinstance(sets, dict):
        raise ValueError("dataset split manifest missing sets")
    for stage in STAGES:
        stage_info = sets.get(stage)
        if not isinstance(stage_info, dict):
            raise ValueError(f"dataset split manifest missing sets.{stage}")
        if not isinstance(stage_info.get("contracts", []), list):
            raise ValueError(f"dataset split manifest sets.{stage}.contracts must be a list")
    return manifest
```

- [x] **Step 3: Replace build_dataset_manifest implementation**

Replace `build_dataset_manifest(...)` with:

```python
def build_dataset_manifest(
    split_manifest,
    dataset_split_manifest_path,
    input_root,
    state_features_path,
    output_root,
    symbol,
    target_freq,
    chunk_length,
    early_stop,
):
    manifest = {
        "symbol": symbol,
        "target_freq": target_freq,
        "dataset_split_manifest_path": str(dataset_split_manifest_path),
        "state_features_source_path": str(state_features_path),
        "state_features_path": str(Path(output_root) / symbol / "state_features.npy"),
        "sets": {},
    }

    next_slice = 0
    for stage in STAGES:
        split_stage = split_manifest["sets"].get(stage, {})
        contracts = []
        for item in split_stage.get("contracts", []):
            contract = item["contract"]
            record = {
                "contract": contract,
                "input_path": str(
                    _stage_input_path(input_root, symbol, target_freq, stage, contract)
                ),
                "output_path": str(
                    _stage_output(output_root, symbol, stage, contract)
                ),
            }
            if "range" in item:
                record["range"] = item["range"]
            elif "range" in split_stage:
                record["range"] = split_stage["range"]
            if "trading_days" in item:
                record["trading_days"] = item["trading_days"]
            if stage == "train":
                row_count = int(item.get("output_row_count", 0))
                slices, next_slice = _build_slice_plan(
                    row_count,
                    output_root,
                    symbol,
                    contract,
                    next_slice,
                    chunk_length,
                    early_stop,
                )
                record["slice_outputs"] = slices
            contracts.append(record)
        manifest["sets"][stage] = {
            "range": split_stage.get("range"),
            "contracts": contracts,
            "skipped_contracts": split_stage.get("skipped_contracts", []),
        }
    return manifest
```

- [x] **Step 4: Replace _build_slice_plan to work from row counts**

Replace `_build_slice_plan(...)` with:

```python
def _build_slice_plan(
    row_count, output_root, symbol, contract, start_index, chunk_length, early_stop
):
    outputs = []
    if row_count <= 0:
        return outputs, start_index

    row_start = 0
    index = start_index
    while row_start < row_count:
        row_end = min(row_start + chunk_length + early_stop, row_count)
        if row_end > row_start:
            outputs.append(
                {
                    "index": index,
                    "contract": contract,
                    "path": str(
                        Path(output_root)
                        / symbol
                        / "train"
                        / "slice"
                        / f"df_{index}.feather"
                    ),
                    "source_output": str(_stage_output(output_root, symbol, "train", contract)),
                    "row_start": row_start,
                    "row_end": row_end,
                }
            )
            index += 1
        row_start += chunk_length
    return outputs, index
```

- [x] **Step 5: Replace write_stage_datasets copy logic**

Replace `write_stage_datasets(manifest)` with:

```python
def write_stage_datasets(manifest):
    state_features_source_path = Path(manifest["state_features_source_path"])
    if not state_features_source_path.exists():
        raise FileNotFoundError(
            f"Missing selected state_features.npy: {state_features_source_path}"
        )
    state_features = np.load(state_features_source_path, allow_pickle=True).tolist()
    if not state_features:
        raise ValueError(f"state feature list is empty: {state_features_source_path}")

    state_features_path = Path(manifest["state_features_path"])
    state_features_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(state_features_source_path, state_features_path)

    for stage, set_info in manifest["sets"].items():
        contracts_total_count = 0
        for contract in set_info["contracts"]:
            input_path = Path(contract["input_path"])
            if not input_path.exists():
                raise FileNotFoundError(
                    f"Missing SCALE_SAVE file for stage={stage} contract={contract['contract']}: {input_path}"
                )
            output_path = Path(contract["output_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(input_path, output_path)
            output_df = pd.read_feather(output_path)
            if output_df.empty:
                raise ValueError(
                    f"copied empty stage dataset: stage={stage} contract={contract['contract']} output={output_path}"
                )
            contract["output_row_count"] = int(len(output_df))
            contracts_total_count += contract["output_row_count"]
        set_info["contracts_total_count"] = contracts_total_count
```

- [x] **Step 6: Replace run_dataset_generation signature and body**

Replace `run_dataset_generation(...)` with:

```python
def run_dataset_generation(
    dataset_split_manifest_path,
    input_root,
    state_features_path,
    output_root,
    symbol,
    target_freq,
    chunk_length=3200,
    early_stop=320,
):
    split_manifest = load_dataset_split_manifest(
        dataset_split_manifest_path,
        symbol=symbol,
        target_freq=target_freq,
    )
    dataset_root = Path(output_root) / symbol
    manifest = build_dataset_manifest(
        split_manifest=split_manifest,
        dataset_split_manifest_path=dataset_split_manifest_path,
        input_root=input_root,
        state_features_path=state_features_path,
        output_root=output_root,
        symbol=symbol,
        target_freq=target_freq,
        chunk_length=chunk_length,
        early_stop=early_stop,
    )
    dataset_root.mkdir(parents=True, exist_ok=True)
    write_stage_datasets(manifest)
    rebuild_train_slice_plan(manifest, chunk_length=chunk_length, early_stop=early_stop)
    write_train_slices(manifest)
    (dataset_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
```

- [x] **Step 7: Replace parser arguments**

Replace `build_parser()` contents with:

```python
def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_split_manifest_path", type=Path, required=True)
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--state_features_path", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--target_freq", type=str, required=True)
    parser.add_argument("--chunk_length", type=int, default=3200)
    parser.add_argument("--early_stop", type=int, default=320)
    return parser
```

Update `main()` call:

```python
def main(args=None):
    parsed = build_parser().parse_args(args)
    run_dataset_generation(
        dataset_split_manifest_path=parsed.dataset_split_manifest_path,
        input_root=parsed.input_root,
        state_features_path=parsed.state_features_path,
        output_root=parsed.output_root,
        symbol=parsed.symbol,
        target_freq=parsed.target_freq,
        chunk_length=parsed.chunk_length,
        early_stop=parsed.early_stop,
    )
```

- [x] **Step 8: Run tests**

Run: `conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py -q`

Expected: FAIL only on train slice tests or script text assertions if those have not been updated yet; new loader/build/copy tests should pass.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 3: Keep train slices working from contract-named files

> **trace:** plan-ready.md → `### Task 3: Keep train slices working from contract-named files` | tasks.md → ``- [ ] 1.3 Keep train slice generation working from `train/{contract}.feather`, with continuous slice indices and manifest row counts.``
> **sync:** tasks.md → ``- [ ] 1.3 Keep train slice generation working from `train/{contract}.feather`, with continuous slice indices and manifest row counts.`` | plan-ready.md → `### Task 3: Keep train slices working from contract-named files`

**Files:**
- Modify: `FineFT/datahandler/commodity_contract_dataset.py`
- Modify: `FineFT/tests/datahandler/test_commodity_contract_dataset.py`

- [x] **Step 1: Update train slice unit test file names**

In `test_write_train_slices_uses_contiguous_indices_and_single_contract_files`, replace writes to `df_fu2601.feather` and `df_fu2605.feather` with contract-named files:

```python
pd.DataFrame(
    {
        "timestamp": pd.date_range("2026-01-01", periods=5, freq="D"),
        "symbol": ["fu2508"] * 5,
        "feature_a": range(5),
    }
).to_feather(train_dir / "fu2508.feather")
pd.DataFrame(
    {
        "timestamp": pd.date_range("2026-01-01", periods=3, freq="D"),
        "symbol": ["fu2509"] * 3,
        "feature_a": range(10, 13),
    }
).to_feather(train_dir / "fu2509.feather")
```

Update the manifest in that test:

```python
manifest = {
    "sets": {
        "train": {
            "contracts": [
                {
                    "contract": "fu2508",
                    "output_path": str(train_dir / "fu2508.feather"),
                    "slice_outputs": [
                        {
                            "index": 0,
                            "path": str(train_dir / "slice" / "df_0.feather"),
                            "row_start": 0,
                            "row_end": 3,
                        },
                        {
                            "index": 1,
                            "path": str(train_dir / "slice" / "df_1.feather"),
                            "row_start": 2,
                            "row_end": 5,
                        },
                    ],
                },
                {
                    "contract": "fu2509",
                    "output_path": str(train_dir / "fu2509.feather"),
                    "slice_outputs": [
                        {
                            "index": 2,
                            "path": str(train_dir / "slice" / "df_2.feather"),
                            "row_start": 0,
                            "row_end": 3,
                        }
                    ],
                },
            ]
        }
    }
}
```

- [x] **Step 2: Ensure rebuild_train_slice_plan refreshes source_output**

Keep `rebuild_train_slice_plan` but ensure it reads row counts from copied `output_path` and writes `source_output` as the contract-named file:

```python
def rebuild_train_slice_plan(manifest, chunk_length, early_stop):
    next_index = 0
    for contract in manifest["sets"]["train"]["contracts"]:
        output_path = Path(contract["output_path"])
        slice_dir = output_path.parent / "slice"
        row_count = int(contract.get("output_row_count", 0))
        row_start = 0
        slice_outputs = []
        while row_start < row_count:
            row_end = min(row_start + chunk_length + early_stop, row_count)
            if row_end > row_start:
                slice_outputs.append(
                    {
                        "index": next_index,
                        "contract": contract["contract"],
                        "path": str(slice_dir / f"df_{next_index}.feather"),
                        "source_output": str(output_path),
                        "row_start": row_start,
                        "row_end": row_end,
                    }
                )
                next_index += 1
            row_start += chunk_length
        contract["slice_outputs"] = slice_outputs
```

- [x] **Step 3: Run focused train slice test**

Run: `conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py::test_write_train_slices_uses_contiguous_indices_and_single_contract_files -q`

Expected: PASS.

- [x] **Step 4: Run full commodity contract dataset tests**

Run: `conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py -q`

Expected: FAIL only on script assertions if data handler scripts are not updated yet.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 4: Update commodity data handler scripts

> **trace:** plan-ready.md → `### Task 4: Update commodity data handler scripts` | tasks.md → ``- [ ] 1.4 Update `FineFT/script/data/commodity_data_handler_fu.sh` and `FineFT/script/data/commodity_data_handler_al.sh` to pass `--dataset_split_manifest_path` and `--state_features_path`, and to scan `valid/*.feather` for `slice_model.py`.``
> **sync:** tasks.md → ``- [ ] 1.4 Update `FineFT/script/data/commodity_data_handler_fu.sh` and `FineFT/script/data/commodity_data_handler_al.sh` to pass `--dataset_split_manifest_path` and `--state_features_path`, and to scan `valid/*.feather` for `slice_model.py`.`` | plan-ready.md → `### Task 4: Update commodity data handler scripts`

**Files:**
- Modify: `FineFT/script/data/commodity_data_handler_fu.sh`
- Modify: `FineFT/script/data/commodity_data_handler_al.sh`
- Modify: `FineFT/tests/datahandler/test_commodity_contract_dataset.py`

- [x] **Step 1: Update script assertion test**

Replace the relevant assertions in `test_commodity_data_handler_scripts_use_contract_dataset_tool` with:

```python
assert "commodity_contract_dataset.py" in text
assert f"--symbol {symbol}" in text or '--symbol "${SYMBOL}"' in text
assert "--dataset_split_manifest_path" in text
assert "SPLIT-TRAIN-VALID-TEST/${TARGET_FREQ}/${SYMBOL}/dataset_split_manifest.json" in text
assert "--state_features_path" in text
assert "FEATURE_SELECTION/${TARGET_FREQ}/${SYMBOL}/train/state_features.npy" in text
assert 'for valid_file in "dataset/${TARGET_FREQ}/${SYMBOL}/valid"/*.feather' in text
assert 'slice_model.py --data_path "${valid_file}" --timestamp timestamp' in text
assert "--summary_path" not in text
assert "--feature_union_path" not in text
assert "preprocess_data.py --trading_pair" not in text
assert "slice_model.py --data_path dataset/" not in text
```

- [x] **Step 2: Update `commodity_data_handler_fu.sh` CLI call**

Replace the Python invocation arguments with:

```bash
python FineFT/datahandler/commodity_contract_dataset.py \
  --dataset_split_manifest_path "PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/${TARGET_FREQ}/${SYMBOL}/dataset_split_manifest.json" \
  --input_root "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE" \
  --state_features_path "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/${TARGET_FREQ}/${SYMBOL}/train/state_features.npy" \
  --output_root "dataset/${TARGET_FREQ}" \
  --symbol "${SYMBOL}" \
  --target_freq "${TARGET_FREQ}" \
  --chunk_length "${CHUNK_LENGTH}" \
  --early_stop "${EARLY_STOP}"
```

Replace the valid loop with:

```bash
for valid_file in "dataset/${TARGET_FREQ}/${SYMBOL}/valid"/*.feather; do
  [ -e "${valid_file}" ] || continue
  python FineFT/datahandler/slice_model.py --data_path "${valid_file}" --timestamp timestamp
done
```

- [x] **Step 3: Apply the same script update to `commodity_data_handler_al.sh`**

Use the same Python invocation and valid loop from Step 2. Keep `SYMBOL=${SYMBOL:-al}` in the aluminum script.

- [x] **Step 4: Run script assertion test**

Run: `conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py::test_commodity_data_handler_scripts_use_contract_dataset_tool -q`

Expected: PASS.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 5: Remove old path assertions from FineFT commodity tests

> **trace:** plan-ready.md → `### Task 5: Remove old path assertions from FineFT commodity tests` | tasks.md → ``- [ ] 1.5 Update any FineFT commodity dataset tests that assert old `df_<contract>.feather`, `--summary_path`, `--feature_union_path`, or valid `df_*.feather` contracts.``
> **sync:** tasks.md → ``- [ ] 1.5 Update any FineFT commodity dataset tests that assert old `df_<contract>.feather`, `--summary_path`, `--feature_union_path`, or valid `df_*.feather` contracts.`` | plan-ready.md → `### Task 5: Remove old path assertions from FineFT commodity tests`

**Files:**
- Modify: `FineFT/tests/datahandler/test_commodity_contract_dataset.py`

- [x] **Step 1: Replace end-to-end generation test setup**

Replace `test_run_dataset_generation_writes_manifest_stage_files_and_train_slices` setup with:

```python
def test_run_dataset_generation_writes_manifest_stage_files_and_train_slices(tmp_path):
    manifest_path = _write_dataset_split_manifest(
        tmp_path
        / "PREPROCESS_DATASET"
        / "commodity-futures"
        / "SPLIT-TRAIN-VALID-TEST"
        / "10min"
        / "fu"
        / "dataset_split_manifest.json"
    )
    _write_scale_save_file(tmp_path, "train", "fu2508", rows=10)
    _write_scale_save_file(tmp_path, "train", "fu2509", rows=2)
    _write_scale_save_file(tmp_path, "valid", "fu2508", rows=4)
    _write_scale_save_file(tmp_path, "test", "fu2509", rows=3)
    state_features = (
        tmp_path
        / "PREPROCESS_DATASET"
        / "commodity-futures"
        / "FEATURE_SELECTION"
        / "10min"
        / "fu"
        / "train"
        / "state_features.npy"
    )
    state_features.parent.mkdir(parents=True)
    np.save(state_features, np.array(["feature_a"]))

    run_dataset_generation(
        dataset_split_manifest_path=manifest_path,
        input_root=tmp_path / "SCALE_SAVE",
        state_features_path=state_features,
        output_root=tmp_path / "dataset" / "10min",
        symbol="fu",
        target_freq="10min",
        chunk_length=4,
        early_stop=1,
    )

    dataset_root = tmp_path / "dataset" / "10min" / "fu"
    assert (dataset_root / "dataset_manifest.json").exists()
    assert (dataset_root / "train" / "fu2508.feather").exists()
    assert (dataset_root / "train" / "fu2509.feather").exists()
    assert (dataset_root / "valid" / "fu2508.feather").exists()
    assert (dataset_root / "test" / "fu2509.feather").exists()
    assert not (dataset_root / "train" / "df_fu2508.feather").exists()
    assert not (dataset_root / "train.feather").exists()
    manifest = json.loads((dataset_root / "dataset_manifest.json").read_text())
    assert manifest["sets"]["train"]["contracts_total_count"] == 12
    assert manifest["sets"]["valid"]["contracts_total_count"] == 4
    assert manifest["sets"]["test"]["contracts_total_count"] == 3
    assert np.load(dataset_root / "state_features.npy", allow_pickle=True).tolist() == [
        "feature_a"
    ]
    assert sorted(path.name for path in (dataset_root / "train" / "slice").glob("df_*.feather")) == [
        "df_0.feather",
        "df_1.feather",
        "df_2.feather",
        "df_3.feather",
    ]
    assert not (dataset_root / "valid" / "label_0").exists()
```

- [x] **Step 2: Search for obsolete assertions**

Run: `rg -n "df_|summary_path|feature_union_path|calculate_split_boundaries|start_date|end_date|train_ratio|valid_ratio|test_ratio" FineFT/tests/datahandler/test_commodity_contract_dataset.py`

Expected: Only allowed hits are `df_*.feather` train slice paths and JSON trading day audit fields. There should be no old CLI assertions or stage contract output paths using `df_<contract>.feather`.

- [x] **Step 3: Run full test file**

Run: `conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py -q`

Expected: PASS.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 6: Run commodity contract dataset test suite

> **trace:** plan-ready.md → `### Task 6: Run commodity contract dataset test suite` | tasks.md → ``- [ ] 2.1 Run `conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py -q`.``
> **sync:** tasks.md → ``- [ ] 2.1 Run `conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py -q`.`` | plan-ready.md → `### Task 6: Run commodity contract dataset test suite`

**Files:**
- No file changes.

- [x] **Step 1: Run the verification command**

Run: `conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py -q`

Expected: PASS for all tests in `FineFT/tests/datahandler/test_commodity_contract_dataset.py`.

- [x] **Step 2: Record any failure before proceeding**

Expected: Terminal output reports all tests passed and does not include `FAILED`.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 7: Validate OpenSpec change

> **trace:** plan-ready.md → `### Task 7: Validate OpenSpec change` | tasks.md → ``- [ ] 2.2 Run `openspec validate adapt-commodity-contract-dataset-inputs --strict`.``
> **sync:** tasks.md → ``- [ ] 2.2 Run `openspec validate adapt-commodity-contract-dataset-inputs --strict`.`` | plan-ready.md → `### Task 7: Validate OpenSpec change`

**Files:**
- No file changes.

- [x] **Step 1: Run OpenSpec strict validation**

Run: `openspec validate adapt-commodity-contract-dataset-inputs --strict`

Expected: PASS with `Change 'adapt-commodity-contract-dataset-inputs' is valid`.

- [x] **Step 2: Check worktree diff**

Run: `git diff -- openspec/changes/adapt-commodity-contract-dataset-inputs docs/superpowers/plans/2026-07-19-adapt-commodity-contract-dataset-inputs.md FineFT/datahandler/commodity_contract_dataset.py FineFT/tests/datahandler/test_commodity_contract_dataset.py FineFT/script/data/commodity_data_handler_fu.sh FineFT/script/data/commodity_data_handler_al.sh`

Expected: Diff contains only this change's spec, plan, tests, dataset tool, and commodity data handler script updates.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）
