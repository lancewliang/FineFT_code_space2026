# Support Commodity VAE Cross Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a commodity VAE workflow that trains each label from a materialized cross-contract training set and reports per-contract plus aggregate test logpx outputs.

**Architecture:** `FineFT/RL/DiHFT/VAE/merge_vae_train.py` owns commodity VAE contract path discovery, train-set materialization, input validation, and manifest writing. `FineFT/RL/DiHFT/VAE/main.py` owns CLI workflow orchestration and test source discovery. `FineFT/RL/DiHFT/VAE/process.py` owns dataset loader preparation and VAE analysis output writing. Tests in `FineFT/tests/rl/test_commodity_vae_cross_contract.py` define the multi-contract data contract without running long VAE training.

**Tech Stack:** Python, NumPy, pandas CSV output, PyTorch DataLoader, pytest, bash, OpenSpec.

**Traceability (sddflow):**
- plan-ready: `openspec/changes/support-commodity-vae-cross-contract/plan-ready.md`
- tasks: `openspec/changes/support-commodity-vae-cross-contract/tasks.md`
- plan: `docs/superpowers/plans/2026-07-22-support-commodity-vae-cross-contract.md`

---

### Task 1: Add cross-contract VAE training materialization tests

> **trace:** plan-ready.md → `### Task 1: Add cross-contract VAE training materialization tests` | tasks.md → `- [ ] 1.1 Add tests for commodity VAE cross-contract train-set materialization, manifest contents, missing label handling, and array dimension validation.`
> **sync:** tasks.md → `- [ ] 1.1 Add tests for commodity VAE cross-contract train-set materialization, manifest contents, missing label handling, and array dimension validation.` | plan-ready.md → `### Task 1: Add cross-contract VAE training materialization tests`

**Files:**
- Create: `FineFT/tests/rl/test_commodity_vae_cross_contract.py`

- [x] **Step 1: Create the test module with imports and path setup**

Add this file:

```python
import json
import sys
from pathlib import Path

import numpy as np
import pytest


FINEFT_ROOT = Path(__file__).resolve().parents[2]
VAE_ROOT = FINEFT_ROOT / "RL" / "DiHFT" / "VAE"
if str(FINEFT_ROOT) not in sys.path:
    sys.path.insert(0, str(FINEFT_ROOT))
if str(VAE_ROOT) not in sys.path:
    sys.path.insert(0, str(VAE_ROOT))

from RL.DiHFT.VAE import main as vae_main
from RL.DiHFT.VAE import merge_vae_train


def _dataset_root(tmp_path):
    return tmp_path / "dataset" / "10min"


def _vae_dir(tmp_path):
    return _dataset_root(tmp_path) / "fu" / "VAE_data"


def _save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(data, dtype=float))
```

- [x] **Step 2: Add the materialization success test**

Append this test:

```python
def test_materialize_label_training_data_merges_contract_arrays_and_writes_manifest(tmp_path):
    vae_dir = _vae_dir(tmp_path)
    _save(vae_dir / "fu2505" / "label_0.npy", [[1.0, 2.0], [3.0, 4.0]])
    _save(vae_dir / "fu2509" / "label_0.npy", [[5.0, 6.0]])
    (vae_dir / "fu2510").mkdir(parents=True)
    (vae_dir / "test").mkdir()

    result = merge_vae_train.materialize_label_training_data(
        data_base_path=str(_dataset_root(tmp_path)),
        dataset_name="fu",
        label_index=0,
    )

    merged = np.load(vae_dir / "train" / "label_0.npy")
    np.testing.assert_array_equal(
        merged,
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    )
    assert result["merged_path"] == str(vae_dir / "train" / "label_0.npy")
    assert result["total_samples"] == 3
    assert result["feature_dim"] == 2

    manifest = json.loads((vae_dir / "train" / "label_0_manifest.json").read_text())
    assert manifest["dataset_name"] == "fu"
    assert manifest["label"] == "label_0"
    assert manifest["total_samples"] == 3
    assert manifest["feature_dim"] == 2
    assert [item["contract"] for item in manifest["included_contracts"]] == [
        "fu2505",
        "fu2509",
    ]
    assert manifest["included_contracts"][0]["sample_count"] == 2
    assert manifest["included_contracts"][1]["sample_count"] == 1
    assert manifest["missing_contracts"] == ["fu2510"]
```

- [x] **Step 3: Add fail-fast tests for no sources and bad dimensions**

Append these tests:

```python
def test_materialize_label_training_data_fails_when_no_contract_has_label(tmp_path):
    vae_dir = _vae_dir(tmp_path)
    (vae_dir / "fu2505").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="label_4"):
        merge_vae_train.materialize_label_training_data(
            data_base_path=str(_dataset_root(tmp_path)),
            dataset_name="fu",
            label_index=4,
        )

    assert not (vae_dir / "train" / "label_4.npy").exists()


def test_materialize_label_training_data_rejects_non_2d_arrays(tmp_path):
    vae_dir = _vae_dir(tmp_path)
    _save(vae_dir / "fu2505" / "label_0.npy", [1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="two-dimensional"):
        merge_vae_train.materialize_label_training_data(
            data_base_path=str(_dataset_root(tmp_path)),
            dataset_name="fu",
            label_index=0,
        )


def test_materialize_label_training_data_rejects_feature_dim_mismatch(tmp_path):
    vae_dir = _vae_dir(tmp_path)
    _save(vae_dir / "fu2505" / "label_0.npy", [[1.0, 2.0]])
    _save(vae_dir / "fu2509" / "label_0.npy", [[3.0, 4.0, 5.0]])

    with pytest.raises(ValueError, match="feature dimension"):
        merge_vae_train.materialize_label_training_data(
            data_base_path=str(_dataset_root(tmp_path)),
            dataset_name="fu",
            label_index=0,
        )
```

- [x] **Step 4: Run the focused test and verify RED**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_commodity_vae_cross_contract.py -q
```

Expected: FAIL with an `AttributeError` or import failure showing `merge_vae_train.materialize_label_training_data` is not implemented yet.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）


### Task 2: Implement VAE data discovery and train-set materialization

> **trace:** plan-ready.md → `### Task 2: Implement VAE data discovery and train-set materialization` | tasks.md → - [ ] 1.2 Implement commodity VAE data discovery and train-set materialization in `FineFT/RL/DiHFT/VAE/merge_vae_train.py` and integrate it from `main.py`.
> **sync:** tasks.md → - [ ] 1.2 Implement commodity VAE data discovery and train-set materialization in `FineFT/RL/DiHFT/VAE/merge_vae_train.py` and integrate it from `main.py`. | plan-ready.md → `### Task 2: Implement VAE data discovery and train-set materialization`

**Files:**
- Create: `FineFT/RL/DiHFT/VAE/merge_vae_train.py`
- Modify: `FineFT/RL/DiHFT/VAE/main.py`
- Test: `FineFT/tests/rl/test_commodity_vae_cross_contract.py`

- [x] **Step 1: Add train materialization module imports and path helpers**

In `FineFT/RL/DiHFT/VAE/merge_vae_train.py`, add these imports near the top:

```python
import json
from pathlib import Path
```

Add these helpers:

```python
RESERVED_VAE_DIRS = {"test", "train", "processed", "__pycache__"}


def label_name_from_index(label_index):
    return "label_{}".format(label_index)


def vae_data_dir(data_base_path, dataset_name):
    return Path(data_base_path) / dataset_name / "VAE_data"


def contract_dirs(root):
    if not root.exists():
        raise FileNotFoundError(f"missing VAE_data path: {root}")
    return [
        path
        for path in sorted(root.iterdir(), key=lambda item: item.name)
        if path.is_dir() and path.name not in RESERVED_VAE_DIRS
    ]
```

- [x] **Step 2: Add array validation and source discovery**

Add:

```python
def load_2d_array(path, contract):
    data = np.load(path)
    if data.ndim != 2:
        raise ValueError(
            f"{path} for contract {contract} must be a two-dimensional array"
        )
    if data.shape[0] == 0:
        raise ValueError(f"{path} for contract {contract} has no samples")
    return data


def discover_label_sources(data_base_path, dataset_name, label_index):
    root = vae_data_dir(data_base_path, dataset_name)
    label_name = label_name_from_index(label_index)
    included = []
    missing = []
    for contract_dir in contract_dirs(root):
        source_file = contract_dir / f"{label_name}.npy"
        if source_file.exists():
            included.append(
                {
                    "contract": contract_dir.name,
                    "source_file": str(source_file),
                }
            )
        else:
            missing.append(contract_dir.name)
    if not included:
        raise FileNotFoundError(
            f"no arrays found for {label_name} under {root}/<contract>/{label_name}.npy"
        )
    return included, missing
```

- [x] **Step 3: Add materialization function**

Add:

```python
def materialize_label_training_data(data_base_path, dataset_name, label_index):
    root = vae_data_dir(data_base_path, dataset_name)
    label_name = label_name_from_index(label_index)
    included_sources, missing_contracts = discover_label_sources(
        data_base_path, dataset_name, label_index
    )
    arrays = []
    included_contracts = []
    feature_dim = None
    for source in included_sources:
        array = load_2d_array(Path(source["source_file"]), source["contract"])
        if feature_dim is None:
            feature_dim = int(array.shape[1])
        elif int(array.shape[1]) != feature_dim:
            raise ValueError(
                "feature dimension mismatch for "
                f"{source['contract']} at {source['source_file']}: "
                f"expected {feature_dim}, got {array.shape[1]}"
            )
        arrays.append(array)
        included_contracts.append(
            {
                "contract": source["contract"],
                "source_file": source["source_file"],
                "sample_count": int(array.shape[0]),
            }
        )
    merged = np.concatenate(arrays, axis=0)
    if merged.shape[0] == 0:
        raise ValueError(f"merged training set for {label_name} has no samples")
    train_dir = root / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    merged_path = train_dir / f"{label_name}.npy"
    manifest_path = train_dir / f"{label_name}_manifest.json"
    np.save(merged_path, merged)
    manifest = {
        "dataset_name": dataset_name,
        "label": label_name,
        "merged_path": str(merged_path),
        "total_samples": int(merged.shape[0]),
        "feature_dim": int(merged.shape[1]),
        "included_contracts": included_contracts,
        "missing_contracts": missing_contracts,
    }
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    return manifest
```

- [x] **Step 4: Run materialization tests**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_commodity_vae_cross_contract.py::test_materialize_label_training_data_merges_contract_arrays_and_writes_manifest FineFT/tests/rl/test_commodity_vae_cross_contract.py::test_materialize_label_training_data_fails_when_no_contract_has_label FineFT/tests/rl/test_commodity_vae_cross_contract.py::test_materialize_label_training_data_rejects_non_2d_arrays FineFT/tests/rl/test_commodity_vae_cross_contract.py::test_materialize_label_training_data_rejects_feature_dim_mismatch -q
```

Expected: PASS.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 3: Add per-contract VAE analysis output tests

> **trace:** plan-ready.md → `### Task 3: Add per-contract VAE analysis output tests` | tasks.md → - [ ] 1.3 Add tests for per-contract VAE analysis outputs, traceable CSV rows, aggregate `ood_logpx_all` outputs, and `summary.json` statistics.
> **sync:** tasks.md → - [ ] 1.3 Add tests for per-contract VAE analysis outputs, traceable CSV rows, aggregate `ood_logpx_all` outputs, and `summary.json` statistics. | plan-ready.md → `### Task 3: Add per-contract VAE analysis output tests`

**Files:**
- Modify: `FineFT/tests/rl/test_commodity_vae_cross_contract.py`

- [x] **Step 1: Add pandas import and process import**

In the test module, add:

```python
import pandas as pd
from RL.DiHFT.VAE import process as vae_process
```

- [x] **Step 2: Add test source discovery test**

Append:

```python
def test_discover_test_sources_reads_contract_test_arrays(tmp_path):
    vae_dir = _vae_dir(tmp_path)
    _save(vae_dir / "test" / "test_fu2508.npy", [[1.0, 2.0]])
    _save(vae_dir / "test" / "test_fu2509.npy", [[3.0, 4.0]])

    sources = vae_main.discover_test_sources(
        data_base_path=str(_dataset_root(tmp_path)),
        dataset_name="fu",
    )

    assert [source["contract"] for source in sources] == ["fu2508", "fu2509"]
    assert sources[0]["source_file"].endswith("test_fu2508.npy")
```

- [x] **Step 3: Add logpx output writer test**

Append:

```python
def test_write_contract_logpx_outputs_writes_per_contract_and_aggregate_files(tmp_path):
    save_path = tmp_path / "result" / "DiHFT" / "vae_results" / "fu" / "label_0"
    results = [
        {
            "contract": "fu2508",
            "source_file": "dataset/10min/fu/VAE_data/test/test_fu2508.npy",
            "logpx": np.array([-1.0, -2.0]),
        },
        {
            "contract": "fu2509",
            "source_file": "dataset/10min/fu/VAE_data/test/test_fu2509.npy",
            "logpx": np.array([-3.0]),
        },
    ]

    summary = vae_process.write_contract_logpx_outputs(
        results,
        save_path=str(save_path),
        dataset_name="fu",
        label="label_0",
    )

    np.testing.assert_array_equal(
        np.load(save_path / "ood_logpx_fu2508.npy"),
        np.array([-1.0, -2.0]),
    )
    np.testing.assert_array_equal(
        np.load(save_path / "ood_logpx_all.npy"),
        np.array([-1.0, -2.0, -3.0]),
    )
    per_contract_csv = pd.read_csv(save_path / "ood_logpx_fu2508.csv")
    assert per_contract_csv.to_dict("records") == [
        {
            "contract": "fu2508",
            "source_file": "dataset/10min/fu/VAE_data/test/test_fu2508.npy",
            "row_index": 0,
            "logpx": -1.0,
        },
        {
            "contract": "fu2508",
            "source_file": "dataset/10min/fu/VAE_data/test/test_fu2508.npy",
            "row_index": 1,
            "logpx": -2.0,
        },
    ]
    all_csv = pd.read_csv(save_path / "ood_logpx_all.csv")
    assert all_csv["contract"].tolist() == ["fu2508", "fu2508", "fu2509"]
    summary_file = json.loads((save_path / "summary.json").read_text())
    assert summary == summary_file
    assert summary_file["dataset_name"] == "fu"
    assert summary_file["label"] == "label_0"
    assert summary_file["test"]["contracts"]["fu2508"]["samples"] == 2
    assert summary_file["test"]["all"]["samples"] == 3
    assert "roc_auc" not in json.dumps(summary_file).lower()
```

- [x] **Step 4: Run analysis tests and verify RED**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_commodity_vae_cross_contract.py::test_discover_test_sources_reads_contract_test_arrays FineFT/tests/rl/test_commodity_vae_cross_contract.py::test_write_contract_logpx_outputs_writes_per_contract_and_aggregate_files -q
```

Expected: FAIL until `discover_test_sources` and `write_contract_logpx_outputs` exist.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 4: Implement per-contract test analysis and aggregate outputs

> **trace:** plan-ready.md → `### Task 4: Implement per-contract test analysis and aggregate outputs` | tasks.md → - [ ] 1.4 Refactor `FineFT/RL/DiHFT/VAE/process.py` and `main.py` analysis flow to read `VAE_data/test/test_<contract>.npy`, emit per-contract `.npy/.csv`, aggregate `.npy/.csv`, and `summary.json`.
> **sync:** tasks.md → - [ ] 1.4 Refactor `FineFT/RL/DiHFT/VAE/process.py` and `main.py` analysis flow to read `VAE_data/test/test_<contract>.npy`, emit per-contract `.npy/.csv`, aggregate `.npy/.csv`, and `summary.json`. | plan-ready.md → `### Task 4: Implement per-contract test analysis and aggregate outputs`

**Files:**
- Modify: `FineFT/RL/DiHFT/VAE/main.py`
- Modify: `FineFT/RL/DiHFT/VAE/process.py`
- Test: `FineFT/tests/rl/test_commodity_vae_cross_contract.py`

- [x] **Step 1: Add test source discovery in main.py**

Add to `main.py`:

```python
def discover_test_sources(data_base_path, dataset_name):
    root = vae_data_dir(data_base_path, dataset_name)
    test_dir = root / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"missing VAE test path: {test_dir}")
    sources = []
    for path in sorted(test_dir.glob("test_*.npy"), key=lambda item: item.name):
        contract = path.stem
        if contract.startswith("test_"):
            contract = contract[len("test_") :]
        sources.append({"contract": contract, "source_file": str(path)})
    if not sources:
        raise FileNotFoundError(f"no test_*.npy files found under {test_dir}")
    return sources
```

- [x] **Step 2: Add output helpers to process.py**

In `process.py`, add imports:

```python
import json
import pandas as pd
```

Add:

```python
def _logpx_stats(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("logpx array has no samples")
    return {
        "samples": int(values.size),
        "logpx_mean": float(np.mean(values)),
        "logpx_std": float(np.std(values)),
        "logpx_min": float(np.min(values)),
        "logpx_max": float(np.max(values)),
    }


def _logpx_rows(contract, source_file, logpx):
    return [
        {
            "contract": contract,
            "source_file": source_file,
            "row_index": int(index),
            "logpx": float(value),
        }
        for index, value in enumerate(np.asarray(logpx, dtype=float))
    ]


def write_contract_logpx_outputs(contract_results, save_path, dataset_name, label):
    os.makedirs(save_path, exist_ok=True)
    all_logpx = []
    all_rows = []
    contract_summary = {}
    for result in sorted(contract_results, key=lambda item: item["contract"]):
        contract = result["contract"]
        source_file = result["source_file"]
        logpx = np.asarray(result["logpx"], dtype=float)
        np.save(os.path.join(save_path, f"ood_logpx_{contract}.npy"), logpx)
        rows = _logpx_rows(contract, source_file, logpx)
        pd.DataFrame(rows, columns=["contract", "source_file", "row_index", "logpx"]).to_csv(
            os.path.join(save_path, f"ood_logpx_{contract}.csv"),
            index=False,
        )
        all_logpx.append(logpx)
        all_rows.extend(rows)
        contract_summary[contract] = {
            "source_file": source_file,
            **_logpx_stats(logpx),
        }
    combined = np.concatenate(all_logpx, axis=0)
    np.save(os.path.join(save_path, "ood_logpx_all.npy"), combined)
    pd.DataFrame(all_rows, columns=["contract", "source_file", "row_index", "logpx"]).to_csv(
        os.path.join(save_path, "ood_logpx_all.csv"),
        index=False,
    )
    summary = {
        "dataset_name": dataset_name,
        "label": label,
        "test": {
            "contracts": contract_summary,
            "all": _logpx_stats(combined),
        },
    }
    with open(os.path.join(save_path, "summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary
```

- [x] **Step 3: Add contract loader and analyzer functions to process.py**

Add:

```python
def prepare_contract_dataset_loader_list(test_sources, expected_feature_dim):
    dataloader_list = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if device.type == "cuda" else {}
    for source in test_sources:
        data = np.load(source["source_file"])
        if data.ndim != 2:
            raise ValueError(
                f"{source['source_file']} for contract {source['contract']} must be two-dimensional"
            )
        if int(data.shape[1]) != int(expected_feature_dim):
            raise ValueError(
                f"feature dimension mismatch for {source['contract']} at {source['source_file']}: "
                f"expected {expected_feature_dim}, got {data.shape[1]}"
            )
        dataset = One_Dim_Dataset(source["source_file"])
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, **kwargs
        )
        dataloader_list.append({**source, "loader": dataloader})
    return dataloader_list


def analyze_contract_tests(
    pretrained_model_path,
    model,
    contract_loader_list,
    device,
    save_path,
    dataset_name,
    label,
):
    print("Start contract analyzing...")
    model.load_state_dict(torch.load(pretrained_model_path))
    contract_results = []
    for item in contract_loader_list:
        ood_mus, ood_logpx = VAEs.analyze(model, item["loader"], device)
        contract_results.append(
            {
                "contract": item["contract"],
                "source_file": item["source_file"],
                "logpx": np.asarray(ood_logpx, dtype=float),
            }
        )
    return write_contract_logpx_outputs(
        contract_results,
        save_path=save_path,
        dataset_name=dataset_name,
        label=label,
    )
```

- [x] **Step 4: Make prepare_model independent of a merged test file**

Change `prepare_model` signature and OOD loader creation in `process.py`:

```python
def prepare_model(
    train_data_path,
    ood_test_dataset_path,
    hidden_dims,
    z_dim,
    loss,
    learning_rate,
    batch_size,
    epochs,
    log_interval,
    prr,
):
    train_data = One_Dim_Dataset(train_data_path)
    test_data = One_Dim_Dataset(train_data_path)
    if ood_test_dataset_path is None:
        ood_test_data = One_Dim_Dataset(train_data_path)
    else:
        ood_test_data = One_Dim_Dataset(ood_test_dataset_path)
```

Keep the rest of `prepare_model` unchanged except using `device.type == "cuda"` for the kwargs check:

```python
kwargs = {"num_workers": 1, "pin_memory": True} if device.type == "cuda" else {}
```

- [x] **Step 5: Run focused tests**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_commodity_vae_cross_contract.py::test_discover_test_sources_reads_contract_test_arrays FineFT/tests/rl/test_commodity_vae_cross_contract.py::test_write_contract_logpx_outputs_writes_per_contract_and_aggregate_files -q
```

Expected: PASS.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 5: Replace ambiguous VAE CLI booleans with explicit workflow flags

> **trace:** plan-ready.md → `### Task 5: Replace ambiguous VAE CLI booleans with explicit workflow flags` | tasks.md → - [ ] 1.5 Replace ambiguous boolean CLI usage in `FineFT/RL/DiHFT/VAE/main.py` with explicit `--train` and `--analyze-only` behavior for the commodity VAE workflow.
> **sync:** tasks.md → - [ ] 1.5 Replace ambiguous boolean CLI usage in `FineFT/RL/DiHFT/VAE/main.py` with explicit `--train` and `--analyze-only` behavior for the commodity VAE workflow. | plan-ready.md → `### Task 5: Replace ambiguous VAE CLI booleans with explicit workflow flags`

**Files:**
- Modify: `FineFT/RL/DiHFT/VAE/main.py`
- Modify: `FineFT/tests/rl/test_commodity_vae_cross_contract.py`

- [x] **Step 1: Add CLI behavior tests**

Append to the test module:

```python
def test_parser_accepts_explicit_train_and_analyze_only_flags():
    train_args = vae_main.parser.parse_args(
        ["--dataset_name", "fu", "--data_base_path", "dataset/10min", "--label_index", "0", "--train"]
    )
    analyze_args = vae_main.parser.parse_args(
        [
            "--dataset_name",
            "fu",
            "--data_base_path",
            "dataset/10min",
            "--label_index",
            "0",
            "--analyze-only",
        ]
    )

    assert train_args.train is True
    assert train_args.analyze_only is False
    assert analyze_args.train is False
    assert analyze_args.analyze_only is True
```

- [x] **Step 2: Replace bool parser arguments with explicit flags**

In `main.py`, replace the `--if_train` and `--if_cross_analyze` arguments with:

```python
parser.add_argument(
    "--train",
    action="store_true",
    help="materialize cross-contract training data, train the VAE, and analyze test contracts",
)
parser.add_argument(
    "--analyze-only",
    action="store_true",
    help="load model_latest.pth and analyze test contracts without retraining",
)
```

Change `--save_interval` to an int:

```python
parser.add_argument(
    "--save_interval",
    type=int,
    default=50,
    help="interval for saving the checkpoints",
)
```

- [x] **Step 3: Refactor Piplineruner initialization**

In `Piplineruner.__init__`, compute label, materialized train path, and test sources:

```python
label_name = label_name_from_index(self.args.label_index)
self.label_name = label_name
self.single_label_save_path = os.path.join(
    args.base_model_path,
    "vae_results",
    self.args.dataset_name,
    label_name,
)
self.args.single_label_save_path = self.single_label_save_path
if self.args.train:
    self.train_manifest = materialize_label_training_data(
        self.args.data_base_path,
        self.args.dataset_name,
        self.args.label_index,
    )
else:
    train_path = (
        vae_data_dir(self.args.data_base_path, self.args.dataset_name)
        / "train"
        / f"{label_name}.npy"
    )
    if not train_path.exists():
        raise FileNotFoundError(f"missing materialized training data: {train_path}")
    train_data = np.load(train_path)
    if train_data.ndim != 2 or train_data.shape[0] == 0:
        raise ValueError(f"invalid materialized training data: {train_path}")
    self.train_manifest = {
        "merged_path": str(train_path),
        "feature_dim": int(train_data.shape[1]),
    }
train_data_path = self.train_manifest["merged_path"]
test_sources = discover_test_sources(self.args.data_base_path, self.args.dataset_name)
```

Pass `None` as `ood_test_dataset_path` to `prepare_model`, then build contract loaders:

```python
(
    self.model,
    self.optimizer,
    self.train_loader,
    self.test_loader,
    self.ood_test_loader,
    self.device,
) = prepare_model(
    train_data_path,
    None,
    hidden_dims,
    z_dim,
    loss,
    learning_rate,
    batch_size,
    epochs,
    log_interval,
    prr,
)
self.contract_loader_list = prepare_contract_dataset_loader_list(
    test_sources,
    expected_feature_dim=self.train_manifest["feature_dim"],
)
```

- [x] **Step 4: Add analyze_contracts method and main entry behavior**

Replace `analyze_test` and `analyze_cross_test` usage with:

```python
def analyze_contracts(self):
    return analyze_contract_tests(
        pretrained_model_path=os.path.join(
            self.single_label_save_path,
            "model_latest.pth",
        ),
        model=self.model,
        contract_loader_list=self.contract_loader_list,
        device=self.device,
        save_path=self.args.single_label_save_path,
        dataset_name=self.args.dataset_name,
        label=self.label_name,
    )
```

Replace the `if __name__ == "__main__":` block with:

```python
if __name__ == "__main__":
    args = parser.parse_args()
    if args.train and args.analyze_only:
        parser.error("--train and --analyze-only are mutually exclusive")
    if not args.train and not args.analyze_only:
        parser.error("choose --train or --analyze-only")
    piplinerunner = Piplineruner(args)
    if args.train:
        piplinerunner.train()
    piplinerunner.analyze_contracts()
```

- [x] **Step 5: Update imports in main.py**

Change the import lists in `main.py` to include the train materialization helper and process functions:

```python
from merge_vae_train import (
    label_name_from_index,
    materialize_label_training_data,
    vae_data_dir,
)
from process import (
    prepare_model,
    train_test,
    analyze_contract_tests,
    prepare_contract_dataset_loader_list,
)
```

- [x] **Step 6: Run focused tests**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_commodity_vae_cross_contract.py -q
```

Expected: PASS.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 6: Update fu VAE shell entry

> **trace:** plan-ready.md → `### Task 6: Update fu VAE shell entry` | tasks.md → - [ ] 1.6 Update `FineFT/script/train/DiHFT/low_level/VAE_util_fu.sh` to activate `finetf`, set `PYTHONPATH`, pass `--dataset_name fu`, pass `--data_base_path dataset/10min`, and launch `label_0..label_4` with the explicit training flag and max-2 default concurrency.
> **sync:** tasks.md → - [ ] 1.6 Update `FineFT/script/train/DiHFT/low_level/VAE_util_fu.sh` to activate `finetf`, set `PYTHONPATH`, pass `--dataset_name fu`, pass `--data_base_path dataset/10min`, and launch `label_0..label_4` with the explicit training flag and max-2 default concurrency. | plan-ready.md → `### Task 6: Update fu VAE shell entry`

**Files:**
- Modify: `FineFT/script/train/DiHFT/low_level/VAE_util_fu.sh`

- [x] **Step 1: Replace shell content with explicit commodity entry**

Use this content:

```bash
#!/usr/bin/env bash

set -euo pipefail

ROOTPATH=${ROOTPATH:-$(pwd)}
cd "$ROOTPATH"

DATASET_NAME=${DATASET_NAME:-fu}
DATA_BASE_PATH=${DATA_BASE_PATH:-dataset/10min}
LABEL_COUNT=${LABEL_COUNT:-5}
MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS:-2}

if ! [[ "${MAX_PARALLEL_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_PARALLEL_JOBS must be a positive integer." >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate finetf
export PYTHONPATH="${ROOTPATH}/FineFT${PYTHONPATH:+:${PYTHONPATH}}"

log_dir="log/DiHFT/${DATASET_NAME}/VAE/${EXPERIMENT_NAME}"
mkdir -p "${log_dir}"

pids=()
failed=0

wait_for_available_slot() {
    while ((${#pids[@]} >= MAX_PARALLEL_JOBS)); do
        if ! wait -n; then
            failed=1
        fi
        prune_finished_jobs
    done
}

prune_finished_jobs() {
    local active_pids=()
    local pid
    for pid in "${pids[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            active_pids+=("${pid}")
        fi
    done
    pids=("${active_pids[@]}")
}

for label_index in $(seq 0 $((LABEL_COUNT - 1))); do
    wait_for_available_slot
    nohup python -u FineFT/RL/DiHFT/VAE/main.py \
        --dataset_name "${DATASET_NAME}" \
        --data_base_path "${DATA_BASE_PATH}" \
        --label_index "${label_index}" \
        --train \
        >"${log_dir}/train_label_${label_index}.log" 2>&1 &
    pids+=("$!")
done

while ((${#pids[@]} > 0)); do
    if ! wait -n; then
        failed=1
    fi
    prune_finished_jobs
done

if ((failed != 0)); then
    echo "${DATASET_NAME} VAE labels 0 to $((LABEL_COUNT - 1)) finished with failures."
    exit 1
fi

echo "${DATASET_NAME} VAE labels 0 to $((LABEL_COUNT - 1)) finished."
```

- [x] **Step 2: Run shell syntax validation**

Run:

```bash
bash -n FineFT/script/train/DiHFT/low_level/VAE_util_fu.sh
```

Expected: exits 0.

- [x] **Step 3: Run static parameter check**

Run:

```bash
rg -n -- '--dataset_name|--data_base_path|--train|conda activate finetf|PYTHONPATH' FineFT/script/train/DiHFT/low_level/VAE_util_fu.sh
```

Expected: output contains all five patterns and `MAX_PARALLEL_JOBS`.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 7: Run focused VAE tests or full FineFT tests

> **trace:** plan-ready.md → `### Task 7: Run focused VAE tests or full FineFT tests` | tasks.md → - [ ] 2.1 Run `conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests -q` or the focused VAE tests if the full suite is blocked by unrelated environment fixtures.
> **sync:** tasks.md → - [ ] 2.1 Run `conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests -q` or the focused VAE tests if the full suite is blocked by unrelated environment fixtures. | plan-ready.md → `### Task 7: Run focused VAE tests or full FineFT tests`

**Files:**
- Verify: `FineFT/tests/rl/test_commodity_vae_cross_contract.py`

- [x] **Step 1: Run full FineFT tests**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests -q
```

Expected: PASS. If collection fails because an unrelated test imports external data that is absent from this machine, capture the failing file and error in the build notes, then run Step 2.

- [x] **Step 2: Confirm focused VAE fallback was not needed because the full suite passed**

Run only if Step 1 is blocked by unrelated external fixture availability:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_commodity_vae_cross_contract.py -q
```

Expected: PASS.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 8: Run VAE py_compile

> **trace:** plan-ready.md → `### Task 8: Run VAE py_compile` | tasks.md → - [ ] 2.2 Run `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/VAE/main.py FineFT/RL/DiHFT/VAE/process.py FineFT/RL/DiHFT/VAE/merge_vae_train.py`.
> **sync:** tasks.md → - [ ] 2.2 Run `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/VAE/main.py FineFT/RL/DiHFT/VAE/process.py FineFT/RL/DiHFT/VAE/merge_vae_train.py`. | plan-ready.md → `### Task 8: Run VAE py_compile`

**Files:**
- Verify: `FineFT/RL/DiHFT/VAE/merge_vae_train.py`
- Verify: `FineFT/RL/DiHFT/VAE/main.py`
- Verify: `FineFT/RL/DiHFT/VAE/process.py`

- [x] **Step 1: Compile VAE Python files**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && python -m py_compile FineFT/RL/DiHFT/VAE/main.py FineFT/RL/DiHFT/VAE/process.py FineFT/RL/DiHFT/VAE/merge_vae_train.py
```

Expected: exits 0 and prints no traceback.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 9: Run shell syntax validation

> **trace:** plan-ready.md → `### Task 9: Run shell syntax validation` | tasks.md → - [ ] 2.3 Run `bash -n FineFT/script/train/DiHFT/low_level/VAE_util_fu.sh`.
> **sync:** tasks.md → - [ ] 2.3 Run `bash -n FineFT/script/train/DiHFT/low_level/VAE_util_fu.sh`. | plan-ready.md → `### Task 9: Run shell syntax validation`

**Files:**
- Verify: `FineFT/script/train/DiHFT/low_level/VAE_util_fu.sh`

- [x] **Step 1: Validate shell syntax**

Run:

```bash
bash -n FineFT/script/train/DiHFT/low_level/VAE_util_fu.sh
```

Expected: exits 0.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 10: Run OpenSpec strict validation

> **trace:** plan-ready.md → `### Task 10: Run OpenSpec strict validation` | tasks.md → - [ ] 2.4 Run `openspec validate support-commodity-vae-cross-contract --strict`.
> **sync:** tasks.md → - [ ] 2.4 Run `openspec validate support-commodity-vae-cross-contract --strict`. | plan-ready.md → `### Task 10: Run OpenSpec strict validation`

**Files:**
- Verify: `openspec/changes/support-commodity-vae-cross-contract/proposal.md`
- Verify: `openspec/changes/support-commodity-vae-cross-contract/design.md`
- Verify: `openspec/changes/support-commodity-vae-cross-contract/specs/commodity-futures-support/spec.md`
- Verify: `openspec/changes/support-commodity-vae-cross-contract/tasks.md`

- [x] **Step 1: Run strict OpenSpec validation**

Run:

```bash
openspec validate support-commodity-vae-cross-contract --strict
```

Expected:

```text
Change 'support-commodity-vae-cross-contract' is valid
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 11: Add enhanced per-label summary metric tests

> **trace:** plan-ready.md -> `### Task 11: Add enhanced per-label summary metric tests` | tasks.md -> `- [ ] 1.7 Add tests for enhanced per-label summary.json metrics`
> **sync:** tasks.md -> `- [ ] 1.7 Add tests for enhanced per-label summary.json metrics` | plan-ready.md -> `### Task 11: Add enhanced per-label summary metric tests`

**Files:**
- Modify: `FineFT/tests/rl/test_commodity_vae_cross_contract.py`

- [x] **Step 1: Add a focused enhanced summary writer test**

Append this test to `FineFT/tests/rl/test_commodity_vae_cross_contract.py`:

```python
def test_write_contract_logpx_outputs_includes_enhanced_summary_metrics(tmp_path):
    save_path = tmp_path / "result" / "DiHFT" / "vae_results" / "fu" / "label_0"
    train_baseline = {
        "source_file": "dataset/10min/fu/VAE_data/train/label_0.npy",
        "input_samples": 4,
        "analyzed_samples": 4,
        "logpx": np.array([-10.0, -8.0, -6.0, -4.0]),
    }
    results = [
        {
            "contract": "fu2508",
            "source_file": "dataset/10min/fu/VAE_data/test/test_fu2508.npy",
            "input_samples": 3,
            "logpx": np.array([-9.0, -7.0]),
        },
        {
            "contract": "fu2509",
            "source_file": "dataset/10min/fu/VAE_data/test/test_fu2509.npy",
            "input_samples": 2,
            "logpx": np.array([-5.0, -3.0]),
        },
    ]

    summary = vae_process.write_contract_logpx_outputs(
        results,
        save_path=str(save_path),
        dataset_name="fu",
        label="label_0",
        train_baseline=train_baseline,
    )

    assert summary["train_baseline"]["source_file"].endswith("label_0.npy")
    assert summary["train_baseline"]["input_samples"] == 4
    assert summary["train_baseline"]["analyzed_samples"] == 4
    assert summary["train_baseline"]["sample_mismatch"] is False
    assert set(summary["train_baseline"]["quantiles"]) == {
        "q01",
        "q05",
        "q25",
        "q50",
        "q75",
        "q95",
        "q99",
    }
    fu2508 = summary["test"]["contracts"]["fu2508"]
    assert fu2508["input_samples"] == 3
    assert fu2508["analyzed_samples"] == 2
    assert fu2508["sample_mismatch"] is True
    assert fu2508["samples"] == 2
    assert set(fu2508["quantiles"]) == set(summary["train_baseline"]["quantiles"])
    assert set(fu2508["acceptance"]) == {
        "ge_train_q01_pct",
        "ge_train_q05_pct",
        "ge_train_q50_pct",
    }
    assert summary["test"]["all"]["analyzed_samples"] == 4
    assert "roc_auc" not in json.dumps(summary).lower()
    assert "accuracy" not in json.dumps(summary).lower()
```

- [x] **Step 2: Run the test and verify RED**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_commodity_vae_cross_contract.py::test_write_contract_logpx_outputs_includes_enhanced_summary_metrics -q
```

Expected: FAIL because `write_contract_logpx_outputs` does not yet accept `train_baseline`.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 12: Implement enhanced per-label summary metrics

> **trace:** plan-ready.md -> `### Task 12: Implement enhanced per-label summary metrics` | tasks.md -> `- [ ] 1.8 Implement enhanced per-label summary statistics`
> **sync:** tasks.md -> `- [ ] 1.8 Implement enhanced per-label summary statistics` | plan-ready.md -> `### Task 12: Implement enhanced per-label summary metrics`

**Files:**
- Modify: `FineFT/RL/DiHFT/VAE/process.py`
- Modify: `FineFT/RL/DiHFT/VAE/main.py`
- Test: `FineFT/tests/rl/test_commodity_vae_cross_contract.py`

- [x] **Step 1: Add quantile and acceptance helpers**

In `FineFT/RL/DiHFT/VAE/process.py`, replace `_logpx_stats` with a version that includes quantiles, and add an acceptance helper:

```python
SUMMARY_QUANTILES = (
    ("q01", 0.01),
    ("q05", 0.05),
    ("q25", 0.25),
    ("q50", 0.50),
    ("q75", 0.75),
    ("q95", 0.95),
    ("q99", 0.99),
)


def _logpx_stats(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("logpx array has no samples")
    return {
        "samples": int(values.size),
        "logpx_mean": float(np.mean(values)),
        "logpx_std": float(np.std(values)),
        "logpx_min": float(np.min(values)),
        "logpx_max": float(np.max(values)),
        "quantiles": {
            name: float(np.quantile(values, quantile))
            for name, quantile in SUMMARY_QUANTILES
        },
    }


def _acceptance_stats(values, train_quantiles):
    values = np.asarray(values, dtype=float).reshape(-1)
    return {
        "ge_train_q01_pct": float(np.mean(values >= train_quantiles["q01"]) * 100.0),
        "ge_train_q05_pct": float(np.mean(values >= train_quantiles["q05"]) * 100.0),
        "ge_train_q50_pct": float(np.mean(values >= train_quantiles["q50"]) * 100.0),
    }
```

- [x] **Step 2: Add sample integrity helper**

Add:

```python
def _sample_integrity(input_samples, analyzed_samples):
    input_samples = int(input_samples)
    analyzed_samples = int(analyzed_samples)
    return {
        "input_samples": input_samples,
        "analyzed_samples": analyzed_samples,
        "sample_mismatch": input_samples != analyzed_samples,
    }
```

- [x] **Step 3: Extend write_contract_logpx_outputs**

Change the function signature to:

```python
def write_contract_logpx_outputs(
    contract_results,
    save_path,
    dataset_name,
    label,
    train_baseline=None,
):
```

At the start of `write_contract_logpx_outputs`, before looping over contracts, compute:

```python
train_summary = None
if train_baseline is not None:
    train_logpx = np.asarray(train_baseline["logpx"], dtype=float).reshape(-1)
    train_summary = {
        "source_file": train_baseline["source_file"],
        **_sample_integrity(
            train_baseline.get("input_samples", train_logpx.size),
            train_baseline.get("analyzed_samples", train_logpx.size),
        ),
        **_logpx_stats(train_logpx),
    }
```

Inside the contract loop, compute:

```python
input_samples = int(result.get("input_samples", logpx.size))
contract_stats = {
    "source_file": source_file,
    **_sample_integrity(input_samples, logpx.size),
    **_logpx_stats(logpx),
}
if train_baseline is not None:
    contract_stats["acceptance"] = _acceptance_stats(
        logpx,
        train_summary["quantiles"],
    )
contract_summary[contract] = contract_stats
```

Before building `summary`, compute:

```python
all_summary = {
    **_sample_integrity(sum(item.get("input_samples", len(np.asarray(item["logpx"]).reshape(-1))) for item in contract_results), combined.size),
    **_logpx_stats(combined),
}
if train_summary is not None:
    all_summary["acceptance"] = _acceptance_stats(combined, train_summary["quantiles"])
summary = {
    "dataset_name": dataset_name,
    "label": label,
    "test": {
        "contracts": contract_summary,
        "all": all_summary,
    },
}
if train_summary is not None:
    summary["train_baseline"] = train_summary
```

- [x] **Step 4: Pass input_samples for test contracts**

In `analyze_contract_tests`, include `input_samples` from the loader dataset:

```python
"input_samples": len(item["loader"].dataset),
```

- [x] **Step 5: Compute train baseline in main.py**

In `Piplineruner.analyze_contracts`, analyze `self.train_loader` or a non-shuffled batch-size-1 train loader before calling `analyze_contract_tests`, then pass:

```python
train_baseline={
    "source_file": self.train_manifest["merged_path"],
    "input_samples": self.train_manifest["total_samples"],
    "analyzed_samples": int(np.asarray(train_logpx).reshape(-1).size),
    "logpx": np.asarray(train_logpx, dtype=float),
}
```

- [x] **Step 6: Run enhanced summary tests**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_commodity_vae_cross_contract.py::test_write_contract_logpx_outputs_writes_per_contract_and_aggregate_files FineFT/tests/rl/test_commodity_vae_cross_contract.py::test_write_contract_logpx_outputs_includes_enhanced_summary_metrics -q
```

Expected: PASS.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 13: Add cross-label routing summary tests

> **trace:** plan-ready.md -> `### Task 13: Add cross-label routing summary tests` | tasks.md -> `- [ ] 1.9 Add tests for cross-label routing_summary.json`
> **sync:** tasks.md -> `- [ ] 1.9 Add tests for cross-label routing_summary.json` | plan-ready.md -> `### Task 13: Add cross-label routing summary tests`

**Files:**
- Modify: `FineFT/tests/rl/test_commodity_vae_cross_contract.py`

- [x] **Step 1: Add routing summary writer test**

Append:

```python
def test_write_routing_summary_compares_labels_by_contract(tmp_path):
    result_root = tmp_path / "result" / "DiHFT" / "vae_results" / "fu"
    for label, values in {
        "label_0": {
            "fu2508": [-1.0, -5.0, -2.0],
            "fu2509": [-3.0, -2.0],
        },
        "label_1": {
            "fu2508": [-2.0, -4.0, -3.0],
            "fu2509": [-2.5, -5.0],
        },
        "label_2": {
            "fu2508": [-4.0, -3.0],
            "fu2509": [-1.0, -4.0],
        },
    }.items():
        label_dir = result_root / label
        label_dir.mkdir(parents=True)
        for contract, logpx in values.items():
            np.save(label_dir / f"ood_logpx_{contract}.npy", np.asarray(logpx))

    summary = vae_process.write_routing_summary(
        result_root=str(result_root),
        dataset_name="fu",
        labels=["label_0", "label_1", "label_2"],
        low_margin_threshold=1.0,
    )

    assert summary["dataset_name"] == "fu"
    assert summary["labels"] == ["label_0", "label_1", "label_2"]
    assert summary["score_type"] == "raw_logpx"
    assert summary["contracts"]["fu2508"]["samples"] == 2
    assert summary["contracts"]["fu2508"]["input_samples_by_label"]["label_0"] == 3
    assert summary["contracts"]["fu2508"]["sample_mismatch"] is True
    assert summary["contracts"]["fu2508"]["winner_counts"] == {
        "label_0": 1,
        "label_1": 0,
        "label_2": 1,
    }
    assert summary["all"]["winner_counts"] == {
        "label_0": 1,
        "label_1": 1,
        "label_2": 2,
    }
    assert (result_root / "routing_summary.json").exists()
```

- [x] **Step 2: Run the routing test and verify RED**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_commodity_vae_cross_contract.py::test_write_routing_summary_compares_labels_by_contract -q
```

Expected: FAIL because `write_routing_summary` does not exist yet.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 14: Implement cross-label routing summary generation

> **trace:** plan-ready.md -> `### Task 14: Implement cross-label routing summary generation` | tasks.md -> `- [ ] 1.10 Implement cross-label routing summary generation`
> **sync:** tasks.md -> `- [ ] 1.10 Implement cross-label routing summary generation` | plan-ready.md -> `### Task 14: Implement cross-label routing summary generation`

**Files:**
- Modify: `FineFT/RL/DiHFT/VAE/process.py`
- Modify: `FineFT/RL/DiHFT/VAE/main.py`
- Modify: `FineFT/script/train/DiHFT/low_level/VAE_util_fu.sh`
- Test: `FineFT/tests/rl/test_commodity_vae_cross_contract.py`

- [x] **Step 1: Add routing summary helpers in process.py**

Add:

```python
def _winner_summary(scores, labels, low_margin_threshold):
    winners = np.argmax(scores, axis=1)
    sorted_scores = np.sort(scores, axis=1)
    margins = sorted_scores[:, -1] - sorted_scores[:, -2]
    counts = {label: int(np.sum(winners == index)) for index, label in enumerate(labels)}
    samples = int(scores.shape[0])
    return {
        "samples": samples,
        "winner_counts": counts,
        "winner_pct": {
            label: float(count / samples * 100.0) if samples else 0.0
            for label, count in counts.items()
        },
        "top1_top2_margin_mean": float(np.mean(margins)) if samples else 0.0,
        "top1_top2_margin_q25": float(np.quantile(margins, 0.25)) if samples else 0.0,
        "low_margin_pct": float(np.mean(margins <= low_margin_threshold) * 100.0)
        if samples
        else 0.0,
    }
```

- [x] **Step 2: Add write_routing_summary in process.py**

Add `Path` to the imports:

```python
from pathlib import Path
```

Then add:

```python
def write_routing_summary(
    result_root,
    dataset_name,
    labels,
    low_margin_threshold=1.0,
):
    result_root = Path(result_root)
    contract_names = None
    by_label = {}
    for label in labels:
        label_dir = result_root / label
        files = sorted(label_dir.glob("ood_logpx_*.npy"))
        contract_values = {
            path.stem.replace("ood_logpx_", ""): np.load(path).reshape(-1)
            for path in files
            if path.name != "ood_logpx_all.npy"
        }
        by_label[label] = contract_values
        names = set(contract_values)
        contract_names = names if contract_names is None else contract_names & names
    contract_summaries = {}
    all_scores = []
    for contract in sorted(contract_names or []):
        arrays = [by_label[label][contract] for label in labels]
        input_samples_by_label = {label: int(array.size) for label, array in zip(labels, arrays)}
        n = min(input_samples_by_label.values())
        scores = np.vstack([array[:n] for array in arrays]).T
        contract_summary = _winner_summary(scores, labels, low_margin_threshold)
        contract_summary["input_samples_by_label"] = input_samples_by_label
        contract_summary["sample_mismatch"] = len(set(input_samples_by_label.values())) != 1
        contract_summaries[contract] = contract_summary
        all_scores.append(scores)
    combined = np.concatenate(all_scores, axis=0) if all_scores else np.empty((0, len(labels)))
    summary = {
        "dataset_name": dataset_name,
        "labels": list(labels),
        "score_type": "raw_logpx",
        "low_margin_threshold": float(low_margin_threshold),
        "contracts": contract_summaries,
        "all": _winner_summary(combined, labels, low_margin_threshold),
    }
    with open(result_root / "routing_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary
```

- [x] **Step 3: Keep routing summary out of main.py workflow flags**

Do not add a `--routing-summary` workflow argument to `FineFT/RL/DiHFT/VAE/main.py`. Keep `main.py` limited to the two explicit model workflows:

```python
workflow_count = sum([args.train, args.analyze_only])
if workflow_count != 1:
    parser.error("choose exactly one of --train or --analyze-only")
```

Add a parser regression test that `--routing-summary` is rejected.

- [x] **Step 4: Call routing summary from VAE_util_fu.sh after all label jobs succeed**

After the label training loop succeeds, call `write_routing_summary` directly from a Python post-processing snippet:

```bash
python - "${BASE_MODEL_PATH}" "${DATASET_NAME}" "${LABEL_COUNT}" <<'PY'
import sys
from pathlib import Path

from RL.DiHFT.VAE.process import write_routing_summary

base_model_path, dataset_name, label_count = sys.argv[1], sys.argv[2], int(sys.argv[3])
labels = [f"label_{index}" for index in range(label_count)]
write_routing_summary(
    result_root=Path(base_model_path) / "vae_results" / dataset_name,
    dataset_name=dataset_name,
    labels=labels,
)
PY
```

- [x] **Step 5: Run routing tests**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_commodity_vae_cross_contract.py::test_write_routing_summary_compares_labels_by_contract -q
```

Expected: PASS.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 15: Re-run summary extension verification

> **trace:** plan-ready.md -> `### Task 15: Re-run summary extension verification` | tasks.md -> `- [ ] 2.5 Re-run focused commodity VAE tests`
> **sync:** tasks.md -> `- [ ] 2.5 Re-run focused commodity VAE tests` | plan-ready.md -> `### Task 15: Re-run summary extension verification`

**Files:**
- Verify: `FineFT/tests/rl/test_commodity_vae_cross_contract.py`
- Verify: `FineFT/RL/DiHFT/VAE/main.py`
- Verify: `FineFT/RL/DiHFT/VAE/process.py`
- Verify: `FineFT/RL/DiHFT/VAE/merge_vae_train.py`
- Verify: `FineFT/script/train/DiHFT/low_level/VAE_util_fu.sh`
- Verify: `openspec/changes/support-commodity-vae-cross-contract`

- [x] **Step 1: Run focused commodity VAE tests**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_commodity_vae_cross_contract.py -q
```

Expected: PASS.

- [x] **Step 2: Compile VAE Python files**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && python -m py_compile FineFT/RL/DiHFT/VAE/main.py FineFT/RL/DiHFT/VAE/process.py FineFT/RL/DiHFT/VAE/merge_vae_train.py
```

Expected: exits 0.

- [x] **Step 3: Validate shell syntax**

Run:

```bash
bash -n FineFT/script/train/DiHFT/low_level/VAE_util_fu.sh
```

Expected: exits 0.

- [x] **Step 4: Run OpenSpec strict validation**

Run:

```bash
openspec validate support-commodity-vae-cross-contract --strict
```

Expected:

```text
Change 'support-commodity-vae-cross-contract' is valid
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）
