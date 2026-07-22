# Refactor Datahandler Manifest Objects Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `FineFT/datahandler` manifest dict plumbing with dataclass objects while preserving existing manifest JSON payloads and datahandler behavior.

**Architecture:** Add a focused `FineFT/datahandler/manifests.py` module that owns split, dataset, and slice manifest data structures plus JSON boundary conversion. Keep pandas/numpy I/O in `commodity_contract_dataset.py` and `slice_model.py`; those modules should mutate manifest objects and call `to_dict()` only at JSON write boundaries.

**Tech Stack:** Python dataclasses, pathlib, json, pytest, pandas, numpy, existing `finetf` conda environment.

**Traceability (sddflow):**
- plan-ready: `openspec/changes/refactor-datahandler-manifest-objects/plan-ready.md`
- tasks: `openspec/changes/refactor-datahandler-manifest-objects/tasks.md`
- plan: `docs/superpowers/plans/2026-07-22-refactor-datahandler-manifest-objects.md`

---

## File Structure

- Create `FineFT/datahandler/manifests.py`: dataclass definitions and manifest conversion/update helpers only.
- Modify `FineFT/datahandler/commodity_contract_dataset.py`: replace dict manifest access with `DatasetSplitManifest` and `DatasetManifest`.
- Modify `FineFT/datahandler/slice_model.py`: replace manifest dict read/update/write code with `SliceManifest`.
- Modify `FineFT/tests/datahandler/test_commodity_contract_dataset.py`: assert object return types, object attributes, and dataset manifest JSON compatibility.
- Modify `FineFT/tests/datahandler/test_slice_model.py`: assert `SliceManifest` object behavior and preserve existing JSON compatibility checks.

### Task 1: Add Focused Datahandler Manifest Object Tests

> **trace:** plan-ready.md → `### Task 1: Add focused datahandler manifest object tests` | tasks.md → `- [ ] 1.1 Add focused tests for datahandler manifest object return types, attribute access, and JSON payload compatibility.`
> **sync:** tasks.md → `- [ ] 1.1 Add focused tests for datahandler manifest object return types, attribute access, and JSON payload compatibility.` | plan-ready.md → `### Task 1: Add focused datahandler manifest object tests`

**Files:**
- Modify: `FineFT/tests/datahandler/test_commodity_contract_dataset.py`
- Modify: `FineFT/tests/datahandler/test_slice_model.py`

- [x] **Step 1: Add expected imports to commodity dataset tests**

In `FineFT/tests/datahandler/test_commodity_contract_dataset.py`, extend the imports from `FineFT.datahandler.commodity_contract_dataset` and add manifest type imports:

```python
from FineFT.datahandler.commodity_contract_dataset import (
    build_dataset_manifest,
    load_dataset_split_manifest,
    run_dataset_generation,
    write_stage_datasets,
    write_train_slices,
)
from FineFT.datahandler.manifests import DatasetManifest, DatasetSplitManifest
```

- [x] **Step 2: Convert split manifest assertions to object attributes**

In `test_load_dataset_split_manifest_validates_symbol_and_target_freq`, replace the dict assertions with:

```python
    assert isinstance(manifest, DatasetSplitManifest)
    assert manifest.symbol == "fu"
    assert manifest.target_freq == "10min"
    assert [item.contract for item in manifest.sets["train"].contracts] == [
        "fu2508",
        "fu2509",
    ]
```

- [x] **Step 3: Convert dataset manifest builder assertions to object attributes**

In `test_build_dataset_manifest_uses_split_manifest_and_stage_scale_save_paths`, construct the split manifest through the loader object path and assert `DatasetManifest`:

```python
    split_manifest_path = _write_dataset_split_manifest(
        tmp_path / "dataset_split_manifest.json",
        manifest=_dataset_split_manifest(),
    )
    split_manifest = load_dataset_split_manifest(
        split_manifest_path,
        symbol="fu",
        target_freq="10min",
    )

    manifest = build_dataset_manifest(
        split_manifest=split_manifest,
        dataset_split_manifest_path=split_manifest_path,
        input_root=tmp_path / "SCALE_SAVE",
        state_features_path=tmp_path / "FEATURE_SELECTION" / "state_features.npy",
        output_root=tmp_path / "dataset" / "10min",
        symbol="fu",
        target_freq="10min",
        chunk_length=2,
        early_stop=1,
    )

    assert isinstance(manifest, DatasetManifest)
    assert manifest.dataset_split_manifest_path.endswith("dataset_split_manifest.json")
    assert manifest.state_features_path.endswith("dataset/10min/fu/state_features.npy")
    train_contracts = {
        item.contract: item for item in manifest.sets["train"].contracts
    }
    assert train_contracts["fu2508"].input_path.endswith(
        "SCALE_SAVE/fu/10min/train/fu2508.feather"
    )
    assert train_contracts["fu2508"].output_path.endswith(
        "dataset/10min/fu/train/fu2508.feather"
    )
    assert train_contracts["fu2508"].slice_outputs[0].path.endswith(
        "dataset/10min/fu/train/slice/df_0.feather"
    )
    assert manifest.sets["valid"].skipped_contracts == [
        {"contract": "fu2509", "reason": "no trading days in valid range"}
    ]
```

- [x] **Step 4: Update direct write helper tests to build DatasetManifest from dict**

For tests that currently create a minimal manifest dict and call `write_stage_datasets()` or `write_train_slices()`, import and wrap with `DatasetManifest.from_dict(...)` before calling the production function:

```python
    manifest = DatasetManifest.from_dict(
        {
            "symbol": "fu",
            "target_freq": "10min",
            "dataset_split_manifest_path": str(
                tmp_path / "dataset_split_manifest.json"
            ),
            "state_features_source_path": str(state_features),
            "state_features_path": str(dataset_root / "state_features.npy"),
            "sets": {
                "train": {
                    "range": None,
                    "contracts": [
                        {
                            "contract": "fu2508",
                            "input_path": str(train_file),
                            "output_path": str(
                                dataset_root / "train" / "fu2508.feather"
                            ),
                        }
                    ],
                    "skipped_contracts": [],
                }
            },
        }
    )
```

When asserting mutated values, use attributes:

```python
    assert manifest.sets["train"].contracts[0].output_row_count == 3
    assert manifest.sets["valid"].contracts_total_count == 2
    assert manifest.sets["test"].contracts_total_count == 2
```

- [x] **Step 5: Assert run_dataset_generation returns an object and JSON matches to_dict**

In `test_run_dataset_generation_writes_manifest_stage_files_and_train_slices`, keep the existing JSON assertions and capture the return value:

```python
    returned_manifest = run_dataset_generation(
        dataset_split_manifest_path=manifest_path,
        input_root=tmp_path / "SCALE_SAVE",
        state_features_path=state_features,
        output_root=tmp_path / "dataset" / "10min",
        symbol="fu",
        target_freq="10min",
        chunk_length=4,
        early_stop=1,
    )

    assert isinstance(returned_manifest, DatasetManifest)
```

After loading `dataset_manifest.json`, add:

```python
    assert manifest == returned_manifest.to_dict()
```

- [x] **Step 6: Add SliceManifest focused test**

In `FineFT/tests/datahandler/test_slice_model.py`, add imports and a focused unit test near the helpers:

```python
from FineFT.datahandler.manifests import (
    SliceContractManifest,
    SliceFileManifest,
    SliceLabelManifest,
    SliceManifest,
)
```

```python
def test_slice_manifest_replaces_contract_and_rebuilds_label_view(tmp_path):
    valid_dir = tmp_path / "dataset" / "10min" / "fu" / "valid"
    manifest = SliceManifest(valid_path=str(valid_dir))
    manifest.replace_contract(
        SliceContractManifest(
            contract="fu2505",
            processed_path=str(valid_dir / "processed" / "valid_processed_fu2505.feather"),
            labels={
                "label_0": SliceLabelManifest(
                    label="label_0",
                    file_count=1,
                    total_row_count=2,
                    files=[
                        SliceFileManifest(
                            path=str(valid_dir / "fu2505" / "label_0" / "df_0.feather"),
                            output_row_count=2,
                        )
                    ],
                )
            },
        )
    )
    manifest.record_skipped_contract(
        contract="fu2509",
        processed_path=str(valid_dir / "processed" / "valid_processed_fu2509.feather"),
        reason="insufficient rows",
        input_row_count=4,
    )

    payload = manifest.to_dict()

    assert sorted(payload["contracts"].keys()) == ["fu2505"]
    assert sorted(payload["labels"].keys()) == ["label_0"]
    assert payload["labels"]["label_0"]["files"][0]["contract"] == "fu2505"
    assert payload["labels"]["label_0"]["total_row_count"] == 2
    assert payload["skipped_contracts"]["fu2509"]["input_row_count"] == 4
```

- [x] **Step 7: Run tests to confirm the new expectations fail before implementation**

Run:

```bash
conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py FineFT/tests/datahandler/test_slice_model.py
```

Expected: FAIL with import errors for `FineFT.datahandler.manifests` or assertion failures because production functions still return dicts.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 2: Add Datahandler Manifest Dataclass Models

> **trace:** plan-ready.md → `### Task 2: Add datahandler manifest dataclass models` | tasks.md → `- [ ] 1.2 Add `FineFT/datahandler/manifests.py` with dataclass models for split, dataset, and slice manifests.`
> **sync:** tasks.md → `- [ ] 1.2 Add `FineFT/datahandler/manifests.py` with dataclass models for split, dataset, and slice manifests.` | plan-ready.md → `### Task 2: Add datahandler manifest dataclass models`

**Files:**
- Create: `FineFT/datahandler/manifests.py`

- [x] **Step 1: Create the manifest model module**

Create `FineFT/datahandler/manifests.py` with this structure:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


STAGES = ("train", "valid", "test")


def _copy_optional_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    return list(value)


@dataclass
class DatasetSplitContract:
    contract: str
    range: list[str] | None = None
    trading_days: list[str] | None = None
    output_row_count: int | None = None

    @classmethod
    def from_dict(cls, item: dict[str, Any], stage: str) -> "DatasetSplitContract":
        if not isinstance(item, dict) or not isinstance(item.get("contract"), str):
            raise ValueError(
                f"dataset split manifest sets.{stage}.contracts items need contract"
            )
        output_row_count = item.get("output_row_count")
        return cls(
            contract=item["contract"],
            range=_copy_optional_list(item.get("range")),
            trading_days=_copy_optional_list(item.get("trading_days")),
            output_row_count=(
                int(output_row_count) if output_row_count is not None else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"contract": self.contract}
        if self.range is not None:
            payload["range"] = list(self.range)
        if self.trading_days is not None:
            payload["trading_days"] = list(self.trading_days)
        if self.output_row_count is not None:
            payload["output_row_count"] = self.output_row_count
        return payload


@dataclass
class DatasetSplitSet:
    range: list[str] | None = None
    contracts: list[DatasetSplitContract] = field(default_factory=list)
    skipped_contracts: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, stage: str, stage_info: Any) -> "DatasetSplitSet":
        if not isinstance(stage_info, dict):
            raise ValueError(f"dataset split manifest missing sets.{stage}")
        if "contracts" not in stage_info or not isinstance(
            stage_info["contracts"], list
        ):
            raise ValueError(
                f"dataset split manifest sets.{stage}.contracts must be a list"
            )
        return cls(
            range=_copy_optional_list(stage_info.get("range")),
            contracts=[
                DatasetSplitContract.from_dict(item, stage)
                for item in stage_info["contracts"]
            ],
            skipped_contracts=[
                dict(item) for item in stage_info.get("skipped_contracts", [])
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "range": list(self.range) if self.range is not None else None,
            "contracts": [contract.to_dict() for contract in self.contracts],
            "skipped_contracts": [dict(item) for item in self.skipped_contracts],
        }


@dataclass
class DatasetSplitManifest:
    symbol: str
    target_freq: str
    sets: dict[str, DatasetSplitSet]

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        symbol: str,
        target_freq: str,
    ) -> "DatasetSplitManifest":
        if payload.get("symbol") != symbol:
            raise ValueError(
                "dataset split manifest symbol mismatch: "
                f"expected={symbol} actual={payload.get('symbol')}"
            )
        if payload.get("target_freq") != target_freq:
            raise ValueError(
                "dataset split manifest target_freq mismatch: "
                f"expected={target_freq} actual={payload.get('target_freq')}"
            )
        sets = payload.get("sets")
        if not isinstance(sets, dict):
            raise ValueError("dataset split manifest missing sets")
        return cls(
            symbol=symbol,
            target_freq=target_freq,
            sets={
                stage: DatasetSplitSet.from_dict(stage, sets.get(stage))
                for stage in STAGES
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "target_freq": self.target_freq,
            "sets": {stage: self.sets[stage].to_dict() for stage in STAGES},
        }
```

- [x] **Step 2: Add dataset output dataclasses**

Append these classes to `FineFT/datahandler/manifests.py`:

```python
@dataclass
class DatasetSliceOutput:
    index: int
    contract: str | None
    path: str
    source_output: str | None
    row_start: int
    row_end: int
    output_row_count: int | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetSliceOutput":
        return cls(
            index=int(payload["index"]),
            contract=payload.get("contract"),
            path=payload["path"],
            source_output=payload.get("source_output"),
            row_start=int(payload["row_start"]),
            row_end=int(payload["row_end"]),
            output_row_count=(
                int(payload["output_row_count"])
                if "output_row_count" in payload
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "index": self.index,
            "path": self.path,
            "row_start": self.row_start,
            "row_end": self.row_end,
        }
        if self.contract is not None:
            payload["contract"] = self.contract
        if self.source_output is not None:
            payload["source_output"] = self.source_output
        if self.output_row_count is not None:
            payload["output_row_count"] = self.output_row_count
        return payload


@dataclass
class DatasetContractManifest:
    contract: str
    input_path: str
    output_path: str
    range: list[str] | None = None
    trading_days: list[str] | None = None
    output_row_count: int | None = None
    slice_outputs: list[DatasetSliceOutput] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetContractManifest":
        return cls(
            contract=payload["contract"],
            input_path=payload["input_path"],
            output_path=payload["output_path"],
            range=_copy_optional_list(payload.get("range")),
            trading_days=_copy_optional_list(payload.get("trading_days")),
            output_row_count=(
                int(payload["output_row_count"])
                if "output_row_count" in payload
                else None
            ),
            slice_outputs=[
                DatasetSliceOutput.from_dict(item)
                for item in payload.get("slice_outputs", [])
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contract": self.contract,
            "input_path": self.input_path,
            "output_path": self.output_path,
        }
        if self.range is not None:
            payload["range"] = list(self.range)
        if self.trading_days is not None:
            payload["trading_days"] = list(self.trading_days)
        if self.output_row_count is not None:
            payload["output_row_count"] = self.output_row_count
        if self.slice_outputs:
            payload["slice_outputs"] = [
                slice_output.to_dict() for slice_output in self.slice_outputs
            ]
        return payload


@dataclass
class DatasetSetManifest:
    range: list[str] | None = None
    contracts: list[DatasetContractManifest] = field(default_factory=list)
    skipped_contracts: list[dict[str, Any]] = field(default_factory=list)
    contracts_total_count: int | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetSetManifest":
        return cls(
            range=_copy_optional_list(payload.get("range")),
            contracts=[
                DatasetContractManifest.from_dict(item)
                for item in payload.get("contracts", [])
            ],
            skipped_contracts=[
                dict(item) for item in payload.get("skipped_contracts", [])
            ],
            contracts_total_count=(
                int(payload["contracts_total_count"])
                if "contracts_total_count" in payload
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "range": list(self.range) if self.range is not None else None,
            "contracts": [contract.to_dict() for contract in self.contracts],
            "skipped_contracts": [dict(item) for item in self.skipped_contracts],
        }
        if self.contracts_total_count is not None:
            payload["contracts_total_count"] = self.contracts_total_count
        return payload


@dataclass
class DatasetManifest:
    symbol: str
    target_freq: str
    dataset_split_manifest_path: str
    state_features_source_path: str
    state_features_path: str
    sets: dict[str, DatasetSetManifest]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetManifest":
        return cls(
            symbol=payload.get("symbol", ""),
            target_freq=payload.get("target_freq", ""),
            dataset_split_manifest_path=payload.get(
                "dataset_split_manifest_path", ""
            ),
            state_features_source_path=payload["state_features_source_path"],
            state_features_path=payload["state_features_path"],
            sets={
                stage: DatasetSetManifest.from_dict(stage_payload)
                for stage, stage_payload in payload.get("sets", {}).items()
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "target_freq": self.target_freq,
            "dataset_split_manifest_path": self.dataset_split_manifest_path,
            "state_features_source_path": self.state_features_source_path,
            "state_features_path": self.state_features_path,
            "sets": {
                stage: set_info.to_dict() for stage, set_info in self.sets.items()
            },
        }
```

- [x] **Step 3: Add slice manifest dataclasses**

Append these classes to `FineFT/datahandler/manifests.py`:

```python
@dataclass
class SliceFileManifest:
    path: str
    output_row_count: int
    contract: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SliceFileManifest":
        return cls(
            path=payload["path"],
            output_row_count=int(payload["output_row_count"]),
            contract=payload.get("contract"),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "path": self.path,
            "output_row_count": self.output_row_count,
        }
        if self.contract is not None:
            payload["contract"] = self.contract
        return payload


@dataclass
class SliceLabelManifest:
    label: str
    file_count: int = 0
    total_row_count: int = 0
    files: list[SliceFileManifest] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SliceLabelManifest":
        return cls(
            label=payload["label"],
            file_count=int(payload.get("file_count", 0)),
            total_row_count=int(payload.get("total_row_count", 0)),
            files=[
                SliceFileManifest.from_dict(item)
                for item in payload.get("files", [])
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "file_count": self.file_count,
            "total_row_count": self.total_row_count,
            "files": [file_info.to_dict() for file_info in self.files],
        }


@dataclass
class SliceContractManifest:
    contract: str
    processed_path: str
    file_count: int = 0
    total_row_count: int = 0
    labels: dict[str, SliceLabelManifest] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SliceContractManifest":
        return cls(
            contract=payload["contract"],
            processed_path=payload["processed_path"],
            file_count=int(payload.get("file_count", 0)),
            total_row_count=int(payload.get("total_row_count", 0)),
            labels={
                label: SliceLabelManifest.from_dict(label_info)
                for label, label_info in payload.get("labels", {}).items()
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "processed_path": self.processed_path,
            "file_count": self.file_count,
            "total_row_count": self.total_row_count,
            "labels": {
                label: label_info.to_dict()
                for label, label_info in sorted(self.labels.items())
            },
        }


@dataclass
class SkippedContractManifest:
    contract: str
    processed_path: str
    reason: str
    input_row_count: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SkippedContractManifest":
        return cls(
            contract=payload["contract"],
            processed_path=payload["processed_path"],
            reason=payload["reason"],
            input_row_count=int(payload["input_row_count"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "processed_path": self.processed_path,
            "reason": self.reason,
            "input_row_count": self.input_row_count,
        }


@dataclass
class SliceManifest:
    valid_path: str
    contracts: dict[str, SliceContractManifest] = field(default_factory=dict)
    labels: dict[str, SliceLabelManifest] = field(default_factory=dict)
    skipped_contracts: dict[str, SkippedContractManifest] = field(
        default_factory=dict
    )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SliceManifest":
        manifest = cls(
            valid_path=payload["valid_path"],
            contracts={
                contract: SliceContractManifest.from_dict(contract_info)
                for contract, contract_info in payload.get("contracts", {}).items()
            },
            skipped_contracts={
                contract: SkippedContractManifest.from_dict(skip_info)
                for contract, skip_info in payload.get("skipped_contracts", {}).items()
            },
        )
        manifest.rebuild_labels()
        return manifest

    @classmethod
    def new(cls, valid_root: object) -> "SliceManifest":
        return cls(valid_path=str(valid_root))

    def replace_contract(self, contract_record: SliceContractManifest) -> None:
        self.skipped_contracts.pop(contract_record.contract, None)
        if contract_record.file_count:
            self.contracts[contract_record.contract] = contract_record
        else:
            self.contracts.pop(contract_record.contract, None)
        self.rebuild_labels()
        self.sort()

    def record_skipped_contract(
        self,
        *,
        contract: str,
        processed_path: str,
        reason: str,
        input_row_count: int,
    ) -> None:
        self.contracts.pop(contract, None)
        self.skipped_contracts[contract] = SkippedContractManifest(
            contract=contract,
            processed_path=processed_path,
            reason=reason,
            input_row_count=int(input_row_count),
        )
        self.rebuild_labels()
        self.sort()

    def rebuild_labels(self) -> None:
        labels: dict[str, SliceLabelManifest] = {}
        for contract_record in self.contracts.values():
            for label, label_info in contract_record.labels.items():
                target = labels.setdefault(label, SliceLabelManifest(label=label))
                target.file_count += label_info.file_count
                target.total_row_count += label_info.total_row_count
                for file_info in label_info.files:
                    target.files.append(
                        SliceFileManifest(
                            contract=contract_record.contract,
                            path=file_info.path,
                            output_row_count=file_info.output_row_count,
                        )
                    )
        self.labels = dict(sorted(labels.items()))

    def sort(self) -> None:
        self.contracts = dict(sorted(self.contracts.items()))
        self.labels = dict(sorted(self.labels.items()))
        self.skipped_contracts = dict(sorted(self.skipped_contracts.items()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid_path": self.valid_path,
            "contracts": {
                contract: contract_info.to_dict()
                for contract, contract_info in sorted(self.contracts.items())
            },
            "labels": {
                label: label_info.to_dict()
                for label, label_info in sorted(self.labels.items())
            },
            "skipped_contracts": {
                contract: skip_info.to_dict()
                for contract, skip_info in sorted(self.skipped_contracts.items())
            },
        }
```

- [x] **Step 4: Compile the new module**

Run:

```bash
conda activate finetf && python -m py_compile FineFT/datahandler/manifests.py
```

Expected: command exits with status 0 and prints no syntax errors.

- [x] **Step 5: Run the focused SliceManifest unit test**

Run:

```bash
conda activate finetf && pytest FineFT/tests/datahandler/test_slice_model.py::test_slice_manifest_replaces_contract_and_rebuilds_label_view -v
```

Expected: PASS for the focused object-model test.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 3: Refactor commodity_contract_dataset.py to Use Manifest Objects

> **trace:** plan-ready.md → `### Task 3: Refactor commodity_contract_dataset.py to use manifest objects` | tasks.md → `- [ ] 1.3 Refactor `FineFT/datahandler/commodity_contract_dataset.py` to use `DatasetSplitManifest` and `DatasetManifest` objects internally and at public return boundaries.`
> **sync:** tasks.md → `- [ ] 1.3 Refactor `FineFT/datahandler/commodity_contract_dataset.py` to use `DatasetSplitManifest` and `DatasetManifest` objects internally and at public return boundaries.` | plan-ready.md → `### Task 3: Refactor commodity_contract_dataset.py to use manifest objects`

**Files:**
- Modify: `FineFT/datahandler/commodity_contract_dataset.py`
- Modify: `FineFT/tests/datahandler/test_commodity_contract_dataset.py`

- [x] **Step 1: Import manifest dataclasses**

In `FineFT/datahandler/commodity_contract_dataset.py`, add these imports after the pandas/numpy imports:

```python
from FineFT.datahandler.manifests import (
    DatasetContractManifest,
    DatasetManifest,
    DatasetSetManifest,
    DatasetSliceOutput,
    DatasetSplitManifest,
)
```

- [x] **Step 2: Return DatasetSplitManifest from load_dataset_split_manifest**

Replace the validation loop in `load_dataset_split_manifest()` with object construction:

```python
def load_dataset_split_manifest(path, symbol, target_freq):
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing dataset split manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    return DatasetSplitManifest.from_dict(
        manifest,
        symbol=symbol,
        target_freq=target_freq,
    )
```

- [x] **Step 3: Return DatasetSliceOutput objects from _build_slice_plan**

Replace the dict append in `_build_slice_plan()` with:

```python
        outputs.append(
            DatasetSliceOutput(
                index=index,
                contract=contract,
                path=str(
                    Path(output_root)
                    / symbol
                    / "train"
                    / "slice"
                    / f"df_{index}.feather"
                ),
                source_output=str(
                    _stage_output(output_root, symbol, "train", contract)
                ),
                row_start=row_start,
                row_end=row_end,
            )
        )
```

- [x] **Step 4: Build DatasetManifest objects instead of dicts**

Replace the body of `build_dataset_manifest()` with:

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
    manifest = DatasetManifest(
        symbol=symbol,
        target_freq=target_freq,
        dataset_split_manifest_path=str(dataset_split_manifest_path),
        state_features_source_path=str(state_features_path),
        state_features_path=str(Path(output_root) / symbol / "state_features.npy"),
        sets={},
    )

    next_slice = 0
    for set_name in STAGES:
        split_set = split_manifest.sets[set_name]
        set_contracts = []
        for item in split_set.contracts:
            contract_name = item.contract
            record = DatasetContractManifest(
                contract=contract_name,
                input_path=str(
                    _stage_input_path(
                        input_root, symbol, target_freq, set_name, contract_name
                    )
                ),
                output_path=str(
                    _stage_output(output_root, symbol, set_name, contract_name)
                ),
                range=item.range if item.range is not None else split_set.range,
                trading_days=item.trading_days,
            )
            if set_name == "train":
                row_count = int(item.output_row_count or 0)
                slices, next_slice = _build_slice_plan(
                    row_count,
                    output_root,
                    symbol,
                    contract_name,
                    next_slice,
                    chunk_length,
                    early_stop,
                )
                record.slice_outputs = slices
            set_contracts.append(record)
        manifest.sets[set_name] = DatasetSetManifest(
            range=split_set.range,
            contracts=set_contracts,
            skipped_contracts=split_set.skipped_contracts,
        )
    return manifest
```

- [x] **Step 5: Update stage dataset and slice writers to use attributes**

Replace dict access in `write_stage_datasets()`, `rebuild_train_slice_plan()`, and `write_train_slices()` with object attribute access:

```python
def write_stage_datasets(manifest):
    state_features_source_path = Path(manifest.state_features_source_path)
    if not state_features_source_path.exists():
        raise FileNotFoundError(
            f"Missing selected state_features.npy: {state_features_source_path}"
        )
    state_features = np.load(state_features_source_path, allow_pickle=True).tolist()
    if not state_features:
        raise ValueError(f"state feature list is empty: {state_features_source_path}")

    state_features_path = Path(manifest.state_features_path)
    state_features_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(state_features_source_path, state_features_path)

    for stage, set_info in manifest.sets.items():
        contracts_total_count = 0
        for contract in set_info.contracts:
            input_path = Path(contract.input_path)
            if not input_path.exists():
                raise FileNotFoundError(
                    "Missing SCALE_SAVE file for "
                    f"stage={stage} contract={contract.contract}: {input_path}"
                )
            output_path = Path(contract.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(input_path, output_path)
            output_df = pd.read_feather(output_path)
            if output_df.empty:
                raise ValueError(
                    "copied empty stage dataset: "
                    f"stage={stage} contract={contract.contract} output={output_path}"
                )
            contract.output_row_count = int(len(output_df))
            contracts_total_count += contract.output_row_count
        set_info.contracts_total_count = contracts_total_count


def rebuild_train_slice_plan(manifest, chunk_length, early_stop):
    next_index = 0
    for contract in manifest.sets["train"].contracts:
        output_path = Path(contract.output_path)
        slice_dir = output_path.parent / "slice"
        row_count = int(contract.output_row_count or 0)
        row_start = 0
        slice_outputs = []
        while row_start < row_count:
            row_end = min(row_start + chunk_length + early_stop, row_count)
            slice_outputs.append(
                DatasetSliceOutput(
                    index=next_index,
                    contract=contract.contract,
                    path=str(slice_dir / f"df_{next_index}.feather"),
                    source_output=str(output_path),
                    row_start=row_start,
                    row_end=row_end,
                )
            )
            next_index += 1
            row_start += chunk_length
        contract.slice_outputs = slice_outputs


def write_train_slices(manifest):
    expected_index = 0
    for contract in manifest.sets["train"].contracts:
        df = pd.read_feather(contract.output_path)
        for slice_info in contract.slice_outputs:
            if int(slice_info.index) != expected_index:
                raise ValueError("train slice indices must be continuous")
            sliced = df.iloc[slice_info.row_start:slice_info.row_end].reset_index(
                drop=True
            )
            if sliced.empty:
                continue
            output_path = Path(slice_info.path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sliced.to_feather(output_path)
            slice_info.output_row_count = int(len(sliced))
            expected_index += 1
```

- [x] **Step 6: Serialize DatasetManifest only at the JSON boundary**

In `run_dataset_generation()`, replace the `json.dumps(manifest, ...)` call with:

```python
    (dataset_root / "dataset_manifest.json").write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
```

- [x] **Step 7: Run commodity dataset focused tests**

Run:

```bash
conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py -v
```

Expected: PASS. Existing failure tests for missing split manifest, symbol mismatch, target frequency mismatch, missing stage contracts, missing state features, empty state features, missing SCALE_SAVE file, empty stage data, and contiguous train slice indices remain covered.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 4: Refactor slice_model.py to Use SliceManifest

> **trace:** plan-ready.md → `### Task 4: Refactor slice_model.py to use SliceManifest` | tasks.md → `- [ ] 1.4 Refactor `FineFT/datahandler/slice_model.py` to use `SliceManifest` for manifest reads, contract updates, skip updates, label aggregation, sorting, and JSON serialization.`
> **sync:** tasks.md → `- [ ] 1.4 Refactor `FineFT/datahandler/slice_model.py` to use `SliceManifest` for manifest reads, contract updates, skip updates, label aggregation, sorting, and JSON serialization.` | plan-ready.md → `### Task 4: Refactor slice_model.py to use SliceManifest`

**Files:**
- Modify: `FineFT/datahandler/slice_model.py`
- Modify: `FineFT/tests/datahandler/test_slice_model.py`

- [x] **Step 1: Import slice manifest dataclasses**

In `FineFT/datahandler/slice_model.py`, keep the existing `json` import for reading/writing the file boundary and add:

```python
from FineFT.datahandler.manifests import (
    SliceContractManifest,
    SliceFileManifest,
    SliceLabelManifest,
    SliceManifest,
)
```

- [x] **Step 2: Add a local manifest loader helper**

Inside `Linear_Market_Dynamics_Model`, add this helper near `_contract_name()`:

```python
    def _load_slice_manifest(self, manifest_path, valid_root):
        if manifest_path.exists():
            return SliceManifest.from_dict(
                json.loads(manifest_path.read_text(encoding="utf-8"))
            )
        return SliceManifest.new(valid_root)
```

- [x] **Step 3: Replace _write_slice_manifest with object update logic**

Replace `_write_slice_manifest()` with:

```python
    def _write_slice_manifest(
        self,
        manifest_path,
        valid_root,
        contract_name,
        processed_path,
        contract_labels,
    ):
        manifest = self._load_slice_manifest(manifest_path, valid_root)
        contract_file_count = sum(
            label_info.file_count for label_info in contract_labels.values()
        )
        contract_total_rows = sum(
            label_info.total_row_count for label_info in contract_labels.values()
        )
        manifest.replace_contract(
            SliceContractManifest(
                contract=contract_name,
                processed_path=str(processed_path),
                file_count=contract_file_count,
                total_row_count=contract_total_rows,
                labels=contract_labels,
            )
        )
        self._write_manifest(manifest_path, manifest)
```

- [x] **Step 4: Remove manual label aggregation from slice_model.py**

Delete `_build_label_manifest()` from `Linear_Market_Dynamics_Model`. `SliceManifest.rebuild_labels()` now owns the contract-to-label aggregation behavior.

- [x] **Step 5: Serialize SliceManifest at the JSON boundary**

Replace `_write_manifest()` with:

```python
    def _write_manifest(self, manifest_path, manifest):
        manifest_path.write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
```

- [x] **Step 6: Replace _write_skip_manifest with object update logic**

Replace `_write_skip_manifest()` with:

```python
    def _write_skip_manifest(
        self,
        manifest_path,
        valid_root,
        contract_name,
        processed_path,
        reason,
        input_row_count,
    ):
        manifest = self._load_slice_manifest(manifest_path, valid_root)
        manifest.record_skipped_contract(
            contract=contract_name,
            processed_path=str(processed_path),
            reason=reason,
            input_row_count=input_row_count,
        )
        self._write_manifest(manifest_path, manifest)
```

- [x] **Step 7: Build contract label records as SliceLabelManifest objects**

In `run()`, keep `label_counter` as a plain list, but make `contract_labels` store `SliceLabelManifest` objects. Replace the `label_info` block inside `write_segment()` with:

```python
            label_info = contract_labels.setdefault(
                label_name,
                SliceLabelManifest(label=label_name),
            )
            output_row_count = int(len(segment))
            label_info.file_count += 1
            label_info.total_row_count += output_row_count
            label_info.files.append(
                SliceFileManifest(
                    path=str(output_file),
                    output_row_count=output_row_count,
                )
            )
```

At the `_write_slice_manifest()` call site, pass the object mapping directly:

```python
        self._write_slice_manifest(
            ticker_name_path / "slice_manifest.json",
            ticker_name_path,
            contract_name,
            process_data_path,
            dict(sorted(contract_labels.items())),
        )
```

- [x] **Step 8: Run slice model focused tests**

Run:

```bash
conda activate finetf && pytest FineFT/tests/datahandler/test_slice_model.py -v
```

Expected: PASS. Existing tests still prove `slice_manifest.json` contract accumulation, rerun replacement behavior, skipped contract recording, and small slope segment handling.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 5: Run Focused Verification

> **trace:** plan-ready.md → `### Task 5: Run focused verification` | tasks.md → `- [ ] 1.5 Run focused verification for datahandler tests, Python compilation, and OpenSpec validation.`
> **sync:** tasks.md → `- [ ] 1.5 Run focused verification for datahandler tests, Python compilation, and OpenSpec validation.` | plan-ready.md → `### Task 5: Run focused verification`

**Files:**
- Modify only if verification exposes an issue in files changed by Tasks 1-4.

- [x] **Step 1: Run focused datahandler tests**

Run:

```bash
conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py FineFT/tests/datahandler/test_slice_model.py
```

Expected: PASS for all selected tests.

- [x] **Step 2: Run Python compilation checks**

Run:

```bash
conda activate finetf && python -m py_compile FineFT/datahandler/manifests.py FineFT/datahandler/commodity_contract_dataset.py FineFT/datahandler/slice_model.py
```

Expected: command exits with status 0 and prints no syntax errors.

- [x] **Step 3: Run OpenSpec strict validation**

Run:

```bash
openspec validate refactor-datahandler-manifest-objects --strict
```

Expected:

```text
Change 'refactor-datahandler-manifest-objects' is valid
```

- [x] **Step 4: Inspect git diff for scope**

Run:

```bash
git diff -- FineFT/datahandler/manifests.py FineFT/datahandler/commodity_contract_dataset.py FineFT/datahandler/slice_model.py FineFT/tests/datahandler/test_commodity_contract_dataset.py FineFT/tests/datahandler/test_slice_model.py openspec/changes/refactor-datahandler-manifest-objects docs/superpowers/plans/2026-07-22-refactor-datahandler-manifest-objects.md
```

Expected: diff only contains datahandler manifest object refactor changes, matching the proposal scope. Existing unrelated untracked VAE files are not modified by this work.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

## Amendments

### 2026-07-22: typed skipped contract records

This amendment extends the same manifest object seam to dataset split/output `skipped_contracts`, which still use `list[dict]` after Tasks 1-5. Existing completed tasks remain complete; Task 6 is the only new implementation work.

### Task 6: Refactor Dataset Skipped Contracts to Dataclass Records

> **trace:** plan-ready.md → `### Task 6: Refactor dataset skipped contracts to dataclass records` | tasks.md → `- [ ] 1.6 Refactor dataset split/output skipped_contracts from list[dict] to list[DatasetSkippedContract] while preserving JSON compatibility.`
> **sync:** tasks.md → `- [ ] 1.6 Refactor dataset split/output skipped_contracts from list[dict] to list[DatasetSkippedContract] while preserving JSON compatibility.` | plan-ready.md → `### Task 6: Refactor dataset skipped contracts to dataclass records`

**Files:**
- Modify: `FineFT/datahandler/manifests.py`
- Modify: `FineFT/datahandler/commodity_contract_dataset.py`
- Modify: `FineFT/tests/datahandler/test_commodity_contract_dataset.py`

- [x] **Step 1: Write failing skipped-contract object assertions**

In `FineFT/tests/datahandler/test_commodity_contract_dataset.py`, import the new class:

```python
from FineFT.datahandler.manifests import (
    DatasetManifest,
    DatasetSkippedContract,
    DatasetSplitManifest,
)
```

In `test_build_dataset_manifest_uses_split_manifest_and_stage_scale_save_paths`, replace the skipped contract assertion with object and JSON compatibility checks:

```python
    skipped_contract = manifest.sets["valid"].skipped_contracts[0]
    assert isinstance(skipped_contract, DatasetSkippedContract)
    assert skipped_contract.contract == "fu2509"
    assert skipped_contract.reason == "no trading days in valid range"
    assert manifest.to_dict()["sets"]["valid"]["skipped_contracts"] == [
        {"contract": "fu2509", "reason": "no trading days in valid range"}
    ]
```

Run:

```bash
conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py::test_build_dataset_manifest_uses_split_manifest_and_stage_scale_save_paths -v
```

Expected: FAIL because `DatasetSkippedContract` is not defined or `skipped_contracts[0]` is still a dict.

- [x] **Step 2: Add DatasetSkippedContract to manifests.py**

In `FineFT/datahandler/manifests.py`, add this dataclass before `DatasetSplitSet`:

```python
@dataclass
class DatasetSkippedContract:
    contract: str
    reason: str | None = None
    extra_fields: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetSkippedContract":
        extra_fields = {
            key: value
            for key, value in payload.items()
            if key not in {"contract", "reason"}
        }
        return cls(
            contract=payload["contract"],
            reason=payload.get("reason"),
            extra_fields=extra_fields,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"contract": self.contract}
        if self.reason is not None:
            payload["reason"] = self.reason
        payload.update(self.extra_fields)
        return payload
```

- [x] **Step 3: Change dataset split/output skipped_contracts fields to typed lists**

In `FineFT/datahandler/manifests.py`, update `DatasetSplitSet`:

```python
@dataclass
class DatasetSplitSet:
    range: list[str] | None = None
    contracts: list[DatasetSplitContract] = field(default_factory=list)
    skipped_contracts: list[DatasetSkippedContract] = field(default_factory=list)
```

Replace the `skipped_contracts` constructor and serializer lines with:

```python
            skipped_contracts=[
                DatasetSkippedContract.from_dict(item)
                for item in stage_info.get("skipped_contracts", [])
            ],
```

```python
            "skipped_contracts": [
                skipped_contract.to_dict()
                for skipped_contract in self.skipped_contracts
            ],
```

Then update `DatasetSetManifest` the same way:

```python
@dataclass
class DatasetSetManifest:
    range: list[str] | None = None
    contracts: list[DatasetContractManifest] = field(default_factory=list)
    skipped_contracts: list[DatasetSkippedContract] = field(default_factory=list)
    contracts_total_count: int | None = None
```

Use `DatasetSkippedContract.from_dict(item)` in `from_dict()` and `skipped_contract.to_dict()` in `to_dict()`.

- [x] **Step 4: Preserve object transfer in commodity_contract_dataset.py**

In `FineFT/datahandler/commodity_contract_dataset.py`, keep the existing assignment:

```python
        manifest.sets[set_name] = DatasetSetManifest(
            range=split_set.range,
            contracts=set_contracts,
            skipped_contracts=split_set.skipped_contracts,
        )
```

No new dict conversion should be added here. The transfer from split manifest to dataset manifest remains object-to-object.

- [x] **Step 5: Run focused verification for the amendment**

Run:

```bash
conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py
conda activate finetf && python -m py_compile FineFT/datahandler/manifests.py FineFT/datahandler/commodity_contract_dataset.py
openspec validate refactor-datahandler-manifest-objects --strict
```

Expected: all commands pass. Existing `dataset_manifest.json` assertions remain compatible because `DatasetSkippedContract.to_dict()` emits the same JSON payload.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）
