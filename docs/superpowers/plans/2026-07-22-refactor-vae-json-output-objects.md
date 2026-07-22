# Refactor VAE JSON Output Objects Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor VAE manifest, label summary, and routing summary generation to pass dataclass objects internally while preserving the existing JSON files.

**Architecture:** `FineFT/RL/DiHFT/VAE/manifests.py` owns the dataclass object model and JSON-compatible `to_dict()` serialization. `merge_vae_train.py`, `main.py`, `process.py`, and `summary.py` keep their existing workflow responsibilities, but exchange VAE manifest and summary data as objects instead of dicts. Focused tests assert both object attributes and exact JSON payload compatibility.

**Tech Stack:** Python dataclasses, NumPy, pandas CSV output, PyTorch DataLoader, pytest, OpenSpec.

**Traceability (sddflow):**
- plan-ready: `openspec/changes/refactor-vae-json-output-objects/plan-ready.md`
- tasks: `openspec/changes/refactor-vae-json-output-objects/tasks.md`
- plan: `docs/superpowers/plans/2026-07-22-refactor-vae-json-output-objects.md`

---

### Task 1: Add focused VAE JSON object tests

> **trace:** plan-ready.md → `### Task 1: Add focused VAE JSON object tests` | tasks.md → ``- [ ] 1.1 Add focused tests for VAE JSON output objects covering `LabelTrainingManifest`, `LabelSummary`, `RoutingSummary`, object attribute access, `maybe_write_routing_summary_after_analysis()` return type, and `to_dict()` equality with written JSON files.``
> **sync:** tasks.md → ``- [ ] 1.1 Add focused tests for VAE JSON output objects covering `LabelTrainingManifest`, `LabelSummary`, `RoutingSummary`, object attribute access, `maybe_write_routing_summary_after_analysis()` return type, and `to_dict()` equality with written JSON files.`` | plan-ready.md → `### Task 1: Add focused VAE JSON object tests`

**Files:**
- Modify: `FineFT/tests/rl/test_commodity_vae_cross_contract.py`

- [x] **Step 1: Import the expected object types**

Add this import after the existing VAE imports:

```python
from RL.DiHFT.VAE.manifests import (
    ContractLogpxResult,
    LabelSummary,
    LabelTrainingManifest,
    RoutingSummary,
    TrainBaselineLogpx,
)
```

- [x] **Step 2: Update the train manifest success test to assert object access**

In `test_materialize_label_training_data_merges_contract_arrays_and_writes_manifest`, replace the current `result[...]` assertions and manifest JSON assertions with:

```python
    assert isinstance(result, LabelTrainingManifest)
    assert result.merged_path == str(vae_dir / "train" / "label_0.npy")
    assert result.total_samples == 3
    assert result.feature_dim == 2
    assert [item.contract for item in result.included_contracts] == [
        "fu2505",
        "fu2509",
    ]
    assert result.included_contracts[0].sample_count == 2
    assert result.included_contracts[1].sample_count == 1
    assert result.missing_contracts == ["fu2510"]

    manifest = json.loads((vae_dir / "train" / "label_0_manifest.json").read_text())
    assert manifest == result.to_dict()
    assert manifest["dataset_name"] == "fu"
    assert manifest["label"] == "label_0"
    assert manifest["included_contracts"][0]["source_file"].endswith(
        "fu2505/label_0.npy"
    )
```

- [x] **Step 3: Update the basic label summary test to assert object return and JSON compatibility**

In `test_write_contract_logpx_outputs_writes_per_contract_and_aggregate_files`, replace the `results = [...]` block with:

```python
    results = [
        ContractLogpxResult(
            contract="fu2508",
            source_file="dataset/10min/fu/VAE_data/test/test_fu2508.npy",
            logpx=np.array([-1.0, -2.0]),
        ),
        ContractLogpxResult(
            contract="fu2509",
            source_file="dataset/10min/fu/VAE_data/test/test_fu2509.npy",
            logpx=np.array([-3.0]),
        ),
    ]
```

Then replace `assert summary == summary_file` and subsequent summary dict reads with:

```python
    assert isinstance(summary, LabelSummary)
    summary_file = json.loads((save_path / "summary.json").read_text())
    assert summary_file == summary.to_dict()
    assert summary.dataset_name == "fu"
    assert summary.label == "label_0"
    assert summary.test.contracts["fu2508"].summary.stats.samples == 2
    assert summary.test.all.stats.samples == 3
    assert "roc_auc" not in json.dumps(summary_file).lower()
```

- [x] **Step 4: Update the enhanced summary test to use object attributes**

In `test_write_contract_logpx_outputs_includes_enhanced_summary_metrics`, replace `train_baseline = {...}` with:

```python
    train_baseline = TrainBaselineLogpx(
        source_file="dataset/10min/fu/VAE_data/train/label_0.npy",
        input_samples=4,
        analyzed_samples=4,
        logpx=np.array([-10.0, -8.0, -6.0, -4.0]),
    )
```

Replace the `results = [...]` block with:

```python
    results = [
        ContractLogpxResult(
            contract="fu2508",
            source_file="dataset/10min/fu/VAE_data/test/test_fu2508.npy",
            input_samples=3,
            logpx=np.array([-9.0, -7.0]),
        ),
        ContractLogpxResult(
            contract="fu2509",
            source_file="dataset/10min/fu/VAE_data/test/test_fu2509.npy",
            input_samples=2,
            logpx=np.array([-5.0, -3.0]),
        ),
    ]
```

Then replace the return-value assertions with:

```python
    assert isinstance(summary, LabelSummary)
    summary_file = json.loads((save_path / "summary.json").read_text())
    assert summary_file == summary.to_dict()
    assert summary.train_baseline is not None
    assert summary.train_baseline.source_file.endswith("label_0.npy")
    assert summary.train_baseline.summary.integrity.input_samples == 4
    assert summary.train_baseline.summary.integrity.analyzed_samples == 4
    assert summary.train_baseline.summary.integrity.sample_mismatch is False
    assert set(summary.train_baseline.summary.stats.quantiles) == {
        "q01",
        "q05",
        "q25",
        "q50",
        "q75",
        "q95",
        "q99",
    }
    fu2508 = summary.test.contracts["fu2508"]
    assert fu2508.summary.integrity.input_samples == 3
    assert fu2508.summary.integrity.analyzed_samples == 2
    assert fu2508.summary.integrity.sample_mismatch is True
    assert fu2508.summary.stats.samples == 2
    assert set(fu2508.summary.stats.quantiles) == set(
        summary.train_baseline.summary.stats.quantiles
    )
    assert fu2508.summary.acceptance is not None
    assert set(fu2508.summary.acceptance.to_dict()) == {
        "ge_train_q01_pct",
        "ge_train_q05_pct",
        "ge_train_q50_pct",
    }
    assert summary.test.all.integrity.analyzed_samples == 4
    assert "roc_auc" not in json.dumps(summary_file).lower()
    assert "accuracy" not in json.dumps(summary_file).lower()
```

- [x] **Step 5: Update routing summary tests to assert object return and JSON compatibility**

In `test_write_routing_summary_compares_labels_by_contract`, replace summary dict assertions with:

```python
    assert isinstance(summary, RoutingSummary)
    summary_file = json.loads((result_root / "routing_summary.json").read_text())
    assert summary_file == summary.to_dict()
    assert summary.dataset_name == "fu"
    assert summary.labels == ["label_0", "label_1", "label_2"]
    assert summary.score_type == "raw_logpx"
    assert summary.contracts["fu2508"].winner.samples == 2
    assert summary.contracts["fu2508"].input_samples_by_label["label_0"] == 3
    assert summary.contracts["fu2508"].sample_mismatch is True
    assert summary.contracts["fu2508"].winner.winner_counts == {
        "label_0": 1,
        "label_1": 0,
        "label_2": 1,
    }
    assert summary.all.winner_counts == {
        "label_0": 2,
        "label_1": 0,
        "label_2": 2,
    }
```

In `test_main_writes_routing_summary_after_analysis_when_all_labels_ready`, replace summary dict assertions with:

```python
    assert isinstance(summary, RoutingSummary)
    assert summary.dataset_name == "fu"
    assert summary.all.winner_counts == {"label_0": 1, "label_1": 1}
    summary_file = json.loads((result_root / "routing_summary.json").read_text())
    assert summary_file == summary.to_dict()
```

- [x] **Step 6: Run focused tests and verify RED**

Run:

```bash
conda activate finetf && pytest FineFT/tests/rl/test_commodity_vae_cross_contract.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'RL.DiHFT.VAE.manifests'` or assertion failures showing functions still return dicts.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）


### Task 2: Add VAE manifest dataclasses

> **trace:** plan-ready.md → `### Task 2: Add VAE manifest dataclasses` | tasks.md → ``- [ ] 1.2 Add `FineFT/RL/DiHFT/VAE/manifests.py` with dataclass models for training manifest, logpx summary, sample integrity, acceptance, winner summary, contract routing summary, and routing summary serialization.``
> **sync:** tasks.md → ``- [ ] 1.2 Add `FineFT/RL/DiHFT/VAE/manifests.py` with dataclass models for training manifest, logpx summary, sample integrity, acceptance, winner summary, contract routing summary, and routing summary serialization.`` | plan-ready.md → `### Task 2: Add VAE manifest dataclasses`

**Files:**
- Create: `FineFT/RL/DiHFT/VAE/manifests.py`
- Test: `FineFT/tests/rl/test_commodity_vae_cross_contract.py`

- [x] **Step 1: Create the dataclass module**

Create `FineFT/RL/DiHFT/VAE/manifests.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LabelContractSource:
    contract: str
    source_file: str
    sample_count: int

    def to_dict(self) -> dict:
        return {
            "contract": self.contract,
            "source_file": self.source_file,
            "sample_count": int(self.sample_count),
        }


@dataclass(frozen=True)
class LabelTrainingManifest:
    dataset_name: str
    label: str
    merged_path: str
    total_samples: int
    feature_dim: int
    included_contracts: list[LabelContractSource]
    missing_contracts: list[str]

    def to_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "label": self.label,
            "merged_path": self.merged_path,
            "total_samples": int(self.total_samples),
            "feature_dim": int(self.feature_dim),
            "included_contracts": [
                item.to_dict() for item in self.included_contracts
            ],
            "missing_contracts": list(self.missing_contracts),
        }


@dataclass(frozen=True)
class ContractLogpxResult:
    contract: str
    source_file: str
    logpx: object
    input_samples: int | None = None


@dataclass(frozen=True)
class TrainBaselineLogpx:
    source_file: str
    logpx: object
    input_samples: int
    analyzed_samples: int


@dataclass(frozen=True)
class SampleIntegrity:
    input_samples: int
    analyzed_samples: int

    @property
    def sample_mismatch(self) -> bool:
        return int(self.input_samples) != int(self.analyzed_samples)

    def to_dict(self) -> dict:
        return {
            "input_samples": int(self.input_samples),
            "analyzed_samples": int(self.analyzed_samples),
            "sample_mismatch": self.sample_mismatch,
        }


@dataclass(frozen=True)
class LogpxStats:
    samples: int
    logpx_mean: float
    logpx_std: float
    logpx_min: float
    logpx_max: float
    quantiles: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "samples": int(self.samples),
            "logpx_mean": float(self.logpx_mean),
            "logpx_std": float(self.logpx_std),
            "logpx_min": float(self.logpx_min),
            "logpx_max": float(self.logpx_max),
            "quantiles": {
                key: float(value) for key, value in self.quantiles.items()
            },
        }


@dataclass(frozen=True)
class AcceptanceStats:
    ge_train_q01_pct: float
    ge_train_q05_pct: float
    ge_train_q50_pct: float

    def to_dict(self) -> dict:
        return {
            "ge_train_q01_pct": float(self.ge_train_q01_pct),
            "ge_train_q05_pct": float(self.ge_train_q05_pct),
            "ge_train_q50_pct": float(self.ge_train_q50_pct),
        }


@dataclass(frozen=True)
class LogpxSummary:
    integrity: SampleIntegrity
    stats: LogpxStats
    acceptance: AcceptanceStats | None = None

    def to_dict(self) -> dict:
        payload = {
            **self.integrity.to_dict(),
            **self.stats.to_dict(),
        }
        if self.acceptance is not None:
            payload["acceptance"] = self.acceptance.to_dict()
        return payload


@dataclass(frozen=True)
class ContractLogpxSummary:
    source_file: str
    summary: LogpxSummary

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            **self.summary.to_dict(),
        }


@dataclass(frozen=True)
class LabelTestSummary:
    contracts: dict[str, ContractLogpxSummary]
    all: LogpxSummary

    def to_dict(self) -> dict:
        return {
            "contracts": {
                contract: summary.to_dict()
                for contract, summary in self.contracts.items()
            },
            "all": self.all.to_dict(),
        }


@dataclass(frozen=True)
class LabelSummary:
    dataset_name: str
    label: str
    test: LabelTestSummary
    train_baseline: ContractLogpxSummary | None = None

    def to_dict(self) -> dict:
        payload = {
            "dataset_name": self.dataset_name,
            "label": self.label,
            "test": self.test.to_dict(),
        }
        if self.train_baseline is not None:
            payload["train_baseline"] = self.train_baseline.to_dict()
        return payload


@dataclass(frozen=True)
class WinnerSummary:
    samples: int
    winner_counts: dict[str, int]
    winner_pct: dict[str, float]
    top1_top2_margin_mean: float
    top1_top2_margin_q25: float
    low_margin_pct: float

    def to_dict(self) -> dict:
        return {
            "samples": int(self.samples),
            "winner_counts": {
                label: int(count) for label, count in self.winner_counts.items()
            },
            "winner_pct": {
                label: float(value) for label, value in self.winner_pct.items()
            },
            "top1_top2_margin_mean": float(self.top1_top2_margin_mean),
            "top1_top2_margin_q25": float(self.top1_top2_margin_q25),
            "low_margin_pct": float(self.low_margin_pct),
        }


@dataclass(frozen=True)
class ContractRoutingSummary:
    winner: WinnerSummary
    input_samples_by_label: dict[str, int]
    sample_mismatch: bool

    def to_dict(self) -> dict:
        return {
            **self.winner.to_dict(),
            "input_samples_by_label": {
                label: int(samples)
                for label, samples in self.input_samples_by_label.items()
            },
            "sample_mismatch": bool(self.sample_mismatch),
        }


@dataclass(frozen=True)
class RoutingSummary:
    dataset_name: str
    labels: list[str]
    score_type: str
    low_margin_threshold: float
    contracts: dict[str, ContractRoutingSummary]
    all: WinnerSummary

    def to_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "labels": list(self.labels),
            "score_type": self.score_type,
            "low_margin_threshold": float(self.low_margin_threshold),
            "contracts": {
                contract: summary.to_dict()
                for contract, summary in self.contracts.items()
            },
            "all": self.all.to_dict(),
        }
```

- [x] **Step 2: Compile the new module**

Run:

```bash
conda activate finetf && python -m py_compile FineFT/RL/DiHFT/VAE/manifests.py
```

Expected: no output and exit code 0.

- [x] **Step 3: Re-run focused tests and verify remaining failures are implementation failures**

Run:

```bash
conda activate finetf && pytest FineFT/tests/rl/test_commodity_vae_cross_contract.py -q
```

Expected: FAIL because `materialize_label_training_data()`, `write_contract_logpx_outputs()`, and `write_routing_summary()` still return dicts.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）


### Task 3: Refactor VAE train manifest flow to objects

> **trace:** plan-ready.md → `### Task 3: Refactor VAE train manifest flow to objects` | tasks.md → ``- [ ] 1.3 Refactor `FineFT/RL/DiHFT/VAE/merge_vae_train.py` and `main.py` so materialized train data is represented and passed as objects, not dicts, while preserving `label_k_manifest.json`.``
> **sync:** tasks.md → ``- [ ] 1.3 Refactor `FineFT/RL/DiHFT/VAE/merge_vae_train.py` and `main.py` so materialized train data is represented and passed as objects, not dicts, while preserving `label_k_manifest.json`.`` | plan-ready.md → `### Task 3: Refactor VAE train manifest flow to objects`

**Files:**
- Modify: `FineFT/RL/DiHFT/VAE/merge_vae_train.py`
- Modify: `FineFT/RL/DiHFT/VAE/main.py`
- Test: `FineFT/tests/rl/test_commodity_vae_cross_contract.py`

- [x] **Step 1: Import manifest dataclasses in merge_vae_train.py**

Add imports after `import numpy as np`:

```python
try:
    from .manifests import LabelContractSource, LabelTrainingManifest
except ImportError:
    from manifests import LabelContractSource, LabelTrainingManifest
```

- [x] **Step 2: Build included contract sources as objects**

Inside `materialize_label_training_data()`, replace the `included_contracts.append({...})` block with:

```python
        included_contracts.append(
            LabelContractSource(
                contract=source["contract"],
                source_file=source["source_file"],
                sample_count=int(array.shape[0]),
            )
        )
```

- [x] **Step 3: Return LabelTrainingManifest and write JSON through to_dict()**

Replace the final `manifest = {...}` block in `materialize_label_training_data()` with:

```python
    manifest = LabelTrainingManifest(
        dataset_name=dataset_name,
        label=label_name,
        merged_path=str(merged_path),
        total_samples=int(merged.shape[0]),
        feature_dim=int(merged.shape[1]),
        included_contracts=included_contracts,
        missing_contracts=missing_contracts,
    )
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest.to_dict(), file, ensure_ascii=False, indent=2)
    return manifest
```

- [x] **Step 4: Import LabelTrainingManifest and TrainBaselineLogpx in main.py**

Add this import near the other VAE imports:

```python
from RL.DiHFT.VAE.manifests import LabelTrainingManifest, TrainBaselineLogpx
```

- [x] **Step 5: Replace analyze-only temporary dict with a LabelTrainingManifest object**

In `Piplineruner.__init__`, replace the analyze-only `train_manifest = {...}` block with:

```python
            train_manifest = LabelTrainingManifest(
                dataset_name=self.args.dataset_name,
                label=label_name,
                merged_path=str(train_path),
                total_samples=int(train_data.shape[0]),
                feature_dim=int(train_data.shape[1]),
                included_contracts=[],
                missing_contracts=[],
            )
```

Then replace subsequent dict-key access:

```python
        train_data_path = train_manifest.merged_path
```

and:

```python
            expected_feature_dim=train_manifest.feature_dim,
```

- [x] **Step 6: Replace train manifest dict-key access in analyze_contracts()**

In `analyze_contracts()`, replace `self.train_manifest[...]` reads with object attributes:

```python
        train_dataset = One_Dim_Dataset(self.train_manifest.merged_path)
```

and pass the train baseline as an object:

```python
            train_baseline=TrainBaselineLogpx(
                source_file=self.train_manifest.merged_path,
                input_samples=self.train_manifest.total_samples,
                analyzed_samples=int(np.asarray(train_logpx).reshape(-1).size),
                logpx=np.asarray(train_logpx, dtype=float),
            ),
```

- [x] **Step 7: Run the train manifest focused test**

Run:

```bash
conda activate finetf && pytest FineFT/tests/rl/test_commodity_vae_cross_contract.py::test_materialize_label_training_data_merges_contract_arrays_and_writes_manifest -q
```

Expected: PASS.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）


### Task 4: Refactor VAE summary and routing flow to objects

> **trace:** plan-ready.md → `### Task 4: Refactor VAE summary and routing flow to objects` | tasks.md → ``- [ ] 1.4 Refactor `FineFT/RL/DiHFT/VAE/process.py` and `summary.py` so per-label summary inputs, summary outputs, and routing summary data are represented and returned as objects, not dicts, while preserving `summary.json` and `routing_summary.json`.``
> **sync:** tasks.md → ``- [ ] 1.4 Refactor `FineFT/RL/DiHFT/VAE/process.py` and `summary.py` so per-label summary inputs, summary outputs, and routing summary data are represented and returned as objects, not dicts, while preserving `summary.json` and `routing_summary.json`.`` | plan-ready.md → `### Task 4: Refactor VAE summary and routing flow to objects`

**Files:**
- Modify: `FineFT/RL/DiHFT/VAE/process.py`
- Modify: `FineFT/RL/DiHFT/VAE/summary.py`
- Test: `FineFT/tests/rl/test_commodity_vae_cross_contract.py`

- [x] **Step 1: Import summary dataclasses in process.py**

Update the summary import in `FineFT/RL/DiHFT/VAE/process.py`:

```python
from RL.DiHFT.VAE.manifests import ContractLogpxResult
from RL.DiHFT.VAE.summary import write_contract_logpx_outputs
```

- [x] **Step 2: Build ContractLogpxResult objects in analyze_contract_tests()**

Replace the `contract_results.append({...})` block with:

```python
        contract_results.append(
            ContractLogpxResult(
                contract=item["contract"],
                source_file=item["source_file"],
                input_samples=len(item["loader"].dataset),
                logpx=np.asarray(ood_logpx, dtype=float),
            )
        )
```

- [x] **Step 3: Import dataclasses in summary.py**

Add these imports after the existing merge_vae_train import block:

```python
try:
    from .manifests import (
        AcceptanceStats,
        ContractLogpxResult,
        ContractLogpxSummary,
        ContractRoutingSummary,
        LabelSummary,
        LabelTestSummary,
        LogpxStats,
        LogpxSummary,
        RoutingSummary,
        SampleIntegrity,
        TrainBaselineLogpx,
        WinnerSummary,
    )
except ImportError:
    from manifests import (
        AcceptanceStats,
        ContractLogpxResult,
        ContractLogpxSummary,
        ContractRoutingSummary,
        LabelSummary,
        LabelTestSummary,
        LogpxStats,
        LogpxSummary,
        RoutingSummary,
        SampleIntegrity,
        TrainBaselineLogpx,
        WinnerSummary,
    )
```

- [x] **Step 4: Convert summary helper return values to objects**

Replace `_logpx_stats`, `_acceptance_stats`, and `_sample_integrity` with:

```python
def _logpx_stats(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("logpx array has no samples")
    return LogpxStats(
        samples=int(values.size),
        logpx_mean=float(np.mean(values)),
        logpx_std=float(np.std(values)),
        logpx_min=float(np.min(values)),
        logpx_max=float(np.max(values)),
        quantiles={
            name: float(np.quantile(values, quantile))
            for name, quantile in SUMMARY_QUANTILES
        },
    )


def _acceptance_stats(values, train_quantiles):
    values = np.asarray(values, dtype=float).reshape(-1)
    return AcceptanceStats(
        ge_train_q01_pct=float(np.mean(values >= train_quantiles["q01"]) * 100.0),
        ge_train_q05_pct=float(np.mean(values >= train_quantiles["q05"]) * 100.0),
        ge_train_q50_pct=float(np.mean(values >= train_quantiles["q50"]) * 100.0),
    )


def _sample_integrity(input_samples, analyzed_samples):
    return SampleIntegrity(
        input_samples=int(input_samples),
        analyzed_samples=int(analyzed_samples),
    )
```

- [x] **Step 5: Add a small helper for complete logpx summaries**

Add below `_sample_integrity`:

```python
def _logpx_summary(input_samples, logpx, acceptance=None):
    flat_logpx = np.asarray(logpx, dtype=float).reshape(-1)
    return LogpxSummary(
        integrity=_sample_integrity(input_samples, flat_logpx.size),
        stats=_logpx_stats(flat_logpx),
        acceptance=acceptance,
    )
```

- [x] **Step 6: Return LabelSummary from write_contract_logpx_outputs()**

Inside `write_contract_logpx_outputs()`, keep the file writes and replace summary construction with object construction. The core object-building logic should be:

```python
    train_summary = None
    if train_baseline is not None:
        train_logpx = np.asarray(train_baseline.logpx, dtype=float).reshape(-1)
        train_summary = ContractLogpxSummary(
            source_file=train_baseline.source_file,
            summary=LogpxSummary(
                integrity=_sample_integrity(
                    train_baseline.input_samples,
                    train_baseline.analyzed_samples,
                ),
                stats=_logpx_stats(train_logpx),
            ),
        )
```

For each contract result, use attributes:

```python
    for result in sorted(contract_results, key=lambda item: item.contract):
        contract = result.contract
        source_file = result.source_file
        logpx = np.asarray(result.logpx, dtype=float)
        flat_logpx = logpx.reshape(-1)
        input_samples = int(
            result.input_samples
            if result.input_samples is not None
            else flat_logpx.size
        )
```

Build the contract summary object:

```python
        acceptance = None
        if train_summary is not None:
            acceptance = _acceptance_stats(
                flat_logpx, train_summary.summary.stats.quantiles
            )
        contract_summary[contract] = ContractLogpxSummary(
            source_file=source_file,
            summary=_logpx_summary(input_samples, flat_logpx, acceptance),
        )
```

Replace the existing `total_input_samples = sum(...)` block with:

```python
    total_input_samples = sum(
        int(
            item.input_samples
            if item.input_samples is not None
            else np.asarray(item.logpx, dtype=float).reshape(-1).size
        )
        for item in contract_results
    )
```

Build and write the final object:

```python
    all_summary = _logpx_summary(total_input_samples, combined_flat)
    if train_summary is not None:
        all_summary = _logpx_summary(
            total_input_samples,
            combined_flat,
            _acceptance_stats(combined_flat, train_summary.summary.stats.quantiles),
        )
    summary = LabelSummary(
        dataset_name=dataset_name,
        label=label,
        test=LabelTestSummary(contracts=contract_summary, all=all_summary),
        train_baseline=train_summary,
    )
    with open(os.path.join(save_path, "summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary.to_dict(), file, ensure_ascii=False, indent=2)
    return summary
```

- [x] **Step 7: Convert winner summary helpers to objects**

Replace `_winner_summary()` returns with `WinnerSummary` objects:

```python
def _winner_summary(scores, labels, low_margin_threshold):
    scores = np.asarray(scores, dtype=float)
    samples = int(scores.shape[0])
    if samples == 0:
        return WinnerSummary(
            samples=0,
            winner_counts={label: 0 for label in labels},
            winner_pct={label: 0.0 for label in labels},
            top1_top2_margin_mean=0.0,
            top1_top2_margin_q25=0.0,
            low_margin_pct=0.0,
        )
    winners = np.argmax(scores, axis=1)
    sorted_scores = np.sort(scores, axis=1)
    margins = sorted_scores[:, -1] - sorted_scores[:, -2]
    counts = {
        label: int(np.sum(winners == index)) for index, label in enumerate(labels)
    }
    return WinnerSummary(
        samples=samples,
        winner_counts=counts,
        winner_pct={
            label: float(count / samples * 100.0) for label, count in counts.items()
        },
        top1_top2_margin_mean=float(np.mean(margins)),
        top1_top2_margin_q25=float(np.quantile(margins, 0.25)),
        low_margin_pct=float(np.mean(margins <= low_margin_threshold) * 100.0),
    )
```

- [x] **Step 8: Return RoutingSummary from write_routing_summary()**

In the per-contract routing loop, replace dict mutation with:

```python
        contract_summaries[contract] = ContractRoutingSummary(
            winner=_winner_summary(scores, labels, low_margin_threshold),
            input_samples_by_label=input_samples_by_label,
            sample_mismatch=len(set(input_samples_by_label.values())) != 1,
        )
```

Replace the final summary dict with:

```python
    summary = RoutingSummary(
        dataset_name=dataset_name,
        labels=list(labels),
        score_type="raw_logpx",
        low_margin_threshold=float(low_margin_threshold),
        contracts=contract_summaries,
        all=_winner_summary(combined, labels, low_margin_threshold),
    )
    with open(result_root / "routing_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary.to_dict(), file, ensure_ascii=False, indent=2)
    return summary
```

- [x] **Step 9: Run focused summary and routing tests**

Run:

```bash
conda activate finetf && pytest \
  FineFT/tests/rl/test_commodity_vae_cross_contract.py::test_write_contract_logpx_outputs_writes_per_contract_and_aggregate_files \
  FineFT/tests/rl/test_commodity_vae_cross_contract.py::test_write_contract_logpx_outputs_includes_enhanced_summary_metrics \
  FineFT/tests/rl/test_commodity_vae_cross_contract.py::test_write_routing_summary_compares_labels_by_contract \
  FineFT/tests/rl/test_commodity_vae_cross_contract.py::test_main_writes_routing_summary_after_analysis_when_all_labels_ready \
  -q
```

Expected: PASS.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）


### Task 5: Run focused VAE tests

> **trace:** plan-ready.md → `### Task 5: Run focused VAE tests` | tasks.md → ``- [ ] 2.1 Run `conda activate finetf && pytest FineFT/tests/rl/test_commodity_vae_cross_contract.py`.``
> **sync:** tasks.md → ``- [ ] 2.1 Run `conda activate finetf && pytest FineFT/tests/rl/test_commodity_vae_cross_contract.py`.`` | plan-ready.md → `### Task 5: Run focused VAE tests`

**Files:**
- Test: `FineFT/tests/rl/test_commodity_vae_cross_contract.py`

- [x] **Step 1: Run the focused VAE test module**

Run:

```bash
conda activate finetf && pytest FineFT/tests/rl/test_commodity_vae_cross_contract.py
```

Expected: all tests in `FineFT/tests/rl/test_commodity_vae_cross_contract.py` pass.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）


### Task 6: Run VAE py_compile

> **trace:** plan-ready.md → `### Task 6: Run VAE py_compile` | tasks.md → ``- [ ] 2.2 Run `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/VAE/manifests.py FineFT/RL/DiHFT/VAE/merge_vae_train.py FineFT/RL/DiHFT/VAE/main.py FineFT/RL/DiHFT/VAE/summary.py`.``
> **sync:** tasks.md → ``- [ ] 2.2 Run `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/VAE/manifests.py FineFT/RL/DiHFT/VAE/merge_vae_train.py FineFT/RL/DiHFT/VAE/main.py FineFT/RL/DiHFT/VAE/summary.py`.`` | plan-ready.md → `### Task 6: Run VAE py_compile`

**Files:**
- Test: `FineFT/RL/DiHFT/VAE/manifests.py`
- Test: `FineFT/RL/DiHFT/VAE/merge_vae_train.py`
- Test: `FineFT/RL/DiHFT/VAE/main.py`
- Test: `FineFT/RL/DiHFT/VAE/summary.py`

- [x] **Step 1: Compile changed VAE Python files**

Run:

```bash
conda activate finetf && python -m py_compile \
  FineFT/RL/DiHFT/VAE/manifests.py \
  FineFT/RL/DiHFT/VAE/merge_vae_train.py \
  FineFT/RL/DiHFT/VAE/main.py \
  FineFT/RL/DiHFT/VAE/summary.py
```

Expected: no output and exit code 0.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）


### Task 7: Run OpenSpec strict validation

> **trace:** plan-ready.md → `### Task 7: Run OpenSpec strict validation` | tasks.md → ``- [ ] 2.3 Run `openspec validate refactor-vae-json-output-objects --strict`.``
> **sync:** tasks.md → ``- [ ] 2.3 Run `openspec validate refactor-vae-json-output-objects --strict`.`` | plan-ready.md → `### Task 7: Run OpenSpec strict validation`

**Files:**
- Test: `openspec/changes/refactor-vae-json-output-objects/`

- [x] **Step 1: Validate the OpenSpec change**

Run:

```bash
openspec validate refactor-vae-json-output-objects --strict
```

Expected:

```text
Change 'refactor-vae-json-output-objects' is valid
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 8: Add focused source/loader object tests

> **trace:** plan-ready.md → `### Task 8: Add focused source/loader object tests` | tasks.md → ``- [ ] 1.5 Add focused tests for VAE source discovery and loader preparation objects covering `LabelArraySource`, `TestContractSource`, `ContractDatasetLoader`, object attribute access, and preservation of existing discovery/validation behavior.``
> **sync:** tasks.md → ``- [ ] 1.5 Add focused tests for VAE source discovery and loader preparation objects covering `LabelArraySource`, `TestContractSource`, `ContractDatasetLoader`, object attribute access, and preservation of existing discovery/validation behavior.`` | plan-ready.md → `### Task 8: Add focused source/loader object tests`

**Files:**
- Modify: `FineFT/tests/rl/test_commodity_vae_cross_contract.py`

- [x] **Step 1: Add source and loader object expectations to tests**

Add imports and assertions for `LabelArraySource`, `TestContractSource`, and `ContractDatasetLoader`, then replace test source/loader dict access with object attribute access.

- [x] **Step 2: Run the focused test module and verify RED**

Run:

```bash
eval "$(conda shell.bash hook)" && conda activate finetf && pytest FineFT/tests/rl/test_commodity_vae_cross_contract.py -q
```

Expected: FAIL until the source/loader discovery flow is refactored.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 9: Refactor VAE source and loader flow to objects

> **trace:** plan-ready.md → `### Task 9: Refactor VAE source and loader flow to objects` | tasks.md → ``- [ ] 1.6 Refactor `FineFT/RL/DiHFT/VAE/merge_vae_train.py`, `main.py`, and `process.py` so label source discovery, test source discovery, and contract loader preparation return and pass dataclass objects instead of `list[dict]`.``
> **sync:** tasks.md → ``- [ ] 1.6 Refactor `FineFT/RL/DiHFT/VAE/merge_vae_train.py`, `main.py`, and `process.py` so label source discovery, test source discovery, and contract loader preparation return and pass dataclass objects instead of `list[dict]`.`` | plan-ready.md → `### Task 9: Refactor VAE source and loader flow to objects`

**Files:**
- Modify: `FineFT/RL/DiHFT/VAE/merge_vae_train.py`
- Modify: `FineFT/RL/DiHFT/VAE/main.py`
- Modify: `FineFT/RL/DiHFT/VAE/process.py`
- Modify: `FineFT/RL/DiHFT/VAE/manifests.py`
- Test: `FineFT/tests/rl/test_commodity_vae_cross_contract.py`

- [x] **Step 1: Add source and loader dataclasses**

Extend `manifests.py` with dataclasses for `LabelArraySource`, `TestContractSource`, and `ContractDatasetLoader`, each with `to_dict()` only if needed at a JSON boundary.

- [x] **Step 2: Refactor label source discovery and train manifest construction**

Change `discover_label_sources()` to return `list[LabelArraySource]` and update `materialize_label_training_data()` to use attribute access only.

- [x] **Step 3: Refactor test source discovery and contract loader preparation**

Change `discover_test_sources()` to return `list[TestContractSource]` and `prepare_contract_dataset_loader_list()` to return `list[ContractDatasetLoader]`, then update `analyze_contract_tests()` to read object attributes.

- [x] **Step 4: Run focused source/loader tests and verify GREEN**

Run:

```bash
eval "$(conda shell.bash hook)" && conda activate finetf && pytest FineFT/tests/rl/test_commodity_vae_cross_contract.py
```

Expected: PASS.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 10: Re-run verification for the expanded scope

> **trace:** plan-ready.md → `### Task 10: Re-run verification for the expanded scope` | tasks.md → ``- [ ] 2.4 Re-run `conda activate finetf && pytest FineFT/tests/rl/test_commodity_vae_cross_contract.py`.`` | ``- [ ] 2.5 Re-run `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/VAE/manifests.py FineFT/RL/DiHFT/VAE/merge_vae_train.py FineFT/RL/DiHFT/VAE/main.py FineFT/RL/DiHFT/VAE/process.py FineFT/RL/DiHFT/VAE/summary.py`.`` | ``- [ ] 2.6 Re-run `openspec validate refactor-vae-json-output-objects --strict`.``
> **sync:** tasks.md → ``- [ ] 2.4 Re-run `conda activate finetf && pytest FineFT/tests/rl/test_commodity_vae_cross_contract.py`.`` | ``- [ ] 2.5 Re-run `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/VAE/manifests.py FineFT/RL/DiHFT/VAE/merge_vae_train.py FineFT/RL/DiHFT/VAE/main.py FineFT/RL/DiHFT/VAE/process.py FineFT/RL/DiHFT/VAE/summary.py`.`` | ``- [ ] 2.6 Re-run `openspec validate refactor-vae-json-output-objects --strict`.`` | plan-ready.md → `### Task 10: Re-run verification for the expanded scope`

**Files:**
- Test: `FineFT/tests/rl/test_commodity_vae_cross_contract.py`
- Test: `FineFT/RL/DiHFT/VAE/manifests.py`
- Test: `FineFT/RL/DiHFT/VAE/merge_vae_train.py`
- Test: `FineFT/RL/DiHFT/VAE/main.py`
- Test: `FineFT/RL/DiHFT/VAE/process.py`
- Test: `FineFT/RL/DiHFT/VAE/summary.py`
- Test: `openspec/changes/refactor-vae-json-output-objects/`

- [x] **Step 1: Re-run the focused VAE test module**

Run:

```bash
eval "$(conda shell.bash hook)" && conda activate finetf && pytest FineFT/tests/rl/test_commodity_vae_cross_contract.py
```

Expected: PASS after the source/loader refactor.

- [x] **Step 2: Re-run VAE py_compile**

Run:

```bash
eval "$(conda shell.bash hook)" && conda activate finetf && python -m py_compile FineFT/RL/DiHFT/VAE/manifests.py FineFT/RL/DiHFT/VAE/merge_vae_train.py FineFT/RL/DiHFT/VAE/main.py FineFT/RL/DiHFT/VAE/process.py FineFT/RL/DiHFT/VAE/summary.py
```

Expected: no output and exit code 0.

- [x] **Step 3: Re-run OpenSpec strict validation**

Run:

```bash
openspec validate refactor-vae-json-output-objects --strict
```

Expected: `Change 'refactor-vae-json-output-objects' is valid`

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）
