# Refactor Commodity Main Contract Objects Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace nested dict handoff in commodity main-contract summary generation with dataclass-based source discovery and build state objects while keeping `main_contract_summary.json` unchanged.

**Architecture:** `main_contract.py` keeps the public summary models and owns the JSON boundary. A small set of internal dataclasses represents discovered source files and mutable build state, so monthly selection and day-window clipping are expressed through object methods instead of nested dict mutation. Tests stay in `data_preprocess/tests/test_commodity_main_contract.py` and verify both object access and JSON compatibility.

**Tech Stack:** Python, dataclasses, Polars, pytest, argparse CLIs, OpenSpec.

**Traceability (sddflow):**
- plan-ready: `openspec/changes/refactor-commodity-main-contract-objects/plan-ready.md`
- tasks: `openspec/changes/refactor-commodity-main-contract-objects/tasks.md`
- plan: `docs/superpowers/plans/2026-07-22-refactor-commodity-main-contract-objects.md`

---

### Task 1: Add focused commodity main-contract tests

> **trace:** plan-ready.md -> `### Task 1: Add focused commodity main-contract tests` | tasks.md -> `- [ ] 1.1 Add focused tests for commodity main contract object return types, object attribute access, and JSON payload compatibility.`
> **sync:** tasks.md -> `- [ ] 1.1 Add focused tests for commodity main contract object return types, object attribute access, and JSON payload compatibility.` | plan-ready.md -> `### Task 1: Add focused commodity main-contract tests`

**Files:**
- Modify: `data_preprocess/tests/test_commodity_main_contract.py`

- [x] **Step 1: Add the failing object-assertion tests**

Replace the dict-based assertions in the existing test file with object assertions and add a JSON-equality check for the writer:

```python
from operator_futures.commodity.main_contract import (
    ContractSourceFile,
    TradingDayContractSources,
    MainContractSummary,
    build_main_contract_summary_for_date_range,
    load_contract_files_by_trading_day_for_years,
    write_main_contract_summary_for_date_range,
)


def test_load_contract_files_by_trading_day_for_years_returns_paths(tmp_path):
    raw_root = tmp_path / "data" / "原始下载"
    first = _write_contract_file(
        raw_root, "燃料油", "2026", "01", "20260105", "fu2602", "20260104", [0, 30]
    )
    second = _write_contract_file(
        raw_root, "燃料油", "2026", "01", "20260106", "fu2603", "20260105", [0, 20]
    )

    days = load_contract_files_by_trading_day_for_years(raw_root, "燃料油", ["2026"])

    assert isinstance(days[0], TradingDayContractSources)
    assert isinstance(days[0].contract_files[0], ContractSourceFile)
    assert days[0].trading_day == "20260105"
    assert days[0].contract_files[0].contract == "fu2602"
    assert days[0].contract_files[0].source_file == first
    assert days[1].contract_files[0].source_file == second
```

Add a summary return-type assertion:

```python
summary = build_main_contract_summary_for_date_range(
    raw_root=raw_root,
    commodity_name="燃料油",
    start_date="2026-01-01",
    end_date="2026-03-01",
    symbol="fu",
)

assert isinstance(summary, MainContractSummary)
assert summary.contracts[0].contract == "fu2601"
```

- [x] **Step 2: Run the focused test subset to verify it fails**

Run:

```bash
conda activate finetf && pytest \
  data_preprocess/tests/test_commodity_main_contract.py::test_load_contract_files_by_trading_day_for_years_returns_paths \
  data_preprocess/tests/test_commodity_main_contract.py::test_build_main_contract_summary_selects_monthly_top_two_contracts \
  data_preprocess/tests/test_commodity_main_contract.py::test_write_main_contract_summary_for_date_range_writes_json -q
```

Expected: FAIL because the loader still returns nested dicts and the builder still returns a dict.

- [x] **Step 3: Commit the red test baseline**（skipped commit because the worktree already contains unrelated staged changes; changes are left in the working tree）

```bash
git add data_preprocess/tests/test_commodity_main_contract.py
git commit -m "test: pin commodity main-contract object interfaces"
```

- [x] **Task complete**（object assertion tests are in place; commit skipped to avoid mixing unrelated staged changes）

### Task 2: Add main-contract build state dataclasses

> **trace:** plan-ready.md -> `### Task 2: Add main-contract build state dataclasses` | tasks.md -> `- [ ] 1.2 Add internal dataclass models in main_contract.py for source discovery and build state so summary assembly no longer depends on nested dict handoff.`
> **sync:** tasks.md -> `- [ ] 1.2 Add internal dataclass models in main_contract.py for source discovery and build state so summary assembly no longer depends on nested dict handoff.` | plan-ready.md -> `### Task 2: Add main-contract build state dataclasses`

**Files:**
- Modify: `data_preprocess/operator_futures/commodity/main_contract.py`

- [x] **Step 1: Add the internal dataclass section**

Introduce a small source-index model and a mutable build-state model near the existing summary dataclasses:

```python
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ContractSourceFile:
    contract: str
    source_file: Path


@dataclass(frozen=True)
class TradingDayContractSources:
    trading_day: str
    contract_files: tuple[ContractSourceFile, ...]


@dataclass
class MainContractBuildState:
    monthly_volumes: dict[str, dict[str, float]] = field(default_factory=dict)
    monthly_high_volume_days: dict[str, dict[str, int]] = field(default_factory=dict)
    contract_days: dict[str, list[MainContractSummaryTradingDay]] = field(default_factory=dict)
    selected_months_by_contract: dict[str, set[str]] = field(default_factory=dict)
```

Add methods for recording source discovery results and monthly selection, for example:

```python
def record_contract_day(
    self,
    contract: str,
    trading_day: str,
    source_file: Path,
    daily_volume: float,
) -> None:
    days = self.contract_days.setdefault(contract, [])
    days.append(
        MainContractSummaryTradingDay(
            trading_day=trading_day,
            date=_format_trading_day_file_date(trading_day),
            source_file=str(source_file),
            daily_volume=daily_volume,
        )
    )

def record_monthly_volume(self, month: str, contract: str, daily_volume: float) -> None:
    monthly = self.monthly_volumes.setdefault(month, {})
    monthly[contract] = monthly.get(contract, 0.0) + daily_volume
```

- [x] **Step 2: Run the failing test again**

Run:

```bash
conda activate finetf && pytest \
  data_preprocess/tests/test_commodity_main_contract.py::test_load_contract_files_by_trading_day_for_years_returns_paths -q
```

Expected: FAIL until the loader returns `TradingDayContractSources` and `ContractSourceFile` objects.

- [x] **Step 3: Commit the dataclass model baseline**（skipped commit because the worktree already contains unrelated staged changes; changes are left in the working tree）

```bash
git add data_preprocess/operator_futures/commodity/main_contract.py
git commit -m "refactor: add commodity main-contract build state objects"
```

- [x] **Task complete**（build-state dataclasses are implemented; commit skipped to avoid mixing unrelated staged changes）

### Task 3: Refactor summary build and write boundaries

> **trace:** plan-ready.md -> `### Task 3: Refactor summary build and write boundaries` | tasks.md -> `- [ ] 1.3 Refactor build_main_contract_summary_for_date_range() and write_main_contract_summary_for_date_range() to return MainContractSummary and serialize only at the JSON boundary.`
> **sync:** tasks.md -> `- [ ] 1.3 Refactor build_main_contract_summary_for_date_range() and write_main_contract_summary_for_date_range() to return MainContractSummary and serialize only at the JSON boundary.` | plan-ready.md -> `### Task 3: Refactor summary build and write boundaries`

**Files:**
- Modify: `data_preprocess/operator_futures/commodity/main_contract.py`
- Modify: `data_preprocess/tests/test_commodity_main_contract.py`

- [x] **Step 1: Switch the loader and builder to object handoff**

Refactor the loader to return ordered `TradingDayContractSources` objects and make the builder consume `MainContractBuildState`:

```python
def load_contract_files_by_trading_day_for_years(
    raw_root: Path, commodity_name: str, years: Sequence[str]
) -> tuple[TradingDayContractSources, ...]:
    days: list[TradingDayContractSources] = []
    days.append(
        TradingDayContractSources(
            trading_day=trading_day,
            contract_files=tuple(
                ContractSourceFile(contract=contract, source_file=source_file)
                for contract, source_file in sorted(trading_day_contract_map.items())
            ),
        )
    )
    return tuple(days)


def build_main_contract_summary_for_date_range(
    raw_root: Path,
    commodity_name: str,
    start_date: str,
    end_date: str,
    symbol: str,
) -> MainContractSummary:
    return build_main_contract_summary_model_for_date_range(
        raw_root=raw_root,
        commodity_name=commodity_name,
        start_date=start_date,
        end_date=end_date,
        symbol=symbol,
    )
```

The writer should serialize only at the boundary:

```python
summary = build_main_contract_summary_for_date_range(
    raw_root=raw_root,
    commodity_name=commodity_name,
    start_date=start_date,
    end_date=end_date,
    symbol=symbol,
)
path.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
```

- [x] **Step 2: Update the existing tests to assert object access**

Change the body of `test_build_main_contract_summary_selects_monthly_top_two_contracts` and `test_write_main_contract_summary_for_date_range_writes_json` to use attributes:

```python
contracts = {item.contract: item for item in summary.contracts}
assert contracts["fu2601"].selected_months == ["2026-01"]
assert contracts["fu2601"].trading_days[0].source_file == str(jan_fu2601)

written = json.loads(path.read_text(encoding="utf-8"))
assert written == summary.to_dict()
```

- [x] **Step 3: Run the summary tests until they pass**

Run:

```bash
conda activate finetf && pytest \
  data_preprocess/tests/test_commodity_main_contract.py::test_build_main_contract_summary_selects_monthly_top_two_contracts \
  data_preprocess/tests/test_commodity_main_contract.py::test_build_main_contract_summary_selects_contract_with_ten_high_volume_days \
  data_preprocess/tests/test_commodity_main_contract.py::test_build_main_contract_summary_rejects_duplicate_contract_day \
  data_preprocess/tests/test_commodity_main_contract.py::test_write_main_contract_summary_for_date_range_writes_json -q
```

Expected: PASS with `MainContractSummary` return values and matching JSON payloads.

- [x] **Step 4: Commit the refactor**（skipped commit because the worktree already contains unrelated staged changes; changes are left in the working tree）

```bash
git add data_preprocess/operator_futures/commodity/main_contract.py data_preprocess/tests/test_commodity_main_contract.py
git commit -m "refactor: objectify commodity main-contract summary flow"
```

- [x] **Task complete**（builder returns objects and writer serializes with to_dict(); commit skipped to avoid mixing unrelated staged changes）

### Task 4: Run focused verification

> **trace:** plan-ready.md -> `### Task 4: Run focused verification` | tasks.md -> `- [ ] 1.4 Update the existing commodity main-contract tests to assert object access on source discovery, summary assembly, and written JSON equality.`
> **sync:** tasks.md -> `- [ ] 1.4 Update the existing commodity main-contract tests to assert object access on source discovery, summary assembly, and written JSON equality.` | plan-ready.md -> `### Task 4: Run focused verification`

**Files:**
- Modify: `data_preprocess/tests/test_commodity_main_contract.py`
- Modify: `data_preprocess/operator_futures/commodity/main_contract.py`

- [x] **Step 1: Run the full focused test module**

Run:

```bash
conda activate finetf && pytest data_preprocess/tests/test_commodity_main_contract.py
```

Expected: PASS.

- [x] **Step 2: Run syntax validation**

Run:

```bash
conda activate finetf && python -m py_compile data_preprocess/operator_futures/commodity/main_contract.py
```

Expected: no output and exit code 0.

- [x] **Step 3: Run OpenSpec strict validation**

Run:

```bash
openspec validate refactor-commodity-main-contract-objects --strict
```

Expected: `Change 'refactor-commodity-main-contract-objects' is valid`.

- [x] **Step 4: Final commit**（skipped commit because the worktree already contains unrelated staged changes; changes are left in the working tree）

```bash
git add data_preprocess/tests/test_commodity_main_contract.py data_preprocess/operator_futures/commodity/main_contract.py openspec/changes/refactor-commodity-main-contract-objects
git commit -m "refactor: finalize commodity main-contract object plan"
```

- [x] **Task complete**（focused verification and OpenSpec validation both pass; commit skipped to avoid mixing unrelated staged changes）
