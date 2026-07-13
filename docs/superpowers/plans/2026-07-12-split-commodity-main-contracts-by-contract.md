# Split Commodity Main Contracts By Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace commodity continuous main-contract files with a JSON summary and generate all commodity factors per selected contract.

**Architecture:** `main_contract.py` owns raw-file scanning, monthly top-2 selection, and summary construction. `downscale_continuous_by_trading_day.py` consumes that summary and writes contract-scoped daily outputs. Shared downstream feature scripts keep legacy paths unless `--contract` is supplied, while commodity shell scripts parse the summary and run the full pipeline per contract.

**Tech Stack:** Python, Polars, pytest, argparse CLIs, Bash, OpenSpec.

**Traceability (sddflow):**
- plan-ready: `openspec/changes/split-commodity-main-contracts-by-contract/plan-ready.md`
- tasks: `openspec/changes/split-commodity-main-contracts-by-contract/tasks.md`
- plan: `docs/superpowers/plans/2026-07-12-split-commodity-main-contracts-by-contract.md`

**Amendments:**
- 2026-07-12: Add typed summary model bean construction before final verification. External `main_contract_summary.json` schema stays unchanged.
- 2026-07-12: Add typed summary model bean deserialization for readers. External `main_contract_summary.json` schema stays unchanged.
- 2026-07-12: Add `daily_volume` to each `contracts[].trading_days[]` summary entry. External `main_contract_summary.json` schema gains this required day-level numeric field.
- 2026-07-13: Clip each selected contract's summary trading days from the first selected month start through the inclusive cutoff 10 contract trading days before the last trading day in the requested date range.
- 2026-07-13: Add configured high-volume-day selection rule. A contract is selected for a month if it is monthly top 2 or has at least 10 actual trading days with `daily_volume > threshold`; `fu` threshold is `15000`.

---

### Task 1: Main-contract summary generation

> **trace:** plan-ready.md -> `### Task 1: Main-contract summary generation` | tasks.md -> `- [ ] 1.0 Main-contract summary generation complete（与 plan-ready.md Task 1 和 superpowers plan Task 1 同步）`
> **sync:** tasks.md -> `- [ ] 1.0 Main-contract summary generation complete（与 plan-ready.md Task 1 和 superpowers plan Task 1 同步）` | plan-ready.md -> `### Task 1: Main-contract summary generation`

**Files:**
- Modify: `data_preprocess/tests/test_commodity_main_contract.py`
- Modify: `data_preprocess/tests/test_commodity_main_contract_cli.py`
- Modify: `data_preprocess/operator_futures/commodity/main_contract.py`
- Modify: `data_preprocess/operator_futures/commodity/stitch_main_contract.py`

- [x] **Step 1: Add failing summary unit tests**

Append focused tests to `data_preprocess/tests/test_commodity_main_contract.py` and update the import list to include the new public functions:

```python
from operator_futures.commodity.main_contract import (
    build_main_contract_summary_for_date_range,
    write_main_contract_summary_for_date_range,
)
```

Add this test body:

```python
def test_build_main_contract_summary_selects_monthly_top_two_contracts(tmp_path):
    raw_root = tmp_path / "data" / "原始下载"
    jan_fu2601 = _write_contract_file(
        raw_root, "燃料油", "2026", "01", "20260105", "fu2601", "20260105", [0, 100]
    )
    jan_fu2602 = _write_contract_file(
        raw_root, "燃料油", "2026", "01", "20260106", "fu2602", "20260106", [0, 80]
    )
    _write_contract_file(
        raw_root, "燃料油", "2026", "01", "20260107", "fu2603", "20260107", [0, 10]
    )
    feb_fu2603 = _write_contract_file(
        raw_root, "燃料油", "2026", "02", "20260203", "fu2603", "20260203", [0, 90]
    )
    _write_contract_file(
        raw_root, "燃料油", "2026", "02", "20260204", "fu2604", "20260204", [0, 90]
    )

    summary = build_main_contract_summary_for_date_range(
        raw_root=raw_root,
        commodity_name="燃料油",
        start_date="2026-01-01",
        end_date="2026-03-01",
        symbol="fu",
    )

    contracts = {item["contract"]: item for item in summary["contracts"]}
    assert list(contracts) == ["fu2601", "fu2602", "fu2603", "fu2604"]
    assert contracts["fu2601"]["selected_months"] == ["2026-01"]
    assert contracts["fu2603"]["selected_months"] == ["2026-02"]
    assert contracts["fu2601"]["start_trading_day"] == "20260105"
    assert contracts["fu2601"]["end_trading_day"] == "20260105"
    assert contracts["fu2601"]["trading_day_count"] == 1
    assert contracts["fu2601"]["trading_days"] == [
        {
            "trading_day": "20260105",
            "date": "2026-01-05",
            "source_file": str(jan_fu2601),
        }
    ]
    assert contracts["fu2602"]["trading_days"][0]["source_file"] == str(jan_fu2602)
    assert contracts["fu2603"]["trading_days"][0]["source_file"] == str(feb_fu2603)
```

- [x] **Step 2: Add failing deterministic and error tests**

Append these tests to the same file:

```python
def test_build_main_contract_summary_ties_sort_by_contract_name(tmp_path):
    raw_root = tmp_path / "data" / "原始下载"
    _write_contract_file(raw_root, "燃料油", "2026", "01", "20260105", "fu2603", "20260105", [0, 90])
    _write_contract_file(raw_root, "燃料油", "2026", "01", "20260105", "fu2601", "20260105", [0, 90])
    _write_contract_file(raw_root, "燃料油", "2026", "01", "20260105", "fu2602", "20260105", [0, 90])

    summary = build_main_contract_summary_for_date_range(
        raw_root, "燃料油", "2026-01-01", "2026-02-01", "fu"
    )

    assert [item["contract"] for item in summary["contracts"]] == ["fu2601", "fu2602"]


def test_build_main_contract_summary_rejects_duplicate_contract_day(tmp_path):
    raw_root = tmp_path / "data" / "原始下载"
    _write_contract_file(raw_root, "燃料油", "2026", "01", "20260105", "fu2601", "20260105", [0, 10])
    duplicate = raw_root / "燃料油" / "2026" / "fu2601-copy.csv"
    _frame("fu2601", "20260105", "20260105", [0, 11]).write_csv(duplicate)

    with pytest.raises(ValueError, match="Duplicate contract data for TradingDay 20260105"):
        build_main_contract_summary_for_date_range(
            raw_root, "燃料油", "2026-01-01", "2026-02-01", "fu"
        )
```

Also add `import pytest` at the top of the file if it is not already present.

- [x] **Step 3: Run the summary unit tests to verify failure**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py::test_build_main_contract_summary_selects_monthly_top_two_contracts data_preprocess/tests/test_commodity_main_contract.py::test_build_main_contract_summary_ties_sort_by_contract_name data_preprocess/tests/test_commodity_main_contract.py::test_build_main_contract_summary_rejects_duplicate_contract_day -q
```

Expected: FAIL because `build_main_contract_summary_for_date_range` and `write_main_contract_summary_for_date_range` are not implemented.

- [x] **Step 4: Implement summary helpers in main_contract.py**

In `data_preprocess/operator_futures/commodity/main_contract.py`, add imports:

```python
import json
from collections import defaultdict
from dataclasses import dataclass
```

Add these helpers near the date-range helpers:

```python
def _month_key_from_trading_day(trading_day: str) -> str:
    return datetime.strptime(trading_day, "%Y%m%d").strftime("%Y-%m")


def _summary_contract_sort_key(item: dict) -> tuple:
    return (item["contract"], item["start_trading_day"])


def _contract_summary_entry(
    contract: str,
    selected_months: list[str],
    trading_days: list[dict],
) -> dict:
    ordered_days = sorted(trading_days, key=lambda item: item["trading_day"])
    return {
        "contract": contract,
        "start_trading_day": ordered_days[0]["trading_day"],
        "end_trading_day": ordered_days[-1]["trading_day"],
        "trading_day_count": len(ordered_days),
        "selected_months": sorted(selected_months),
        "trading_days": ordered_days,
    }
```

Then add the public summary builder:

```python
def build_main_contract_summary_for_date_range(
    raw_root: Path,
    commodity_name: str,
    start_date: str,
    end_date: str,
    symbol: str,
) -> dict:
    years = infer_years_for_date_range(start_date, end_date)
    days = load_contract_files_by_trading_day_for_years(raw_root, commodity_name, years)
    monthly_volumes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    contract_days: dict[str, list[dict]] = defaultdict(list)

    for trading_day in sorted(days):
        if not _trading_day_in_range(trading_day, start_date, end_date):
            continue
        for contract, source_file in sorted(days[trading_day].items()):
            frame = pl.read_csv(source_file)
            eligible = _eligible_contracts({contract: frame}, symbol)
            if contract not in eligible:
                continue
            month = _month_key_from_trading_day(trading_day)
            monthly_volumes[month][contract] += calculate_contract_volume(frame)
            contract_days[contract].append(
                {
                    "trading_day": trading_day,
                    "date": _format_trading_day_file_date(trading_day),
                    "source_file": str(source_file),
                }
            )

    selected_months_by_contract: dict[str, set[str]] = defaultdict(set)
    for month, volumes in sorted(monthly_volumes.items()):
        positive = {
            contract: volume for contract, volume in volumes.items() if volume > 0
        }
        for contract, _volume in sorted(
            positive.items(), key=lambda item: (-item[1], item[0])
        )[:2]:
            selected_months_by_contract[contract].add(month)

    if not selected_months_by_contract:
        raise ValueError(f"No monthly top-2 contracts found for symbol {symbol!r}")

    contracts = [
        _contract_summary_entry(
            contract,
            sorted(months),
            contract_days[contract],
        )
        for contract, months in selected_months_by_contract.items()
    ]
    contracts = sorted(contracts, key=_summary_contract_sort_key)
    return {
        "symbol": symbol,
        "commodity_name": commodity_name,
        "start_date": start_date,
        "end_date": end_date,
        "selection_rule": "monthly_top_2_by_sum_daily_volume_delta",
        "contracts": contracts,
    }
```

Add the writer:

```python
def write_main_contract_summary_for_date_range(
    raw_root: Path,
    commodity_name: str,
    output_dir: Path,
    start_date: str,
    end_date: str,
    symbol: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_main_contract_summary_for_date_range(
        raw_root=raw_root,
        commodity_name=commodity_name,
        start_date=start_date,
        end_date=end_date,
        symbol=symbol,
    )
    output_path = output_dir / "main_contract_summary.json"
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Wrote commodity main-contract summary: output=%s contracts=%d",
        output_path,
        len(summary["contracts"]),
    )
    return output_path
```

- [x] **Step 5: Update stitch CLI and CLI test**

In `data_preprocess/operator_futures/commodity/stitch_main_contract.py`, replace the import with:

```python
from .main_contract import write_main_contract_summary_for_date_range
```

Replace the call in `main()` with:

```python
summary_path = write_main_contract_summary_for_date_range(
    raw_root=Path(args.raw_root),
    commodity_name=args.commodity_name,
    output_dir=output_dir,
    start_date=args.start_date,
    end_date=args.end_date,
    symbol=args.symbol,
)
logger.info(
    "Wrote commodity main-contract summary: output=%s elapsed_seconds=%.2f",
    summary_path,
    time.monotonic() - started_at,
)
```

In `data_preprocess/tests/test_commodity_main_contract_cli.py`, replace `test_stitch_main_contract_cli_outputs_daily_files` with:

```python
def test_stitch_main_contract_cli_outputs_summary_json(tmp_path):
    raw_root = tmp_path / "data" / "原始下载"
    _write_contract(raw_root / "燃料油" / "2026" / "01" / "20260105" / "fu2602.csv", "fu2602", "20260105", "20260104", [0, 30])
    _write_contract(raw_root / "燃料油" / "2026" / "01" / "20260106" / "fu2603.csv", "fu2603", "20260106", "20260105", [0, 20])

    output_dir = tmp_path / "continuous" / "fu"
    result = subprocess.run(
        [
            sys.executable, "-m", "operator_futures.commodity.stitch_main_contract",
            "--raw_root", str(raw_root),
            "--commodity_name", "燃料油",
            "--start_date", "2026-01-05",
            "--end_date", "2026-01-07",
            "--symbol", "fu",
            "--output_dir", str(output_dir),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        capture_output=True,
        check=True,
        text=True,
    )

    summary = json.loads((output_dir / "main_contract_summary.json").read_text(encoding="utf-8"))
    assert [item["contract"] for item in summary["contracts"]] == ["fu2602", "fu2603"]
    assert not (output_dir / "2026-01-05.csv").exists()
    assert "Wrote commodity main-contract summary" in result.stderr
```

Add `import json` to the test file.

- [x] **Step 6: Run Task 1 verification**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py::test_stitch_main_contract_cli_outputs_summary_json -q
```

Expected: PASS.

- [x] **Step 7: Commit Task 1**

Run:

```bash
git add data_preprocess/operator_futures/commodity/main_contract.py data_preprocess/operator_futures/commodity/stitch_main_contract.py data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py
git commit -m "feat: generate commodity main contract summary"
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 2: Summary-driven downscale

> **trace:** plan-ready.md -> `### Task 2: Summary-driven downscale` | tasks.md -> `- [ ] 2.0 Summary-driven downscale complete（与 plan-ready.md Task 2 和 superpowers plan Task 2 同步）`
> **sync:** tasks.md -> `- [ ] 2.0 Summary-driven downscale complete（与 plan-ready.md Task 2 和 superpowers plan Task 2 同步）` | plan-ready.md -> `### Task 2: Summary-driven downscale`

**Files:**
- Modify: `data_preprocess/tests/test_commodity_main_contract_cli.py`
- Modify: `data_preprocess/tests/test_commodity_downscale.py`
- Modify: `data_preprocess/operator_futures/commodity/downscale_continuous_by_trading_day.py`

- [x] **Step 1: Add failing summary downscale CLI test**

In `data_preprocess/tests/test_commodity_main_contract_cli.py`, add:

```python
def _write_summary(path: Path, source_file: Path, contract: str = "fu2602"):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "symbol": "fu",
        "commodity_name": "燃料油",
        "start_date": "2026-01-05",
        "end_date": "2026-01-06",
        "selection_rule": "monthly_top_2_by_sum_daily_volume_delta",
        "contracts": [
            {
                "contract": contract,
                "start_trading_day": "20260105",
                "end_trading_day": "20260105",
                "trading_day_count": 1,
                "selected_months": ["2026-01"],
                "trading_days": [
                    {
                        "trading_day": "20260105",
                        "date": "2026-01-05",
                        "source_file": str(source_file),
                    }
                ],
            }
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
```

Add the test:

```python
def test_downscale_continuous_cli_reads_summary_and_writes_contract_outputs(tmp_path):
    raw_file = tmp_path / "raw" / "fu2602.csv"
    _write_continuous_day(raw_file, "fu2602", "20260105", "20260105")
    summary = tmp_path / "continuous" / "fu" / "main_contract_summary.json"
    _write_summary(summary, raw_file)
    output_root = tmp_path / "PREPROCESS_DATASET" / "commodity-futures"

    subprocess.run(
        [
            sys.executable, "-m", "operator_futures.commodity.downscale_continuous_by_trading_day",
            "--summary", str(summary),
            "--output_root", str(output_root),
            "--target_freq", "5min",
            "--symbol", "fu",
            "--depth", "5",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        check=True,
    )

    assert (output_root / "BASE_FEATURE" / "fu" / "fu2602" / "5min" / "2026-01-05.feather").exists()
    assert (output_root / "DOWNSCALE_ORDERBOOK_25" / "fu" / "fu2602" / "5min" / "2026-01-05.feather").exists()
```

- [x] **Step 2: Add failing summary validation tests**

Add:

```python
def test_downscale_continuous_cli_rejects_missing_summary_source_file(tmp_path):
    summary = tmp_path / "continuous" / "fu" / "main_contract_summary.json"
    _write_summary(summary, tmp_path / "missing.csv")

    result = subprocess.run(
        [
            sys.executable, "-m", "operator_futures.commodity.downscale_continuous_by_trading_day",
            "--summary", str(summary),
            "--output_root", str(tmp_path / "out"),
            "--target_freq", "5min",
            "--symbol", "fu",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "source_file does not exist" in result.stderr


def test_downscale_continuous_cli_filters_contract(tmp_path):
    first = tmp_path / "raw" / "fu2602.csv"
    second = tmp_path / "raw" / "fu2603.csv"
    _write_continuous_day(first, "fu2602", "20260105", "20260105")
    _write_continuous_day(second, "fu2603", "20260105", "20260105")
    summary = tmp_path / "continuous" / "fu" / "main_contract_summary.json"
    payload = {
        "symbol": "fu",
        "commodity_name": "燃料油",
        "start_date": "2026-01-05",
        "end_date": "2026-01-06",
        "selection_rule": "monthly_top_2_by_sum_daily_volume_delta",
        "contracts": [
            {"contract": "fu2602", "start_trading_day": "20260105", "end_trading_day": "20260105", "trading_day_count": 1, "selected_months": ["2026-01"], "trading_days": [{"trading_day": "20260105", "date": "2026-01-05", "source_file": str(first)}]},
            {"contract": "fu2603", "start_trading_day": "20260105", "end_trading_day": "20260105", "trading_day_count": 1, "selected_months": ["2026-01"], "trading_days": [{"trading_day": "20260105", "date": "2026-01-05", "source_file": str(second)}]},
        ],
    }
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    output_root = tmp_path / "PREPROCESS_DATASET" / "commodity-futures"

    subprocess.run(
        [
            sys.executable, "-m", "operator_futures.commodity.downscale_continuous_by_trading_day",
            "--summary", str(summary),
            "--contract", "fu2603",
            "--output_root", str(output_root),
            "--target_freq", "5min",
            "--symbol", "fu",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        check=True,
    )

    assert not (output_root / "BASE_FEATURE" / "fu" / "fu2602").exists()
    assert (output_root / "BASE_FEATURE" / "fu" / "fu2603" / "5min" / "2026-01-05.feather").exists()
```

- [x] **Step 3: Run downscale tests to verify failure**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract_cli.py::test_downscale_continuous_cli_reads_summary_and_writes_contract_outputs data_preprocess/tests/test_commodity_main_contract_cli.py::test_downscale_continuous_cli_rejects_missing_summary_source_file data_preprocess/tests/test_commodity_main_contract_cli.py::test_downscale_continuous_cli_filters_contract -q
```

Expected: FAIL because the CLI still requires `--input_dir`.

- [x] **Step 4: Implement summary loading and validation**

In `data_preprocess/operator_futures/commodity/downscale_continuous_by_trading_day.py`, add imports:

```python
import json
```

Add:

```python
def load_main_contract_summary(summary_path: Path) -> dict:
    if not summary_path.exists():
        raise FileNotFoundError(f"main contract summary does not exist: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    contracts = summary.get("contracts")
    if not isinstance(contracts, list) or not contracts:
        raise ValueError("main contract summary must contain non-empty contracts")
    for contract in contracts:
        trading_days = contract.get("trading_days")
        if not isinstance(trading_days, list):
            raise ValueError(f"summary contract missing trading_days: {contract}")
        if contract.get("trading_day_count") != len(trading_days):
            raise ValueError(
                f"trading_day_count mismatch for contract {contract.get('contract')}"
            )
        for day in trading_days:
            source_file = Path(day["source_file"])
            if not source_file.exists():
                raise FileNotFoundError(f"source_file does not exist: {source_file}")
    return summary


def iter_summary_trading_days(summary: dict, contract_filter: str | None = None):
    matched = False
    for contract in summary["contracts"]:
        contract_name = contract["contract"]
        if contract_filter is not None and contract_name != contract_filter:
            continue
        matched = True
        for day in contract["trading_days"]:
            yield contract_name, day["date"], Path(day["source_file"])
    if contract_filter is not None and not matched:
        raise ValueError(f"contract {contract_filter!r} not found in summary")
```

- [x] **Step 5: Update output writer and CLI**

Change `_write_downscaled_day` signature to include `contract: str`, and change the output path line to:

```python
path = output_root / folder / symbol / contract / target_freq
```

Replace `downscale_continuous_by_trading_day` with a summary-based function:

```python
def downscale_continuous_by_trading_day(
    summary_path: Path,
    output_root: Path,
    target_freq: str,
    symbol: str,
    depth: int = 5,
    contract: str | None = None,
) -> None:
    started_at = time.monotonic()
    summary = load_main_contract_summary(summary_path)
    processed = []
    for contract_name, date, source_file in iter_summary_trading_days(summary, contract):
        raw = pl.read_csv(source_file)
        logger.info(
            "Downscaling commodity contract source file: contract=%s date=%s input=%s rows=%d",
            contract_name,
            date,
            source_file,
            raw.height,
        )
        trading_day = _write_downscaled_day(
            raw, output_root, target_freq, symbol, contract_name, depth
        )
        processed.append((contract_name, trading_day))
    logger.info(
        "Finished commodity summary downscale: contract_days=%d elapsed_seconds=%.2f",
        len(processed),
        time.monotonic() - started_at,
    )
```

Update `parse_args()`:

```python
parser.add_argument("--summary", required=True)
parser.add_argument("--output_root", required=True)
parser.add_argument("--target_freq", default="5min")
parser.add_argument("--symbol", default="fu")
parser.add_argument("--contract")
parser.add_argument("--depth", type=int, default=5)
```

Update `main()` to pass `summary_path=Path(args.summary)` and `contract=args.contract`.

- [x] **Step 6: Update old CLI rejection test**

Replace the old input-dir rejection expectation in `test_downscale_continuous_cli_rejects_old_input_file_argument` or add a new test:

```python
def test_downscale_continuous_cli_rejects_old_input_dir_argument(tmp_path):
    result = subprocess.run(
        [
            sys.executable, "-m", "operator_futures.commodity.downscale_continuous_by_trading_day",
            "--input_dir", str(tmp_path / "continuous"),
            "--start_date", "2026-01-05",
            "--end_date", "2026-01-06",
            "--output_root", str(tmp_path / "out"),
            "--target_freq", "5min",
            "--symbol", "fu",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--summary" in result.stderr
```

- [x] **Step 7: Run Task 2 verification**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract_cli.py data_preprocess/tests/test_commodity_downscale.py -q
```

Expected: PASS.

- [x] **Step 8: Commit Task 2**

Run:

```bash
git add data_preprocess/operator_futures/commodity/downscale_continuous_by_trading_day.py data_preprocess/tests/test_commodity_main_contract_cli.py data_preprocess/tests/test_commodity_downscale.py
git commit -m "feat: downscale commodity contracts from summary"
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 3: Contract-scoped downstream Python paths

> **trace:** plan-ready.md -> `### Task 3: Contract-scoped downstream Python paths` | tasks.md -> `- [ ] 3.0 Contract-scoped downstream Python paths complete（与 plan-ready.md Task 3 和 superpowers plan Task 3 同步）`
> **sync:** tasks.md -> `- [ ] 3.0 Contract-scoped downstream Python paths complete（与 plan-ready.md Task 3 和 superpowers plan Task 3 同步）` | plan-ready.md -> `### Task 3: Contract-scoped downstream Python paths`

**Files:**
- Modify: `data_preprocess/operator_futures/cross_section/create_feature.py`
- Modify: `data_preprocess/operator_futures/merge_concat/merge.py`
- Modify: `data_preprocess/operator_futures/merge_concat/concat.py`
- Modify: `data_preprocess/operator_futures/time_operator/create_feature_multi_processing.py`
- Modify: `data_preprocess/operator_futures/merge_all/merge_clean.py`
- Modify: `data_preprocess/operator_futures/feature_selection/ic_correlation.py`
- Modify: `data_preprocess/operator_futures/scale_describe_save/scale_save.py`
- Modify: `data_preprocess/tests/test_commodity_feature_pipeline.py`

- [x] **Step 1: Add path helper tests**

In `data_preprocess/tests/test_commodity_feature_pipeline.py`, add tests that call a new helper expected from shared scripts:

```python
from operator_futures.util import symbol_contract_path_parts


def test_symbol_contract_path_parts_uses_contract_when_present():
    assert symbol_contract_path_parts("fu", "fu2601") == ("fu", "fu2601")


def test_symbol_contract_path_parts_keeps_legacy_symbol_only_path():
    assert symbol_contract_path_parts("BTCUSDT", None) == ("BTCUSDT",)
```

- [x] **Step 2: Implement shared path helper**

In `data_preprocess/operator_futures/util.py`, add:

```python
def symbol_contract_path_parts(symbol: str, contract: str | None = None) -> tuple[str, ...]:
    if contract:
        return (symbol, contract)
    return (symbol,)
```

- [x] **Step 3: Update cross-section paths**

In `cross_section/create_feature.py`, add:

```python
parser.add_argument("--contract", type=str, default=None)
```

Import:

```python
from operator_futures.util import symbol_contract_path_parts
```

Before path construction:

```python
symbol_parts = symbol_contract_path_parts(args.symbols, args.contract)
```

Replace occurrences shaped like:

```python
os.path.join(args.data_path, "BASE_FEATURE", args.symbols, args.target_freq, args.date + ".feather")
```

with:

```python
os.path.join(args.data_path, "BASE_FEATURE", *symbol_parts, args.target_freq, args.date + ".feather")
```

Apply the same `*symbol_parts` pattern to KLINE, QUOTES, SNAPSHOT input and output directories.

- [x] **Step 4: Update merge and concat paths**

In `merge_concat/merge.py` and `merge_concat/concat.py`, add `--contract`, import `symbol_contract_path_parts`, and use:

```python
symbol_parts = symbol_contract_path_parts(args.symbols, args.contract)
```

For merge output, use:

```python
os.path.join(args.save_path, "MERGED_FEATURE", *symbol_parts, args.target_freq, "CONCURRENT_FEATURE")
os.path.join(args.save_path, "MERGED_FEATURE", *symbol_parts, args.target_freq, "FUTURE_FEATURE")
```

For concat input/output, use:

```python
cocurrent_path = os.path.join(args.data_path, "MERGED_FEATURE", *symbol_parts, args.target_freq, "CONCURRENT_FEATURE")
future_path = os.path.join(args.data_path, "MERGED_FEATURE", *symbol_parts, args.target_freq, "FUTURE_FEATURE")
save_path = os.path.join(args.save_path, "CONCAT_FEATURE", *symbol_parts, args.target_freq)
```

- [x] **Step 5: Update range-based post-merge scripts**

In `time_operator/create_feature_multi_processing.py`, `merge_all/merge_clean.py`, `feature_selection/ic_correlation.py`, and `scale_describe_save/scale_save.py`, add `--contract`, import the helper, and replace path construction with `*symbol_parts`.

Use this pattern for range files:

```python
symbol_parts = symbol_contract_path_parts(args.symbols, args.contract)
input_path = Path(args.data_path).joinpath(*symbol_parts, args.target_freq, f"{args.start_date}-{args.end_date}.feather")
output_dir = Path(args.save_path).joinpath(*symbol_parts, args.target_freq, f"{args.start_date}-{args.end_date}")
```

For scripts using `os.path.join`, the equivalent is:

```python
os.path.join(args.save_path, *symbol_parts, args.target_freq, f"{args.start_date}-{args.end_date}.feather")
```

- [x] **Step 6: Add representative CLI path tests**

Extend `data_preprocess/tests/test_commodity_feature_pipeline.py` with parser/path helper assertions, not full data runs:

```python
def test_contract_path_shape_for_daily_outputs(tmp_path):
    parts = symbol_contract_path_parts("fu", "fu2601")
    path = tmp_path.joinpath("BASE_FEATURE", *parts, "5min", "2026-01-05.feather")
    assert path.as_posix().endswith("BASE_FEATURE/fu/fu2601/5min/2026-01-05.feather")


def test_legacy_path_shape_for_daily_outputs(tmp_path):
    parts = symbol_contract_path_parts("fu", None)
    path = tmp_path.joinpath("BASE_FEATURE", *parts, "5min", "2026-01-05.feather")
    assert path.as_posix().endswith("BASE_FEATURE/fu/5min/2026-01-05.feather")
```

- [x] **Step 7: Run Task 3 verification**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_commodity_feature_pipeline.py -q
python -m compileall data_preprocess/operator_futures/cross_section/create_feature.py data_preprocess/operator_futures/merge_concat/merge.py data_preprocess/operator_futures/merge_concat/concat.py data_preprocess/operator_futures/time_operator/create_feature_multi_processing.py data_preprocess/operator_futures/merge_all/merge_clean.py data_preprocess/operator_futures/feature_selection/ic_correlation.py data_preprocess/operator_futures/scale_describe_save/scale_save.py
```

Expected: PASS and compileall reports no syntax errors.

- [x] **Step 8: Commit Task 3**

Run:

```bash
git add data_preprocess/operator_futures/util.py data_preprocess/operator_futures/cross_section/create_feature.py data_preprocess/operator_futures/merge_concat/merge.py data_preprocess/operator_futures/merge_concat/concat.py data_preprocess/operator_futures/time_operator/create_feature_multi_processing.py data_preprocess/operator_futures/merge_all/merge_clean.py data_preprocess/operator_futures/feature_selection/ic_correlation.py data_preprocess/operator_futures/scale_describe_save/scale_save.py data_preprocess/tests/test_commodity_feature_pipeline.py
git commit -m "feat: add commodity contract scoped feature paths"
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 4: Commodity shell scripts, validation, and docs

> **trace:** plan-ready.md -> `### Task 4: Commodity shell scripts, validation, and docs` | tasks.md -> `- [ ] 4.0 Commodity shell scripts, validation, and docs complete（与 plan-ready.md Task 4 和 superpowers plan Task 4 同步）`
> **sync:** tasks.md -> `- [ ] 4.0 Commodity shell scripts, validation, and docs complete（与 plan-ready.md Task 4 和 superpowers plan Task 4 同步）` | plan-ready.md -> `### Task 4: Commodity shell scripts, validation, and docs`

**Files:**
- Modify: `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`
- Modify: `data_preprocess/script_preprocess/future_upgraded/commodity/main_fu.sh`
- Modify: `data_preprocess/script_preprocess/future_upgraded/commodity/main_al.sh`
- Modify: `data_preprocess/script_preprocess/future_upgraded/commodity/commodity_process.sh`
- Modify: `data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh`
- Inspect/modify if needed: `data_preprocess/script_preprocess/future_upgraded/commodity/flatten_aluminum_raw_csv.sh`
- Modify: `data_preprocess/tests/test_commodity_main_contract_cli.py`
- Modify: `docs/上海商品交易所/commodity_futures_preprocess.md`

- [x] **Step 1: Add shell contract-list helper tests**

In `data_preprocess/tests/test_commodity_main_contract_cli.py`, add a shell test that sources the copied script and parses a summary:

```python
def test_commodity_full_process_reads_contracts_from_summary(tmp_path):
    script_dir = _copy_commodity_script_tree(tmp_path)
    summary = tmp_path / "PREPROCESS_DATASET" / "commodity-futures" / "CONTINUOUS_RAW" / "fu" / "main_contract_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        json.dumps(
            {
                "contracts": [
                    {"contract": "fu2601", "trading_day_count": 1, "trading_days": [{"trading_day": "20260105", "date": "2026-01-05", "source_file": "a.csv"}]},
                    {"contract": "fu2605", "trading_day_count": 1, "trading_days": [{"trading_day": "20260205", "date": "2026-02-05", "source_file": "b.csv"}]},
                ]
            }
        ),
        encoding="utf-8",
    )
    command = f'''
source "{script_dir / "fu_full_process.sh"}"
run_commodity_summary_contracts "{summary}"
'''
    result = subprocess.run(["bash", "-lc", command], capture_output=True, text=True, check=True)
    assert result.stdout.splitlines() == ["fu2601", "fu2605"]
```

- [x] **Step 2: Implement summary contract parsing in fu_full_process.sh**

Add this function to `fu_full_process.sh`:

```bash
run_commodity_summary_contracts() {
    local summary_path=$1
    python - "$summary_path" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
for item in summary.get("contracts", []):
    print(item["contract"])
PY
}
```

- [x] **Step 3: Update downscale and downstream shell functions**

Change `run_commodity_downscale_continuous_by_trading_day` to accept a summary path and optional contract:

```bash
run_commodity_downscale_continuous_by_trading_day() {
    local root_path=$1
    local summary_path=$2
    local target_freq=$3
    local symbol=${4:-fu}
    local contract=${5:-}
    local output_root="${root_path}/PREPROCESS_DATASET/commodity-futures"
    local contract_args=()
    if [ -n "$contract" ]; then
        contract_args=(--contract "$contract")
    fi

    PYTHONPATH="${root_path}/data_preprocess" python -m operator_futures.commodity.downscale_continuous_by_trading_day \
        --summary "${summary_path}" \
        --output_root "${output_root}" \
        --target_freq "${target_freq}" \
        --symbol "${symbol}" \
        --depth 5 \
        "${contract_args[@]}"
}
```

For each downstream function, add a `contract` parameter and pass `--contract "$contract"` to the Python command. Update existence checks to look under `${symbol}/${contract}/${target_freq}`.

- [x] **Step 4: Update full-process orchestration**

In `run_commodity_full_process`, after stitch:

```bash
local summary_path="${root_path}/PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/${symbol}/main_contract_summary.json"
```

Run downscale:

```bash
run_commodity_logged_step \
    "$log_dir" "$symbol" "$target_freq" "$start_date" "$end_date" \
    "downscale_continuous_by_trading_day" \
    run_commodity_downscale_continuous_by_trading_day "$root_path" "$summary_path" "$target_freq" "$symbol"
```

Then loop:

```bash
local contract
while IFS= read -r contract; do
    [ -n "$contract" ] || continue
    run_commodity_logged_step "$log_dir" "${symbol}_${contract}" "$target_freq" "$start_date" "$end_date" \
        "cross_section" \
        run_commodity_cross_section_process "$start_date" "$end_date" "$max_processes" "$target_freq" "$symbol" "$root_path" "$contract"
    run_commodity_logged_step "$log_dir" "${symbol}_${contract}" "$target_freq" "$start_date" "$end_date" \
        "merge" \
        run_commodity_merge_process "$start_date" "$end_date" "$max_processes" "$target_freq" "$symbol" "$root_path" "$contract"
    run_commodity_logged_step "$log_dir" "${symbol}_${contract}" "$target_freq" "$start_date" "$end_date" \
        "concat" \
        run_commodity_concat_process "$target_freq" "$start_date" "$end_date" "$symbol" "$root_path" "$contract"
    run_commodity_logged_step "$log_dir" "${symbol}_${contract}" "$target_freq" "$start_date" "$end_date" \
        "time_feature" \
        run_commodity_time_feature "$target_freq" "$start_date" "$end_date" "$symbol" "$root_path" "$contract"
    run_commodity_logged_step "$log_dir" "${symbol}_${contract}" "$target_freq" "$start_date" "$end_date" \
        "merge_clean" \
        run_commodity_merge_and_clean "$target_freq" "$start_date" "$end_date" "$symbol" "$root_path" "$contract"
    run_commodity_logged_step "$log_dir" "${symbol}_${contract}" "$target_freq" "$start_date" "$end_date" \
        "ic_correlation" \
        run_commodity_ic_correlation "$target_freq" "$start_date" "$end_date" "$symbol" "$root_path" "$contract"
    run_commodity_logged_step "$log_dir" "${symbol}_${contract}" "$target_freq" "$start_date" "$end_date" \
        "scale_save" \
        run_commodity_scale_save "$target_freq" "$start_date" "$end_date" "$symbol" "$root_path" "$contract"
done < <(run_commodity_summary_contracts "$summary_path")
```

- [x] **Step 5: Update validation and docs**

In `validate_features.sh`, add summary parsing using `run_commodity_summary_contracts` or an equivalent Python snippet, then validate paths under:

```bash
"${root_path}/PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/${symbol}/${contract}/${target_freq}/${start_date}-${end_date}/df.feather"
```

In `docs/上海商品交易所/commodity_futures_preprocess.md`, replace examples that mention `CONTINUOUS_RAW/{symbol}/{date}.csv` with:

```bash
PYTHONPATH=data_preprocess python -m operator_futures.commodity.stitch_main_contract \
  --raw_root data/原始下载 \
  --commodity_name 燃料油 \
  --start_date 2026-01-01 \
  --end_date 2026-04-01 \
  --symbol fu \
  --output_dir PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/fu

PYTHONPATH=data_preprocess python -m operator_futures.commodity.downscale_continuous_by_trading_day \
  --summary PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/fu/main_contract_summary.json \
  --output_root PREPROCESS_DATASET/commodity-futures \
  --target_freq 5min \
  --symbol fu
```

- [x] **Step 6: Run Task 4 verification**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract_cli.py -q
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/main_fu.sh
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/main_al.sh
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/commodity_process.sh
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/flatten_aluminum_raw_csv.sh
```

Expected: PASS; no syntax errors.

- [x] **Step 7: Commit Task 4**

Run:

```bash
git add data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh data_preprocess/script_preprocess/future_upgraded/commodity/main_fu.sh data_preprocess/script_preprocess/future_upgraded/commodity/main_al.sh data_preprocess/script_preprocess/future_upgraded/commodity/commodity_process.sh data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh data_preprocess/script_preprocess/future_upgraded/commodity/flatten_aluminum_raw_csv.sh data_preprocess/tests/test_commodity_main_contract_cli.py docs/上海商品交易所/commodity_futures_preprocess.md
git commit -m "feat: run commodity full process per contract"
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 5: Summary model bean refactor

> **trace:** plan-ready.md -> `### Task 5: Summary model bean refactor` | tasks.md -> `- [ ] 5.0 Summary model bean refactor complete（与 plan-ready.md Task 5 和 superpowers plan Task 5 同步）`
> **sync:** tasks.md -> `- [ ] 5.0 Summary model bean refactor complete（与 plan-ready.md Task 5 和 superpowers plan Task 5 同步）` | plan-ready.md -> `### Task 5: Summary model bean refactor`

**Files:**
- Modify: `data_preprocess/tests/test_commodity_main_contract.py`
- Modify: `data_preprocess/operator_futures/commodity/main_contract.py`

- [x] **Step 1: Add summary model serialization test**

In `data_preprocess/tests/test_commodity_main_contract.py`, add a focused test that constructs `MainContractSummary` with one nested `MainContractSummaryContract` and one `MainContractSummaryTradingDay`, then asserts `to_dict()` returns the unchanged JSON-ready structure:

```python
from operator_futures.commodity.main_contract import (
    MainContractSummary,
    MainContractSummaryContract,
    MainContractSummaryTradingDay,
)


def test_main_contract_summary_model_serializes_to_json_contract(tmp_path):
    source_file = tmp_path / "fu2601.csv"
    summary = MainContractSummary(
        symbol="fu",
        commodity_name="燃料油",
        start_date="2026-01-05",
        end_date="2026-01-06",
        contracts=[
            MainContractSummaryContract(
                contract="fu2601",
                selected_months=["2026-01"],
                trading_days=[
                    MainContractSummaryTradingDay(
                        trading_day="20260105",
                        date="2026-01-05",
                        source_file=str(source_file),
                    )
                ],
            )
        ],
    )

    assert summary.to_dict() == {
        "symbol": "fu",
        "commodity_name": "燃料油",
        "start_date": "2026-01-05",
        "end_date": "2026-01-06",
        "selection_rule": "monthly_top_2_by_sum_daily_volume_delta",
        "contracts": [
            {
                "contract": "fu2601",
                "start_trading_day": "20260105",
                "end_trading_day": "20260105",
                "trading_day_count": 1,
                "selected_months": ["2026-01"],
                "trading_days": [
                    {
                        "trading_day": "20260105",
                        "date": "2026-01-05",
                        "source_file": str(source_file),
                    }
                ],
            }
        ],
    }
```

- [x] **Step 2: Add typed summary models**

In `data_preprocess/operator_futures/commodity/main_contract.py`, add frozen dataclasses:

```python
@dataclass(frozen=True)
class MainContractSummaryTradingDay:
    trading_day: str
    date: str
    source_file: str


@dataclass(frozen=True)
class MainContractSummaryContract:
    contract: str
    selected_months: list[str]
    trading_days: list[MainContractSummaryTradingDay]


@dataclass(frozen=True)
class MainContractSummary:
    symbol: str
    commodity_name: str
    start_date: str
    end_date: str
    contracts: list[MainContractSummaryContract]
    selection_rule: str = "monthly_top_2_by_sum_daily_volume_delta"
```

Each class owns its JSON-ready `to_dict()` method. `MainContractSummaryContract.to_dict()` computes `start_trading_day`, `end_trading_day`, and `trading_day_count` from ordered trading-day objects.

- [x] **Step 3: Refactor builder to construct the model first**

Add this model-returning helper and move the monthly top-2 summary construction there:

```python
def build_main_contract_summary_model_for_date_range(
    raw_root: Path,
    commodity_name: str,
    start_date: str,
    end_date: str,
    symbol: str,
) -> MainContractSummary:
    years = infer_years_for_date_range(start_date, end_date)
    # Move the existing monthly volume, contract_days, and selected_months_by_contract
    # construction from build_main_contract_summary_for_date_range into this helper.
    return MainContractSummary(
        symbol=symbol,
        commodity_name=commodity_name,
        start_date=start_date,
        end_date=end_date,
        contracts=contracts,
    )
```

Keep `build_main_contract_summary_for_date_range(raw_root, commodity_name, start_date, end_date, symbol) -> dict` as the existing dict-returning interface by delegating to `.to_dict()`. The writer should continue dumping the dict so CLI JSON output remains unchanged.

- [x] **Step 4: Run Task 5 verification**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py::test_stitch_main_contract_cli_outputs_summary_json -q
```

Expected: PASS.

- [x] **Step 5: Update tracking checkboxes after implementation**

When the refactor passes, update only these tracking checkboxes:

```markdown
docs/superpowers/plans/2026-07-12-split-commodity-main-contracts-by-contract.md
- [x] **Task complete**

openspec/changes/split-commodity-main-contracts-by-contract/tasks.md
- [x] 5.0 Summary model bean refactor complete（与 plan-ready.md Task 5 和 superpowers plan Task 5 同步）

openspec/changes/split-commodity-main-contracts-by-contract/plan-ready.md
- [x] **任务完成**（与 superpowers plan `Task 5`、`tasks.md` 对应条目同步勾选）
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 6: Verification

> **trace:** plan-ready.md -> `### Task 6: Verification` | tasks.md -> `- [ ] 6.0 Verification complete（与 plan-ready.md Task 6 和 superpowers plan Task 6 同步）`
> **sync:** tasks.md -> `- [ ] 6.0 Verification complete（与 plan-ready.md Task 6 和 superpowers plan Task 6 同步）` | plan-ready.md -> `### Task 6: Verification`

**Files:**
- Modify: `openspec/changes/split-commodity-main-contracts-by-contract/tasks.md`
- Modify: `openspec/changes/split-commodity-main-contracts-by-contract/plan-ready.md`
- Modify: `docs/superpowers/plans/2026-07-12-split-commodity-main-contracts-by-contract.md`

- [x] **Step 1: Run focused pytest suite**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py data_preprocess/tests/test_commodity_downscale.py data_preprocess/tests/test_commodity_feature_pipeline.py -q
```

Expected: PASS.

- [x] **Step 2: Run shell syntax checks**

Run:

```bash
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/main_fu.sh
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/main_al.sh
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/commodity_process.sh
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/flatten_aluminum_raw_csv.sh
```

Expected: all commands exit 0.

- [x] **Step 3: Run OpenSpec and diff checks**

Run:

```bash
openspec validate split-commodity-main-contracts-by-contract --strict
git diff --check
```

Expected: OpenSpec reports the change is valid; `git diff --check` prints no whitespace errors.

- [x] **Step 4: Update tracking checkboxes after implementation**

When all verification commands pass, update only these tracking checkboxes:

```markdown
docs/superpowers/plans/2026-07-12-split-commodity-main-contracts-by-contract.md
- [x] **Task complete**

openspec/changes/split-commodity-main-contracts-by-contract/tasks.md
- [x] 6.0 Verification complete（与 plan-ready.md Task 6 和 superpowers plan Task 6 同步）

openspec/changes/split-commodity-main-contracts-by-contract/plan-ready.md
- [x] **任务完成**（与 superpowers plan `Task 6`、`tasks.md` 对应条目同步勾选）
```

- [x] **Step 5: Commit verification tracking**

Run:

```bash
git add openspec/changes/split-commodity-main-contracts-by-contract/tasks.md openspec/changes/split-commodity-main-contracts-by-contract/plan-ready.md docs/superpowers/plans/2026-07-12-split-commodity-main-contracts-by-contract.md
git commit -m "chore: verify commodity contract preprocessing change"
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 7: Summary model bean deserialization

> **trace:** plan-ready.md -> `### Task 7: Summary model bean deserialization` | tasks.md -> `- [ ] 7.0 Summary model bean deserialization complete（与 plan-ready.md Task 7 和 superpowers plan Task 7 同步）`
> **sync:** tasks.md -> `- [ ] 7.0 Summary model bean deserialization complete（与 plan-ready.md Task 7 和 superpowers plan Task 7 同步）` | plan-ready.md -> `### Task 7: Summary model bean deserialization`

**Files:**
- Modify: `data_preprocess/tests/test_commodity_main_contract.py`
- Modify: `data_preprocess/tests/test_commodity_downscale.py`
- Modify: `data_preprocess/tests/test_commodity_main_contract_cli.py`
- Modify: `data_preprocess/operator_futures/commodity/main_contract.py`
- Modify: `data_preprocess/operator_futures/commodity/downscale_continuous_by_trading_day.py`
- Modify: `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`
- Modify: `data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh`

- [x] **Step 1: Add summary model deserialization tests**

In `data_preprocess/tests/test_commodity_main_contract.py`, add tests for `MainContractSummary.from_dict` and `load_main_contract_summary`:

```python
def test_main_contract_summary_model_deserializes_from_json_contract(tmp_path):
    source_file = tmp_path / "fu2601.csv"
    payload = {
        "symbol": "fu",
        "commodity_name": "燃料油",
        "start_date": "2026-01-05",
        "end_date": "2026-01-06",
        "selection_rule": "monthly_top_2_by_sum_daily_volume_delta",
        "contracts": [
            {
                "contract": "fu2601",
                "start_trading_day": "20260105",
                "end_trading_day": "20260105",
                "trading_day_count": 1,
                "selected_months": ["2026-01"],
                "trading_days": [
                    {
                        "trading_day": "20260105",
                        "date": "2026-01-05",
                        "source_file": str(source_file),
                    }
                ],
            }
        ],
    }

    summary = MainContractSummary.from_dict(payload)

    assert summary.symbol == "fu"
    assert summary.contracts[0].contract == "fu2601"
    assert summary.contracts[0].trading_days[0].source_file == str(source_file)
    assert summary.to_dict() == payload
```

Add a loader test:

```python
def test_load_main_contract_summary_reads_model(tmp_path):
    source_file = tmp_path / "fu2601.csv"
    summary_path = tmp_path / "main_contract_summary.json"
    payload = {
        "symbol": "fu",
        "commodity_name": "燃料油",
        "start_date": "2026-01-05",
        "end_date": "2026-01-06",
        "selection_rule": "monthly_top_2_by_sum_daily_volume_delta",
        "contracts": [
            {
                "contract": "fu2601",
                "start_trading_day": "20260105",
                "end_trading_day": "20260105",
                "trading_day_count": 1,
                "selected_months": ["2026-01"],
                "trading_days": [
                    {
                        "trading_day": "20260105",
                        "date": "2026-01-05",
                        "source_file": str(source_file),
                    }
                ],
            }
        ],
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    summary = load_main_contract_summary(summary_path)

    assert isinstance(summary, MainContractSummary)
    assert summary.contracts[0].contract == "fu2601"
```

- [x] **Step 2: Add downscale model-reader test**

In `data_preprocess/tests/test_commodity_downscale.py`, update imports to include `load_main_contract_summary`. Add a test that passes a `MainContractSummary` object to the downscale iterator:

```python
def test_iter_summary_trading_days_accepts_summary_model(tmp_path):
    source_file = tmp_path / "fu2601.csv"
    _write_raw_day(source_file, contract="fu2601", trading_day="20260105")
    summary = MainContractSummary.from_dict(
        {
            "symbol": "fu",
            "commodity_name": "燃料油",
            "start_date": "2026-01-05",
            "end_date": "2026-01-06",
            "selection_rule": "monthly_top_2_by_sum_daily_volume_delta",
            "contracts": [
                {
                    "contract": "fu2601",
                    "start_trading_day": "20260105",
                    "end_trading_day": "20260105",
                    "trading_day_count": 1,
                    "selected_months": ["2026-01"],
                    "trading_days": [
                        {
                            "trading_day": "20260105",
                            "date": "2026-01-05",
                            "source_file": str(source_file),
                        }
                    ],
                }
            ],
        }
    )

    days = list(iter_summary_trading_days(summary))

    assert days[0].contract == "fu2601"
    assert days[0].source_file == source_file
```

If the existing test helper has a different raw writer name, use the local helper already present in that test file. The assertion must prove the iterator returns typed entries or objects derived from the model, not nested dicts.

- [x] **Step 3: Implement summary model from_dict and loader**

In `data_preprocess/operator_futures/commodity/main_contract.py`, add:

```python
@classmethod
def from_dict(cls, payload: dict) -> "MainContractSummary":
    if not isinstance(payload, dict):
        raise ValueError("main contract summary must be a JSON object")
    contracts = [
        MainContractSummaryContract.from_dict(item)
        for item in payload.get("contracts", [])
    ]
    return cls(
        symbol=payload["symbol"],
        commodity_name=payload["commodity_name"],
        start_date=payload["start_date"],
        end_date=payload["end_date"],
        selection_rule=payload.get(
            "selection_rule",
            "monthly_top_2_by_sum_daily_volume_delta",
        ),
        contracts=contracts,
    )
```

Add matching `from_dict` methods to `MainContractSummaryContract` and `MainContractSummaryTradingDay`. `MainContractSummaryContract.from_dict` must validate `trading_day_count == len(trading_days)` and preserve the existing fail-fast error wording expected by tests.

Add:

```python
def load_main_contract_summary(path: Path) -> MainContractSummary:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid main contract summary JSON: {path}") from exc
    return MainContractSummary.from_dict(payload)
```

- [x] **Step 4: Refactor downscale to consume the model**

In `data_preprocess/operator_futures/commodity/downscale_continuous_by_trading_day.py`, replace direct `json.loads` / dict validation with `load_main_contract_summary`. Update `iter_summary_trading_days` to accept `MainContractSummary` and iterate typed `contract` / `trading_day` objects.

Keep CLI behavior and error messages covered by current tests:

```python
summary = load_main_contract_summary(summary_path)
for item in iter_summary_trading_days(summary, contract_filter=contract):
    frame = pl.read_csv(item.source_file)
```

- [x] **Step 5: Refactor shell Python snippets to use the model loader**

In `fu_full_process.sh` and `validate_features.sh`, keep the shell interface unchanged but replace the embedded Python snippet body with:

```python
from pathlib import Path
import sys

from operator_futures.commodity.main_contract import load_main_contract_summary

summary = load_main_contract_summary(Path(sys.argv[1]))
for item in summary.contracts:
    print(item.contract)
```

Ensure the shell invokes Python with `PYTHONPATH="${root_path}/data_preprocess"` where `root_path` is available. For helper calls that only receive a summary path, derive `root_path` from the current full process argument or keep the caller's exported `PYTHONPATH` intact.

- [x] **Step 6: Run Task 7 verification**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py data_preprocess/tests/test_commodity_downscale.py -q
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh
```

Expected: PASS and shell syntax exits 0.

- [x] **Step 7: Update tracking checkboxes after implementation**

When Task 7 verification passes, update only these tracking checkboxes:

```markdown
docs/superpowers/plans/2026-07-12-split-commodity-main-contracts-by-contract.md
- [x] **Task complete**

openspec/changes/split-commodity-main-contracts-by-contract/tasks.md
- [x] 7.0 Summary model bean deserialization complete（与 plan-ready.md Task 7 和 superpowers plan Task 7 同步）

openspec/changes/split-commodity-main-contracts-by-contract/plan-ready.md
- [x] **任务完成**（与 superpowers plan `Task 7`、`tasks.md` 对应条目同步勾选）
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 8: Post-deserialization verification

> **trace:** plan-ready.md -> `### Task 8: Post-deserialization verification` | tasks.md -> `- [ ] 8.0 Post-deserialization verification complete（与 plan-ready.md Task 8 和 superpowers plan Task 8 同步）`
> **sync:** tasks.md -> `- [ ] 8.0 Post-deserialization verification complete（与 plan-ready.md Task 8 和 superpowers plan Task 8 同步）` | plan-ready.md -> `### Task 8: Post-deserialization verification`

**Files:**
- Modify: `openspec/changes/split-commodity-main-contracts-by-contract/tasks.md`
- Modify: `openspec/changes/split-commodity-main-contracts-by-contract/plan-ready.md`
- Modify: `docs/superpowers/plans/2026-07-12-split-commodity-main-contracts-by-contract.md`

- [x] **Step 1: Run focused pytest suite**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py data_preprocess/tests/test_commodity_downscale.py -q
```

Expected: PASS.

- [x] **Step 2: Run shell syntax checks**

Run:

```bash
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/main_fu.sh
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/main_al.sh
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/commodity_process.sh
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/flatten_aluminum_raw_csv.sh
```

Expected: all commands exit 0.

- [x] **Step 3: Run OpenSpec and diff checks**

Run:

```bash
openspec validate split-commodity-main-contracts-by-contract --strict
git diff --check
```

Expected: OpenSpec reports the change is valid; `git diff --check` prints no whitespace errors.

- [x] **Step 4: Update tracking checkboxes after verification**

When all verification commands pass, update only these tracking checkboxes:

```markdown
docs/superpowers/plans/2026-07-12-split-commodity-main-contracts-by-contract.md
- [x] **Task complete**

openspec/changes/split-commodity-main-contracts-by-contract/tasks.md
- [x] 8.0 Post-deserialization verification complete（与 plan-ready.md Task 8 和 superpowers plan Task 8 同步）

openspec/changes/split-commodity-main-contracts-by-contract/plan-ready.md
- [x] **任务完成**（与 superpowers plan `Task 8`、`tasks.md` 对应条目同步勾选）
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 9: Daily volume in summary trading days

> **trace:** plan-ready.md -> `### Task 9: Daily volume in summary trading days` | tasks.md -> `- [ ] 9.0 Daily volume in summary trading days complete（与 plan-ready.md Task 9 和 superpowers plan Task 9 同步）`
> **sync:** tasks.md -> `- [ ] 9.0 Daily volume in summary trading days complete（与 plan-ready.md Task 9 和 superpowers plan Task 9 同步）` | plan-ready.md -> `### Task 9: Daily volume in summary trading days`

**Files:**
- Modify: `data_preprocess/operator_futures/commodity/main_contract.py`
- Modify: `data_preprocess/tests/test_commodity_main_contract.py`
- Modify: `data_preprocess/tests/test_commodity_main_contract_cli.py`
- Modify: `data_preprocess/tests/test_commodity_downscale.py`

- [x] **Step 1: Add failing summary model daily-volume tests**

In `data_preprocess/tests/test_commodity_main_contract.py`, update the typed-model serialization test so `MainContractSummaryTradingDay(...)` requires `daily_volume=100.0`, and assert the serialized dict includes:

```python
{
    "trading_day": "20260105",
    "date": "2026-01-05",
    "source_file": str(source_file),
    "daily_volume": 100.0,
}
```

Update the `MainContractSummary.from_dict` / `load_main_contract_summary` tests so each payload trading-day object includes `"daily_volume": 100.0`, then assert the deserialized day object exposes `daily_volume == 100.0`.

- [x] **Step 2: Add failing summary builder daily-volume assertion**

In the summary builder test that writes a source frame with `Volume` values `[0, 100]`, assert:

```python
assert contracts["fu2601"]["trading_days"] == [
    {
        "trading_day": "20260105",
        "date": "2026-01-05",
        "source_file": str(jan_fu2601),
        "daily_volume": 100.0,
    }
]
```

If the test currently asserts only `source_file`, add a direct assertion:

```python
assert contracts["fu2601"]["trading_days"][0]["daily_volume"] == 100.0
```

- [x] **Step 3: Run the new focused tests to verify failure**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py -q
```

Expected: FAIL because `MainContractSummaryTradingDay` and the summary builder do not yet include `daily_volume`.

- [x] **Step 4: Implement daily_volume in the typed summary model**

In `data_preprocess/operator_futures/commodity/main_contract.py`, update `MainContractSummaryTradingDay`:

```python
@dataclass(frozen=True)
class MainContractSummaryTradingDay:
    trading_day: str
    date: str
    source_file: str
    daily_volume: float
```

Update `to_dict()` to emit `"daily_volume": self.daily_volume`.

Update `from_dict()` to read and validate the required numeric field:

```python
daily_volume = payload["daily_volume"]
if not isinstance(daily_volume, (int, float)):
    raise ValueError("summary trading day daily_volume must be numeric")
```

Return `daily_volume=float(daily_volume)` so JSON integers and floats deserialize consistently.

- [x] **Step 5: Store daily volume during summary construction**

In the raw-file scan loop, compute the daily volume once and reuse it for monthly aggregation and the trading-day model:

```python
daily_volume = calculate_contract_volume(frame)
monthly_volumes[month][contract] += daily_volume
contract_days[contract].append(
    MainContractSummaryTradingDay(
        trading_day=trading_day,
        date=_format_trading_day_file_date(trading_day),
        source_file=str(source_file),
        daily_volume=daily_volume,
    )
)
```

Keep the existing selection rule unchanged: monthly top-2 still uses the sum of these daily values.

- [x] **Step 6: Update summary fixtures that are loaded through the model**

Update `_write_summary` and inline summary payloads in:

```text
data_preprocess/tests/test_commodity_main_contract_cli.py
data_preprocess/tests/test_commodity_downscale.py
```

Every `trading_days` entry should include:

```python
"daily_volume": 100.0
```

Use the fixture's local volume value when the test already implies a different value; otherwise use `100.0`.

- [x] **Step 7: Run Task 9 verification**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py data_preprocess/tests/test_commodity_downscale.py -q
```

Expected: PASS.

- [x] **Step 8: Update tracking checkboxes after implementation**

When Task 9 verification passes, update only these tracking checkboxes:

```markdown
docs/superpowers/plans/2026-07-12-split-commodity-main-contracts-by-contract.md
- [x] **Task complete**

openspec/changes/split-commodity-main-contracts-by-contract/tasks.md
- [x] 9.0 Daily volume in summary trading days complete（与 plan-ready.md Task 9 和 superpowers plan Task 9 同步）

openspec/changes/split-commodity-main-contracts-by-contract/plan-ready.md
- [x] **任务完成**（与 superpowers plan `Task 9`、`tasks.md` 对应条目同步勾选）
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 10: Daily-volume verification

> **trace:** plan-ready.md -> `### Task 10: Daily-volume verification` | tasks.md -> `- [ ] 10.0 Daily-volume verification complete（与 plan-ready.md Task 10 和 superpowers plan Task 10 同步）`
> **sync:** tasks.md -> `- [ ] 10.0 Daily-volume verification complete（与 plan-ready.md Task 10 和 superpowers plan Task 10 同步）` | plan-ready.md -> `### Task 10: Daily-volume verification`

**Files:**
- Modify: `openspec/changes/split-commodity-main-contracts-by-contract/tasks.md`
- Modify: `openspec/changes/split-commodity-main-contracts-by-contract/plan-ready.md`
- Modify: `docs/superpowers/plans/2026-07-12-split-commodity-main-contracts-by-contract.md`

- [x] **Step 1: Run focused pytest suite**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py data_preprocess/tests/test_commodity_downscale.py -q
```

Expected: PASS.

- [x] **Step 2: Run shell syntax checks**

Run:

```bash
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/main_fu.sh
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/main_al.sh
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/commodity_process.sh
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/flatten_aluminum_raw_csv.sh
```

Expected: all commands exit 0.

- [x] **Step 3: Run OpenSpec and diff checks**

Run:

```bash
openspec validate split-commodity-main-contracts-by-contract --strict
git diff --check
```

Expected: OpenSpec reports the change is valid; `git diff --check` prints no whitespace errors.

- [x] **Step 4: Update tracking checkboxes after verification**

When all verification commands pass, update only these tracking checkboxes:

```markdown
docs/superpowers/plans/2026-07-12-split-commodity-main-contracts-by-contract.md
- [x] **Task complete**

openspec/changes/split-commodity-main-contracts-by-contract/tasks.md
- [x] 10.0 Daily-volume verification complete（与 plan-ready.md Task 10 和 superpowers plan Task 10 同步）

openspec/changes/split-commodity-main-contracts-by-contract/plan-ready.md
- [x] **任务完成**（与 superpowers plan `Task 10`、`tasks.md` 对应条目同步勾选）
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 11: Contract trading-window clipping

> **trace:** plan-ready.md -> `### Task 11: Contract trading-window clipping` | tasks.md -> `- [ ] 11.0 Contract trading-window clipping complete（与 plan-ready.md Task 11 和 superpowers plan Task 11 同步）`
> **sync:** tasks.md -> `- [ ] 11.0 Contract trading-window clipping complete（与 plan-ready.md Task 11 和 superpowers plan Task 11 同步）` | plan-ready.md -> `### Task 11: Contract trading-window clipping`

**Files:**
- Modify: `data_preprocess/operator_futures/commodity/main_contract.py`
- Modify: `data_preprocess/tests/test_commodity_main_contract.py`
- Modify if needed: `data_preprocess/tests/test_commodity_main_contract_cli.py`
- Modify if needed: `data_preprocess/tests/test_commodity_downscale.py`

- [x] **Step 1: Add failing tests for first-selected-month start clipping**

In `data_preprocess/tests/test_commodity_main_contract.py`, update the summary builder coverage so a contract with source files before its first selected month does not retain those earlier files.

Create a focused test shaped like:

```python
def test_build_main_contract_summary_starts_contract_on_first_selected_month(tmp_path):
    raw_root = tmp_path / "data" / "原始下载"
    _write_contract_file(raw_root, "燃料油", "2026", "01", "20260105", "fu2601", "20260105", [0, 10])
    _write_contract_file(raw_root, "燃料油", "2026", "02", "20260202", "fu2601", "20260202", [0, 200])
    for day in range(3, 16):
        trading_day = f"202602{day:02d}"
        _write_contract_file(raw_root, "燃料油", "2026", "02", trading_day, "fu2601", trading_day, [0, 100])
    _write_contract_file(raw_root, "燃料油", "2026", "01", "20260105", "fu2602", "20260105", [0, 300])
    _write_contract_file(raw_root, "燃料油", "2026", "01", "20260105", "fu2603", "20260105", [0, 250])

    summary = build_main_contract_summary_for_date_range(
        raw_root, "燃料油", "2026-01-01", "2026-03-01", "fu"
    )

    contract = {item["contract"]: item for item in summary["contracts"]}["fu2601"]
    assert contract["selected_months"] == ["2026-02"]
    assert contract["start_trading_day"] == "20260202"
    assert all(day["trading_day"] >= "20260201" for day in contract["trading_days"])
```

Keep the data minimal but ensure `fu2601` has at least 11 in-range trading days after its first selected month start, so the end clipping rule still leaves retained days.

- [x] **Step 2: Add failing tests for date-range last-trading-day minus 10 clipping**

Add a test where a selected contract has an ordered in-range trading-day sequence and assert the summary excludes the final 10 in-range contract trading days:

```python
def test_build_main_contract_summary_ends_contract_ten_trading_days_before_last_raw_day(tmp_path):
    raw_root = tmp_path / "data" / "原始下载"
    for offset in range(1, 16):
        trading_day = f"202601{offset:02d}"
        _write_contract_file(raw_root, "燃料油", "2026", "01", trading_day, "fu2601", trading_day, [0, 100 + offset])

    summary = build_main_contract_summary_for_date_range(
        raw_root, "燃料油", "2026-01-01", "2026-02-01", "fu"
    )

    contract = summary["contracts"][0]
    assert contract["end_trading_day"] == "20260105"
    assert [day["trading_day"] for day in contract["trading_days"]] == [
        "20260101",
        "20260102",
        "20260103",
        "20260104",
        "20260105",
    ]
```

The expected end is the retained cutoff where the last trading day in the requested date range is `20260115` and the final 10 in-range trading days `20260106` through `20260115` are excluded.

- [x] **Step 3: Add failing empty-window fail-fast test**

Add a test where a selected contract has 10 or fewer raw trading days after its first selected month start and assert summary generation fails:

```python
with pytest.raises(ValueError, match="No retained trading days"):
    build_main_contract_summary_for_date_range(...)
```

The error should identify the contract so users know which selected contract cannot produce a valid clipped summary.

- [x] **Step 4: Run the new focused tests to verify failure**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py -q
```

Expected: FAIL because summary construction still retains all source days for selected contracts.

- [x] **Step 5: Implement contract trading-window clipping**

In `data_preprocess/operator_futures/commodity/main_contract.py`, keep monthly volume selection unchanged. After `selected_months_by_contract` is computed and before constructing `MainContractSummaryContract`, filter `contract_days[contract]`:

```python
first_selected_month = min(selected_months_by_contract[contract])
window_start = datetime.strptime(f"{first_selected_month}-01", "%Y-%m-%d").date()
raw_days = sorted(contract_days[contract], key=lambda item: item.trading_day)
if len(raw_days) <= 10:
    raise ValueError(f"No retained trading days for contract {contract}: fewer than 11 raw trading days")
end_cutoff = date_range_days[-11].trading_day
retained_days = [
    day
    for day in raw_days
    if _parse_date(day.date) >= window_start and day.trading_day <= end_cutoff
]
if not retained_days:
    raise ValueError(f"No retained trading days for contract {contract} after window clipping")
```

Use `retained_days` as `trading_days` for `MainContractSummaryContract`. Preserve existing `daily_volume`, `source_file`, selected month sorting, and deterministic contract ordering.

- [x] **Step 6: Update affected existing expectations and fixtures**

Update existing summary builder assertions in `data_preprocess/tests/test_commodity_main_contract.py` to reflect the clipped window. If a test's selected contract has too few raw trading days, add enough raw fixture days or narrow the assertion to the behavior under test.

Do not change downscale behavior: it should continue reading every `trading_days[]` entry from summary. Only the summary content changes.

- [x] **Step 7: Run Task 11 verification**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py data_preprocess/tests/test_commodity_downscale.py -q
```

Expected: PASS.

- [x] **Step 8: Update tracking checkboxes after implementation**

When Task 11 verification passes, update only these tracking checkboxes:

```markdown
docs/superpowers/plans/2026-07-12-split-commodity-main-contracts-by-contract.md
- [x] **Task complete**

openspec/changes/split-commodity-main-contracts-by-contract/tasks.md
- [x] 11.0 Contract trading-window clipping complete（与 plan-ready.md Task 11 和 superpowers plan Task 11 同步）

openspec/changes/split-commodity-main-contracts-by-contract/plan-ready.md
- [x] **任务完成**（与 superpowers plan `Task 11`、`tasks.md` 对应条目同步勾选）
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 12: Contract trading-window verification

> **trace:** plan-ready.md -> `### Task 12: Contract trading-window verification` | tasks.md -> `- [ ] 12.0 Contract trading-window verification complete（与 plan-ready.md Task 12 和 superpowers plan Task 12 同步）`
> **sync:** tasks.md -> `- [ ] 12.0 Contract trading-window verification complete（与 plan-ready.md Task 12 和 superpowers plan Task 12 同步）` | plan-ready.md -> `### Task 12: Contract trading-window verification`

**Files:**
- Modify: `openspec/changes/split-commodity-main-contracts-by-contract/tasks.md`
- Modify: `openspec/changes/split-commodity-main-contracts-by-contract/plan-ready.md`
- Modify: `docs/superpowers/plans/2026-07-12-split-commodity-main-contracts-by-contract.md`

- [x] **Step 1: Run focused pytest suite**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py data_preprocess/tests/test_commodity_downscale.py -q
```

Expected: PASS.

- [x] **Step 2: Run shell syntax checks**

Run:

```bash
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/main_fu.sh
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/main_al.sh
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/commodity_process.sh
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/flatten_aluminum_raw_csv.sh
```

Expected: all commands exit 0.

- [x] **Step 3: Run OpenSpec and diff checks**

Run:

```bash
openspec validate split-commodity-main-contracts-by-contract --strict
git diff --check
```

Expected: OpenSpec reports the change is valid; `git diff --check` prints no whitespace errors.

- [x] **Step 4: Update tracking checkboxes after verification**

When all verification commands pass, update only these tracking checkboxes:

```markdown
docs/superpowers/plans/2026-07-12-split-commodity-main-contracts-by-contract.md
- [x] **Task complete**

openspec/changes/split-commodity-main-contracts-by-contract/tasks.md
- [x] 12.0 Contract trading-window verification complete（与 plan-ready.md Task 12 和 superpowers plan Task 12 同步）

openspec/changes/split-commodity-main-contracts-by-contract/plan-ready.md
- [x] **任务完成**（与 superpowers plan `Task 12`、`tasks.md` 对应条目同步勾选）
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 13: High-volume-day main contract rule

> **trace:** plan-ready.md -> `### Task 13: High-volume-day main contract rule` | tasks.md -> `- [ ] 13.0 High-volume-day main contract rule complete（与 plan-ready.md Task 13 和 superpowers plan Task 13 同步）`
> **sync:** tasks.md -> `- [ ] 13.0 High-volume-day main contract rule complete（与 plan-ready.md Task 13 和 superpowers plan Task 13 同步）` | plan-ready.md -> `### Task 13: High-volume-day main contract rule`

**Files:**
- Modify: `data_preprocess/operator_futures/commodity/config.py`
- Modify: `data_preprocess/operator_futures/commodity/main_contract.py`
- Modify: `data_preprocess/tests/test_commodity_main_contract.py`
- Modify if needed: `data_preprocess/tests/test_commodity_main_contract_cli.py`

- [x] **Step 1: Add failing config threshold test**

In `data_preprocess/tests/test_commodity_main_contract.py`, import `get_commodity_config` if it is not already imported and add:

```python
def test_fu_config_defines_main_contract_daily_volume_threshold():
    config = get_commodity_config("fu")

    assert config.main_contract_daily_volume_threshold == 15000
```

Run this single test and confirm it fails because the config field does not exist yet.

- [x] **Step 2: Add failing high-volume-day selection test**

Add a focused test proving a contract outside monthly top 2 is still selected when it has 10 high-volume days:

```python
def test_build_main_contract_summary_selects_contract_with_ten_high_volume_days(tmp_path):
    raw_root = tmp_path / "data" / "原始下载"
    for day in range(1, 13):
        trading_day = f"202601{day:02d}"
        _write_contract_file(raw_root, "燃料油", "2026", "01", trading_day, "fu2601", trading_day, [0, 30000])
        _write_contract_file(raw_root, "燃料油", "2026", "01", trading_day, "fu2602", trading_day, [0, 25000])
        volume = 15001 if day <= 10 else 100
        _write_contract_file(raw_root, "燃料油", "2026", "01", trading_day, "fu2603", trading_day, [0, volume])

    summary = build_main_contract_summary_for_date_range(
        raw_root, "燃料油", "2026-01-01", "2026-02-01", "fu"
    )

    contracts = {item["contract"]: item for item in summary["contracts"]}
    assert list(contracts) == ["fu2601", "fu2602", "fu2603"]
    assert contracts["fu2603"]["selected_months"] == ["2026-01"]
```

The fixture keeps `fu2603` outside monthly top 2 by monthly sum but above the configured `15000` threshold for exactly 10 days.

- [x] **Step 3: Add failing strict-greater-than test**

Add a test proving `daily_volume == threshold` does not count:

```python
def test_build_main_contract_summary_high_volume_rule_requires_strictly_greater_than_threshold(tmp_path):
    raw_root = tmp_path / "data" / "原始下载"
    for day in range(1, 13):
        trading_day = f"202601{day:02d}"
        _write_contract_file(raw_root, "燃料油", "2026", "01", trading_day, "fu2601", trading_day, [0, 30000])
        _write_contract_file(raw_root, "燃料油", "2026", "01", trading_day, "fu2602", trading_day, [0, 25000])
        _write_contract_file(raw_root, "燃料油", "2026", "01", trading_day, "fu2603", trading_day, [0, 15000])

    summary = build_main_contract_summary_for_date_range(
        raw_root, "燃料油", "2026-01-01", "2026-02-01", "fu"
    )

    assert [item["contract"] for item in summary["contracts"]] == ["fu2601", "fu2602"]
```

Run the new tests and confirm they fail for the expected missing rule/config reason.

- [x] **Step 4: Implement config threshold**

In `data_preprocess/operator_futures/commodity/config.py`, add a field to the commodity config dataclass:

```python
main_contract_daily_volume_threshold: float | None = None
```

Set the `fu` config value to `15000`. Keep other symbols at `None` unless they already have a known threshold.

- [x] **Step 5: Implement high-volume-day union selection**

In `data_preprocess/operator_futures/commodity/main_contract.py`, keep the existing monthly top-2 selection. Add monthly high-volume day counts while scanning eligible contract days:

```python
threshold = get_commodity_config(symbol).main_contract_daily_volume_threshold
if threshold is not None and daily_volume > threshold:
    monthly_high_volume_days.setdefault(month, {}).setdefault(contract, 0)
    monthly_high_volume_days[month][contract] += 1
```

After selecting monthly top 2, union contracts whose monthly high-volume count is at least 10:

```python
for contract, count in monthly_high_volume_days.get(month, {}).items():
    if count >= 10:
        selected_months_by_contract.setdefault(contract, set()).add(month)
```

Do not require the 10 days to be consecutive. Preserve deterministic contract ordering and the existing clipped trading-window behavior.

Update the summary `selection_rule` value to a combined-rule name such as:

```python
"monthly_top_2_or_10_days_above_configured_daily_volume_threshold"
```

Update existing tests that assert the old `selection_rule` string to the combined-rule string.

- [x] **Step 6: Run Task 13 verification**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py data_preprocess/tests/test_commodity_downscale.py -q
```

Expected: PASS.

- [x] **Step 7: Update tracking checkboxes after implementation**

When Task 13 verification passes, update only these tracking checkboxes:

```markdown
docs/superpowers/plans/2026-07-12-split-commodity-main-contracts-by-contract.md
- [x] **Task complete**

openspec/changes/split-commodity-main-contracts-by-contract/tasks.md
- [x] 13.0 High-volume-day main contract rule complete（与 plan-ready.md Task 13 和 superpowers plan Task 13 同步）

openspec/changes/split-commodity-main-contracts-by-contract/plan-ready.md
- [x] **任务完成**（与 superpowers plan `Task 13`、`tasks.md` 对应条目同步勾选）
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 14: High-volume-day verification

> **trace:** plan-ready.md -> `### Task 14: High-volume-day verification` | tasks.md -> `- [ ] 14.0 High-volume-day verification complete（与 plan-ready.md Task 14 和 superpowers plan Task 14 同步）`
> **sync:** tasks.md -> `- [ ] 14.0 High-volume-day verification complete（与 plan-ready.md Task 14 和 superpowers plan Task 14 同步）` | plan-ready.md -> `### Task 14: High-volume-day verification`

**Files:**
- Modify: `openspec/changes/split-commodity-main-contracts-by-contract/tasks.md`
- Modify: `openspec/changes/split-commodity-main-contracts-by-contract/plan-ready.md`
- Modify: `docs/superpowers/plans/2026-07-12-split-commodity-main-contracts-by-contract.md`

- [x] **Step 1: Run focused pytest suite**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py data_preprocess/tests/test_commodity_downscale.py -q
```

Expected: PASS.

- [x] **Step 2: Run shell syntax checks**

Run:

```bash
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/main_fu.sh
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/main_al.sh
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/commodity_process.sh
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/flatten_aluminum_raw_csv.sh
```

Expected: all commands exit 0.

- [x] **Step 3: Run OpenSpec and diff checks**

Run:

```bash
openspec validate split-commodity-main-contracts-by-contract --strict
git diff --check
```

Expected: OpenSpec reports the change is valid; `git diff --check` prints no whitespace errors.

- [x] **Step 4: Update tracking checkboxes after verification**

When all verification commands pass, update only these tracking checkboxes:

```markdown
docs/superpowers/plans/2026-07-12-split-commodity-main-contracts-by-contract.md
- [x] **Task complete**

openspec/changes/split-commodity-main-contracts-by-contract/tasks.md
- [x] 14.0 High-volume-day verification complete（与 plan-ready.md Task 14 和 superpowers plan Task 14 同步）

openspec/changes/split-commodity-main-contracts-by-contract/plan-ready.md
- [x] **任务完成**（与 superpowers plan `Task 14`、`tasks.md` 对应条目同步勾选）
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 15: Cross-contract training feature union

> **trace:** plan-ready.md -> `### Task 15: Cross-contract training feature union` | tasks.md -> `- [ ] 15.0 Cross-contract training feature union complete（与 plan-ready.md Task 15 和 superpowers plan Task 15 同步）`
> **sync:** tasks.md -> `- [ ] 15.0 Cross-contract training feature union complete（与 plan-ready.md Task 15 和 superpowers plan Task 15 同步）` | plan-ready.md -> `### Task 15: Cross-contract training feature union`

**Files:**
- Create: `data_preprocess/operator_futures/feature_selection/contract_feature_union.py`
- Modify: `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`
- Modify: `data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh`
- Modify: `data_preprocess/tests/test_commodity_feature_pipeline.py`
- Modify: `data_preprocess/tests/test_commodity_main_contract_cli.py`
- Modify: `docs/上海商品交易所/commodity_futures_preprocess.md`

- [x] **Step 1: Add failing feature-union unit tests**

In `data_preprocess/tests/test_commodity_feature_pipeline.py`, add tests for stable first-seen union and missing feature artifacts:

```python
import json
import numpy as np
import pytest

from operator_futures.feature_selection.contract_feature_union import (
    build_union_state_features,
    write_contract_feature_union,
)


def test_build_union_state_features_preserves_first_seen_order():
    result = build_union_state_features(
        [
            ["alpha", "beta", "alpha"],
            ["beta", "gamma"],
            ["delta", "alpha"],
        ]
    )

    assert result == ["alpha", "beta", "gamma", "delta"]


def test_write_contract_feature_union_writes_symbol_level_manifest(tmp_path):
    summary_path = tmp_path / "main_contract_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "symbol": "fu",
                "commodity_name": "燃料油",
                "start_date": "2026-01-01",
                "end_date": "2026-04-01",
                "contracts": [
                    {
                        "contract": "fu2601",
                        "start_trading_day": "20260101",
                        "end_trading_day": "20260102",
                        "trading_day_count": 1,
                        "selected_months": ["2026-01"],
                        "trading_days": [
                            {
                                "trading_day": "20260101",
                                "date": "2026-01-01",
                                "source_file": "fu2601.csv",
                                "daily_volume": 1.0,
                            }
                        ],
                    },
                    {
                        "contract": "fu2605",
                        "start_trading_day": "20260201",
                        "end_trading_day": "20260202",
                        "trading_day_count": 1,
                        "selected_months": ["2026-02"],
                        "trading_days": [
                            {
                                "trading_day": "20260201",
                                "date": "2026-02-01",
                                "source_file": "fu2605.csv",
                                "daily_volume": 1.0,
                            }
                        ],
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    base = tmp_path / "PREPROCESS_DATASET" / "commodity-futures"
    first = base / "SCALE_SAVE" / "fu" / "fu2601" / "5min" / "2026-01-01-2026-04-01"
    second = base / "SCALE_SAVE" / "fu" / "fu2605" / "5min" / "2026-01-01-2026-04-01"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    np.save(first / "state_features.npy", np.array(["alpha", "beta"]))
    np.save(second / "state_features.npy", np.array(["beta", "gamma"]))

    output_dir = write_contract_feature_union(
        root_path=tmp_path,
        summary_path=summary_path,
        symbol="fu",
        target_freq="5min",
        start_date="2026-01-01",
        end_date="2026-04-01",
    )

    assert output_dir == base / "FEATURE_UNION" / "fu" / "5min" / "2026-01-01-2026-04-01"
    assert np.load(output_dir / "state_features.npy", allow_pickle=True).tolist() == [
        "alpha",
        "beta",
        "gamma",
    ]
    manifest = json.loads((output_dir / "feature_union_manifest.json").read_text(encoding="utf-8"))
    assert manifest["contracts"] == ["fu2601", "fu2605"]
    assert manifest["state_features"] == ["alpha", "beta", "gamma"]
    assert manifest["state_feature_count"] == 3


def test_write_contract_feature_union_fails_when_contract_state_features_missing(tmp_path):
    summary_path = tmp_path / "main_contract_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "symbol": "fu",
                "commodity_name": "燃料油",
                "start_date": "2026-01-01",
                "end_date": "2026-04-01",
                "contracts": [
                    {
                        "contract": "fu2605",
                        "start_trading_day": "20260201",
                        "end_trading_day": "20260202",
                        "trading_day_count": 1,
                        "selected_months": ["2026-02"],
                        "trading_days": [
                            {
                                "trading_day": "20260201",
                                "date": "2026-02-01",
                                "source_file": "fu2605.csv",
                                "daily_volume": 1.0,
                            }
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError) as excinfo:
        write_contract_feature_union(
            root_path=tmp_path,
            summary_path=summary_path,
            symbol="fu",
            target_freq="5min",
            start_date="2026-01-01",
            end_date="2026-04-01",
        )

    message = str(excinfo.value)
    assert "fu2605" in message
    assert "state_features.npy" in message
```

- [x] **Step 2: Run feature-union tests to verify RED**

Run:

```bash
conda run -n finetf pytest \
  data_preprocess/tests/test_commodity_feature_pipeline.py::test_build_union_state_features_preserves_first_seen_order \
  data_preprocess/tests/test_commodity_feature_pipeline.py::test_write_contract_feature_union_writes_symbol_level_manifest \
  data_preprocess/tests/test_commodity_feature_pipeline.py::test_write_contract_feature_union_fails_when_contract_state_features_missing \
  -q
```

Expected: FAIL because `operator_futures.feature_selection.contract_feature_union` does not exist yet.

- [x] **Step 3: Implement the feature-union module**

Create `data_preprocess/operator_futures/feature_selection/contract_feature_union.py` with these public functions and CLI:

```python
import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from operator_futures.commodity.main_contract import load_main_contract_summary


logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def build_union_state_features(feature_lists: Iterable[Sequence[str]]) -> list[str]:
    seen: set[str] = set()
    union: list[str] = []
    for features in feature_lists:
        for feature in features:
            name = str(feature)
            if name in seen:
                continue
            seen.add(name)
            union.append(name)
    return union


def _load_state_features(path: Path, contract: str) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing state_features.npy for contract {contract}: {path}"
        )
    return [str(item) for item in np.load(path, allow_pickle=True).tolist()]


def write_contract_feature_union(
    root_path: Path,
    summary_path: Path,
    symbol: str,
    target_freq: str,
    start_date: str,
    end_date: str,
    scale_save_path: str = "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE",
    save_path: str = "PREPROCESS_DATASET/commodity-futures/FEATURE_UNION",
) -> Path:
    summary = load_main_contract_summary(summary_path)
    date_range = f"{start_date}-{end_date}"
    scale_root = root_path / scale_save_path
    output_dir = root_path / save_path / symbol / target_freq / date_range

    contract_features: dict[str, list[str]] = {}
    contract_feature_paths: dict[str, str] = {}
    for contract in summary.contracts:
        feature_path = (
            scale_root
            / symbol
            / contract.contract
            / target_freq
            / date_range
            / "state_features.npy"
        )
        contract_features[contract.contract] = _load_state_features(
            feature_path, contract.contract
        )
        contract_feature_paths[contract.contract] = str(feature_path)

    union = build_union_state_features(contract_features.values())
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "state_features.npy", np.array(union))
    manifest = {
        "symbol": symbol,
        "target_freq": target_freq,
        "start_date": start_date,
        "end_date": end_date,
        "summary_path": str(summary_path),
        "contracts": list(contract_features),
        "contract_state_feature_paths": contract_feature_paths,
        "per_contract_feature_counts": {
            contract: len(features)
            for contract, features in contract_features.items()
        },
        "state_feature_count": len(union),
        "state_features": union,
    }
    (output_dir / "feature_union_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Wrote contract feature union: symbol=%s contracts=%d state_features=%d output_dir=%s",
        symbol,
        len(contract_features),
        len(union),
        output_dir,
    )
    return output_dir


parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=Path, default=Path("."))
parser.add_argument("--summary", type=Path, required=True)
parser.add_argument("--symbols", type=str, required=True)
parser.add_argument("--target_freq", type=str, required=True)
parser.add_argument("--start_date", type=str, required=True)
parser.add_argument("--end_date", type=str, required=True)
parser.add_argument(
    "--scale_save_path",
    type=str,
    default="PREPROCESS_DATASET/commodity-futures/SCALE_SAVE",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="PREPROCESS_DATASET/commodity-futures/FEATURE_UNION",
)


def main(args: argparse.Namespace) -> None:
    write_contract_feature_union(
        root_path=args.root_path,
        summary_path=args.summary,
        symbol=args.symbols,
        target_freq=args.target_freq,
        start_date=args.start_date,
        end_date=args.end_date,
        scale_save_path=args.scale_save_path,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    configure_logging()
    main(parser.parse_args())
```

- [x] **Step 4: Add failing shell orchestration tests**

In `data_preprocess/tests/test_commodity_main_contract_cli.py`, extend the full-process shell stub test so `run_commodity_feature_union` is called after `scale_save`. Add a validation-source assertion that `validate_features.sh` checks `FEATURE_UNION` outputs.

Use this shape in the shell stub:

```bash
run_commodity_feature_union() {
  local summary_path=$1
  local target_freq=$2
  local start_date=$3
  local end_date=$4
  local symbol=$5
  echo "feature_union:${symbol}:${target_freq}:${start_date}:${end_date}:${summary_path}"
}
```

Assert that the captured output contains `feature_union:fu:5min:2026-01-01:2026-02-01:` after the per-contract `scale_save` output.

Add a static text assertion:

```python
def test_validate_features_checks_feature_union_outputs():
    text = Path(
        "data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh"
    ).read_text(encoding="utf-8")

    assert "FEATURE_UNION" in text
    assert "feature_union_manifest.json" in text
    assert "state_features.npy" in text
```

Run the new shell tests and confirm they fail before the shell scripts are updated.

- [x] **Step 5: Update commodity shell scripts**

In `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`, add a function:

```bash
run_commodity_feature_union() {
    local summary_path=$1
    local target_freq=$2
    local start_date=$3
    local end_date=$4
    local symbol=$5
    local root_path=$6

    PYTHONPATH="${root_path}/data_preprocess" python -u -m operator_futures.feature_selection.contract_feature_union \
        --summary "${summary_path}" \
        --symbols "${symbol}" \
        --target_freq "${target_freq}" \
        --start_date "${start_date}" \
        --end_date "${end_date}" \
        --root_path "${root_path}"
}
```

Call it after the contract loop and before `maintenance_margin_dict`:

```bash
    run_commodity_logged_step \
        "$log_dir" "$symbol" "$target_freq" "$start_date" "$end_date" \
        "feature_union" \
        run_commodity_feature_union "$summary_path" "$target_freq" "$start_date" "$end_date" "$symbol" "$root_path"
```

In `data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh`, check both union files:

```bash
feature_union_dir="${ROOTPATH}/PREPROCESS_DATASET/commodity-futures/FEATURE_UNION/${SYMBOL}/${TARGET_FREQ}/${START_DATE}-${END_DATE}"
if [ ! -f "${feature_union_dir}/state_features.npy" ]; then
    echo "Missing commodity feature union state_features.npy: path=${feature_union_dir}/state_features.npy" >&2
    exit 1
fi
if [ ! -f "${feature_union_dir}/feature_union_manifest.json" ]; then
    echo "Missing commodity feature union manifest: path=${feature_union_dir}/feature_union_manifest.json" >&2
    exit 1
fi
echo "Validated commodity feature union: symbol=${SYMBOL} path=${feature_union_dir}"
```

- [x] **Step 6: Update commodity preprocessing docs**

In `docs/上海商品交易所/commodity_futures_preprocess.md`, add the final feature-union output to the output layout section:

```markdown
PREPROCESS_DATASET/commodity-futures/FEATURE_UNION/fu/5min/2026-01-01-2026-02-01/state_features.npy
PREPROCESS_DATASET/commodity-futures/FEATURE_UNION/fu/5min/2026-01-01-2026-02-01/feature_union_manifest.json
```

Add one sentence that this state-feature list is the stable union of all selected contracts and is intended for downstream single-model training.

- [x] **Step 7: Run Task 15 verification**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_commodity_feature_pipeline.py data_preprocess/tests/test_commodity_main_contract_cli.py -q
```

Expected: PASS.

- [x] **Step 8: Update tracking checkboxes after implementation**

When Task 15 verification passes, update only these tracking checkboxes:

```markdown
docs/superpowers/plans/2026-07-12-split-commodity-main-contracts-by-contract.md
- [x] **Task complete**

openspec/changes/split-commodity-main-contracts-by-contract/tasks.md
- [x] 15.0 Cross-contract training feature union complete（与 plan-ready.md Task 15 和 superpowers plan Task 15 同步）

openspec/changes/split-commodity-main-contracts-by-contract/plan-ready.md
- [x] **任务完成**（与 superpowers plan `Task 15`、`tasks.md` 对应条目同步勾选）
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 16: Feature-union verification

> **trace:** plan-ready.md -> `### Task 16: Feature-union verification` | tasks.md -> `- [ ] 16.0 Feature-union verification complete（与 plan-ready.md Task 16 和 superpowers plan Task 16 同步）`
> **sync:** tasks.md -> `- [ ] 16.0 Feature-union verification complete（与 plan-ready.md Task 16 和 superpowers plan Task 16 同步）` | plan-ready.md -> `### Task 16: Feature-union verification`

**Files:**
- Modify: `openspec/changes/split-commodity-main-contracts-by-contract/tasks.md`
- Modify: `openspec/changes/split-commodity-main-contracts-by-contract/plan-ready.md`
- Modify: `docs/superpowers/plans/2026-07-12-split-commodity-main-contracts-by-contract.md`

- [x] **Step 1: Run focused pytest suite**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_commodity_feature_pipeline.py data_preprocess/tests/test_commodity_main_contract_cli.py -q
```

Expected: PASS.

- [x] **Step 2: Run shell syntax checks**

Run:

```bash
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/main_fu.sh
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/main_al.sh
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/commodity_process.sh
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/flatten_aluminum_raw_csv.sh
```

Expected: all commands exit 0.

- [x] **Step 3: Run OpenSpec and diff checks**

Run:

```bash
openspec validate split-commodity-main-contracts-by-contract --strict
git diff --check
```

Expected: OpenSpec reports the change is valid; `git diff --check` prints no whitespace errors.

- [x] **Step 4: Update tracking checkboxes after verification**

When all verification commands pass, update only these tracking checkboxes:

```markdown
docs/superpowers/plans/2026-07-12-split-commodity-main-contracts-by-contract.md
- [x] **Task complete**

openspec/changes/split-commodity-main-contracts-by-contract/tasks.md
- [x] 16.0 Feature-union verification complete（与 plan-ready.md Task 16 和 superpowers plan Task 16 同步）

openspec/changes/split-commodity-main-contracts-by-contract/plan-ready.md
- [x] **任务完成**（与 superpowers plan `Task 16`、`tasks.md` 对应条目同步勾选）
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

## Self-Review

Spec coverage:
- Main-contract JSON summary, monthly top-2 volume rule, `trading_day_count`, source file listing, fail-fast cases: Task 1.
- Summary-driven downscale, `--summary`, optional `--contract`, source-file validation, contract-scoped downscale outputs: Task 2.
- Contract-scoped downstream Python paths and legacy no-contract behavior: Task 3.
- Commodity shell scripts, validation, docs, summary parsing, contract logs and skip checks: Task 4.
- Summary model bean serialization and builder refactor: Task 5.
- Required pytest, shell syntax, OpenSpec, and diff checks: Task 6.
- Summary model bean deserialization and model-based readers: Task 7.
- Post-deserialization pytest, shell syntax, OpenSpec, and diff checks: Task 8.
- Summary trading-day `daily_volume` schema, model serialization/deserialization, builder persistence, and fixture updates: Task 9.
- Daily-volume pytest, shell syntax, OpenSpec, and diff checks: Task 10.
- Contract trading-window clipping by first selected month and 10 trading days before the last trading day in the requested date range: Task 11.
- Contract trading-window pytest, shell syntax, OpenSpec, and diff checks: Task 12.
- High-volume-day configured main-contract selection rule: Task 13.
- High-volume-day pytest, shell syntax, OpenSpec, and diff checks: Task 14.
- Cross-contract training feature union generation, shell orchestration, validation, and docs: Task 15.
- Feature-union pytest, shell syntax, OpenSpec, and diff checks: Task 16.

Placeholder scan:
- No placeholder markers are present.
- Each code-changing step includes concrete code snippets or exact replacement patterns.
- Each verification step includes exact commands and expected results.

Type consistency:
- `contract` is a string CLI option across Python and shell.
- Summary field names match the OpenSpec delta: `contract`, `start_trading_day`, `end_trading_day`, `trading_day_count`, `selected_months`, `trading_days`, `source_file`.
- Summary trading-day field names match the OpenSpec delta: `trading_day`, `date`, `source_file`, `daily_volume`.
- Contract summary dates use retained trading days after clipping, while the last trading day in the requested date range is used to compute the inclusive cutoff before the final 10 in-range trading days.
- High-volume-day selection uses configured thresholds, `fu=15000`, strict `daily_volume > threshold`, and any 10 actual trading days in a month.
- Path helper signature is consistently `symbol_contract_path_parts(symbol: str, contract: str | None = None) -> tuple[str, ...]`.
- Feature union output path is consistently `PREPROCESS_DATASET/commodity-futures/FEATURE_UNION/{symbol}/{target_freq}/{start_date}-{end_date}`.
