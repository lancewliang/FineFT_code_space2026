# Split Commodity Continuous Raw By Day Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace commodity `CONTINUOUS_RAW` date-range CSV handoff with daily `TradingDay` CSV files and update stitch, downscale, shell, and docs to consume the new contract.

**Architecture:** Keep the existing commodity module boundaries. `main_contract.py` owns selection and daily CSV writing, `stitch_main_contract.py` exposes the new output-directory CLI, `downscale_continuous_by_trading_day.py` owns date-range directory iteration, and shell/docs only pass the new contract through.

**Tech Stack:** Python 3.10, Polars, pytest, argparse, bash, OpenSpec.

**Traceability (sddflow):**
- plan-ready: `openspec/changes/split-commodity-continuous-raw-by-day/plan-ready.md`
- tasks: `openspec/changes/split-commodity-continuous-raw-by-day/tasks.md`
- plan: `docs/superpowers/plans/2026-06-22-split-commodity-continuous-raw-by-day.md`

---

## File Structure

- Modify `data_preprocess/operator_futures/commodity/main_contract.py`: add date iteration helpers, return selected daily frames, and write daily `YYYY-MM-DD.csv` files with skip/overwrite logging.
- Modify `data_preprocess/operator_futures/commodity/stitch_main_contract.py`: remove old `--output`/`--year` execution path from the active CLI and require `--output_dir --start_date --end_date`.
- Modify `data_preprocess/operator_futures/commodity/downscale_continuous_by_trading_day.py`: replace single-file input with directory/date-range iteration while reusing existing per-day downscale body.
- Modify `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`: pass `CONTINUOUS_RAW/{symbol}` as a directory and remove `continuous_file` handoff.
- Modify `data_preprocess/tests/test_commodity_main_contract.py`: unit coverage for daily frame/write behavior.
- Modify `data_preprocess/tests/test_commodity_main_contract_cli.py`: CLI, shell, and script text coverage for new arguments and old-argument removal.
- Modify `data_preprocess/tests/test_commodity_downscale.py`: keep existing bad-data and output-contract coverage; add only focused assertions if needed by the implementation.
- Modify `docs/上海商品交易所/commodity_futures_preprocess.md`: document daily file naming and new CLI/script usage.

### Task 1: 主力连续化日文件输出

> **trace:** plan-ready.md → `### Task 1: 主力连续化日文件输出` | tasks.md → `- [x] 1.0 主力连续化日文件输出完成（与 plan-ready.md Task 1 和 superpowers plan Task 1 同步）`
> **sync:** tasks.md → `- [x] 1.0 主力连续化日文件输出完成（与 plan-ready.md Task 1 和 superpowers plan Task 1 同步）` | plan-ready.md → `### Task 1: 主力连续化日文件输出`

**Files:**
- Modify: `data_preprocess/operator_futures/commodity/main_contract.py`
- Modify: `data_preprocess/operator_futures/commodity/stitch_main_contract.py`
- Test: `data_preprocess/tests/test_commodity_main_contract.py`
- Test: `data_preprocess/tests/test_commodity_main_contract_cli.py`

- [x] **Step 1: Write failing unit tests for daily main-contract output**

Add imports in `data_preprocess/tests/test_commodity_main_contract.py`:

```python
from operator_futures.commodity.main_contract import (
    build_main_contract_continuous_frame_for_date_range,
    build_main_contract_daily_frames_for_date_range,
    calculate_contract_volume,
    infer_years_for_date_range,
    iter_contract_files,
    normalize_timestamp,
    select_main_contract_for_day,
    stitch_main_contract_frames,
    write_main_contract_daily_files_for_date_range,
)
```

Append these tests:

```python
def test_build_main_contract_daily_frames_skips_missing_calendar_dates(tmp_path, caplog):
    raw_root = tmp_path / "data" / "原始下载"
    _write_contract_file(
        raw_root, "燃料油", "2026", "01", "20260105", "fu2602", "20260104", [0, 30]
    )
    _write_contract_file(
        raw_root, "燃料油", "2026", "01", "20260107", "fu2602", "20260106", [0, 4]
    )

    with caplog.at_level(logging.INFO, logger="operator_futures.commodity.main_contract"):
        daily = build_main_contract_daily_frames_for_date_range(
            raw_root,
            "燃料油",
            "2026-01-05",
            "2026-01-08",
            "fu",
        )

    assert sorted(daily) == ["2026-01-05", "2026-01-07"]
    assert daily["2026-01-05"]["main_contract_trading_day"].cast(pl.Utf8).to_list() == [
        "20260105",
        "20260105",
    ]
    assert "Skipped commodity main-contract source dates" in caplog.text
    assert "2026-01-06" in caplog.text


def test_write_main_contract_daily_files_overwrites_existing_file(tmp_path, caplog):
    raw_root = tmp_path / "data" / "原始下载"
    output_dir = tmp_path / "PREPROCESS_DATASET" / "commodity-futures" / "CONTINUOUS_RAW" / "fu"
    _write_contract_file(
        raw_root, "燃料油", "2026", "01", "20260105", "fu2602", "20260104", [0, 30]
    )
    output_dir.mkdir(parents=True)
    existing = output_dir / "2026-01-05.csv"
    existing.write_text("old\n", encoding="utf-8")

    with caplog.at_level(logging.INFO, logger="operator_futures.commodity.main_contract"):
        written = write_main_contract_daily_files_for_date_range(
            raw_root=raw_root,
            commodity_name="燃料油",
            output_dir=output_dir,
            start_date="2026-01-05",
            end_date="2026-01-06",
            symbol="fu",
        )

    assert written == [existing]
    result = pl.read_csv(existing)
    assert result["main_contract"].to_list() == ["fu2602", "fu2602"]
    assert "Overwriting commodity main-contract daily file" in caplog.text
    assert str(existing) in caplog.text
```

- [x] **Step 2: Run tests and verify they fail for missing functions**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py::test_build_main_contract_daily_frames_skips_missing_calendar_dates data_preprocess/tests/test_commodity_main_contract.py::test_write_main_contract_daily_files_overwrites_existing_file -q
```

Expected: FAIL with `ImportError` or `AttributeError` for `build_main_contract_daily_frames_for_date_range` / `write_main_contract_daily_files_for_date_range`.

- [x] **Step 3: Implement daily frame and daily file helpers**

In `data_preprocess/operator_futures/commodity/main_contract.py`, add helpers after `_trading_day_in_range`:

```python
def _format_trading_day_file_date(trading_day: str) -> str:
    return datetime.strptime(trading_day, "%Y%m%d").date().isoformat()


def _iter_iso_dates(start_date: str, end_date: str) -> Iterable[str]:
    current = _parse_date(start_date)
    end = _parse_date(end_date)
    if end <= current:
        raise ValueError(
            f"end_date must be greater than start_date for left-open range: "
            f"{start_date} -> {end_date}"
        )
    while current < end:
        yield current.isoformat()
        current += timedelta(days=1)


def build_main_contract_daily_frames_for_date_range(
    raw_root: Path,
    commodity_name: str,
    start_date: str,
    end_date: str,
    symbol: str,
) -> Dict[str, pl.DataFrame]:
    years = infer_years_for_date_range(start_date, end_date)
    logger.info(
        "Building commodity main-contract daily files: symbol=%s commodity=%s start_date=%s end_date=%s years=%s",
        symbol,
        commodity_name,
        start_date,
        end_date,
        ",".join(years),
    )
    days = load_contract_frames_by_trading_day_for_years(raw_root, commodity_name, years)
    daily: Dict[str, pl.DataFrame] = {}
    selected_dates = set()
    previous_frames: Dict[str, pl.DataFrame] = {}

    for trading_day in sorted(days):
        current_items = days[trading_day]
        current_frames = {
            contract: frame for contract, (frame, _) in current_items.items()
        }
        if not _trading_day_in_range(trading_day, start_date, end_date):
            previous_frames = current_frames
            continue

        contract, reason = select_main_contract_for_day(
            previous_frames, current_frames, symbol
        )
        frame, source_file = current_items[contract]
        _log_selected_main_contract_file(
            trading_day,
            contract,
            reason,
            source_file,
            previous_frames,
            current_frames,
        )
        copied = frame.with_columns(
            pl.lit(reason).alias("main_contract_selection_reason")
        )
        daily[_format_trading_day_file_date(trading_day)] = stitch_main_contract_frames(
            [(trading_day, contract, copied, source_file)]
        )
        selected_dates.add(_format_trading_day_file_date(trading_day))
        previous_frames = current_frames

    skipped = [date for date in _iter_iso_dates(start_date, end_date) if date not in selected_dates]
    if skipped:
        logger.info("Skipped commodity main-contract source dates: dates=%s", ",".join(skipped))
    if not daily:
        raise ValueError("No selected main-contract frames to stitch")
    return daily


def write_main_contract_daily_files_for_date_range(
    raw_root: Path,
    commodity_name: str,
    output_dir: Path,
    start_date: str,
    end_date: str,
    symbol: str,
) -> List[Path]:
    daily = build_main_contract_daily_frames_for_date_range(
        raw_root,
        commodity_name,
        start_date,
        end_date,
        symbol,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for date, frame in sorted(daily.items()):
        path = output_dir / f"{date}.csv"
        if path.exists():
            logger.info("Overwriting commodity main-contract daily file: output=%s", path)
        frame.write_csv(path)
        logger.info(
            "Wrote commodity main-contract daily file: output=%s rows=%d",
            path,
            frame.height,
        )
        written.append(path)
    return written
```

Keep `build_main_contract_continuous_frame_for_date_range` for any internal tests until CLI callers are migrated, but do not use it from the new stitch CLI.

- [x] **Step 4: Run unit tests and verify daily helpers pass**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py -q
```

Expected: PASS for all `test_commodity_main_contract.py` tests.

- [x] **Step 5: Write failing stitch CLI tests for output_dir and old output removal**

In `data_preprocess/tests/test_commodity_main_contract_cli.py`, replace `test_stitch_main_contract_cli_outputs_continuous_file` and `test_stitch_main_contract_cli_accepts_date_range` with:

```python
def test_stitch_main_contract_cli_outputs_daily_files(tmp_path):
    raw_root = tmp_path / "data" / "原始下载"
    _write_contract(
        raw_root / "燃料油" / "2026" / "01" / "20260105" / "fu2602.csv",
        "fu2602",
        "20260105",
        "20260104",
        [0, 30],
    )
    _write_contract(
        raw_root / "燃料油" / "2026" / "01" / "20260106" / "fu2602.csv",
        "fu2602",
        "20260106",
        "20260105",
        [0, 2],
    )
    _write_contract(
        raw_root / "燃料油" / "2026" / "01" / "20260106" / "fu2603.csv",
        "fu2603",
        "20260106",
        "20260105",
        [0, 5],
    )

    output_dir = tmp_path / "continuous" / "fu"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "operator_futures.commodity.stitch_main_contract",
            "--raw_root",
            str(raw_root),
            "--commodity_name",
            "燃料油",
            "--start_date",
            "2026-01-05",
            "--end_date",
            "2026-01-07",
            "--symbol",
            "fu",
            "--output_dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        capture_output=True,
        check=True,
        text=True,
    )

    day1 = pd.read_csv(output_dir / "2026-01-05.csv")
    day2 = pd.read_csv(output_dir / "2026-01-06.csv")
    assert day1["main_contract"].tolist() == ["fu2602", "fu2602"]
    assert day2["main_contract"].tolist() == ["fu2602", "fu2602"]
    assert not (output_dir / "fu_2026-01-05_2026-01-07.csv").exists()
    assert "Starting commodity main-contract daily stitch" in result.stderr
    assert "Wrote stitched commodity main-contract daily files" in result.stderr


def test_stitch_main_contract_cli_rejects_old_output_file_argument(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "operator_futures.commodity.stitch_main_contract",
            "--raw_root",
            str(tmp_path),
            "--commodity_name",
            "燃料油",
            "--year",
            "2026",
            "--symbol",
            "fu",
            "--output",
            str(tmp_path / "fu_2026.csv"),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--output_dir" in result.stderr
```

- [x] **Step 6: Run stitch CLI tests and verify they fail on old parser**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract_cli.py::test_stitch_main_contract_cli_outputs_daily_files data_preprocess/tests/test_commodity_main_contract_cli.py::test_stitch_main_contract_cli_rejects_old_output_file_argument -q
```

Expected: first test FAILS because `--output_dir` is not accepted; second test may FAIL because `--output` is still accepted.

- [x] **Step 7: Update stitch CLI to output daily files**

In `data_preprocess/operator_futures/commodity/stitch_main_contract.py`, replace imports and parser/main with:

```python
import argparse
import logging
from pathlib import Path
import time

from .main_contract import write_main_contract_daily_files_for_date_range


logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stitch local commodity futures contracts into daily main-contract files"
    )
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--commodity_name", required=True)
    parser.add_argument("--start_date", required=True)
    parser.add_argument("--end_date", required=True)
    parser.add_argument("--symbol", default="fu")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    started_at = time.monotonic()
    output_dir = Path(args.output_dir)
    logger.info(
        "Starting commodity main-contract daily stitch: raw_root=%s commodity=%s symbol=%s start_date=%s end_date=%s output_dir=%s",
        args.raw_root,
        args.commodity_name,
        args.symbol,
        args.start_date,
        args.end_date,
        output_dir,
    )
    written = write_main_contract_daily_files_for_date_range(
        raw_root=Path(args.raw_root),
        commodity_name=args.commodity_name,
        output_dir=output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        symbol=args.symbol,
    )
    logger.info(
        "Wrote stitched commodity main-contract daily files: output_dir=%s files=%d elapsed_seconds=%.2f",
        output_dir,
        len(written),
        time.monotonic() - started_at,
    )


if __name__ == "__main__":
    main()
```

- [x] **Step 8: Run Task 1 verification**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py -q
```

Expected: PASS for main-contract and CLI tests, except tests for downscale or shell still expecting old `continuous_file` may fail until Tasks 2 and 3. If failures are only in later-task tests, proceed and fix them in the relevant task.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 2: 连续主力日文件下采样

> **trace:** plan-ready.md → `### Task 2: 连续主力日文件下采样` | tasks.md → `- [x] 2.0 连续主力日文件下采样完成（与 plan-ready.md Task 2 和 superpowers plan Task 2 同步）`
> **sync:** tasks.md → `- [x] 2.0 连续主力日文件下采样完成（与 plan-ready.md Task 2 和 superpowers plan Task 2 同步）` | plan-ready.md → `### Task 2: 连续主力日文件下采样`

**Files:**
- Modify: `data_preprocess/operator_futures/commodity/downscale_continuous_by_trading_day.py`
- Test: `data_preprocess/tests/test_commodity_main_contract_cli.py`
- Test: `data_preprocess/tests/test_commodity_downscale.py`

- [x] **Step 1: Write failing downscale CLI tests for input_dir and missing-day skip**

Add helper functions near `_write_contract` in `data_preprocess/tests/test_commodity_main_contract_cli.py`:

```python
def _write_continuous_day(path: Path, contract: str, trading_day: str, action_day: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, volume in enumerate([0, 1, 2]):
        rows.append(
            {
                "InstrumentID": contract,
                "TradingDay": trading_day,
                "ActionDay": action_day,
                "UpdateTime": f"09:00:0{idx}.000",
                "LastPrice": 2600 + idx,
                "Volume": volume,
                "Turnover": volume * (2600 + idx) * 10,
                "BidPrice1": 2599 + idx,
                "BidVolume1": 10,
                "AskPrice1": 2601 + idx,
                "AskVolume1": 10,
                "BidPrice2": 2598 + idx,
                "BidVolume2": 10,
                "AskPrice2": 2602 + idx,
                "AskVolume2": 10,
                "BidPrice3": 2597 + idx,
                "BidVolume3": 10,
                "AskPrice3": 2603 + idx,
                "AskVolume3": 10,
                "BidPrice4": 2596 + idx,
                "BidVolume4": 10,
                "AskPrice4": 2604 + idx,
                "AskVolume4": 10,
                "BidPrice5": 2595 + idx,
                "BidVolume5": 10,
                "AskPrice5": 2605 + idx,
                "AskVolume5": 10,
                "UpperLimitPrice": 3000,
                "LowerLimitPrice": 2000,
                "main_contract": contract,
                "source_contract": contract,
                "source_file": f"{contract}.csv",
                "main_contract_trading_day": trading_day,
                "main_contract_selection_reason": "current_trading_day_fallback",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
```

Append these tests:

```python
def test_downscale_continuous_cli_reads_daily_input_dir_and_skips_missing_day(tmp_path):
    input_dir = tmp_path / "continuous" / "fu"
    output_root = tmp_path / "PREPROCESS_DATASET" / "commodity-futures"
    _write_continuous_day(input_dir / "2026-01-05.csv", "fu2602", "20260105", "20260105")
    _write_continuous_day(input_dir / "2026-01-07.csv", "fu2602", "20260107", "20260107")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "operator_futures.commodity.downscale_continuous_by_trading_day",
            "--input_dir",
            str(input_dir),
            "--start_date",
            "2026-01-05",
            "--end_date",
            "2026-01-08",
            "--output_root",
            str(output_root),
            "--target_freq",
            "5min",
            "--symbol",
            "fu",
            "--depth",
            "5",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        capture_output=True,
        check=True,
        text=True,
    )

    assert (output_root / "DOWNSCALE_DERTIC" / "fu" / "5min" / "20260105.feather").exists()
    assert not (output_root / "DOWNSCALE_DERTIC" / "fu" / "5min" / "20260106.feather").exists()
    assert (output_root / "DOWNSCALE_DERTIC" / "fu" / "5min" / "20260107.feather").exists()
    assert "Missing commodity continuous daily file" in result.stderr
    assert "2026-01-06" in result.stderr


def test_downscale_continuous_cli_rejects_old_input_file_argument(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "operator_futures.commodity.downscale_continuous_by_trading_day",
            "--input",
            str(tmp_path / "fu_2026.csv"),
            "--output_root",
            str(tmp_path / "out"),
            "--target_freq",
            "5min",
            "--symbol",
            "fu",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--input_dir" in result.stderr
```

- [x] **Step 2: Run downscale CLI tests and verify they fail on old parser**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract_cli.py::test_downscale_continuous_cli_reads_daily_input_dir_and_skips_missing_day data_preprocess/tests/test_commodity_main_contract_cli.py::test_downscale_continuous_cli_rejects_old_input_file_argument -q
```

Expected: first test FAILS because `--input_dir` is not accepted; second test may FAIL because `--input` is still accepted.

- [x] **Step 3: Implement directory/date-range downscale**

In `data_preprocess/operator_futures/commodity/downscale_continuous_by_trading_day.py`, add date helpers and split a single day processor:

```python
from datetime import datetime, timedelta
```

```python
def _parse_date(value: str):
    return datetime.strptime(value, "%Y-%m-%d").date()


def _iter_iso_dates(start_date: str, end_date: str):
    current = _parse_date(start_date)
    end = _parse_date(end_date)
    if end <= current:
        raise ValueError(
            f"end_date must be greater than start_date for left-open range: "
            f"{start_date} -> {end_date}"
        )
    while current < end:
        yield current.isoformat()
        current += timedelta(days=1)


def _write_downscaled_day(
    day_frame: pl.DataFrame,
    output_root: Path,
    target_freq: str,
    symbol: str,
    depth: int,
) -> str:
    trading_days = (
        day_frame.select(pl.col("TradingDay").cast(pl.Utf8).unique().sort())
        .to_series()
        .to_list()
    )
    if len(trading_days) != 1:
        raise ValueError(f"Daily continuous file must contain one TradingDay: {trading_days}")
    trading_day = trading_days[0]
    second = create_second_level_snapshots(day_frame)
    outputs = {
        "DOWNSCALE_DERTIC": downscale_derivative_reference(second, target_freq, symbol),
        "DOWNSCALE_ORDERBOOK_25": downscale_orderbook(second, target_freq, depth=depth),
        "BASE_FEATURE": downscale_base_features(second, target_freq, symbol),
        "COMMODITY_QUOTE_FEATURE": downscale_quote_features(second, target_freq),
    }
    for folder, frame in outputs.items():
        path = output_root / folder / symbol / target_freq
        path.mkdir(parents=True, exist_ok=True)
        frame.write_ipc(path / f"{trading_day}.feather")
    return trading_day
```

Replace `downscale_continuous_by_trading_day` with:

```python
def downscale_continuous_by_trading_day(
    input_dir: Path,
    output_root: Path,
    target_freq: str,
    symbol: str,
    start_date: str,
    end_date: str,
    depth: int = 5,
) -> None:
    started_at = time.monotonic()
    logger.info(
        "Starting commodity continuous downscale: input_dir=%s output_root=%s target_freq=%s symbol=%s start_date=%s end_date=%s depth=%d",
        input_dir,
        output_root,
        target_freq,
        symbol,
        start_date,
        end_date,
        depth,
    )
    processed = []
    skipped = []
    for date in _iter_iso_dates(start_date, end_date):
        daily_file = input_dir / f"{date}.csv"
        if not daily_file.exists():
            logger.warning("Missing commodity continuous daily file: date=%s input=%s", date, daily_file)
            skipped.append(date)
            continue
        raw = pl.read_csv(daily_file)
        logger.info("Downscaling commodity continuous daily file: date=%s input=%s rows=%d", date, daily_file, raw.height)
        trading_day = _write_downscaled_day(raw, output_root, target_freq, symbol, depth)
        processed.append(trading_day)
    if skipped:
        logger.warning("Skipped commodity continuous daily files: dates=%s", ",".join(skipped))
    logger.info(
        "Finished commodity continuous downscale: trading_days=%d skipped_dates=%d elapsed_seconds=%.2f",
        len(processed),
        len(skipped),
        time.monotonic() - started_at,
    )
```

Replace `parse_args` and `main` with:

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downscale continuous commodity main-contract daily files by TradingDay"
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--start_date", required=True)
    parser.add_argument("--end_date", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--target_freq", default="5min")
    parser.add_argument("--symbol", default="fu")
    parser.add_argument("--depth", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    downscale_continuous_by_trading_day(
        input_dir=Path(args.input_dir),
        output_root=Path(args.output_root),
        target_freq=args.target_freq,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        depth=args.depth,
    )
```

- [x] **Step 4: Run downscale verification**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract_cli.py::test_downscale_continuous_cli_reads_daily_input_dir_and_skips_missing_day data_preprocess/tests/test_commodity_main_contract_cli.py::test_downscale_continuous_cli_rejects_old_input_file_argument data_preprocess/tests/test_commodity_downscale.py -q
```

Expected: PASS. Existing `test_commodity_downscale.py` bad-data tests must still pass, confirming present malformed data remains fail-fast.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 3: 主流程脚本、文档与验证

> **trace:** plan-ready.md → `### Task 3: 主流程脚本、文档与验证` | tasks.md → `- [x] 3.0 主流程脚本、文档与验证完成（与 plan-ready.md Task 3 和 superpowers plan Task 3 同步）`
> **sync:** tasks.md → `- [x] 3.0 主流程脚本、文档与验证完成（与 plan-ready.md Task 3 和 superpowers plan Task 3 同步）` | plan-ready.md → `### Task 3: 主流程脚本、文档与验证`

**Files:**
- Modify: `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`
- Modify: `data_preprocess/tests/test_commodity_main_contract_cli.py`
- Modify: `docs/上海商品交易所/commodity_futures_preprocess.md`
- Modify: `openspec/changes/split-commodity-continuous-raw-by-day/tasks.md`
- Modify: `openspec/changes/split-commodity-continuous-raw-by-day/plan-ready.md`
- Modify: `docs/superpowers/plans/2026-06-22-split-commodity-continuous-raw-by-day.md`

- [x] **Step 1: Update shell/doc tests for directory handoff**

In `data_preprocess/tests/test_commodity_main_contract_cli.py`, update `test_commodity_full_process_shell_exposes_expected_functions` assertions:

```python
assert "--output_dir" in text
assert "--input_dir" in text
assert "continuous_dir" in text
assert "${symbol}_${start_date}_${end_date}.csv" not in text
assert "continuous_file" not in text
assert "operator_futures.commodity.downscale_continuous_by_trading_day" in text
```

Keep the existing assertions for `--start_date`, `--end_date`, `--market_type commodity_futures`, `--orderbook_depth 5`, and absence of `python - <<`.

- [x] **Step 2: Run shell text test and verify it fails on old handoff**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract_cli.py::test_commodity_full_process_shell_exposes_expected_functions -q
```

Expected: FAIL because `fu_full_process.sh` still constructs `continuous_file` and uses `--output` / `--input`.

- [x] **Step 3: Update fu_full_process.sh**

In `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`, change `run_commodity_stitch_main_contract` to:

```bash
run_commodity_stitch_main_contract() {
    local root_path=$1
    local commodity_name=${2:-燃料油}
    local start_date=$3
    local end_date=$4
    local symbol=${5:-fu}
    local output_dir="${root_path}/PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/${symbol}"

    mkdir -p "${output_dir}"
    PYTHONPATH="${root_path}/data_preprocess" python -m operator_futures.commodity.stitch_main_contract \
        --raw_root "${root_path}/data/原始下载" \
        --commodity_name "${commodity_name}" \
        --start_date "${start_date}" \
        --end_date "${end_date}" \
        --symbol "${symbol}" \
        --output_dir "${output_dir}"
}
```

Change `run_commodity_downscale_continuous_by_trading_day` to:

```bash
run_commodity_downscale_continuous_by_trading_day() {
    local root_path=$1
    local continuous_dir=$2
    local start_date=$3
    local end_date=$4
    local target_freq=$5
    local symbol=${6:-fu}
    local output_root="${root_path}/PREPROCESS_DATASET/commodity-futures"

    PYTHONPATH="${root_path}/data_preprocess" python -m operator_futures.commodity.downscale_continuous_by_trading_day \
        --input_dir "${continuous_dir}" \
        --start_date "${start_date}" \
        --end_date "${end_date}" \
        --output_root "${output_root}" \
        --target_freq "${target_freq}" \
        --symbol "${symbol}" \
        --depth 5
}
```

Change the top of `run_commodity_full_process` to:

```bash
    run_commodity_stitch_main_contract "$root_path" "$commodity_name" "$start_date" "$end_date" "$symbol"
    local continuous_dir="${root_path}/PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/${symbol}"
    run_commodity_downscale_continuous_by_trading_day "$root_path" "$continuous_dir" "$start_date" "$end_date" "$target_freq" "$symbol"
```

- [x] **Step 4: Update commodity preprocessing docs**

In `docs/上海商品交易所/commodity_futures_preprocess.md`, replace the full-process example with:

```bash
source data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh
run_commodity_full_process "$(pwd)" 2026-01-01 2026-02-01 5min fu 燃料油 4
```

Replace the continuous raw output description with:

```text
PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/fu/2026-01-05.csv
PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/fu/2026-01-06.csv
```

Add direct CLI examples:

```bash
PYTHONPATH=data_preprocess python -m operator_futures.commodity.stitch_main_contract \
  --raw_root data/原始下载 \
  --commodity_name 燃料油 \
  --start_date 2026-01-01 \
  --end_date 2026-02-01 \
  --symbol fu \
  --output_dir PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/fu

PYTHONPATH=data_preprocess python -m operator_futures.commodity.downscale_continuous_by_trading_day \
  --input_dir PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/fu \
  --start_date 2026-01-01 \
  --end_date 2026-02-01 \
  --output_root PREPROCESS_DATASET/commodity-futures \
  --symbol fu \
  --target_freq 5min \
  --depth 5
```

- [x] **Step 5: Run shell and focused regression verification**

Run:

```bash
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh
```

Expected: exits 0.

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py data_preprocess/tests/test_commodity_downscale.py -q
```

Expected: PASS.

- [x] **Step 6: Run OpenSpec and diff verification**

Run:

```bash
openspec validate split-commodity-continuous-raw-by-day --strict
```

Expected: PASS.

Run:

```bash
git diff --check
```

Expected: no whitespace errors.

- [x] **Step 7: Update tracking checkboxes after implementation**

After all verification passes, update only tracking checkboxes for this change:

```markdown
# openspec/changes/split-commodity-continuous-raw-by-day/tasks.md
- [x] 1.0 主力连续化日文件输出完成（与 plan-ready.md Task 1 和 superpowers plan Task 1 同步）
- [x] 2.0 连续主力日文件下采样完成（与 plan-ready.md Task 2 和 superpowers plan Task 2 同步）
- [x] 3.0 主流程脚本、文档与验证完成（与 plan-ready.md Task 3 和 superpowers plan Task 3 同步）

# openspec/changes/split-commodity-continuous-raw-by-day/plan-ready.md
- [x] **任务完成**（与 superpowers plan `Task N`、tasks.md 对应条目同步勾选）

# docs/superpowers/plans/2026-06-22-split-commodity-continuous-raw-by-day.md
- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）
```

Expected: each Task is marked complete in all three tracking documents only after its steps and verification are complete.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）
