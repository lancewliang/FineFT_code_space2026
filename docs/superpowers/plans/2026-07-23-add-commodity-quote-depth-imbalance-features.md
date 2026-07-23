# Add Commodity Quote Depth Imbalance Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add depth-aware static quote imbalance features for 1, 3, and 5 levels to commodity quote downscale output while preserving existing `imbalance_volume` compatibility.

**Architecture:** Extend `downscale_quote_features()` in place, with small private helpers for quote volume validation, depth imbalance expressions, and quote window aggregation expressions. Tests stay in the existing commodity downscale test module and reuse the five-depth quote fixture already used by OFI tests.

**Tech Stack:** Python 3.10, Polars, pytest, OpenSpec, conda environment `finetf`.

**Traceability (sddflow):**
- plan-ready: `openspec/changes/add-commodity-quote-depth-imbalance-features/plan-ready.md`
- tasks: `openspec/changes/add-commodity-quote-depth-imbalance-features/tasks.md`
- plan: `docs/superpowers/plans/2026-07-23-add-commodity-quote-depth-imbalance-features.md`

---

### Task 1: Add quote depth imbalance regression tests

> **trace:**
> ```text
> plan-ready.md → ### Task 1: Add quote depth imbalance regression tests | tasks.md → - [ ] 1.1 Add focused quote downscale regression tests in `data_preprocess/tests/test_commodity_downscale.py`, covering `imbalance_1/3/5` OHLC/TWAP/AWAP/STD outputs, `imbalance_1` compatibility with `imbalance_volume`, zero-denominator neutral output, non-finite volume fail-fast, and missing depth-column fail-fast.
> ```
> **sync:**
> ```text
> tasks.md → - [ ] 1.1 Add focused quote downscale regression tests in `data_preprocess/tests/test_commodity_downscale.py`, covering `imbalance_1/3/5` OHLC/TWAP/AWAP/STD outputs, `imbalance_1` compatibility with `imbalance_volume`, zero-denominator neutral output, non-finite volume fail-fast, and missing depth-column fail-fast. | plan-ready.md → ### Task 1: Add quote depth imbalance regression tests
> ```

**Files:**
- Modify: `data_preprocess/tests/test_commodity_downscale.py`
- Test: `data_preprocess/tests/test_commodity_downscale.py`

- [x] **Step 1: Add test utilities for finite checks and one-depth manual fixtures**

In `data_preprocess/tests/test_commodity_downscale.py`, change the import block and add helper functions below `_five_depth_quote_frame()`:

```python
from datetime import datetime
import logging
import math
from types import SimpleNamespace
```

```python
def _quote_frame_with_depth(data: dict) -> pl.DataFrame:
    row_count = len(data["timestamp"])
    columns = dict(data)
    for level in range(2, 6):
        columns.setdefault(f"BidVolume{level}", [0.0] * row_count)
        columns.setdefault(f"AskVolume{level}", [0.0] * row_count)
    return pl.DataFrame(columns)


def _assert_finite_imbalance_outputs(row: dict) -> None:
    for column, value in row.items():
        if "imbalance" in column and value is not None:
            assert math.isfinite(value), column
```

- [x] **Step 2: Add regression tests for depth imbalance outputs**

Add these tests near the existing quote downscale tests in `data_preprocess/tests/test_commodity_downscale.py`:

```python
def test_downscale_quote_features_outputs_depth_imbalance_statistics():
    frame = _five_depth_quote_frame(
        [
            {
                "timestamp": datetime(2026, 2, 2, 9, 0, 1),
                "BidVolume1": 2.0,
                "BidVolume2": 3.0,
                "BidVolume3": 5.0,
                "BidVolume4": 7.0,
                "BidVolume5": 11.0,
                "AskVolume1": 1.0,
                "AskVolume2": 1.0,
                "AskVolume3": 2.0,
                "AskVolume4": 2.0,
                "AskVolume5": 4.0,
            },
            {
                "timestamp": datetime(2026, 2, 2, 9, 4, 59),
                "BidVolume1": 1.0,
                "BidVolume2": 1.0,
                "BidVolume3": 2.0,
                "BidVolume4": 2.0,
                "BidVolume5": 4.0,
                "AskVolume1": 2.0,
                "AskVolume2": 3.0,
                "AskVolume3": 5.0,
                "AskVolume4": 7.0,
                "AskVolume5": 11.0,
            },
        ]
    )

    result = downscale_quote_features(frame, "5min")
    row = result.filter(
        pl.col("timestamp") == datetime(2026, 2, 2, 9, 5, 0)
    ).row(0, named=True)

    expected_imbalance_1_open = 1.0 / 3.0
    expected_imbalance_3_open = 3.0 / 7.0
    expected_imbalance_5_open = 9.0 / 19.0
    expected_imbalance_1_close = -1.0 / 3.0
    expected_imbalance_3_close = -3.0 / 7.0
    expected_imbalance_5_close = -9.0 / 19.0

    assert row["open_imbalance_1"] == pytest.approx(expected_imbalance_1_open)
    assert row["high_imbalance_1"] == pytest.approx(expected_imbalance_1_open)
    assert row["low_imbalance_1"] == pytest.approx(expected_imbalance_1_close)
    assert row["close_imbalance_1"] == pytest.approx(expected_imbalance_1_close)
    assert row["awap_imbalance_1"] == pytest.approx(0.0)
    assert row["twap_imbalance_1"] == pytest.approx(0.0)
    assert row["std_imbalance_1"] == pytest.approx(
        pl.Series([expected_imbalance_1_open, expected_imbalance_1_close]).std()
    )

    assert row["open_imbalance_3"] == pytest.approx(expected_imbalance_3_open)
    assert row["close_imbalance_3"] == pytest.approx(expected_imbalance_3_close)
    assert row["awap_imbalance_3"] == pytest.approx(0.0)
    assert row["twap_imbalance_3"] == pytest.approx(0.0)
    assert row["std_imbalance_3"] == pytest.approx(
        pl.Series([expected_imbalance_3_open, expected_imbalance_3_close]).std()
    )

    assert row["open_imbalance_5"] == pytest.approx(expected_imbalance_5_open)
    assert row["close_imbalance_5"] == pytest.approx(expected_imbalance_5_close)
    assert row["awap_imbalance_5"] == pytest.approx(0.0)
    assert row["twap_imbalance_5"] == pytest.approx(0.0)
    assert row["std_imbalance_5"] == pytest.approx(
        pl.Series([expected_imbalance_5_open, expected_imbalance_5_close]).std()
    )

    for stat in ["open", "high", "low", "close", "awap", "twap", "std"]:
        assert row[f"{stat}_imbalance_volume"] == pytest.approx(
            row[f"{stat}_imbalance_1"]
        )


def test_downscale_quote_features_depth_imbalance_zero_denominator_is_neutral():
    zero_volumes = {
        **{f"BidVolume{level}": 0.0 for level in range(1, 6)},
        **{f"AskVolume{level}": 0.0 for level in range(1, 6)},
    }
    frame = _five_depth_quote_frame(
        [
            {"timestamp": datetime(2026, 2, 2, 9, 0, 1), **zero_volumes},
            {"timestamp": datetime(2026, 2, 2, 9, 4, 59), **zero_volumes},
        ]
    )

    result = downscale_quote_features(frame, "5min")
    row = result.filter(
        pl.col("timestamp") == datetime(2026, 2, 2, 9, 5, 0)
    ).row(0, named=True)

    assert row["open_imbalance_1"] == 0.0
    assert row["open_imbalance_3"] == 0.0
    assert row["open_imbalance_5"] == 0.0
    assert row["std_imbalance_volume"] == 0.0
    _assert_finite_imbalance_outputs(row)


def test_downscale_quote_features_rejects_non_finite_depth_volume():
    frame = _five_depth_quote_frame([{"BidVolume3": float("nan")}])

    with pytest.raises(
        ValueError,
        match="Quote volume columns contain non-finite values: BidVolume3",
    ):
        downscale_quote_features(frame, "5min")


def test_downscale_quote_features_rejects_missing_depth_volume_column():
    frame = _five_depth_quote_frame([{}]).drop("AskVolume5")

    with pytest.raises(
        ValueError,
        match="Missing quote depth volume columns: AskVolume5",
    ):
        downscale_quote_features(frame, "5min")
```

- [x] **Step 3: Update existing one-depth quote fixtures to include depth volume columns**

In `data_preprocess/tests/test_commodity_downscale.py`, update these four tests so the manual quote fixture is built with `_quote_frame_with_depth(...)` instead of `pl.DataFrame(...)`:

```python
def test_limit_down_single_sided_quote_window_counts_as_quote():
    second = _quote_frame_with_depth(
        {
            "timestamp": [
                datetime(2026, 2, 2, 13, 50, 1),
                datetime(2026, 2, 2, 13, 52, 0),
                datetime(2026, 2, 2, 13, 54, 59),
            ],
            "LastPrice": [2679.0, 2679.0, 2679.0],
            "LowPrice": [2679.0, 2679.0, 2679.0],
            "LowerLimitPrice": [2679.0, 2679.0, 2679.0],
            "BidPrice1": [None, None, None],
            "BidVolume1": [0, 0, 0],
            "AskPrice1": [2679.0, 2679.0, 2679.0],
            "AskVolume1": [601, 900, 1383],
        }
    )
```

```python
def test_limit_up_single_sided_quote_window_counts_as_quote():
    second = _quote_frame_with_depth(
        {
            "timestamp": [
                datetime(2026, 2, 2, 13, 50, 1),
                datetime(2026, 2, 2, 13, 52, 0),
                datetime(2026, 2, 2, 13, 54, 59),
            ],
            "LastPrice": [2905.0, 2905.0, 2905.0],
            "HighPrice": [2905.0, 2905.0, 2905.0],
            "UpperLimitPrice": [2905.0, 2905.0, 2905.0],
            "BidPrice1": [2905.0, 2905.0, 2905.0],
            "BidVolume1": [601, 900, 1383],
            "AskPrice1": [None, None, None],
            "AskVolume1": [0, 0, 0],
        }
    )
```

```python
def test_cross_session_quote_gap_does_not_fail():
    second = _quote_frame_with_depth(
        {
            "timestamp": [
                datetime(2025, 10, 31, 23, 0, 0),
                datetime(2025, 11, 3, 9, 0, 0),
            ],
            "BidPrice1": [2600.0, 2601.0],
            "AskPrice1": [2602.0, 2603.0],
            "BidVolume1": [1.0, 1.0],
            "AskVolume1": [1.0, 1.0],
        }
    )
```

```python
def test_intermediate_empty_quote_window_in_same_session_fails_fast():
    second = _quote_frame_with_depth(
        {
            "timestamp": [
                datetime(2023, 1, 3, 9, 0, 0),
                datetime(2023, 1, 3, 9, 10, 0),
            ],
            "BidPrice1": [2600.0, 2601.0],
            "AskPrice1": [2602.0, 2603.0],
            "BidVolume1": [1.0, 1.0],
            "AskVolume1": [1.0, 1.0],
        }
    )
```

Leave each test's existing assertions unchanged after the fixture construction.

- [x] **Step 4: Run the new focused tests and confirm old implementation fails**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest data_preprocess/tests/test_commodity_downscale.py -k "depth_imbalance or quote_depth" -q
```

Expected: FAIL before implementation because `downscale_quote_features()` does not output `imbalance_1`, `imbalance_3`, `imbalance_5`, or `std_imbalance_volume`, and does not raise the new quote depth volume validation errors.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 2: Implement quote depth imbalance helpers and aggregation

> **trace:**
> ```text
> plan-ready.md → ### Task 2: Implement quote depth imbalance helpers and aggregation | tasks.md → - [ ] 1.2 Implement private quote-depth imbalance helpers and extend `downscale_quote_features()` in `data_preprocess/operator_futures/commodity/downscale.py`, preserving existing quote gap and limit single-sided behavior while adding `imbalance_1/3/5` and `std_imbalance_volume`.
> ```
> **sync:**
> ```text
> tasks.md → - [ ] 1.2 Implement private quote-depth imbalance helpers and extend `downscale_quote_features()` in `data_preprocess/operator_futures/commodity/downscale.py`, preserving existing quote gap and limit single-sided behavior while adding `imbalance_1/3/5` and `std_imbalance_volume`. | plan-ready.md → ### Task 2: Implement quote depth imbalance helpers and aggregation
> ```

**Files:**
- Modify: `data_preprocess/operator_futures/commodity/downscale.py`
- Test: `data_preprocess/tests/test_commodity_downscale.py`

- [x] **Step 1: Add quote depth imbalance constants and helper functions**

In `data_preprocess/operator_futures/commodity/downscale.py`, add these helpers after `_safe_divide()` and before `_normalize_limit_single_sided_quote_prices()`:

```python
QUOTE_DEPTH_IMBALANCE_LEVELS = (1, 3, 5)


def _quote_depth_volume_columns(depth: int = 5) -> list[str]:
    columns: list[str] = []
    for level in range(1, depth + 1):
        columns.extend([f"BidVolume{level}", f"AskVolume{level}"])
    return columns


def _validate_quote_depth_imbalance_input(second_df: pl.DataFrame) -> None:
    volume_columns = _quote_depth_volume_columns()
    missing = [column for column in volume_columns if column not in second_df.columns]
    if missing:
        raise ValueError(
            f"Missing quote depth volume columns: {', '.join(missing)}"
        )

    non_finite_counts = second_df.select(
        [_non_finite_expr(column).sum().alias(column) for column in volume_columns]
    ).row(0, named=True)
    non_finite_columns = [
        column for column, count in non_finite_counts.items() if count > 0
    ]
    if non_finite_columns:
        raise ValueError(
            "Quote volume columns contain non-finite values: "
            f"{', '.join(non_finite_columns)}"
        )


def _quote_volume_expr(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.Float64, strict=False).fill_null(0.0)


def _depth_imbalance_expr(depth: int) -> pl.Expr:
    bid_total = pl.sum_horizontal(
        [_quote_volume_expr(f"BidVolume{level}") for level in range(1, depth + 1)]
    )
    ask_total = pl.sum_horizontal(
        [_quote_volume_expr(f"AskVolume{level}") for level in range(1, depth + 1)]
    )
    denominator = bid_total + ask_total
    return _safe_divide(bid_total - ask_total, denominator)


def _quote_window_stat_aggs(
    names: list[str], std_names: set[str] | None = None
) -> list[pl.Expr]:
    std_names = std_names or set()
    aggs: list[pl.Expr] = []
    for name in names:
        aggs.extend(
            [
                pl.col(name).first().alias(f"open_{name}"),
                pl.col(name).max().alias(f"high_{name}"),
                pl.col(name).min().alias(f"low_{name}"),
                pl.col(name).last().alias(f"close_{name}"),
                pl.col(name).mean().alias(f"awap_{name}"),
                pl.col(name).mean().alias(f"twap_{name}"),
            ]
        )
        if name in std_names:
            aggs.append(pl.col(name).std().alias(f"std_{name}"))
    return aggs
```

- [x] **Step 2: Harden `_safe_divide()` for null and non-finite denominators**

In `data_preprocess/operator_futures/commodity/downscale.py`, replace `_safe_divide()` with:

```python
def _safe_divide(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    denominator = denominator.cast(pl.Float64, strict=False)
    return (
        pl.when(
            denominator.is_not_null()
            & denominator.is_finite()
            & (denominator != 0)
        )
        .then(numerator / denominator)
        .otherwise(0.0)
    )
```

- [x] **Step 3: Extend `downscale_quote_features()` select and derived columns**

In `data_preprocess/operator_futures/commodity/downscale.py`, replace the first part of `downscale_quote_features()` through the first `quote.with_columns(...)` block with:

```python
def downscale_quote_features(
    second_df: pl.DataFrame, target_freq: str, symbol: str = "fu"
) -> pl.DataFrame:
    if second_df.height == 0:
        raise ValueError("Target window has no quote snapshots")
    _validate_quote_depth_imbalance_input(second_df)

    quote = (
        _normalize_limit_single_sided_quote_prices(second_df)
        .sort("timestamp")
        .select(
            "timestamp",
            pl.col("BidPrice1").alias("bid_price"),
            pl.col("AskPrice1").alias("ask_price"),
            pl.col("BidVolume1").alias("bid_amount"),
            pl.col("AskVolume1").alias("ask_amount"),
            *BID_VOLUME_COLUMNS,
            *ASK_VOLUME_COLUMNS,
        )
    )
    quote = quote.with_columns(
        (pl.col("ask_price") - pl.col("bid_price")).alias("spread"),
        ((pl.col("ask_price") + pl.col("bid_price")) / 2).alias("mid"),
        _depth_imbalance_expr(1).alias("imbalance_volume"),
        *[
            _depth_imbalance_expr(level).alias(f"imbalance_{level}")
            for level in QUOTE_DEPTH_IMBALANCE_LEVELS
        ],
        pl.col("bid_price").alias("bid"),
        pl.col("ask_price").alias("ask"),
        pl.col("bid_amount").alias("bidsize"),
        pl.col("ask_amount").alias("asksize"),
    )
```

Leave the existing quote change count block below this code unchanged.

- [x] **Step 4: Replace manual aggregation loop with `_quote_window_stat_aggs()`**

In `data_preprocess/operator_futures/commodity/downscale.py`, replace the existing `for name in [...]` aggregation loop with:

```python
    quote_stat_names = [
        "spread",
        "mid",
        "imbalance_volume",
        "bid",
        "ask",
        "bidsize",
        "asksize",
        *[f"imbalance_{level}" for level in QUOTE_DEPTH_IMBALANCE_LEVELS],
    ]
    aggs.extend(
        _quote_window_stat_aggs(
            quote_stat_names,
            std_names={
                "imbalance_volume",
            }
            | {
                f"imbalance_{level}"
                for level in QUOTE_DEPTH_IMBALANCE_LEVELS
            },
        )
    )
```

- [x] **Step 5: Run focused tests and fix only failures caused by this change**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest data_preprocess/tests/test_commodity_downscale.py -k "depth_imbalance or quote_depth or limit_single_sided_quote or quote_gap" -q
```

Expected: PASS after implementation. If a failure is unrelated to quote depth imbalance, record the failure and continue with the narrower command from Step 6 before broadening validation.

- [x] **Step 6: Run the whole commodity downscale test file**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest data_preprocess/tests/test_commodity_downscale.py -q
```

Expected: PASS.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 3: Validate quote depth imbalance artifacts

> **trace:**
> ```text
> plan-ready.md → ### Task 3: Validate quote depth imbalance artifacts | tasks.md → - [ ] 1.3 Run focused validation for the quote-depth imbalance change, including targeted pytest, Python compile checks for changed Python files, and `openspec validate add-commodity-quote-depth-imbalance-features --strict`.
> ```
> **sync:**
> ```text
> tasks.md → - [ ] 1.3 Run focused validation for the quote-depth imbalance change, including targeted pytest, Python compile checks for changed Python files, and `openspec validate add-commodity-quote-depth-imbalance-features --strict`. | plan-ready.md → ### Task 3: Validate quote depth imbalance artifacts
> ```

**Files:**
- Modify: `openspec/changes/add-commodity-quote-depth-imbalance-features/tasks.md`
- Modify: `openspec/changes/add-commodity-quote-depth-imbalance-features/plan-ready.md`
- Modify: `docs/superpowers/plans/2026-07-23-add-commodity-quote-depth-imbalance-features.md`
- Test: `data_preprocess/operator_futures/commodity/downscale.py`
- Test: `data_preprocess/tests/test_commodity_downscale.py`

- [x] **Step 1: Run Python compile checks**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && python -m py_compile data_preprocess/operator_futures/commodity/downscale.py data_preprocess/tests/test_commodity_downscale.py
```

Expected: command exits with status 0 and prints no syntax errors.

- [x] **Step 2: Run focused quote validation**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest data_preprocess/tests/test_commodity_downscale.py -k "depth_imbalance or quote_depth or limit_single_sided_quote or quote_gap" -q
```

Expected: PASS.

- [x] **Step 3: Run OpenSpec strict validation**

Run:

```bash
openspec validate add-commodity-quote-depth-imbalance-features --strict
```

Expected: `Change 'add-commodity-quote-depth-imbalance-features' is valid`.

- [x] **Step 4: Update task-level checkboxes after implementation verification**

After Steps 1-3 pass during build, update:

```markdown
- [x] 1.1 Add focused quote downscale regression tests in `data_preprocess/tests/test_commodity_downscale.py`, covering `imbalance_1/3/5` OHLC/TWAP/AWAP/STD outputs, `imbalance_1` compatibility with `imbalance_volume`, zero-denominator neutral output, non-finite volume fail-fast, and missing depth-column fail-fast.
- [x] 1.2 Implement private quote-depth imbalance helpers and extend `downscale_quote_features()` in `data_preprocess/operator_futures/commodity/downscale.py`, preserving existing quote gap and limit single-sided behavior while adding `imbalance_1/3/5` and `std_imbalance_volume`.
- [x] 1.3 Run focused validation for the quote-depth imbalance change, including targeted pytest, Python compile checks for changed Python files, and `openspec validate add-commodity-quote-depth-imbalance-features --strict`.
```

Also update the corresponding `- [ ] **任务完成**` lines in `plan-ready.md` and the corresponding `- [ ] **Task complete**` lines in this plan.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

## Self-Review

- Spec coverage: Task 1 covers the new output columns, compatibility, zero denominator, non-finite input, and missing depth column scenarios. Task 2 covers helper implementation and `downscale_quote_features()` extension. Task 3 covers compile, pytest, and OpenSpec validation.
- Placeholder scan: The plan contains concrete file paths, test names, code snippets, commands, and expected outcomes.
- Type consistency: Helper names match the proposal and implementation snippets: `_depth_imbalance_expr`, `_quote_window_stat_aggs`, `_validate_quote_depth_imbalance_input`, `_quote_depth_volume_columns`, and `_quote_volume_expr`.
