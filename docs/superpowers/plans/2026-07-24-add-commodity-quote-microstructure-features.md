# Add Commodity Quote Microstructure Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independent fixed-row-window commodity quote microstructure feature function that computes microprice pressure, relative spread, and spread direction counts.

**Architecture:** Keep the new microstructure path separate from both `downscale_quote_features()` and `downscale_quote_ofi_features()`. Add a small validator plus one row-window function in `data_preprocess/operator_futures/commodity/downscale.py`, and cover behavior with focused tests in the existing commodity downscale test module.

**Tech Stack:** Python, Polars, pytest, OpenSpec, conda environment `finetf`.

**Traceability (sddflow):**
- plan-ready: `openspec/changes/add-commodity-quote-microstructure-features/plan-ready.md`
- tasks: `openspec/changes/add-commodity-quote-microstructure-features/tasks.md`
- plan: `docs/superpowers/plans/2026-07-24-add-commodity-quote-microstructure-features.md`

---

### Task 1: Add quote microstructure regression tests

> **trace:**
> ```text
> plan-ready.md -> ### Task 1: Add quote microstructure regression tests | tasks.md -> - [ ] 1.1 Add focused quote microstructure regression tests in `data_preprocess/tests/test_commodity_downscale.py`, covering formula means, spread widen/narrow/flat counts and ratio, default 12-row aggregation with retained tail window, fail-fast input validation, non-finite input rejection, and derived zero-denominator neutral outputs.
> ```
> **sync:**
> ```text
> tasks.md -> - [ ] 1.1 Add focused quote microstructure regression tests in `data_preprocess/tests/test_commodity_downscale.py`, covering formula means, spread widen/narrow/flat counts and ratio, default 12-row aggregation with retained tail window, fail-fast input validation, non-finite input rejection, and derived zero-denominator neutral outputs. | plan-ready.md -> ### Task 1: Add quote microstructure regression tests
> ```

**Files:**
- Modify: `data_preprocess/tests/test_commodity_downscale.py`
- Test: `data_preprocess/tests/test_commodity_downscale.py`

- [x] **Step 1: Import the new feature function and add a positive-path regression test**

Update the test import block to include the new function:

```python
from operator_futures.commodity.downscale import (
    create_second_level_snapshots,
    downscale_base_features,
    downscale_derivative_reference,
    downscale_orderbook,
    downscale_quote_features,
    downscale_quote_microstructure_features,
    downscale_quote_ofi_features,
    validate_best_quotes,
)
```

Add this test using the existing `_five_depth_quote_frame()` helper:

```python
def test_downscale_quote_microstructure_features_computes_pressure_and_spread_counts():
    frame = _five_depth_quote_frame(
        [
            {
                "BidPrice1": 99.0,
                "AskPrice1": 101.0,
                "BidVolume1": 30.0,
                "AskVolume1": 10.0,
            },
            {
                "BidPrice1": 99.0,
                "AskPrice1": 102.0,
                "BidVolume1": 10.0,
                "AskVolume1": 30.0,
            },
            {
                "BidPrice1": 99.0,
                "AskPrice1": 102.0,
                "BidVolume1": 10.0,
                "AskVolume1": 30.0,
            },
            {
                "BidPrice1": 100.0,
                "AskPrice1": 101.0,
                "BidVolume1": 10.0,
                "AskVolume1": 10.0,
            },
        ]
    )

    result = downscale_quote_microstructure_features(frame, window_rows=12)
    row = result.row(0, named=True)

    assert row["timestamp"] == datetime(2026, 2, 2, 9, 0, 3)
    assert row["nquote"] == 4
    assert row["mean_microprice_pressure"] == pytest.approx(-0.0625)
    assert row["mean_relative_spread"] == pytest.approx(
        (0.02 + (3.0 / 100.5) + (3.0 / 100.5) + (1.0 / 100.5)) / 4.0
    )
    assert row["spread_widen_count"] == 1
    assert row["spread_narrow_count"] == 1
    assert row["spread_flat_count"] == 2
    assert row["spread_widen_ratio"] == pytest.approx(0.25)
```

- [x] **Step 2: Add fixed-window and zero-denominator regression tests**

Add this aggregation test:

```python
def test_downscale_quote_microstructure_features_aggregates_every_twelve_rows_and_keeps_tail():
    frame = _five_depth_quote_frame(
        [{"BidVolume1": float(10 + index)} for index in range(13)]
    )

    result = downscale_quote_microstructure_features(frame)

    assert result["timestamp"].to_list() == [
        datetime(2026, 2, 2, 9, 0, 11),
        datetime(2026, 2, 2, 9, 0, 12),
    ]
    assert result["nquote"].to_list() == [12, 1]
    assert result["spread_flat_count"].to_list() == [12, 1]
    assert result["spread_widen_count"].to_list() == [0, 0]
    assert result["spread_narrow_count"].to_list() == [0, 0]
    assert result["spread_widen_ratio"].to_list() == [0.0, 0.0]
```

Add this neutral-output test:

```python
def test_downscale_quote_microstructure_features_zeroes_division_results_when_spread_is_zero():
    frame = _five_depth_quote_frame(
        [
            {
                "BidPrice1": 100.0,
                "AskPrice1": 100.0,
                "BidVolume1": 12.0,
                "AskVolume1": 18.0,
            }
        ]
    )

    result = downscale_quote_microstructure_features(frame, window_rows=12)
    row = result.row(0, named=True)

    assert row["mean_microprice_pressure"] == 0.0
    assert row["mean_relative_spread"] == 0.0
    assert row["spread_widen_ratio"] == 0.0
```

- [x] **Step 3: Add input validation regression tests**

Add these tests:

```python
def test_downscale_quote_microstructure_features_rejects_empty_input():
    with pytest.raises(
        ValueError, match="Microstructure input has no quote snapshots"
    ):
        downscale_quote_microstructure_features(pl.DataFrame())


def test_downscale_quote_microstructure_features_rejects_missing_depth_columns():
    frame = _five_depth_quote_frame([{}]).drop("AskVolume1")

    with pytest.raises(
        ValueError, match="Missing microstructure columns: AskVolume1"
    ):
        downscale_quote_microstructure_features(frame)


def test_downscale_quote_microstructure_features_rejects_non_finite_depth_values():
    frame = _five_depth_quote_frame(
        [{"BidVolume1": float("nan"), "AskPrice1": float("inf")}]
    )

    with pytest.raises(
        ValueError,
        match="Microstructure columns contain non-finite values: BidVolume1, AskPrice1",
    ):
        downscale_quote_microstructure_features(frame)


def test_downscale_quote_microstructure_features_rejects_invalid_window_rows():
    frame = _five_depth_quote_frame([{}])

    with pytest.raises(ValueError, match="window_rows must be positive"):
        downscale_quote_microstructure_features(frame, window_rows=0)
```

- [x] **Step 4: Run the focused test subset and confirm it fails before implementation**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_downscale.py -k "quote_microstructure" -q
```

Expected: FAIL because `downscale_quote_microstructure_features` is not implemented yet.

- [x] **Step 5: Commit the red test change**

```bash
git add data_preprocess/tests/test_commodity_downscale.py
git commit -m "test: cover commodity quote microstructure features"
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 2: Implement quote microstructure row-window function

> **trace:**
> ```text
> plan-ready.md -> ### Task 2: Implement quote microstructure row-window function | tasks.md -> - [ ] 1.2 Implement `downscale_quote_microstructure_features()` and its small private helpers in `data_preprocess/operator_futures/commodity/downscale.py`, preserving existing `downscale_quote_features()` and `downscale_quote_ofi_features()` behavior.
> ```
> **sync:**
> ```text
> tasks.md -> - [ ] 1.2 Implement `downscale_quote_microstructure_features()` and its small private helpers in `data_preprocess/operator_futures/commodity/downscale.py`, preserving existing `downscale_quote_features()` and `downscale_quote_ofi_features()` behavior. | plan-ready.md -> ### Task 2: Implement quote microstructure row-window function
> ```

**Files:**
- Modify: `data_preprocess/operator_futures/commodity/downscale.py`
- Test: `data_preprocess/tests/test_commodity_downscale.py`

- [x] **Step 1: Add the validation helper and the new public function**

Insert these helpers near the existing OFI helpers in `data_preprocess/operator_futures/commodity/downscale.py`:

```python
def _quote_microstructure_required_columns() -> list[str]:
    return ["timestamp", "BidPrice1", "AskPrice1", "BidVolume1", "AskVolume1"]


def _validate_quote_microstructure_input(
    second_df: pl.DataFrame, window_rows: int
) -> list[str]:
    if window_rows <= 0:
        raise ValueError("window_rows must be positive")
    if second_df.height == 0:
        raise ValueError("Microstructure input has no quote snapshots")

    required_columns = _quote_microstructure_required_columns()
    missing = [column for column in required_columns if column not in second_df.columns]
    if missing:
        raise ValueError(
            f"Missing microstructure columns: {', '.join(missing)}"
        )

    non_finite_counts = second_df.select(
        [
            _non_finite_expr(column).sum().alias(column)
            for column in required_columns
            if column != "timestamp"
        ]
    ).row(0, named=True)
    non_finite_columns = [
        column for column, count in non_finite_counts.items() if count > 0
    ]
    if non_finite_columns:
        raise ValueError(
            "Microstructure columns contain non-finite values: "
            f"{', '.join(non_finite_columns)}"
        )

    return required_columns


def downscale_quote_microstructure_features(
    second_df: pl.DataFrame, window_rows: int = 12
) -> pl.DataFrame:
    required_columns = _validate_quote_microstructure_input(second_df, window_rows)
    quote = second_df.sort("timestamp").select(
        "timestamp",
        pl.col("BidPrice1").cast(pl.Float64, strict=False).alias("bid_price"),
        pl.col("AskPrice1").cast(pl.Float64, strict=False).alias("ask_price"),
        pl.col("BidVolume1").cast(pl.Float64, strict=False).alias("bid_size"),
        pl.col("AskVolume1").cast(pl.Float64, strict=False).alias("ask_size"),
    )
    quote = quote.with_columns(
        (pl.col("ask_price") - pl.col("bid_price")).alias("spread"),
        ((pl.col("ask_price") + pl.col("bid_price")) / 2).alias("mid"),
    ).with_columns(
        _safe_divide(
            pl.col("ask_price") * pl.col("bid_size")
            + pl.col("bid_price") * pl.col("ask_size"),
            pl.col("bid_size") + pl.col("ask_size"),
        ).alias("microprice"),
        _safe_divide(
            pl.col("spread"),
            pl.col("mid"),
        ).alias("relative_spread"),
    ).with_columns(
        _safe_divide(
            pl.col("microprice") - pl.col("mid"),
            pl.col("spread"),
        ).alias("microprice_pressure"),
        pl.col("spread").diff().alias("_spread_diff"),
    ).with_columns(
        (pl.col("_spread_diff") > 0).fill_null(False).alias("_spread_widen"),
        (pl.col("_spread_diff") < 0).fill_null(False).alias("_spread_narrow"),
        ((pl.col("_spread_diff").is_null()) | (pl.col("_spread_diff") == 0))
        .fill_null(False)
        .alias("_spread_flat"),
    )

    grouped = (
        quote.with_row_index("_microstructure_row_index")
        .with_columns(
            (pl.col("_microstructure_row_index") // window_rows).alias(
                "_microstructure_window"
            )
        )
        .group_by("_microstructure_window", maintain_order=True)
        .agg(
            pl.col("timestamp").last().alias("timestamp"),
            pl.len().alias("nquote"),
            pl.col("microprice_pressure").mean().alias("mean_microprice_pressure"),
            pl.col("relative_spread").mean().alias("mean_relative_spread"),
            pl.col("_spread_widen").sum().alias("spread_widen_count"),
            pl.col("_spread_narrow").sum().alias("spread_narrow_count"),
            pl.col("_spread_flat").sum().alias("spread_flat_count"),
        )
        .sort("_microstructure_window")
    )
    grouped = grouped.with_columns(
        _safe_divide(
            pl.col("spread_widen_count"),
            pl.col("nquote").cast(pl.Float64, strict=False),
        ).alias("spread_widen_ratio")
    )
    return grouped.select(
        "timestamp",
        "nquote",
        "mean_microprice_pressure",
        "mean_relative_spread",
        "spread_widen_count",
        "spread_narrow_count",
        "spread_flat_count",
        "spread_widen_ratio",
    )
```

- [x] **Step 2: Confirm the new function passes the focused regression tests**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_downscale.py -k "quote_microstructure" -q
```

Expected: PASS after the implementation is in place.

- [x] **Step 3: Verify the new function does not affect OFI or quote downscale behavior**

Run the nearby regression subset:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_downscale.py -k "quote_microstructure or quote_ofi" -q
```

Expected: PASS with the new microstructure tests and existing OFI tests both green.

- [x] **Step 4: Commit the implementation**

```bash
git add data_preprocess/operator_futures/commodity/downscale.py data_preprocess/tests/test_commodity_downscale.py
git commit -m "feat: add commodity quote microstructure features"
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 3: Validate quote microstructure artifacts

> **trace:**
> ```text
> plan-ready.md -> ### Task 3: Validate quote microstructure artifacts | tasks.md -> - [ ] 1.3 Run focused validation for the quote microstructure change, including Python compile checks for changed Python files, targeted pytest, and `openspec validate add-commodity-quote-microstructure-features --strict`.
> ```
> **sync:**
> ```text
> tasks.md -> - [ ] 1.3 Run focused validation for the quote microstructure change, including Python compile checks for changed Python files, targeted pytest, and `openspec validate add-commodity-quote-microstructure-features --strict`. | plan-ready.md -> ### Task 3: Validate quote microstructure artifacts
> ```

**Files:**
- Modify: `data_preprocess/operator_futures/commodity/downscale.py`
- Modify: `data_preprocess/tests/test_commodity_downscale.py`
- Modify: `openspec/changes/add-commodity-quote-microstructure-features/{proposal.md,specs/,tasks.md,plan-ready.md}`
- Modify: `docs/superpowers/plans/2026-07-24-add-commodity-quote-microstructure-features.md`

- [x] **Step 1: Compile the changed Python files**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && python -m py_compile data_preprocess/operator_futures/commodity/downscale.py data_preprocess/tests/test_commodity_downscale.py
```

Expected: PASS with no syntax errors.

- [x] **Step 2: Run the focused regression suite**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_downscale.py -k "quote_microstructure or quote_ofi" -q
```

Expected: PASS.

- [x] **Step 3: Validate the OpenSpec change**

Run:

```bash
openspec validate add-commodity-quote-microstructure-features --strict
```

Expected: PASS.

- [x] **Step 4: Commit the validation checkpoint**

```bash
git add openspec/changes/add-commodity-quote-microstructure-features docs/superpowers/plans/2026-07-24-add-commodity-quote-microstructure-features.md
git commit -m "docs: finalize commodity quote microstructure plan"
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）
