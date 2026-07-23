# Add Commodity Quote Queue Pressure Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add queue refill/deplete and single-sided quote state ratio features to the existing commodity quote microstructure row-window output.

**Architecture:** Extend the existing `downscale_quote_microstructure_features()` path in `data_preprocess/operator_futures/commodity/downscale.py`. Keep the row-window grouping model, add a small set of private helpers for queue-event detection and single-sided state flags, and keep all normalization behind safe division so no `NaN` or infinity escapes the function.

**Tech Stack:** Python, Polars, pytest, OpenSpec, conda environment `finetf`.

**Traceability (sddflow):**
- plan-ready: `openspec/changes/add-commodity-quote-queue-pressure-features/plan-ready.md`
- tasks: `openspec/changes/add-commodity-quote-queue-pressure-features/tasks.md`
- plan: `docs/superpowers/plans/2026-07-24-add-commodity-quote-queue-pressure-features.md`

---

### Task 1: Add quote queue pressure regression tests

> **trace:**
> ```text
> plan-ready.md -> ### Task 1: Add quote queue pressure regression tests | tasks.md -> - [ ] 1.1 Add focused quote queue pressure regression tests in `data_preprocess/tests/test_commodity_downscale.py`, covering refill/deplete counts, `queue_refill_imbalance`, zero-event neutral output, single-sided state ratios, missing limit-column fail-fast, non-finite input rejection, and preservation of existing microstructure outputs.
> ```
> **sync:**
> ```text
> tasks.md -> - [ ] 1.1 Add focused quote queue pressure regression tests in `data_preprocess/tests/test_commodity_downscale.py`, covering refill/deplete counts, `queue_refill_imbalance`, zero-event neutral output, single-sided state ratios, missing limit-column fail-fast, non-finite input rejection, and preservation of existing microstructure outputs. | plan-ready.md -> ### Task 1: Add quote queue pressure regression tests
> ```

**Files:**
- Modify: `data_preprocess/tests/test_commodity_downscale.py`
- Test: `data_preprocess/tests/test_commodity_downscale.py`

- [x] **Step 1: Extend the microstructure test fixture and add the new queue-pressure assertions**

Update the existing helper so the microstructure tests can supply limit columns without duplicating boilerplate:

```python
def _five_depth_quote_frame(row_overrides: list[dict]) -> pl.DataFrame:
    rows = []
    for index, overrides in enumerate(row_overrides):
        row = {"timestamp": datetime(2026, 2, 2, 9, 0, index)}
        for level in range(1, 6):
            row[f"BidPrice{level}"] = 100.0 - level
            row[f"AskPrice{level}"] = 100.0 + level
            row[f"BidVolume{level}"] = float(level * 10)
            row[f"AskVolume{level}"] = float(level * 20)
        row.setdefault("LastPrice", 100.0)
        row.setdefault("LowPrice", 99.0)
        row.setdefault("HighPrice", 101.0)
        row.setdefault("LowerLimitPrice", 95.0)
        row.setdefault("UpperLimitPrice", 105.0)
        row.update(overrides)
        rows.append(row)
    return pl.DataFrame(rows)
```

Add a positive-path test that exercises both queue events and state ratios:

```python
def test_downscale_quote_microstructure_features_computes_queue_pressure_and_state_ratios():
    frame = _five_depth_quote_frame(
        [
            {
                "BidPrice1": 99.0,
                "AskPrice1": 101.0,
                "BidVolume1": 10.0,
                "AskVolume1": 20.0,
            },
            {
                "BidPrice1": 99.0,
                "AskPrice1": 101.0,
                "BidVolume1": 15.0,
                "AskVolume1": 15.0,
            },
            {
                "BidPrice1": 99.0,
                "AskPrice1": 101.0,
                "BidVolume1": 10.0,
                "AskVolume1": 25.0,
            },
            {
                "BidPrice1": 105.0,
                "AskPrice1": 105.0,
                "BidVolume1": 10.0,
                "AskVolume1": 0.0,
                "LastPrice": 105.0,
                "HighPrice": 105.0,
                "UpperLimitPrice": 105.0,
            },
            {
                "BidPrice1": 95.0,
                "AskPrice1": 100.0,
                "BidVolume1": 0.0,
                "AskVolume1": 10.0,
                "LastPrice": 95.0,
                "LowPrice": 95.0,
                "LowerLimitPrice": 95.0,
            },
        ]
    )

    result = downscale_quote_microstructure_features(frame, window_rows=12)
    row = result.row(0, named=True)

    assert row["bid_refill_count"] == 1
    assert row["bid_deplete_count"] == 1
    assert row["ask_refill_count"] == 1
    assert row["ask_deplete_count"] == 1
    assert row["queue_refill_imbalance"] == pytest.approx(0.0)
    assert row["bid_side_empty_ratio"] == pytest.approx(0.2)
    assert row["ask_side_empty_ratio"] == pytest.approx(0.2)
    assert row["limit_up_single_sided_ratio"] == pytest.approx(0.2)
    assert row["limit_down_single_sided_ratio"] == pytest.approx(0.2)
    assert math.isfinite(float(row["mean_microprice_pressure"]))
    assert row["spread_widen_count"] >= 0
```

- [x] **Step 2: Add the zero-event and fail-fast coverage**

Add a neutral-output test:

```python
def test_downscale_quote_microstructure_features_zeroes_queue_refill_imbalance_when_no_queue_events():
    frame = _five_depth_quote_frame(
        [
            {"BidVolume1": 10.0, "AskVolume1": 20.0},
            {"BidVolume1": 10.0, "AskVolume1": 20.0},
        ]
    )

    result = downscale_quote_microstructure_features(frame, window_rows=12)
    row = result.row(0, named=True)

    assert row["bid_refill_count"] == 0
    assert row["bid_deplete_count"] == 0
    assert row["ask_refill_count"] == 0
    assert row["ask_deplete_count"] == 0
    assert row["queue_refill_imbalance"] == 0.0
```

Add a missing-column validation test, and keep the existing non-finite microstructure test as coverage for non-finite queue-pressure inputs:

```python
def test_downscale_quote_microstructure_features_rejects_missing_limit_columns():
    frame = _five_depth_quote_frame([{}]).drop("UpperLimitPrice")

    with pytest.raises(
        ValueError, match="Missing microstructure columns: UpperLimitPrice"
    ):
        downscale_quote_microstructure_features(frame)
```

- [x] **Step 3: Run the focused tests and confirm the new assertions fail before implementation**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_downscale.py -k "quote_queue_pressure or quote_microstructure" -q
```

Expected: FAIL because the queue pressure columns are not present yet and the limit-column validation has not been extended.

- [x] **Step 4: Commit the red test change**

```bash
git add data_preprocess/tests/test_commodity_downscale.py
git commit -m "test: cover commodity quote queue pressure features"
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 2: Implement quote queue pressure row-window outputs

> **trace:**
> ```text
> plan-ready.md -> ### Task 2: Implement quote queue pressure row-window outputs | tasks.md -> - [ ] 1.2 Extend `downscale_quote_microstructure_features()` and its private helpers in `data_preprocess/operator_futures/commodity/downscale.py`, preserving existing `downscale_quote_features()` and `downscale_quote_ofi_features()` behavior while adding queue pressure and single-sided ratio outputs.
> ```
> **sync:**
> ```text
> tasks.md -> - [ ] 1.2 Extend `downscale_quote_microstructure_features()` and its private helpers in `data_preprocess/operator_futures/commodity/downscale.py`, preserving existing `downscale_quote_features()` and `downscale_quote_ofi_features()` behavior while adding queue pressure and single-sided ratio outputs. | plan-ready.md -> ### Task 2: Implement quote queue pressure row-window outputs
> ```

**Files:**
- Modify: `data_preprocess/operator_futures/commodity/downscale.py`
- Test: `data_preprocess/tests/test_commodity_downscale.py`

- [x] **Step 1: Extend the input validator to require the limit columns and safe numeric inputs**

Update the required-column helper and validation path so the row-window function can identify limit-up and limit-down single-sided states:

```python
def _quote_microstructure_required_columns() -> list[str]:
    return [
        "timestamp",
        "BidPrice1",
        "AskPrice1",
        "BidVolume1",
        "AskVolume1",
        "LastPrice",
        "LowPrice",
        "HighPrice",
        "LowerLimitPrice",
        "UpperLimitPrice",
    ]
```

Keep the existing fail-fast shape for empty input, missing columns, nulls, and non-finite values.

- [x] **Step 2: Add small private helpers for queue event flags and single-sided state flags**

Add helpers near the existing microstructure functions:

```python
def _queue_change_expr(price_column: str, size_column: str, direction: str) -> pl.Expr:
    price = pl.col(price_column).cast(pl.Float64, strict=False)
    size = pl.col(size_column).cast(pl.Float64, strict=False)
    previous_price = price.shift(1)
    previous_size = size.shift(1)
    unchanged_price = price == previous_price

    if direction == "refill":
        return (unchanged_price & (size > previous_size)).fill_null(False)
    return (unchanged_price & (size < previous_size)).fill_null(False)
```

Add a state helper that flags bid-side empty, ask-side empty, limit-up single-sided, and limit-down single-sided rows from the normalized quote columns and limit columns. Use safe boolean expressions and avoid emitting null booleans.

```python
def _bid_side_empty_expr() -> pl.Expr:
    return (
        pl.col("BidPrice1").is_null() | (pl.col("BidVolume1").fill_null(0) == 0)
    ).fill_null(False)


def _ask_side_empty_expr() -> pl.Expr:
    return (
        pl.col("AskPrice1").is_null() | (pl.col("AskVolume1").fill_null(0) == 0)
    ).fill_null(False)


def _limit_up_single_sided_expr() -> pl.Expr:
    return ((
        pl.col("AskVolume1").fill_null(0) == 0
    ) & (
        (pl.col("LastPrice") == pl.col("UpperLimitPrice"))
        | (pl.col("HighPrice") == pl.col("UpperLimitPrice"))
    ) & pl.col("BidPrice1").is_not_null()).fill_null(False)


def _limit_down_single_sided_expr() -> pl.Expr:
    return ((
        pl.col("BidVolume1").fill_null(0) == 0
    ) & (
        (pl.col("LastPrice") == pl.col("LowerLimitPrice"))
        | (pl.col("LowPrice") == pl.col("LowerLimitPrice"))
    ) & pl.col("AskPrice1").is_not_null()).fill_null(False)
```

Attach the per-row flags with one `with_columns(...)` call before the window aggregation:

```python
quote = quote.with_columns(
    _queue_change_expr("BidPrice1", "BidVolume1", "refill").alias("bid_refill"),
    _queue_change_expr("BidPrice1", "BidVolume1", "deplete").alias("bid_deplete"),
    _queue_change_expr("AskPrice1", "AskVolume1", "refill").alias("ask_refill"),
    _queue_change_expr("AskPrice1", "AskVolume1", "deplete").alias("ask_deplete"),
    _bid_side_empty_expr().alias("bid_side_empty"),
    _ask_side_empty_expr().alias("ask_side_empty"),
    _limit_up_single_sided_expr().alias("limit_up_single_sided"),
    _limit_down_single_sided_expr().alias("limit_down_single_sided"),
)
```

- [x] **Step 3: Extend the row-window aggregation**

Inside `downscale_quote_microstructure_features()`, keep the existing `microprice_pressure` / `relative_spread` / spread-direction logic, then add the new per-row columns and aggregate them:

```python
grouped = quote.group_by("_microstructure_window", maintain_order=True).agg(
    pl.col("timestamp").last().alias("timestamp"),
    pl.len().alias("nquote"),
    pl.col("microprice_pressure").mean().alias("mean_microprice_pressure"),
    pl.col("relative_spread").mean().alias("mean_relative_spread"),
    pl.col("_spread_widen").sum().alias("spread_widen_count"),
    pl.col("_spread_narrow").sum().alias("spread_narrow_count"),
    pl.col("_spread_flat").sum().alias("spread_flat_count"),
    pl.col("bid_refill").sum().alias("bid_refill_count"),
    pl.col("bid_deplete").sum().alias("bid_deplete_count"),
    pl.col("ask_refill").sum().alias("ask_refill_count"),
    pl.col("ask_deplete").sum().alias("ask_deplete_count"),
    pl.col("bid_side_empty").sum().alias("_bid_side_empty_count"),
    pl.col("ask_side_empty").sum().alias("_ask_side_empty_count"),
    pl.col("limit_up_single_sided").sum().alias("_limit_up_single_sided_count"),
    pl.col("limit_down_single_sided").sum().alias("_limit_down_single_sided_count"),
)
```

Then add `total_queue_events`, `queue_refill_imbalance`, and the four state ratios with `_safe_divide(...)`. Keep the output selection ordered with the new columns after the existing spread metrics and do not expose temporary `_..._count` columns.

- [x] **Step 4: Run the focused microstructure and queue-pressure tests**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_downscale.py -k "quote_queue_pressure or quote_microstructure" -q
```

Expected: PASS for the new queue-pressure tests and the existing microstructure regression tests.

- [x] **Step 5: Commit the implementation change**

```bash
git add data_preprocess/operator_futures/commodity/downscale.py data_preprocess/tests/test_commodity_downscale.py
git commit -m "feat: add commodity quote queue pressure features"
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 3: Validate quote queue pressure artifacts

> **trace:**
> ```text
> plan-ready.md -> ### Task 3: Validate quote queue pressure artifacts | tasks.md -> - [ ] 1.3 Run focused validation for the quote queue pressure change, including Python compile checks for changed Python files, targeted pytest, and `openspec validate add-commodity-quote-queue-pressure-features --strict`.
> ```
> **sync:**
> ```text
> tasks.md -> - [ ] 1.3 Run focused validation for the quote queue pressure change, including Python compile checks for changed Python files, targeted pytest, and `openspec validate add-commodity-quote-queue-pressure-features --strict`. | plan-ready.md -> ### Task 3: Validate quote queue pressure artifacts
> ```

**Files:**
- Modify: `data_preprocess/operator_futures/commodity/downscale.py`
- Modify: `data_preprocess/tests/test_commodity_downscale.py`
- Modify: `openspec/changes/add-commodity-quote-queue-pressure-features/{proposal.md,specs/,tasks.md,plan-ready.md}`
- Modify: `docs/superpowers/plans/2026-07-24-add-commodity-quote-queue-pressure-features.md`

- [x] **Step 1: Compile the changed Python files**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && python -m py_compile data_preprocess/operator_futures/commodity/downscale.py data_preprocess/tests/test_commodity_downscale.py
```

Expected: PASS with no syntax errors.

- [x] **Step 2: Run the focused pytest subset**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_downscale.py -k "quote_queue_pressure or quote_microstructure or quote_ofi" -q
```

Expected: PASS for the queue-pressure tests plus the existing microstructure and OFI regressions.

- [x] **Step 3: Validate the OpenSpec change**

Run:

```bash
openspec validate add-commodity-quote-queue-pressure-features --strict
```

Expected: PASS with no delta or task checkbox errors.

- [x] **Step 4: Check the final diff for accidental churn**

Run:

```bash
git diff --check
```

Expected: PASS with no whitespace or patch-format errors.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）
