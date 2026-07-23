# Add Commodity Quote OFI Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independent five-depth commodity quote OFI feature function that computes continuous adjacent-snapshot OFI and aggregates every 12 input rows.

**Architecture:** Keep OFI separate from the existing time-window `downscale_quote_features()` path. Add small Polars expression helpers and a new `downscale_quote_ofi_features(second_df, window_rows=12, depth=5)` function in `data_preprocess/operator_futures/commodity/downscale.py`; cover behavior with focused tests in the existing commodity downscale test module.

**Tech Stack:** Python, Polars, pytest, OpenSpec, conda environment `finetf`.

**Traceability (sddflow):**
- plan-ready: `openspec/changes/add-commodity-quote-ofi-features/plan-ready.md`
- tasks: `openspec/changes/add-commodity-quote-ofi-features/tasks.md`
- plan: `docs/superpowers/plans/2026-07-23-add-commodity-quote-ofi-features.md`

---

### Task 1: Add five-depth OFI direction tests

> **trace:** plan-ready.md -> `### Task 1: Add five-depth OFI direction tests` | tasks.md -> `- [ ] 1.1 Add focused tests in `data_preprocess/tests/test_commodity_downscale.py` for five-depth OFI direction math, summary columns, and first-row zero behavior.`
> **sync:** tasks.md -> `- [ ] 1.1 Add focused tests in `data_preprocess/tests/test_commodity_downscale.py` for five-depth OFI direction math, summary columns, and first-row zero behavior.` | plan-ready.md -> `### Task 1: Add five-depth OFI direction tests`

**Files:**
- Modify: `data_preprocess/tests/test_commodity_downscale.py`

- [x] **Step 1: Import the new OFI function in the test module**

Change the import from `operator_futures.commodity.downscale` to include `downscale_quote_ofi_features`:

```python
from operator_futures.commodity.downscale import (
    create_second_level_snapshots,
    downscale_base_features,
    downscale_derivative_reference,
    downscale_orderbook,
    downscale_quote_features,
    downscale_quote_ofi_features,
    validate_best_quotes,
)
```

- [x] **Step 2: Add a five-depth quote frame helper**

Add this helper near the other commodity downscale test helpers:

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
        row.update(overrides)
        rows.append(row)
    return pl.DataFrame(rows)
```

- [x] **Step 3: Add the direction math test**

Add this test to `data_preprocess/tests/test_commodity_downscale.py`:

```python
def test_downscale_quote_ofi_features_computes_five_depth_direction_math():
    frame = _five_depth_quote_frame(
        [
            {},
            {
                "BidPrice1": 100.0,
                "BidVolume1": 11.0,
                "BidVolume2": 25.0,
                "BidPrice3": 96.0,
                "BidVolume3": 35.0,
                "BidVolume4": 35.0,
                "AskPrice1": 100.0,
                "AskVolume1": 21.0,
                "AskVolume2": 42.0,
                "AskPrice3": 104.0,
                "AskVolume3": 65.0,
                "AskVolume4": 70.0,
            },
        ]
    )

    result = downscale_quote_ofi_features(frame, window_rows=12)
    row = result.row(0, named=True)

    assert row["timestamp"] == datetime(2026, 2, 2, 9, 0, 1)
    assert row["nquote"] == 2
    assert row["ofi_bid1"] == 11.0
    assert row["ofi_bid2"] == 5.0
    assert row["ofi_bid3"] == -30.0
    assert row["ofi_bid4"] == -5.0
    assert row["ofi_bid5"] == 0.0
    assert row["ofi_ask1"] == -21.0
    assert row["ofi_ask2"] == -2.0
    assert row["ofi_ask3"] == 60.0
    assert row["ofi_ask4"] == 10.0
    assert row["ofi_ask5"] == 0.0
    assert row["ofi_bid"] == -19.0
    assert row["ofi_ask"] == 47.0
    assert row["ofi"] == 28.0
```

- [x] **Step 4: Run the new test and confirm the expected failure**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_computes_five_depth_direction_math -q
```

Expected: FAIL during collection or import because `downscale_quote_ofi_features` is not implemented yet.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 2: Add fixed-row aggregation and boundary tests

> **trace:** plan-ready.md -> `### Task 2: Add fixed-row aggregation and boundary tests` | tasks.md -> `- [ ] 1.2 Add focused tests in `data_preprocess/tests/test_commodity_downscale.py` for fixed 12-row aggregation, tail groups, and cross-window continuous comparison.`
> **sync:** tasks.md -> `- [ ] 1.2 Add focused tests in `data_preprocess/tests/test_commodity_downscale.py` for fixed 12-row aggregation, tail groups, and cross-window continuous comparison.` | plan-ready.md -> `### Task 2: Add fixed-row aggregation and boundary tests`

**Files:**
- Modify: `data_preprocess/tests/test_commodity_downscale.py`

- [x] **Step 1: Add the fixed 12-row aggregation test**

Add this test:

```python
def test_downscale_quote_ofi_features_aggregates_every_twelve_rows_and_keeps_tail():
    frame = _five_depth_quote_frame(
        [{"BidVolume1": float(10 + index)} for index in range(13)]
    )

    result = downscale_quote_ofi_features(frame)

    assert result["timestamp"].to_list() == [
        datetime(2026, 2, 2, 9, 0, 11),
        datetime(2026, 2, 2, 9, 0, 12),
    ]
    assert result["nquote"].to_list() == [12, 1]
    assert result["ofi_bid1"].to_list() == [11.0, 1.0]
    assert result["ofi_bid"].to_list() == [11.0, 1.0]
    assert result["ofi_ask"].to_list() == [0.0, 0.0]
    assert result["ofi"].to_list() == [11.0, 1.0]
```

- [x] **Step 2: Add the cross-window continuous comparison test**

Add this test:

```python
def test_downscale_quote_ofi_features_compares_across_row_window_boundary():
    frame = _five_depth_quote_frame([{} for _ in range(12)] + [{"BidPrice1": 100.0, "BidVolume1": 77.0}])

    result = downscale_quote_ofi_features(frame)
    boundary_row = result.row(1, named=True)

    assert boundary_row["timestamp"] == datetime(2026, 2, 2, 9, 0, 12)
    assert boundary_row["nquote"] == 1
    assert boundary_row["ofi_bid1"] == 77.0
    assert boundary_row["ofi_bid"] == 77.0
    assert boundary_row["ofi"] == 77.0
```

- [x] **Step 3: Run the aggregation tests and confirm the expected failure**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_aggregates_every_twelve_rows_and_keeps_tail data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_compares_across_row_window_boundary -q
```

Expected: FAIL because `downscale_quote_ofi_features` is still missing.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 3: Add OFI input validation tests

> **trace:** plan-ready.md -> `### Task 3: Add OFI input validation tests` | tasks.md -> `- [ ] 1.3 Add focused tests in `data_preprocess/tests/test_commodity_downscale.py` for empty input, missing required depth columns, null depth values, and invalid `window_rows`.`
> **sync:** tasks.md -> `- [ ] 1.3 Add focused tests in `data_preprocess/tests/test_commodity_downscale.py` for empty input, missing required depth columns, null depth values, and invalid `window_rows`.` | plan-ready.md -> `### Task 3: Add OFI input validation tests`

**Files:**
- Modify: `data_preprocess/tests/test_commodity_downscale.py`

- [x] **Step 1: Add the empty input validation test**

Add this test:

```python
def test_downscale_quote_ofi_features_rejects_empty_input():
    with pytest.raises(ValueError, match="OFI input has no quote snapshots"):
        downscale_quote_ofi_features(pl.DataFrame())
```

- [x] **Step 2: Add the missing depth column validation test**

Add this test:

```python
def test_downscale_quote_ofi_features_rejects_missing_depth_columns():
    frame = _five_depth_quote_frame([{}]).drop("BidPrice5")

    with pytest.raises(ValueError, match="Missing OFI columns: BidPrice5"):
        downscale_quote_ofi_features(frame)
```

- [x] **Step 3: Add the null depth value validation test**

Add this test:

```python
def test_downscale_quote_ofi_features_rejects_null_depth_values():
    frame = _five_depth_quote_frame([{"AskVolume4": None}])

    with pytest.raises(ValueError, match="OFI columns contain null values: AskVolume4"):
        downscale_quote_ofi_features(frame)
```

- [x] **Step 4: Add the invalid window_rows validation test**

Add this test:

```python
def test_downscale_quote_ofi_features_rejects_invalid_window_rows():
    frame = _five_depth_quote_frame([{}])

    with pytest.raises(ValueError, match="window_rows must be positive"):
        downscale_quote_ofi_features(frame, window_rows=0)
```

- [x] **Step 5: Run the validation tests and confirm the expected failure**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_rejects_empty_input data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_rejects_missing_depth_columns data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_rejects_null_depth_values data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_rejects_invalid_window_rows -q
```

Expected: FAIL because `downscale_quote_ofi_features` is still missing.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 4: Implement row-window five-depth OFI

> **trace:** plan-ready.md -> `### Task 4: Implement row-window five-depth OFI` | tasks.md -> `- [ ] 1.4 Implement `downscale_quote_ofi_features()` and its OFI expression helpers in `data_preprocess/operator_futures/commodity/downscale.py`.`
> **sync:** tasks.md -> `- [ ] 1.4 Implement `downscale_quote_ofi_features()` and its OFI expression helpers in `data_preprocess/operator_futures/commodity/downscale.py`.` | plan-ready.md -> `### Task 4: Implement row-window five-depth OFI`

**Files:**
- Modify: `data_preprocess/operator_futures/commodity/downscale.py`

- [x] **Step 1: Add OFI helper constants and validation helpers**

Add this code after `_change_count_expr()`:

```python
def _ofi_required_columns(depth: int) -> list[str]:
    columns = ["timestamp"]
    for level in range(1, depth + 1):
        columns.extend(
            [
                f"BidPrice{level}",
                f"AskPrice{level}",
                f"BidVolume{level}",
                f"AskVolume{level}",
            ]
        )
    return columns


def _validate_ofi_input(
    second_df: pl.DataFrame, window_rows: int, depth: int
) -> list[str]:
    if window_rows <= 0:
        raise ValueError("window_rows must be positive")
    if second_df.height == 0:
        raise ValueError("OFI input has no quote snapshots")

    required_columns = _ofi_required_columns(depth)
    missing = [column for column in required_columns if column not in second_df.columns]
    if missing:
        raise ValueError(f"Missing OFI columns: {', '.join(missing)}")

    null_counts = second_df.select(
        [pl.col(column).null_count().alias(column) for column in required_columns]
    ).row(0, named=True)
    null_columns = [column for column, count in null_counts.items() if count > 0]
    if null_columns:
        raise ValueError(
            f"OFI columns contain null values: {', '.join(null_columns)}"
        )

    return required_columns
```

- [x] **Step 2: Add OFI bid and ask expression helpers**

Add this code after `_validate_ofi_input()`:

```python
def _ofi_bid_expr(level: int) -> pl.Expr:
    price = pl.col(f"BidPrice{level}").cast(pl.Float64, strict=False)
    size = pl.col(f"BidVolume{level}").cast(pl.Float64, strict=False)
    previous_price = price.shift(1)
    previous_size = size.shift(1)
    return (
        pl.when(previous_price.is_null())
        .then(pl.lit(0.0))
        .when(price > previous_price)
        .then(size)
        .when(price == previous_price)
        .then(size - previous_size)
        .otherwise(-previous_size)
        .alias(f"ofi_bid{level}")
    )


def _ofi_ask_expr(level: int) -> pl.Expr:
    price = pl.col(f"AskPrice{level}").cast(pl.Float64, strict=False)
    size = pl.col(f"AskVolume{level}").cast(pl.Float64, strict=False)
    previous_price = price.shift(1)
    previous_size = size.shift(1)
    return (
        pl.when(previous_price.is_null())
        .then(pl.lit(0.0))
        .when(price < previous_price)
        .then(-size)
        .when(price == previous_price)
        .then(-(size - previous_size))
        .otherwise(previous_size)
        .alias(f"ofi_ask{level}")
    )
```

- [x] **Step 3: Add the public row-window OFI function**

Add this function before `downscale_quote_features()`:

```python
def downscale_quote_ofi_features(
    second_df: pl.DataFrame, window_rows: int = 12, depth: int = 5
) -> pl.DataFrame:
    required_columns = _validate_ofi_input(second_df, window_rows, depth)
    quote = second_df.sort("timestamp").select(required_columns)

    bid_columns = [f"ofi_bid{level}" for level in range(1, depth + 1)]
    ask_columns = [f"ofi_ask{level}" for level in range(1, depth + 1)]
    ofi_columns = bid_columns + ask_columns

    quote = quote.with_columns(
        *[_ofi_bid_expr(level) for level in range(1, depth + 1)],
        *[_ofi_ask_expr(level) for level in range(1, depth + 1)],
    ).with_columns(
        pl.sum_horizontal([pl.col(column) for column in bid_columns]).alias("ofi_bid"),
        pl.sum_horizontal([pl.col(column) for column in ask_columns]).alias("ofi_ask"),
    )
    quote = quote.with_columns((pl.col("ofi_bid") + pl.col("ofi_ask")).alias("ofi"))

    grouped = (
        quote.with_row_index("_ofi_row_index")
        .with_columns((pl.col("_ofi_row_index") // window_rows).alias("_ofi_window"))
        .group_by("_ofi_window", maintain_order=True)
        .agg(
            pl.col("timestamp").last().alias("timestamp"),
            pl.len().alias("nquote"),
            *[pl.col(column).sum().alias(column) for column in ofi_columns],
            pl.col("ofi_bid").sum().alias("ofi_bid"),
            pl.col("ofi_ask").sum().alias("ofi_ask"),
            pl.col("ofi").sum().alias("ofi"),
        )
        .sort("_ofi_window")
    )
    return grouped.select(
        "timestamp",
        "nquote",
        *ofi_columns,
        "ofi_bid",
        "ofi_ask",
        "ofi",
    )
```

- [x] **Step 4: Run the focused OFI tests**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_computes_five_depth_direction_math data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_aggregates_every_twelve_rows_and_keeps_tail data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_compares_across_row_window_boundary data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_rejects_empty_input data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_rejects_missing_depth_columns data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_rejects_null_depth_values data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_rejects_invalid_window_rows -q
```

Expected: PASS for all OFI tests.

- [x] **Step 5: Commit the implementation slice**（skipped: user did not request a commit）

Run:

```bash
git add data_preprocess/operator_futures/commodity/downscale.py data_preprocess/tests/test_commodity_downscale.py
git commit -m "feat: add commodity quote ofi features"
```

Expected: commit succeeds if the working tree is ready for commits. If the user did not request commits during build, skip this step and leave the files staged state unchanged.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 5: Verify OFI change

> **trace:** plan-ready.md -> `### Task 5: Verify OFI change` | tasks.md -> `- [ ] 1.5 Verify the new OFI tests and run the commodity downscale test module under the `finetf` conda environment.`
> **sync:** tasks.md -> `- [ ] 1.5 Verify the new OFI tests and run the commodity downscale test module under the `finetf` conda environment.` | plan-ready.md -> `### Task 5: Verify OFI change`

**Files:**
- Verify: `openspec/changes/add-commodity-quote-ofi-features/proposal.md`
- Verify: `openspec/changes/add-commodity-quote-ofi-features/specs/commodity-futures-support/spec.md`
- Verify: `openspec/changes/add-commodity-quote-ofi-features/tasks.md`
- Verify: `openspec/changes/add-commodity-quote-ofi-features/plan-ready.md`
- Verify: `docs/superpowers/plans/2026-07-23-add-commodity-quote-ofi-features.md`
- Verify: `data_preprocess/operator_futures/commodity/downscale.py`
- Verify: `data_preprocess/tests/test_commodity_downscale.py`

- [x] **Step 1: Validate the OpenSpec change**

Run:

```bash
openspec validate add-commodity-quote-ofi-features --strict
```

Expected: `Change 'add-commodity-quote-ofi-features' is valid`.

- [x] **Step 2: Run the full commodity downscale test module**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_downscale.py -q
```

Expected: PASS for the full module.

- [x] **Step 3: Compile the changed Python files**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && python -m py_compile data_preprocess/operator_futures/commodity/downscale.py data_preprocess/tests/test_commodity_downscale.py
```

Expected: command exits with status 0 and prints no syntax errors.

- [x] **Step 4: Review the final diff**

Run:

```bash
git diff -- data_preprocess/operator_futures/commodity/downscale.py data_preprocess/tests/test_commodity_downscale.py openspec/changes/add-commodity-quote-ofi-features docs/superpowers/plans/2026-07-23-add-commodity-quote-ofi-features.md
```

Expected: diff only contains the OFI function, OFI tests, and this change's planning/specification documents.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 6: Add normalized OFI tests

> **trace:** plan-ready.md -> `### Task 6: Add normalized OFI tests` | tasks.md -> `- [ ] 1.6 Add focused tests in `data_preprocess/tests/test_commodity_downscale.py` for `ofi_norm`, `ofi_bid_norm`, `ofi_ask_norm`, and zero-denominator behavior.`
> **sync:** tasks.md -> `- [ ] 1.6 Add focused tests in `data_preprocess/tests/test_commodity_downscale.py` for `ofi_norm`, `ofi_bid_norm`, `ofi_ask_norm`, and zero-denominator behavior.` | plan-ready.md -> `### Task 6: Add normalized OFI tests`

**Files:**
- Modify: `data_preprocess/tests/test_commodity_downscale.py`

- [x] **Step 1: Add normalized OFI ratio test**

Add this test near the existing OFI tests:

```python
def test_downscale_quote_ofi_features_outputs_normalized_ofi():
    frame = _five_depth_quote_frame(
        [
            {},
            {
                "BidPrice1": 100.0,
                "BidVolume1": 11.0,
                "BidVolume2": 25.0,
                "BidPrice3": 96.0,
                "BidVolume3": 35.0,
                "BidVolume4": 35.0,
                "AskPrice1": 100.0,
                "AskVolume1": 21.0,
                "AskVolume2": 42.0,
                "AskPrice3": 104.0,
                "AskVolume3": 65.0,
                "AskVolume4": 70.0,
            },
        ]
    )

    result = downscale_quote_ofi_features(frame, window_rows=12)
    row = result.row(0, named=True)

    assert row["ofi_bid_norm"] == pytest.approx(-19.0 / 306.0)
    assert row["ofi_ask_norm"] == pytest.approx(47.0 / 598.0)
    assert row["ofi_norm"] == pytest.approx(28.0 / 904.0)
```

- [x] **Step 2: Add zero-denominator normalization test**

Add this test:

```python
def test_downscale_quote_ofi_features_zeroes_normalized_ofi_when_denominator_is_zero():
    frame = _five_depth_quote_frame(
        [
            {
                "BidVolume1": 0.0,
                "BidVolume2": 0.0,
                "BidVolume3": 0.0,
                "BidVolume4": 0.0,
                "BidVolume5": 0.0,
                "AskVolume1": 0.0,
                "AskVolume2": 0.0,
                "AskVolume3": 0.0,
                "AskVolume4": 0.0,
                "AskVolume5": 0.0,
            }
        ]
    )

    result = downscale_quote_ofi_features(frame, window_rows=12)
    row = result.row(0, named=True)

    assert row["ofi_bid_norm"] == 0.0
    assert row["ofi_ask_norm"] == 0.0
    assert row["ofi_norm"] == 0.0
```

- [x] **Step 3: Run normalized OFI tests and confirm RED**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_outputs_normalized_ofi data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_zeroes_normalized_ofi_when_denominator_is_zero -q
```

Expected: FAIL because the normalized OFI columns are not implemented yet.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 7: Implement normalized OFI outputs

> **trace:** plan-ready.md -> `### Task 7: Implement normalized OFI outputs` | tasks.md -> `- [ ] 1.7 Implement row-window OFI normalization outputs in `data_preprocess/operator_futures/commodity/downscale.py`.`
> **sync:** tasks.md -> `- [ ] 1.7 Implement row-window OFI normalization outputs in `data_preprocess/operator_futures/commodity/downscale.py`.` | plan-ready.md -> `### Task 7: Implement normalized OFI outputs`

**Files:**
- Modify: `data_preprocess/operator_futures/commodity/downscale.py`

- [x] **Step 1: Add safe division helper and volume denominator columns**

Add a small helper near the OFI expression helpers and compute per-snapshot denominator columns in `downscale_quote_ofi_features()`:

```python
def _safe_divide(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return pl.when(denominator != 0).then(numerator / denominator).otherwise(0.0)
```

Inside `downscale_quote_ofi_features()` after `bid_columns` and `ask_columns`, add:

```python
bid_size_columns = [f"BidVolume{level}" for level in range(1, depth + 1)]
ask_size_columns = [f"AskVolume{level}" for level in range(1, depth + 1)]
```

Before grouping, add:

```python
quote = quote.with_columns(
    pl.sum_horizontal([pl.col(column).cast(pl.Float64, strict=False) for column in bid_size_columns]).alias("_ofi_bid_volume"),
    pl.sum_horizontal([pl.col(column).cast(pl.Float64, strict=False) for column in ask_size_columns]).alias("_ofi_ask_volume"),
)
quote = quote.with_columns(
    (pl.col("_ofi_bid_volume") + pl.col("_ofi_ask_volume")).alias("_ofi_total_volume")
)
```

- [x] **Step 2: Aggregate denominators and output normalized columns**

Extend the group aggregation with denominator sums:

```python
pl.col("_ofi_bid_volume").sum().alias("_ofi_bid_volume"),
pl.col("_ofi_ask_volume").sum().alias("_ofi_ask_volume"),
pl.col("_ofi_total_volume").sum().alias("_ofi_total_volume"),
```

Then add normalized columns after grouping:

```python
grouped = grouped.with_columns(
    _safe_divide(pl.col("ofi_bid"), pl.col("_ofi_bid_volume")).alias("ofi_bid_norm"),
    _safe_divide(pl.col("ofi_ask"), pl.col("_ofi_ask_volume")).alias("ofi_ask_norm"),
    _safe_divide(pl.col("ofi"), pl.col("_ofi_total_volume")).alias("ofi_norm"),
)
```

Return the normalized output columns after raw OFI columns:

```python
return grouped.select(
    "timestamp",
    "nquote",
    *ofi_columns,
    "ofi_bid",
    "ofi_ask",
    "ofi",
    "ofi_bid_norm",
    "ofi_ask_norm",
    "ofi_norm",
)
```

- [x] **Step 3: Run normalized OFI tests and confirm GREEN**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_outputs_normalized_ofi data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_zeroes_normalized_ofi_when_denominator_is_zero -q
```

Expected: PASS for both normalized OFI tests.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 8: Verify normalized OFI change

> **trace:** plan-ready.md -> `### Task 8: Verify normalized OFI change` | tasks.md -> `- [ ] 1.8 Verify the normalized OFI tests, OpenSpec strict validation, and the commodity downscale test module under the `finetf` conda environment.`
> **sync:** tasks.md -> `- [ ] 1.8 Verify the normalized OFI tests, OpenSpec strict validation, and the commodity downscale test module under the `finetf` conda environment.` | plan-ready.md -> `### Task 8: Verify normalized OFI change`

**Files:**
- Verify: `openspec/changes/add-commodity-quote-ofi-features/proposal.md`
- Verify: `openspec/changes/add-commodity-quote-ofi-features/specs/commodity-futures-support/spec.md`
- Verify: `openspec/changes/add-commodity-quote-ofi-features/tasks.md`
- Verify: `openspec/changes/add-commodity-quote-ofi-features/plan-ready.md`
- Verify: `docs/superpowers/plans/2026-07-23-add-commodity-quote-ofi-features.md`
- Verify: `data_preprocess/operator_futures/commodity/downscale.py`
- Verify: `data_preprocess/tests/test_commodity_downscale.py`

- [x] **Step 1: Validate the amended OpenSpec change**

Run:

```bash
openspec validate add-commodity-quote-ofi-features --strict
```

Expected: `Change 'add-commodity-quote-ofi-features' is valid`.

- [x] **Step 2: Run the full commodity downscale test module**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_downscale.py -q
```

Expected: PASS for the full module.

- [x] **Step 3: Compile the changed Python files**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && python -m py_compile data_preprocess/operator_futures/commodity/downscale.py data_preprocess/tests/test_commodity_downscale.py
```

Expected: command exits with status 0 and prints no syntax errors.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 9: Add non-finite OFI input test

> **trace:** plan-ready.md -> `### Task 9: Add non-finite OFI input test` | tasks.md -> `- [x] 1.9 Add a focused test in `data_preprocess/tests/test_commodity_downscale.py` for rejecting NaN and infinite OFI depth values. <!-- 已实现: 添加 NaN/inf 输入 RED 测试 -->`
> **sync:** tasks.md -> `- [x] 1.9 Add a focused test in `data_preprocess/tests/test_commodity_downscale.py` for rejecting NaN and infinite OFI depth values. <!-- 已实现: 添加 NaN/inf 输入 RED 测试 -->` | plan-ready.md -> `### Task 9: Add non-finite OFI input test`

**Files:**
- Modify: `data_preprocess/tests/test_commodity_downscale.py`

- [x] **Step 1: Add NaN and infinite depth value test**

```python
def test_downscale_quote_ofi_features_rejects_non_finite_depth_values():
    frame = _five_depth_quote_frame(
        [{"BidVolume2": float("nan"), "AskPrice3": float("inf")}]
    )

    with pytest.raises(
        ValueError,
        match="OFI columns contain non-finite values: BidVolume2, AskPrice3",
    ):
        downscale_quote_ofi_features(frame)
```

- [x] **Step 2: Run non-finite test and confirm RED**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_rejects_non_finite_depth_values -q
```

Expected before implementation: FAIL with `Failed: DID NOT RAISE ValueError`.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 10: Implement non-finite OFI input validation

> **trace:** plan-ready.md -> `### Task 10: Implement non-finite OFI input validation` | tasks.md -> `- [x] 1.10 Implement non-finite OFI input validation in `data_preprocess/operator_futures/commodity/downscale.py`. <!-- 已实现: 拦截五档价格/数量列中的 NaN、inf 和 -inf -->`
> **sync:** tasks.md -> `- [x] 1.10 Implement non-finite OFI input validation in `data_preprocess/operator_futures/commodity/downscale.py`. <!-- 已实现: 拦截五档价格/数量列中的 NaN、inf 和 -inf -->` | plan-ready.md -> `### Task 10: Implement non-finite OFI input validation`

**Files:**
- Modify: `data_preprocess/operator_futures/commodity/downscale.py`

- [x] **Step 1: Add non-finite expression helper**

```python
def _non_finite_expr(column: str) -> pl.Expr:
    value = pl.col(column).cast(pl.Float64, strict=False)
    return (
        value.is_nan()
        | (value == float("inf"))
        | (value == -float("inf"))
    ).fill_null(False)
```

- [x] **Step 2: Validate non-finite numeric OFI columns**

```python
numeric_columns = [column for column in required_columns if column != "timestamp"]
non_finite_counts = second_df.select(
    [_non_finite_expr(column).sum().alias(column) for column in numeric_columns]
).row(0, named=True)
non_finite_columns = [
    column for column, count in non_finite_counts.items() if count > 0
]
if non_finite_columns:
    raise ValueError(
        f"OFI columns contain non-finite values: {', '.join(non_finite_columns)}"
    )
```

- [x] **Step 3: Run non-finite test and confirm GREEN**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_rejects_non_finite_depth_values -q
```

Expected: PASS.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 11: Verify non-finite OFI validation

> **trace:** plan-ready.md -> `### Task 11: Verify non-finite OFI validation` | tasks.md -> `- [x] 1.11 Verify non-finite validation, OpenSpec strict validation, and the commodity downscale test module under the `finetf` conda environment. <!-- 已实现: 坏数据测试通过，最终验证通过 -->`
> **sync:** tasks.md -> `- [x] 1.11 Verify non-finite validation, OpenSpec strict validation, and the commodity downscale test module under the `finetf` conda environment. <!-- 已实现: 坏数据测试通过，最终验证通过 -->` | plan-ready.md -> `### Task 11: Verify non-finite OFI validation`

**Files:**
- Verify: `openspec/changes/add-commodity-quote-ofi-features/proposal.md`
- Verify: `openspec/changes/add-commodity-quote-ofi-features/specs/commodity-futures-support/spec.md`
- Verify: `openspec/changes/add-commodity-quote-ofi-features/tasks.md`
- Verify: `openspec/changes/add-commodity-quote-ofi-features/plan-ready.md`
- Verify: `docs/superpowers/plans/2026-07-23-add-commodity-quote-ofi-features.md`
- Verify: `data_preprocess/operator_futures/commodity/downscale.py`
- Verify: `data_preprocess/tests/test_commodity_downscale.py`

- [x] **Step 1: Validate OpenSpec**

Run:

```bash
openspec validate add-commodity-quote-ofi-features --strict
```

Expected: `Change 'add-commodity-quote-ofi-features' is valid`.

- [x] **Step 2: Run full commodity downscale tests**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_downscale.py -q
```

Expected: PASS.

- [x] **Step 3: Compile changed Python files**

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && python -m py_compile data_preprocess/operator_futures/commodity/downscale.py data_preprocess/tests/test_commodity_downscale.py
```

Expected: command exits with status 0.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）
