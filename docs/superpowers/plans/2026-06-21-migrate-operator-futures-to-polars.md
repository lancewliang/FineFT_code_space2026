# Migrate Operator Futures To Polars Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the core `data_preprocess/operator_futures` pandas preprocessing engine with Polars while preserving existing CLI, file, timestamp, column, and Feather output contracts.

**Architecture:** Keep the existing script modules and staged preprocessing workflow. Introduce a small test compatibility helper first, then migrate downscale, feature-generation, merge/scale/selection, and commodity-specific paths in dependency order. Treat schema or historical behavior differences as explicit compatibility notes instead of silent behavior changes.

**Tech Stack:** Python 3.10, Polars, PyArrow Feather/IPC, NumPy, pytest, existing `conda run -n finetf` test workflow.

**Traceability (sddflow):**
- plan-ready: `openspec/changes/migrate-operator-futures-to-polars/plan-ready.md`
- tasks: `openspec/changes/migrate-operator-futures-to-polars/tasks.md`
- plan: `docs/superpowers/plans/2026-06-21-migrate-operator-futures-to-polars.md`

---

### Task 1: Dependency and compatibility harness

> **trace:** plan-ready.md -> `### Task 1: Dependency and compatibility harness` | tasks.md -> ``- [ ] 1.0 Dependency and compatibility harness complete（与 `plan-ready.md` Task 1 和 superpowers plan Task 1 同步）``
> **sync:** tasks.md -> ``- [ ] 1.0 Dependency and compatibility harness complete（与 `plan-ready.md` Task 1 和 superpowers plan Task 1 同步）`` | plan-ready.md -> `### Task 1: Dependency and compatibility harness`

**Files:**
- Modify: `data_preprocess/requirements.txt`
- Create: `data_preprocess/tests/polars_compat.py`
- Create: `data_preprocess/tests/test_polars_compat.py`
- Modify: `openspec/changes/migrate-operator-futures-to-polars/compatibility-notes.md`

- [x] **Step 1: Add failing compatibility helper tests**

Create `data_preprocess/tests/test_polars_compat.py` with:

```python
import numpy as np
import polars as pl
import pytest
from datetime import datetime

from polars_compat import assert_frame_contract, assert_no_pandas_engine


def test_assert_frame_contract_accepts_matching_frames():
    expected = pl.DataFrame(
        {
            "timestamp": [
                datetime(2023, 1, 1, 9, 0, 0),
                datetime(2023, 1, 1, 9, 0, 10),
            ],
            "value": [1.0, 2.0],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
    actual = expected.clone()

    assert_frame_contract(actual, expected)


def test_assert_frame_contract_rejects_column_order_change():
    expected = pl.DataFrame({"timestamp": [1, 2], "value": [1.0, 2.0]})
    actual = pl.DataFrame({"value": [1.0, 2.0], "timestamp": [1, 2]})

    with pytest.raises(AssertionError, match="column order"):
        assert_frame_contract(actual, expected)


def test_assert_frame_contract_uses_strict_float_tolerance():
    expected = pl.DataFrame({"timestamp": [1, 2], "value": [1.0, 2.0]})
    actual = pl.DataFrame({"timestamp": [1, 2], "value": [1.0, 2.0 + 1e-9]})

    with pytest.raises(AssertionError, match="float column"):
        assert_frame_contract(actual, expected, rtol=1e-12, atol=1e-12)


def test_assert_no_pandas_engine_rejects_pandas_import(tmp_path):
    module = tmp_path / "module.py"
    module.write_text("import pandas as pd\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="pandas import"):
        assert_no_pandas_engine([module])
```

- [x] **Step 2: Run helper tests to verify they fail before helper exists**

Run: `conda run -n finetf pytest data_preprocess/tests/test_polars_compat.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'polars_compat'`.

- [x] **Step 3: Add Polars dependency**

Modify `data_preprocess/requirements.txt` by inserting `polars==1.31.0` after `pandas==2.0.3`:

```text
pandas==2.0.3
polars==1.31.0
pyarrow==16.0.0
```

Keep the existing pandas pin for tests and non-target code that still imports pandas outside the migrated engine.

- [x] **Step 4: Add compatibility helper implementation**

Create `data_preprocess/tests/polars_compat.py` with:

```python
from pathlib import Path

import numpy as np
import polars as pl


def assert_frame_contract(
    actual: pl.DataFrame,
    expected: pl.DataFrame,
    *,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> None:
    if actual.columns != expected.columns:
        raise AssertionError(
            f"column order mismatch: actual={actual.columns}, expected={expected.columns}"
        )

    if actual.height != expected.height:
        raise AssertionError(
            f"row count mismatch: actual={actual.height}, expected={expected.height}"
        )

    for column in expected.columns:
        actual_dtype = actual.schema[column]
        expected_dtype = expected.schema[column]
        if actual_dtype != expected_dtype:
            raise AssertionError(
                f"dtype mismatch for {column}: actual={actual_dtype}, expected={expected_dtype}"
            )

        actual_series = actual[column]
        expected_series = expected[column]
        if actual_dtype in (pl.Float32, pl.Float64):
            actual_values = actual_series.to_numpy()
            expected_values = expected_series.to_numpy()
            if not np.allclose(
                actual_values,
                expected_values,
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            ):
                raise AssertionError(f"float column {column} differs beyond tolerance")
            continue

        if actual_series.to_list() != expected_series.to_list():
            raise AssertionError(f"column {column} values differ")


def assert_no_pandas_engine(paths: list[Path]) -> None:
    offenders: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if "import pandas" in text or "from pandas" in text:
            offenders.append(str(path))

    if offenders:
        joined = ", ".join(offenders)
        raise AssertionError(f"pandas import remains in migrated engine files: {joined}")
```

- [x] **Step 5: Run helper tests and dependency import check**

Run: `conda run -n finetf python -c "import polars; print(polars.__version__)"`

Expected: command exits 0 and prints `1.31.0`.

Run: `conda run -n finetf pytest data_preprocess/tests/test_polars_compat.py -q`

Expected: PASS, `4 passed`.

- [x] **Step 6: Commit Task 1**

```bash
git add data_preprocess/requirements.txt data_preprocess/tests/polars_compat.py data_preprocess/tests/test_polars_compat.py openspec/changes/migrate-operator-futures-to-polars/compatibility-notes.md
git commit -m "test: add polars preprocessing compatibility harness"
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 2: Binance downscale and IO paths

> **trace:** plan-ready.md -> `### Task 2: Binance downscale and IO paths` | tasks.md -> ``- [ ] 2.0 Binance downscale and IO paths complete（与 `plan-ready.md` Task 2 和 superpowers plan Task 2 同步）``
> **sync:** tasks.md -> ``- [ ] 2.0 Binance downscale and IO paths complete（与 `plan-ready.md` Task 2 和 superpowers plan Task 2 同步）`` | plan-ready.md -> `### Task 2: Binance downscale and IO paths`

**Files:**
- Modify: `data_preprocess/operator_futures/orderbook_25/down_scale_single_shot.py`
- Modify: `data_preprocess/operator_futures/orderbook_25/down_scale_single_shot_base_other.py`
- Modify: `data_preprocess/operator_futures/derivative_ticker/down_scale_single_shot.py`
- Modify: `data_preprocess/operator_futures/derivative_ticker/down_scale_single_shot_base_other.py`
- Test: `data_preprocess/tests/test_polars_downscale_io.py`

- [x] **Step 1: Add downscale compatibility tests**

Create `data_preprocess/tests/test_polars_downscale_io.py` with:

```python
import polars as pl

from operator_futures.derivative_ticker.down_scale_single_shot import down_scale_single_dertick
from operator_futures.orderbook_25.down_scale_single_shot import down_scale_single_oe_snapshot


def test_orderbook_downscale_preserves_first_and_depth_column_names():
    raw = pl.DataFrame(
        {
            "timestamp": [1_000_000, 1_000_001, 1_010_000],
            "local_timestamp": [1, 2, 3],
            "exchange": ["binance", "binance", "binance"],
            "asks[0].price": [101.0, 102.0, 103.0],
            "asks[0].amount": [1.0, 2.0, 3.0],
            "bids[0].price": [100.0, 99.0, 98.0],
            "bids[0].amount": [4.0, 5.0, 6.0],
        }
    )
    for level in range(1, 25):
        raw = raw.with_columns(
            pl.lit(101.0 + level).alias(f"asks[{level}].price"),
            pl.lit(1.0 + level).alias(f"asks[{level}].amount"),
            pl.lit(100.0 - level).alias(f"bids[{level}].price"),
            pl.lit(4.0 + level).alias(f"bids[{level}].amount"),
        )

    out = down_scale_single_oe_snapshot(raw, "10s")

    assert out.columns[:5] == [
        "timestamp",
        "ask1_price",
        "ask1_size",
        "bid1_price",
        "bid1_size",
    ]
    assert out["ask1_price"].to_list()[0] == 101.0
    assert out["bid1_price"].to_list()[0] == 100.0


def test_derivative_downscale_preserves_selected_columns_and_first():
    raw = pl.DataFrame(
        {
            "timestamp": [1_000_000, 1_000_001, 1_010_000],
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "funding_timestamp": [2_000_000, 2_000_001, 2_010_000],
            "funding_rate": [0.01, 0.02, 0.03],
            "index_price": [100.0, 101.0, 102.0],
            "mark_price": [100.5, 101.5, 102.5],
        }
    )

    out = down_scale_single_dertick(raw, "10s")

    assert out.columns == [
        "timestamp",
        "symbol",
        "funding_timestamp",
        "funding_rate",
        "index_price",
        "mark_price",
    ]
    assert out["funding_rate"].to_list()[0] == 0.01
```

- [x] **Step 2: Run tests to verify current pandas functions reject Polars input**

Run: `conda run -n finetf pytest data_preprocess/tests/test_polars_downscale_io.py -q`

Expected: FAIL because current downscale functions expect pandas operations such as `.set_index()` or pandas dtype conversion.

- [x] **Step 3: Replace orderbook downscale with Polars implementation**

In `data_preprocess/operator_futures/orderbook_25/down_scale_single_shot.py`, replace pandas import with `import polars as pl`, update the type annotation to `pl.DataFrame`, and make `down_scale_single_oe_snapshot` return a Polars DataFrame:

```python
def _rename_orderbook_columns(df: pl.DataFrame) -> pl.DataFrame:
    new_column_names = {}
    for i in range(25):
        new_column_names[f"asks[{i}].price"] = f"ask{i + 1}_price"
        new_column_names[f"asks[{i}].amount"] = f"ask{i + 1}_size"
        new_column_names[f"bids[{i}].price"] = f"bid{i + 1}_price"
        new_column_names[f"bids[{i}].amount"] = f"bid{i + 1}_size"
    return df.rename(new_column_names)


def down_scale_single_oe_snapshot(orderbook_df: pl.DataFrame, agg_freq: str) -> pl.DataFrame:
    return (
        orderbook_df.drop(["local_timestamp", "exchange"])
        .with_columns(
            (pl.col("timestamp") * 1000)
            .cast(pl.Datetime("us"))
            .alias("timestamp")
        )
        .sort("timestamp")
        .group_by_dynamic("timestamp", every=agg_freq, closed="left", label="left")
        .agg(pl.all().first())
        .pipe(_rename_orderbook_columns)
        .fill_null(strategy="forward")
    )
```

Update `main(args)` to use:

```python
single_df = pl.read_csv(os.path.join(orderbook_dir, date))
orderbook_df = down_scale_single_oe_snapshot(single_df, args.target_freq)
orderbook_df.write_ipc(
    os.path.join(args.save_path, args.symbols, args.target_freq, args.date + ".feather")
)
```

Apply the same import, function, and IO pattern to `data_preprocess/operator_futures/orderbook_25/down_scale_single_shot_base_other.py`, using `pl.read_ipc` for existing Feather input.

- [x] **Step 4: Replace derivative ticker downscale with Polars implementation**

In `data_preprocess/operator_futures/derivative_ticker/down_scale_single_shot.py`, replace pandas import with `import polars as pl` and implement:

```python
def down_scale_single_dertick(derivative_ticker: pl.DataFrame, agg_freq: str) -> pl.DataFrame:
    return (
        derivative_ticker.select(
            "timestamp",
            "symbol",
            "funding_timestamp",
            "funding_rate",
            "index_price",
            "mark_price",
        )
        .with_columns(
            pl.col("timestamp").cast(pl.Datetime("us")),
            pl.col("funding_timestamp").cast(pl.Datetime("us")),
        )
        .sort("timestamp")
        .group_by_dynamic("timestamp", every=agg_freq, closed="left", label="left")
        .agg(pl.all().first())
        .fill_null(strategy="forward")
    )
```

Update `main(args)` to use `pl.read_csv(...)` and `derivative_ticker_target.write_ipc(...)`. Apply the same pattern to `data_preprocess/operator_futures/derivative_ticker/down_scale_single_shot_base_other.py`, using `pl.read_ipc`.

- [x] **Step 5: Run downscale tests**

Run: `conda run -n finetf pytest data_preprocess/tests/test_polars_downscale_io.py -q`

Expected: PASS.

- [x] **Step 6: Run migrated-engine import scan for Task 2 files**

Run:

```bash
conda run -n finetf python - <<'PY'
from pathlib import Path
from data_preprocess.tests.polars_compat import assert_no_pandas_engine
assert_no_pandas_engine([
    Path("data_preprocess/operator_futures/orderbook_25/down_scale_single_shot.py"),
    Path("data_preprocess/operator_futures/orderbook_25/down_scale_single_shot_base_other.py"),
    Path("data_preprocess/operator_futures/derivative_ticker/down_scale_single_shot.py"),
    Path("data_preprocess/operator_futures/derivative_ticker/down_scale_single_shot_base_other.py"),
])
PY
```

Expected: exits 0.

- [x] **Step 7: Commit Task 2**

```bash
git add data_preprocess/operator_futures/orderbook_25/down_scale_single_shot.py data_preprocess/operator_futures/orderbook_25/down_scale_single_shot_base_other.py data_preprocess/operator_futures/derivative_ticker/down_scale_single_shot.py data_preprocess/operator_futures/derivative_ticker/down_scale_single_shot_base_other.py data_preprocess/tests/test_polars_downscale_io.py
git commit -m "refactor: migrate futures downscale io to polars"
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 3: Binance feature generation paths

> **trace:** plan-ready.md -> `### Task 3: Binance feature generation paths` | tasks.md -> ``- [ ] 3.0 Binance feature generation paths complete（与 `plan-ready.md` Task 3 和 superpowers plan Task 3 同步）``
> **sync:** tasks.md -> ``- [ ] 3.0 Binance feature generation paths complete（与 `plan-ready.md` Task 3 和 superpowers plan Task 3 同步）`` | plan-ready.md -> `### Task 3: Binance feature generation paths`

**Files:**
- Modify: `data_preprocess/operator_futures/features_related/base_feature.py`
- Modify: `data_preprocess/operator_futures/features_related/feature_util.py`
- Modify: `data_preprocess/operator_futures/cross_section/base_feature_util.py`
- Modify: `data_preprocess/operator_futures/cross_section/create_feature.py`
- Modify: `data_preprocess/operator_futures/time_operator/create_feature.py`
- Modify: `data_preprocess/operator_futures/time_operator/create_feature_multi_processing.py`
- Modify: `data_preprocess/operator_futures/time_operator/multi_processing_util.py`
- Modify: `data_preprocess/operator_futures/time_operator/time_operator_util.py`
- Test: `data_preprocess/tests/test_polars_feature_generation.py`

- [ ] **Step 1: Add feature generation tests for Polars helper behavior**

Create `data_preprocess/tests/test_polars_feature_generation.py` with:

```python
import polars as pl

from operator_futures.features_related.feature_util import (
    create_ohlc_quotes_feature,
    create_quotes_feature,
    intial_process_trades,
    preprocess_quotes,
    preprocess_trades,
    side_group_trades,
)


def test_quote_feature_counts_preserve_column_names():
    quotes = pl.DataFrame(
        {
            "timestamp": [1_000_000, 2_000_000, 3_000_000],
            "exchange": ["binance", "binance", "binance"],
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "bid_price": [100.0, 101.0, 101.0],
            "ask_price": [101.0, 101.0, 102.0],
            "bid_amount": [1.0, 2.0, 2.0],
            "ask_amount": [3.0, 3.0, 4.0],
        }
    )
    quotes = preprocess_quotes(quotes)

    out = create_quotes_feature(quotes, "10s")

    assert out.columns == [
        "timestamp",
        "nquote",
        "nquote_bid",
        "nquote_ask",
        "nquote_bid_up",
        "nquote_bid_down",
        "nquote_ask_up",
        "nquote_ask_down",
        "nquote_bidsize",
        "nquote_asksize",
        "nquote_bidsize_up",
        "nquote_bidsize_down",
        "nquote_asksize_up",
        "nquote_asksize_down",
        "nquote_bid_askflat",
        "nquote_bidup_askflat",
        "nquote_biddown_askflat",
        "nquote_ask_bidflat",
        "nquote_askup_bidflat",
        "nquote_askdown_bidflat",
    ]


def test_trade_feature_side_group_columns_exist():
    trades = pl.DataFrame(
        {
            "timestamp": [1_000_000, 2_000_000, 3_000_000],
            "exchange": ["binance", "binance", "binance"],
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "side": ["buy", "sell", "buy"],
            "price": [100.0, 101.0, 102.0],
            "amount": [1.0, 2.0, 3.0],
        }
    )
    trades = preprocess_trades(trades)

    base, processed = intial_process_trades(trades, "10s")
    side = side_group_trades(processed, "10s")

    assert "timestamp" in base.columns
    assert "buy_volume" in side.columns
    assert "sell_volume" in side.columns
```

- [ ] **Step 2: Run feature tests to verify failure before migration**

Run: `conda run -n finetf pytest data_preprocess/tests/test_polars_feature_generation.py -q`

Expected: FAIL because current feature utility functions expect pandas DataFrames.

- [ ] **Step 3: Migrate quote/trade utility functions to Polars expressions**

In `data_preprocess/operator_futures/features_related/feature_util.py`, replace pandas with `import polars as pl`. Preserve public function names. Implement preprocessing helpers with Polars:

```python
def preprocess_trades(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
        .sort("timestamp")
    )


def preprocess_quotes(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
        .sort("timestamp")
    )
```

Implement quote count generation with group windows and shifted columns inside each window. The output must start with `timestamp` and use the exact existing `nquote*` column names:

```python
def create_quotes_feature(quotes: pl.DataFrame, target_freq: str) -> pl.DataFrame:
    prepared = quotes.sort("timestamp").with_columns(
        (pl.col("bid_price") != pl.col("bid_price").shift(1)).alias("_bid_change"),
        (pl.col("ask_price") != pl.col("ask_price").shift(1)).alias("_ask_change"),
        (pl.col("bid_price") > pl.col("bid_price").shift(1)).alias("_bid_up"),
        (pl.col("bid_price") < pl.col("bid_price").shift(1)).alias("_bid_down"),
        (pl.col("ask_price") > pl.col("ask_price").shift(1)).alias("_ask_up"),
        (pl.col("ask_price") < pl.col("ask_price").shift(1)).alias("_ask_down"),
        (pl.col("bid_amount") != pl.col("bid_amount").shift(1)).alias("_bidsize_change"),
        (pl.col("ask_amount") != pl.col("ask_amount").shift(1)).alias("_asksize_change"),
        (pl.col("bid_amount") > pl.col("bid_amount").shift(1)).alias("_bidsize_up"),
        (pl.col("bid_amount") < pl.col("bid_amount").shift(1)).alias("_bidsize_down"),
        (pl.col("ask_amount") > pl.col("ask_amount").shift(1)).alias("_asksize_up"),
        (pl.col("ask_amount") < pl.col("ask_amount").shift(1)).alias("_asksize_down"),
        (pl.col("ask_price") == pl.col("ask_price").shift(1)).alias("_ask_flat"),
        (pl.col("bid_price") == pl.col("bid_price").shift(1)).alias("_bid_flat"),
    )
    return (
        prepared.group_by_dynamic("timestamp", every=target_freq, closed="left", label="left")
        .agg(
            pl.len().alias("nquote"),
            pl.col("_bid_change").sum().alias("nquote_bid"),
            pl.col("_ask_change").sum().alias("nquote_ask"),
            pl.col("_bid_up").sum().alias("nquote_bid_up"),
            pl.col("_bid_down").sum().alias("nquote_bid_down"),
            pl.col("_ask_up").sum().alias("nquote_ask_up"),
            pl.col("_ask_down").sum().alias("nquote_ask_down"),
            pl.col("_bidsize_change").sum().alias("nquote_bidsize"),
            pl.col("_asksize_change").sum().alias("nquote_asksize"),
            pl.col("_bidsize_up").sum().alias("nquote_bidsize_up"),
            pl.col("_bidsize_down").sum().alias("nquote_bidsize_down"),
            pl.col("_asksize_up").sum().alias("nquote_asksize_up"),
            pl.col("_asksize_down").sum().alias("nquote_asksize_down"),
            (pl.col("_ask_flat") & pl.col("_bid_change")).sum().alias("nquote_bid_askflat"),
            (pl.col("_ask_flat") & pl.col("_bid_up")).sum().alias("nquote_bidup_askflat"),
            (pl.col("_ask_flat") & pl.col("_bid_down")).sum().alias("nquote_biddown_askflat"),
            (pl.col("_bid_flat") & pl.col("_ask_change")).sum().alias("nquote_ask_bidflat"),
            (pl.col("_bid_flat") & pl.col("_ask_up")).sum().alias("nquote_askup_bidflat"),
            (pl.col("_bid_flat") & pl.col("_ask_down")).sum().alias("nquote_askdown_bidflat"),
        )
        .sort("timestamp")
    )
```

- [ ] **Step 4: Migrate base feature script IO and concatenation**

In `data_preprocess/operator_futures/features_related/base_feature.py`, use `pl.read_csv`, Polars horizontal joins on `timestamp`, and `write_ipc`. Preserve `exchange` and `symbol` as the first two non-timestamp columns after reset-equivalent output:

```python
quotes = pl.read_csv(os.path.join(quotes_path, date))
quotes = preprocess_quotes(quotes)
trades = pl.read_csv(os.path.join(trades_path, date))
trades = preprocess_trades(trades)

quotes_df = create_quotes_feature(quotes, target_freq).join(
    create_ohlc_quotes_feature(quotes, target_freq),
    on="timestamp",
    how="left",
)
trades_df, trades = intial_process_trades(trades, target_freq)
trades_df = trades_df.join(side_group_trades(trades, target_freq), on="timestamp", how="left")

exchange = trades.select("exchange").item(0, 0)
symbol = trades.select("symbol").item(0, 0)
indicators_df = (
    trades_df.join(quotes_df, on="timestamp", how="left")
    .with_columns(pl.lit(exchange).alias("exchange"), pl.lit(symbol).alias("symbol"))
    .select(["timestamp", "exchange", "symbol"] + [c for c in trades_df.join(quotes_df, on="timestamp", how="left").columns if c != "timestamp"])
    .fill_null(strategy="forward")
)
indicators_df.write_ipc(output_path)
```

If the repeated join expression becomes unwieldy, assign it once to `merged` before `select`.

- [ ] **Step 5: Migrate cross-section and time feature modules incrementally**

For each function in `cross_section/base_feature_util.py` and `time_operator/*` that currently creates an index-aligned pandas DataFrame, replace the return value with a Polars DataFrame containing an explicit `timestamp` column. Use Polars rolling APIs for window features:

```python
def _rolling_rank_last(expr: str, window: int) -> pl.Expr:
    return (
        pl.col(expr)
        .rolling_map(lambda values: float(pl.Series(values).rank().tail(1).item()) / len(values), window_size=window)
    )
```

Where Polars does not support an exact vectorized form, use `rolling_map` with a small, named helper and record the path in `compatibility-notes.md` if dtype or null handling differs from pandas.

- [ ] **Step 6: Run feature generation tests**

Run: `conda run -n finetf pytest data_preprocess/tests/test_polars_feature_generation.py -q`

Expected: PASS.

Run: `conda run -n finetf pytest data_preprocess/tests/test_commodity_feature_pipeline.py -q`

Expected: PASS.

- [ ] **Step 7: Commit Task 3**

```bash
git add data_preprocess/operator_futures/features_related/base_feature.py data_preprocess/operator_futures/features_related/feature_util.py data_preprocess/operator_futures/cross_section/base_feature_util.py data_preprocess/operator_futures/cross_section/create_feature.py data_preprocess/operator_futures/time_operator/create_feature.py data_preprocess/operator_futures/time_operator/create_feature_multi_processing.py data_preprocess/operator_futures/time_operator/multi_processing_util.py data_preprocess/operator_futures/time_operator/time_operator_util.py data_preprocess/tests/test_polars_feature_generation.py openspec/changes/migrate-operator-futures-to-polars/compatibility-notes.md
git commit -m "refactor: migrate futures feature generation to polars"
```

- [ ] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 4: Merge, scale, and feature-selection paths

> **trace:** plan-ready.md -> `### Task 4: Merge, scale, and feature-selection paths` | tasks.md -> ``- [ ] 4.0 Merge, scale, and feature-selection paths complete（与 `plan-ready.md` Task 4 和 superpowers plan Task 4 同步）``
> **sync:** tasks.md -> ``- [ ] 4.0 Merge, scale, and feature-selection paths complete（与 `plan-ready.md` Task 4 和 superpowers plan Task 4 同步）`` | plan-ready.md -> `### Task 4: Merge, scale, and feature-selection paths`

**Files:**
- Modify: `data_preprocess/operator_futures/merge_concat/merge.py`
- Modify: `data_preprocess/operator_futures/merge_concat/concat.py`
- Modify: `data_preprocess/operator_futures/merge_all/merge_clean.py`
- Modify: `data_preprocess/operator_futures/scale_describe_save/scale_save.py`
- Modify: `data_preprocess/operator_futures/feature_selection/cor_util.py`
- Modify: `data_preprocess/operator_futures/feature_selection/ic_correlation.py`
- Modify: `data_preprocess/operator_futures/feature_selection/lasso_linear.py`
- Modify: `data_preprocess/operator_futures/feature_selection/rank_ic_correlation.py`
- Modify: `data_preprocess/operator_futures/feature_selection/remove_duplicates_feature.py`
- Modify: `data_preprocess/operator_futures/feature_selection/catbooost.py`
- Test: `data_preprocess/tests/test_polars_merge_contract.py`

- [ ] **Step 1: Add merge contract tests**

Create `data_preprocess/tests/test_polars_merge_contract.py` with:

```python
import polars as pl

from operator_futures.merge_concat.concat import concat_concurrent_future_frames
from operator_futures.merge_concat.merge import build_daily_feature_frames


def test_concat_concurrent_future_frames_preserves_shift_and_inner_join():
    concurrent = pl.DataFrame(
        {
            "timestamp": [1, 1, 2, 3],
            "mark_price": [10.0, 11.0, 12.0, 13.0],
        }
    )
    future = pl.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "exchange": ["binance", "binance", "binance"],
            "feature": [100.0, 200.0, 300.0],
        }
    )

    out = concat_concurrent_future_frames(concurrent, future)

    assert out.columns == ["timestamp", "mark_price", "feature"]
    assert out["timestamp"].to_list() == [2, 3]
    assert out["feature"].to_list() == [100.0, 200.0]


def test_build_daily_feature_frames_drops_derivative_symbol_from_reward():
    snapshot = pl.DataFrame({"timestamp": [1], "ask1_price": [101.0]})
    der = pl.DataFrame({"timestamp": [1], "symbol": ["BTCUSDT"], "mark_price": [100.0]})
    base = pl.DataFrame({"timestamp": [1], "symbol": ["BTCUSDT"], "exchange": ["binance"], "volume": [1.0]})
    snapshot_feature = pl.DataFrame({"timestamp": [1], "snapshot_feature": [2.0]})
    quotes_feature = pl.DataFrame({"timestamp": [1], "quote_feature": [3.0]})
    kline_feature = pl.DataFrame({"timestamp": [1], "kline_feature": [4.0]})

    reward, future = build_daily_feature_frames(
        snapshot,
        der,
        base,
        snapshot_feature,
        quotes_feature,
        kline_feature,
    )

    assert "symbol" not in reward.columns
    assert future.columns[:3] == ["timestamp", "symbol", "exchange"]
```

- [ ] **Step 2: Run merge tests to verify failure before helper extraction**

Run: `conda run -n finetf pytest data_preprocess/tests/test_polars_merge_contract.py -q`

Expected: FAIL because `concat_concurrent_future_frames` and `build_daily_feature_frames` do not exist yet.

- [ ] **Step 3: Extract Polars merge helpers and update daily merge IO**

In `data_preprocess/operator_futures/merge_concat/merge.py`, add:

```python
import polars as pl


def build_daily_feature_frames(
    snapshot: pl.DataFrame,
    der: pl.DataFrame,
    base_feature: pl.DataFrame,
    snapshot_feature: pl.DataFrame,
    quotes_feature: pl.DataFrame,
    kline_feature: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    der_without_symbol = der.drop("symbol") if "symbol" in der.columns else der
    reward_feature = (
        snapshot.join(der_without_symbol, on="timestamp", how="left")
        .join(snapshot_feature, on="timestamp", how="left")
    )
    future_feature = (
        base_feature.join(quotes_feature, on="timestamp", how="left")
        .join(kline_feature, on="timestamp", how="left")
    )
    return reward_feature, future_feature
```

Replace `pd.read_feather` calls with `pl.read_ipc` and `to_feather` calls with `write_ipc`, keeping output paths unchanged.

- [ ] **Step 4: Extract Polars concat helper and update cross-day concat IO**

In `data_preprocess/operator_futures/merge_concat/concat.py`, add:

```python
import polars as pl


def _first_by_timestamp(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort("timestamp").group_by("timestamp", maintain_order=True).first()


def concat_concurrent_future_frames(
    concurrent_df: pl.DataFrame,
    future_df: pl.DataFrame,
) -> pl.DataFrame:
    concurrent_df = _first_by_timestamp(concurrent_df)
    future_df = _first_by_timestamp(future_df)
    future_df = future_df.drop([c for c in ["symbol", "exchange"] if c in future_df.columns])
    future_df = future_df.with_columns(pl.all().exclude("timestamp").shift(1)).filter(
        pl.all_horizontal(pl.all().exclude("timestamp").is_not_null())
    )
    return (
        concurrent_df.join(future_df, on="timestamp", how="inner")
        .sort("timestamp")
        .fill_null(strategy="forward")
    )
```

Use `pl.read_ipc` for each input file, `pl.concat(..., how="vertical")` for cross-day stacking, and `write_ipc` for the output Feather path.

- [ ] **Step 5: Migrate merge_clean, scale_save, and feature-selection readers**

Use Polars joins and explicit selects:

```python
all_feature_df = cross_section_df.join(time_feature_df, on="timestamp", how="inner")
all_feature_df.write_ipc(output_path)
```

For `scale_save.py`, keep the reward/execution manifest branch intact:

```python
reward_columns = get_reward_execution_columns(orderbook_depth=5) if args.market_type == "commodity_futures" else df.columns[:106]
df_reward = df.select(reward_columns)
df_state = df.select([c for c in df.columns if c not in reward_columns])
```

For feature-selection scripts, replace Feather/CSV reads with Polars and convert only at scikit-learn or CatBoost boundaries:

```python
features = df.select(feature_columns).to_pandas()
target = df.select(target_column).to_pandas()[target_column]
```

Record each third-party pandas boundary in a short inline comment.

- [ ] **Step 6: Run merge contract tests and commodity branch tests**

Run: `conda run -n finetf pytest data_preprocess/tests/test_polars_merge_contract.py data_preprocess/tests/test_commodity_feature_pipeline.py -q`

Expected: PASS.

- [ ] **Step 7: Commit Task 4**

```bash
git add data_preprocess/operator_futures/merge_concat/merge.py data_preprocess/operator_futures/merge_concat/concat.py data_preprocess/operator_futures/merge_all/merge_clean.py data_preprocess/operator_futures/scale_describe_save/scale_save.py data_preprocess/operator_futures/feature_selection/cor_util.py data_preprocess/operator_futures/feature_selection/ic_correlation.py data_preprocess/operator_futures/feature_selection/lasso_linear.py data_preprocess/operator_futures/feature_selection/rank_ic_correlation.py data_preprocess/operator_futures/feature_selection/remove_duplicates_feature.py data_preprocess/operator_futures/feature_selection/catbooost.py data_preprocess/tests/test_polars_merge_contract.py openspec/changes/migrate-operator-futures-to-polars/compatibility-notes.md
git commit -m "refactor: migrate merge scale and selection paths to polars"
```

- [ ] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 5: Commodity futures Polars migration

> **trace:** plan-ready.md -> `### Task 5: Commodity futures Polars migration` | tasks.md -> ``- [ ] 5.0 Commodity futures Polars migration complete（与 `plan-ready.md` Task 5 和 superpowers plan Task 5 同步）``
> **sync:** tasks.md -> ``- [ ] 5.0 Commodity futures Polars migration complete（与 `plan-ready.md` Task 5 和 superpowers plan Task 5 同步）`` | plan-ready.md -> `### Task 5: Commodity futures Polars migration`

**Files:**
- Modify: `data_preprocess/operator_futures/commodity/main_contract.py`
- Modify: `data_preprocess/operator_futures/commodity/stitch_main_contract.py`
- Modify: `data_preprocess/operator_futures/commodity/downscale.py`
- Modify: `data_preprocess/operator_futures/commodity/downscale_single_day.py`
- Modify: `data_preprocess/operator_futures/commodity/downscale_continuous_by_trading_day.py`
- Modify: `data_preprocess/operator_futures/commodity/schema.py`
- Modify: `data_preprocess/operator_futures/commodity/config.py`
- Modify: `data_preprocess/tests/test_commodity_config_schema.py`
- Modify: `data_preprocess/tests/test_commodity_downscale.py`
- Modify: `data_preprocess/tests/test_commodity_feature_pipeline.py`
- Modify: `data_preprocess/tests/test_commodity_main_contract.py`
- Modify: `data_preprocess/tests/test_commodity_main_contract_cli.py`

- [ ] **Step 1: Convert commodity tests to accept Polars outputs at public helpers**

Update `data_preprocess/tests/test_commodity_downscale.py` so helper calls use `pl.read_csv` and Polars assertions. Keep pandas only when reading CLI CSV output is clearer for filesystem assertions. Example conversion:

```python
import polars as pl


def test_create_second_level_snapshots_uses_last_row_per_second():
    raw = pl.read_csv(SAMPLE_PATH).head(20)
    out = create_second_level_snapshots(raw)

    assert "timestamp" in out.columns
    assert out["timestamp"].is_sorted()
```

Run: `conda run -n finetf pytest data_preprocess/tests/test_commodity_downscale.py -q`

Expected: FAIL until commodity helpers accept Polars DataFrames.

- [ ] **Step 2: Migrate main-contract selection to Polars**

In `data_preprocess/operator_futures/commodity/main_contract.py`, replace pandas DataFrame operations with Polars. Preserve public function names. Implement timestamp normalization with a row-dict-compatible helper for tests and a vectorized Polars helper for frames:

```python
def normalize_timestamp(row) -> object:
    action_day = str(row["ActionDay"])
    update_time = str(row["UpdateTime"])
    return pl.Series([f"{action_day} {update_time}"]).str.strptime(
        pl.Datetime("us"), format="%Y%m%d %H:%M:%S%.f", strict=True
    )[0]


def with_normalized_timestamp(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        (pl.col("ActionDay").cast(pl.Utf8) + pl.lit(" ") + pl.col("UpdateTime").cast(pl.Utf8))
        .str.strptime(pl.Datetime("us"), format="%Y%m%d %H:%M:%S%.f", strict=True)
        .alias("timestamp")
    )
```

Use `pl.read_csv(file_path)` in loaders. Use `pl.concat(output, how="vertical")` in `stitch_main_contract_frames`. Preserve metadata columns with `with_columns(pl.lit(...).alias(...))`.

- [ ] **Step 3: Migrate commodity downscale calculations to Polars**

In `data_preprocess/operator_futures/commodity/downscale.py`, replace pandas with Polars. Preserve these public functions and return Polars DataFrames:

```python
def create_second_level_snapshots(df: pl.DataFrame) -> pl.DataFrame:
    copied = with_normalized_timestamp(df)
    contract = copied.select("InstrumentID").item(0, 0) if "InstrumentID" in copied.columns else "unknown"
    validate_best_quotes(copied, contract)
    return (
        copied.sort("timestamp")
        .group_by_dynamic("timestamp", every="1s", closed="left", label="left")
        .agg(pl.all().last())
    )
```

For right-closed target windows, use:

```python
RESAMPLE_KWARGS = {"closed": "right", "label": "right"}
```

Implement `downscale_derivative_reference`, `downscale_orderbook`, `downscale_base_features`, and `downscale_quote_features` with Polars group-window aggregations. Preserve fail-fast checks by raising `ValueError` with the same contract, timestamp, and field names currently asserted in tests.

- [ ] **Step 4: Update commodity CLI modules**

In `downscale_single_day.py` and `downscale_continuous_by_trading_day.py`, use `pl.read_csv` and Polars write methods:

```python
raw = pl.read_csv(args.input)
second_df = create_second_level_snapshots(raw)
downscale_orderbook(second_df, args.target_freq, depth=args.orderbook_depth).write_ipc(orderbook_output)
downscale_derivative_reference(second_df, args.target_freq, args.symbol).write_ipc(derivative_output)
downscale_base_features(second_df, args.target_freq).write_ipc(base_output)
downscale_quote_features(second_df, args.target_freq).write_ipc(quote_output)
```

In `stitch_main_contract.py`, write CSV output from Polars with `stitched.write_csv(args.output)`.

- [ ] **Step 5: Preserve commodity schema/config and market-type branches**

Keep `schema.py` and `config.py` public APIs unchanged. If tests currently pass pandas Series into schema helpers, update those helpers to accept Python sequences or Polars Series without changing returned column names:

```python
def get_reward_execution_columns(orderbook_depth: int = 5) -> list[str]:
    return [
        "timestamp",
        "symbol",
        "funding_timestamp",
        "funding_rate",
        "index_price",
        "mark_price",
    ] + depth_columns(orderbook_depth)
```

Do not add depth 6-25 columns for commodity futures.

- [ ] **Step 6: Run commodity tests**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_commodity_config_schema.py data_preprocess/tests/test_commodity_downscale.py data_preprocess/tests/test_commodity_feature_pipeline.py data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 5**

```bash
git add data_preprocess/operator_futures/commodity/main_contract.py data_preprocess/operator_futures/commodity/stitch_main_contract.py data_preprocess/operator_futures/commodity/downscale.py data_preprocess/operator_futures/commodity/downscale_single_day.py data_preprocess/operator_futures/commodity/downscale_continuous_by_trading_day.py data_preprocess/operator_futures/commodity/schema.py data_preprocess/operator_futures/commodity/config.py data_preprocess/tests/test_commodity_config_schema.py data_preprocess/tests/test_commodity_downscale.py data_preprocess/tests/test_commodity_feature_pipeline.py data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py openspec/changes/migrate-operator-futures-to-polars/compatibility-notes.md
git commit -m "refactor: migrate commodity futures preprocessing to polars"
```

- [ ] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 6: Validation and migration closure

> **trace:** plan-ready.md -> `### Task 6: Validation and migration closure` | tasks.md -> ``- [ ] 6.0 Validation and migration closure complete（与 `plan-ready.md` Task 6 和 superpowers plan Task 6 同步）``
> **sync:** tasks.md -> ``- [ ] 6.0 Validation and migration closure complete（与 `plan-ready.md` Task 6 和 superpowers plan Task 6 同步）`` | plan-ready.md -> `### Task 6: Validation and migration closure`

**Files:**
- Modify: `openspec/changes/migrate-operator-futures-to-polars/tasks.md`
- Modify: `openspec/changes/migrate-operator-futures-to-polars/plan-ready.md`
- Modify: `docs/superpowers/plans/2026-06-21-migrate-operator-futures-to-polars.md`
- Modify: `openspec/changes/migrate-operator-futures-to-polars/compatibility-notes.md`

- [ ] **Step 1: Run OpenSpec strict validation**

Run: `openspec validate migrate-operator-futures-to-polars --strict`

Expected: PASS with `Change 'migrate-operator-futures-to-polars' is valid`.

- [ ] **Step 2: Run full data_preprocess tests**

Run: `conda run -n finetf pytest data_preprocess/tests -q`

Expected: PASS for all tests in `data_preprocess/tests`.

- [ ] **Step 3: Run migrated-engine pandas import scan**

Run:

```bash
conda run -n finetf python - <<'PY'
from pathlib import Path
from data_preprocess.tests.polars_compat import assert_no_pandas_engine

paths = [
    Path("data_preprocess/operator_futures/orderbook_25/down_scale_single_shot.py"),
    Path("data_preprocess/operator_futures/orderbook_25/down_scale_single_shot_base_other.py"),
    Path("data_preprocess/operator_futures/derivative_ticker/down_scale_single_shot.py"),
    Path("data_preprocess/operator_futures/derivative_ticker/down_scale_single_shot_base_other.py"),
    Path("data_preprocess/operator_futures/features_related/base_feature.py"),
    Path("data_preprocess/operator_futures/features_related/feature_util.py"),
    Path("data_preprocess/operator_futures/cross_section/base_feature_util.py"),
    Path("data_preprocess/operator_futures/cross_section/create_feature.py"),
    Path("data_preprocess/operator_futures/time_operator/create_feature.py"),
    Path("data_preprocess/operator_futures/time_operator/create_feature_multi_processing.py"),
    Path("data_preprocess/operator_futures/time_operator/multi_processing_util.py"),
    Path("data_preprocess/operator_futures/time_operator/time_operator_util.py"),
    Path("data_preprocess/operator_futures/merge_concat/merge.py"),
    Path("data_preprocess/operator_futures/merge_concat/concat.py"),
    Path("data_preprocess/operator_futures/merge_all/merge_clean.py"),
    Path("data_preprocess/operator_futures/scale_describe_save/scale_save.py"),
    Path("data_preprocess/operator_futures/commodity/main_contract.py"),
    Path("data_preprocess/operator_futures/commodity/stitch_main_contract.py"),
    Path("data_preprocess/operator_futures/commodity/downscale.py"),
    Path("data_preprocess/operator_futures/commodity/downscale_single_day.py"),
    Path("data_preprocess/operator_futures/commodity/downscale_continuous_by_trading_day.py"),
]
assert_no_pandas_engine(paths)
PY
```

Expected: exits 0, or fails only for documented third-party boundary files that are recorded in `compatibility-notes.md`.

- [ ] **Step 4: Run smoke command checks**

Run commodity smoke:

```bash
conda run -n finetf python -m operator_futures.commodity.downscale_single_day \
  --input docs/上海商品交易所/fu2302.csv \
  --output_root /tmp/finetf-polars-smoke \
  --symbol fu \
  --target_freq 5min
```

Expected: exits 0 and writes commodity downscale outputs under `/tmp/finetf-polars-smoke`.

For Binance smoke, use the smallest available local raw Binance sample. If no `DOWNLOAD_DATASET/binance-futures/*` sample exists in the workspace, append this exact note to `compatibility-notes.md`:

```markdown
## Smoke limitation: Binance futures raw sample

The workspace does not contain a small local `DOWNLOAD_DATASET/binance-futures` raw sample for CLI smoke execution. Compatibility is covered by focused Polars unit tests for downscale, feature generation, merge, scale, and feature-selection semantics.
```

- [ ] **Step 5: Record manual timing evidence**

Run the selected representative preprocessing command before and after the Polars migration if a representative input dataset exists. Append the result to `compatibility-notes.md` in this format:

```markdown
## Manual timing: representative preprocessing path

- Command: `<exact command>`
- Input dataset: `<symbol/date/frequency/path>`
- Before pandas runtime: `<seconds>`
- After Polars runtime: `<seconds>`
- Improvement: `<percentage>`
- Meets expected 30% improvement: `<yes/no>`
```

If the representative dataset is unavailable, write the exact missing path and do not mark the 30% acceptance item complete.

- [ ] **Step 6: Run diff hygiene**

Run: `git diff --check`

Expected: no whitespace errors.

- [ ] **Step 7: Commit Task 6 documentation sync**

```bash
git add openspec/changes/migrate-operator-futures-to-polars/tasks.md openspec/changes/migrate-operator-futures-to-polars/plan-ready.md docs/superpowers/plans/2026-06-21-migrate-operator-futures-to-polars.md openspec/changes/migrate-operator-futures-to-polars/compatibility-notes.md
git commit -m "docs: record polars migration validation plan"
```

- [ ] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

---

## Self-Review

**Spec coverage:** Task 1 covers the required Polars dependency and compatibility helper. Task 2 covers Binance orderbook and derivative ticker downscale. Task 3 covers Binance base, cross-section, and time features. Task 4 covers merge, concat, scale/save, and feature-selection paths. Task 5 covers commodity main-contract, downscale, schema/config, and market-type branches. Task 6 covers strict validation, full tests, smoke commands, and manual timing evidence.

**Placeholder scan:** The plan contains no unresolved placeholder tokens. Any unavailable dataset path must be recorded with the exact missing path during Task 6 rather than left implicit.

**Type consistency:** Public migrated helpers consistently accept and return `polars.DataFrame`. Feather IO uses Polars IPC read/write methods. Compatibility tests use `rtol=1e-12, atol=1e-12`, matching the OpenSpec delta.
