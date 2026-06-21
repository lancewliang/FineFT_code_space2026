# Fix Commodity Quote Session Gap Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make commodity quote gap validation session-aware so non-trading gaps such as `2025-10-31 23:05:00` after a night session do not fail preprocessing, while gaps inside a valid trading session still fail fast.

**Architecture:** Extend commodity configuration with explicit trading sessions, then use that session model only in quote gap validation. Keep `TradingDay` date-range filtering and `ActionDay + UpdateTime` event timestamps unchanged.

**Tech Stack:** Python dataclasses, `datetime.time`, Polars, pytest, OpenSpec.

**Traceability (sddflow):**
- plan-ready: `openspec/changes/fix-commodity-quote-session-gap-validation/plan-ready.md`
- tasks: `openspec/changes/fix-commodity-quote-session-gap-validation/tasks.md`
- plan: `docs/superpowers/plans/2026-06-22-fix-commodity-quote-session-gap-validation.md`

---

### Task 1: 商品交易 session 配置

> **trace:** plan-ready.md → `### Task 1: 商品交易 session 配置` | tasks.md → `- [ ] 1.0 商品交易 session 配置完成（与 `plan-ready.md` Task 1 和 superpowers plan Task 1 同步）`
> **sync:** tasks.md → `- [ ] 1.0 商品交易 session 配置完成（与 `plan-ready.md` Task 1 和 superpowers plan Task 1 同步）` | plan-ready.md → `### Task 1: 商品交易 session 配置`

**Files:**
- Modify: `data_preprocess/operator_futures/commodity/config.py:1-35`
- Test: `data_preprocess/tests/test_commodity_config_schema.py:1-39`

- [x] **Step 1: Write failing config tests for trading sessions**

Add these assertions and tests to `data_preprocess/tests/test_commodity_config_schema.py`:

```python
from datetime import time
```

Update `test_fu_config_contract()`:

```python
    assert tuple((session.start, session.end) for session in config.trading_sessions) == (
        (time(9, 0), time(10, 15)),
        (time(10, 30), time(11, 30)),
        (time(13, 30), time(15, 0)),
        (time(21, 0), time(23, 0)),
    )
```

Add these tests after `test_commodity_config_rejects_non_positive_contract_unit()`:

```python
def test_commodity_config_rejects_empty_trading_sessions():
    with pytest.raises(ValueError, match="trading_sessions must not be empty"):
        CommodityConfig(
            symbol="bad",
            display_name="bad",
            dataset_name="bad",
            orderbook_depth=5,
            funding_enabled=False,
            buy_fee_rate=0.0001,
            sell_fee_rate=0.0003,
            main_contract_months=(1,),
            contract_unit=10,
            use_contract_multiplier=False,
            trading_sessions=(),
        )


def test_commodity_config_rejects_invalid_trading_session_bounds():
    with pytest.raises(ValueError, match="trading session start must be before end"):
        CommodityConfig(
            symbol="bad",
            display_name="bad",
            dataset_name="bad",
            orderbook_depth=5,
            funding_enabled=False,
            buy_fee_rate=0.0001,
            sell_fee_rate=0.0003,
            main_contract_months=(1,),
            contract_unit=10,
            use_contract_multiplier=False,
            trading_sessions=((time(9, 0), time(9, 0)),),
        )
```

- [x] **Step 2: Run config tests to verify they fail**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_commodity_config_schema.py -q
```

Expected: FAIL because `CommodityConfig` has no `trading_sessions` field or because tuple session values have no `.start` / `.end`.

- [x] **Step 3: Add TradingSession and fu session config**

Modify `data_preprocess/operator_futures/commodity/config.py`:

```python
from dataclasses import dataclass
from datetime import time
from typing import Dict, Tuple


@dataclass(frozen=True)
class TradingSession:
    start: time
    end: time

    def __post_init__(self) -> None:
        if self.start >= self.end:
            raise ValueError("trading session start must be before end")
```

Add the field to `CommodityConfig` after `use_contract_multiplier`:

```python
    trading_sessions: Tuple[TradingSession, ...]
```

Extend `CommodityConfig.__post_init__()`:

```python
        if not self.trading_sessions:
            raise ValueError("trading_sessions must not be empty")
```

Update the `fu` config:

```python
        use_contract_multiplier=False,
        trading_sessions=(
            TradingSession(time(9, 0), time(10, 15)),
            TradingSession(time(10, 30), time(11, 30)),
            TradingSession(time(13, 30), time(15, 0)),
            TradingSession(time(21, 0), time(23, 0)),
        ),
```

- [x] **Step 4: Update invalid config tests to pass TradingSession values**

In `data_preprocess/tests/test_commodity_config_schema.py`, import `TradingSession`:

```python
from operator_futures.commodity.config import (
    CommodityConfig,
    TradingSession,
    get_commodity_config,
)
```

In `test_commodity_config_rejects_non_positive_contract_unit()`, add:

```python
            trading_sessions=(TradingSession(time(9, 0), time(10, 15)),),
```

In the empty-session test, keep `trading_sessions=()`.

In the invalid-bounds test, replace the raw tuple with:

```python
            trading_sessions=(TradingSession(time(9, 0), time(9, 0)),),
```

- [x] **Step 5: Run config tests to verify they pass**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_commodity_config_schema.py -q
```

Expected: PASS for all tests in `test_commodity_config_schema.py`.

- [x] **Step 6: Commit Task 1**

```bash
git add data_preprocess/operator_futures/commodity/config.py data_preprocess/tests/test_commodity_config_schema.py
git commit -m "feat: add commodity trading sessions"
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 2: Quote gap session-aware 校验

> **trace:** plan-ready.md → `### Task 2: Quote gap session-aware 校验` | tasks.md → `- [ ] 2.0 Quote gap session-aware 校验完成（与 `plan-ready.md` Task 2 和 superpowers plan Task 2 同步）`
> **sync:** tasks.md → `- [ ] 2.0 Quote gap session-aware 校验完成（与 `plan-ready.md` Task 2 和 superpowers plan Task 2 同步）` | plan-ready.md → `### Task 2: Quote gap session-aware 校验`

**Files:**
- Modify: `data_preprocess/operator_futures/commodity/downscale.py:1-410`
- Test: `data_preprocess/tests/test_commodity_downscale.py:247-270`

- [x] **Step 1: Write failing tests for cross-session and in-session gaps**

In `data_preprocess/tests/test_commodity_downscale.py`, replace `test_intermediate_empty_quote_window_fails_fast()` with:

```python
def test_cross_session_quote_gap_does_not_fail():
    second = pl.DataFrame(
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

    result = downscale_quote_features(second, "5min")

    assert result["timestamp"].to_list() == [
        datetime(2025, 10, 31, 23, 0, 0),
        datetime(2025, 11, 3, 9, 0, 0),
    ]


def test_intermediate_empty_quote_window_in_same_session_fails_fast():
    second = pl.DataFrame(
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

    with pytest.raises(ValueError, match="2023-01-03 09:05:00"):
        downscale_quote_features(second, "5min")
```

- [x] **Step 2: Run downscale tests to verify the new cross-session test fails**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_commodity_downscale.py::test_cross_session_quote_gap_does_not_fail data_preprocess/tests/test_commodity_downscale.py::test_intermediate_empty_quote_window_in_same_session_fails_fast -q
```

Expected: FAIL for `test_cross_session_quote_gap_does_not_fail` with `Target window has no quote snapshots: 2025-10-31 23:05:00`; PASS or still fail-fast for the same-session gap test.

- [x] **Step 3: Add session helper functions**

Modify imports in `data_preprocess/operator_futures/commodity/downscale.py`:

```python
from datetime import datetime, timedelta
from typing import List
```

Change the config import:

```python
from .config import TradingSession, get_commodity_config
```

Add these helpers after `_target_freq_delta()`:

```python
def _session_for_timestamp(
    timestamp: datetime, sessions: tuple[TradingSession, ...]
) -> TradingSession | None:
    timestamp_time = timestamp.time()
    for session in sessions:
        if session.start <= timestamp_time <= session.end:
            return session
    return None


def _same_trading_session(
    previous: datetime,
    current: datetime,
    sessions: tuple[TradingSession, ...],
) -> bool:
    previous_session = _session_for_timestamp(previous, sessions)
    current_session = _session_for_timestamp(current, sessions)
    return (
        previous_session is not None
        and previous_session == current_session
        and previous.date() == current.date()
    )
```

- [x] **Step 4: Make quote gap validation session-aware**

Change the function signature:

```python
def downscale_quote_features(
    second_df: pl.DataFrame, target_freq: str, symbol: str = "fu"
) -> pl.DataFrame:
```

Replace the gap validation block with:

```python
    result = _resample(quote, target_freq, aggs)
    timestamps = result["timestamp"].to_list()
    target_delta = _target_freq_delta(target_freq)
    trading_sessions = get_commodity_config(symbol).trading_sessions
    for previous, current in zip(timestamps, timestamps[1:]):
        missing = previous + target_delta
        if (
            current - previous > target_delta
            and _same_trading_session(previous, current, trading_sessions)
        ):
            raise ValueError(f"Target window has no quote snapshots: {missing}")

    empty_windows = result.filter(pl.col("nquote") == 0)
    if empty_windows.height:
        first = empty_windows.item(0, "timestamp")
        if _session_for_timestamp(first, trading_sessions) is not None:
            raise ValueError(f"Target window has no quote snapshots: {first}")
    return result
```

- [x] **Step 5: Run focused downscale tests**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_commodity_downscale.py::test_empty_quote_window_fails_fast data_preprocess/tests/test_commodity_downscale.py::test_cross_session_quote_gap_does_not_fail data_preprocess/tests/test_commodity_downscale.py::test_intermediate_empty_quote_window_in_same_session_fails_fast -q
```

Expected: PASS. Empty input still raises, cross-session gap does not raise, same-session gap raises.

- [x] **Step 6: Run full commodity downscale tests**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_commodity_downscale.py -q
```

Expected: PASS for the full commodity downscale test file.

- [x] **Step 7: Commit Task 2**

```bash
git add data_preprocess/operator_futures/commodity/downscale.py data_preprocess/tests/test_commodity_downscale.py
git commit -m "fix: validate commodity quote gaps by session"
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 3: 规格与回归验证

> **trace:** plan-ready.md → `### Task 3: 规格与回归验证` | tasks.md → `- [ ] 3.0 规格与回归验证完成（与 `plan-ready.md` Task 3 和 superpowers plan Task 3 同步）`
> **sync:** tasks.md → `- [ ] 3.0 规格与回归验证完成（与 `plan-ready.md` Task 3 和 superpowers plan Task 3 同步）` | plan-ready.md → `### Task 3: 规格与回归验证`

**Files:**
- Modify: `openspec/changes/fix-commodity-quote-session-gap-validation/tasks.md`
- Modify: `openspec/changes/fix-commodity-quote-session-gap-validation/plan-ready.md`
- Modify: `docs/superpowers/plans/2026-06-22-fix-commodity-quote-session-gap-validation.md`

- [x] **Step 1: Run OpenSpec strict validation**

Run:

```bash
openspec validate fix-commodity-quote-session-gap-validation --strict
```

Expected:

```text
Change 'fix-commodity-quote-session-gap-validation' is valid
```

- [x] **Step 2: Run commodity config and downscale regression tests**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_commodity_config_schema.py data_preprocess/tests/test_commodity_downscale.py -q
```

Expected: PASS for both test files.

- [x] **Step 3: Run diff whitespace check**

Run:

```bash
git diff --check
```

Expected: no output and exit code 0.

- [x] **Step 4: Mark OpenSpec task checkboxes complete after verification passes**

In `openspec/changes/fix-commodity-quote-session-gap-validation/tasks.md`, update the task-level lines:

```markdown
- [x] 1.0 商品交易 session 配置完成（与 `plan-ready.md` Task 1 和 superpowers plan Task 1 同步）
- [x] 2.0 Quote gap session-aware 校验完成（与 `plan-ready.md` Task 2 和 superpowers plan Task 2 同步）
- [x] 3.0 规格与回归验证完成（与 `plan-ready.md` Task 3 和 superpowers plan Task 3 同步）
```

Also mark the completed substeps `[x]` if they were executed successfully.

- [x] **Step 5: Mark plan-ready task checkboxes complete after verification passes**

In `openspec/changes/fix-commodity-quote-session-gap-validation/plan-ready.md`, update:

```markdown
- [x] **任务完成**（与 superpowers plan `Task 1`、`tasks.md` 对应条目同步勾选）
- [x] **任务完成**（与 superpowers plan `Task 2`、`tasks.md` 对应条目同步勾选）
- [x] **任务完成**（与 superpowers plan `Task 3`、`tasks.md` 对应条目同步勾选）
```

- [x] **Step 6: Mark this plan's Task complete checkboxes after all steps pass**

In `docs/superpowers/plans/2026-06-22-fix-commodity-quote-session-gap-validation.md`, update each task-ending line:

```markdown
- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）
```

- [x] **Step 7: Commit verification documents if implementation uses commits**

```bash
git add openspec/changes/fix-commodity-quote-session-gap-validation/tasks.md openspec/changes/fix-commodity-quote-session-gap-validation/plan-ready.md docs/superpowers/plans/2026-06-22-fix-commodity-quote-session-gap-validation.md
git commit -m "docs: record commodity quote session validation progress"
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

## Self-Review

Spec coverage:
- 商品配置新增交易 session：Task 1 covers config field, `fu` values, and validation tests.
- 同一 session 内 quote 缺口 fail-fast：Task 2 covers the `09:00 -> 09:10` regression test and validation logic.
- 跨 session、跨自然日、周末或休市间隔不报错：Task 2 covers `2025-10-31 23:00 -> 2025-11-03 09:00`.
- 空 quote 输入 fail-fast：Task 2 focused test keeps `test_empty_quote_window_fails_fast`.
- OpenSpec and regression verification: Task 3 covers strict validation, pytest, and diff checks.

Placeholder scan:
- No placeholder markers are present.
- All code-changing steps include concrete code blocks.
- All verification steps include exact commands and expected results.

Type consistency:
- `TradingSession.start` and `TradingSession.end` are `datetime.time` values.
- `CommodityConfig.trading_sessions` is `Tuple[TradingSession, ...]`.
- `downscale_quote_features()` keeps backward compatibility with default `symbol="fu"`.
