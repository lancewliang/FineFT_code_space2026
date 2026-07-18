# Enhance Limit Single-Sided Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make legal commodity futures limit/single-sided order books produce finite snapshot features while exposing price-limit columns as reward/execution fields.

**Architecture:** Extend the commodity current-market data contract first, then make snapshot feature math total-size aware. Keep illegal-value validators strict and cover the behavior with focused tests before changing implementation.

**Tech Stack:** Python, Polars, NumPy, pytest, OpenSpec. Run Python commands after `conda activate finetf`.

**Traceability (sddflow):**
- plan-ready: `openspec/changes/enhance-limit-single-sided-features/plan-ready.md`
- tasks: `openspec/changes/enhance-limit-single-sided-features/tasks.md`
- plan: `docs/superpowers/plans/2026-07-19-enhance-limit-single-sided-features.md`

---

### Task 1: 扩展商品涨跌停价 reward/execution 合同

> **trace:** plan-ready.md → `### Task 1: 扩展商品涨跌停价 reward/execution 合同` | tasks.md → `- [ ] 1.1 Extend commodity orderbook downscale outputs and reward/execution manifest to include `LowerLimitPrice` and `UpperLimitPrice`.`
> **sync:** tasks.md → `- [ ] 1.1 Extend commodity orderbook downscale outputs and reward/execution manifest to include `LowerLimitPrice` and `UpperLimitPrice`.` | plan-ready.md → `### Task 1: 扩展商品涨跌停价 reward/execution 合同`

**Files:**
- Modify: `data_preprocess/operator_futures/commodity/downscale.py`
- Modify: `data_preprocess/operator_futures/commodity/schema.py`
- Test: `data_preprocess/tests/test_commodity_downscale.py`
- Test: `data_preprocess/tests/test_commodity_config_schema.py`
- Test: `data_preprocess/tests/test_commodity_feature_pipeline.py`

- [x] **Step 1: Write failing tests for price-limit reward columns**

Add or update these assertions.

In `data_preprocess/tests/test_commodity_config_schema.py`, update `test_reward_execution_manifest_for_depth_five`:

```python
def test_reward_execution_manifest_for_depth_five():
    columns = get_reward_execution_columns(depth=5)

    assert columns[0] == "timestamp"
    assert "mark_price" in columns
    assert "funding_rate" in columns
    assert "contract" in columns
    assert "ask5_price" in columns
    assert "bid5_size" in columns
    assert "ask6_price" not in columns
    assert columns.index("LowerLimitPrice") < columns.index("funding_timestamp")
    assert columns.index("UpperLimitPrice") < columns.index("funding_timestamp")
    assert columns.index("LowerLimitPrice") > columns.index("bid5_size")
    assert columns.index("UpperLimitPrice") > columns.index("bid5_size")
    assert len(columns) == 1 + 1 + 20 + 2 + 5
```

In `data_preprocess/tests/test_commodity_feature_pipeline.py`, update `test_manifest_replaces_first_106_reward_columns`:

```python
def test_manifest_replaces_first_106_reward_columns():
    reward_columns = get_reward_execution_columns(depth=5)

    assert len(reward_columns) == 29
    assert "contract" in reward_columns
    assert "ask5_price" in reward_columns
    assert "LowerLimitPrice" in reward_columns
    assert "UpperLimitPrice" in reward_columns
    assert "ask25_price" not in reward_columns
```

Add this test near the existing downscale orderbook tests in `data_preprocess/tests/test_commodity_downscale.py`:

```python
def test_downscale_orderbook_preserves_price_limit_columns():
    second = pl.DataFrame(
        {
            "timestamp": [
                datetime(2026, 2, 2, 9, 0, 1),
                datetime(2026, 2, 2, 9, 4, 59),
            ],
            "LowerLimitPrice": [2500.0, 2501.0],
            "UpperLimitPrice": [3100.0, 3101.0],
            "AskPrice1": [3001.0, 3002.0],
            "AskVolume1": [10, 11],
            "BidPrice1": [2999.0, 3000.0],
            "BidVolume1": [20, 21],
            "AskPrice2": [3003.0, 3004.0],
            "AskVolume2": [12, 13],
            "BidPrice2": [2998.0, 2999.0],
            "BidVolume2": [22, 23],
            "AskPrice3": [3005.0, 3006.0],
            "AskVolume3": [14, 15],
            "BidPrice3": [2997.0, 2998.0],
            "BidVolume3": [24, 25],
            "AskPrice4": [3007.0, 3008.0],
            "AskVolume4": [16, 17],
            "BidPrice4": [2996.0, 2997.0],
            "BidVolume4": [26, 27],
            "AskPrice5": [3009.0, 3010.0],
            "AskVolume5": [18, 19],
            "BidPrice5": [2995.0, 2996.0],
            "BidVolume5": [28, 29],
        }
    )

    out = downscale_orderbook(second, "5min", depth=5)
    row = out.filter(pl.col("timestamp") == datetime(2026, 2, 2, 9, 5, 0))

    assert row.item(0, "LowerLimitPrice") == 2501.0
    assert row.item(0, "UpperLimitPrice") == 3101.0
```

- [x] **Step 2: Run tests and verify they fail**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate finetf
pytest data_preprocess/tests/test_commodity_config_schema.py::test_reward_execution_manifest_for_depth_five data_preprocess/tests/test_commodity_feature_pipeline.py::test_manifest_replaces_first_106_reward_columns data_preprocess/tests/test_commodity_downscale.py::test_downscale_orderbook_preserves_price_limit_columns -q
```

Expected: FAIL because `LowerLimitPrice` and `UpperLimitPrice` are not yet in `downscale_orderbook()` output or `get_reward_execution_columns()`.

- [x] **Step 3: Implement price-limit columns**

In `data_preprocess/operator_futures/commodity/schema.py`, add a constant and include it in the manifest:

```python
PRICE_LIMIT_COLUMNS = [
    "LowerLimitPrice",
    "UpperLimitPrice",
]


def get_reward_execution_columns(depth: int) -> List[str]:
    return [
        "timestamp",
        "contract",
        *build_orderbook_columns(depth),
        *PRICE_LIMIT_COLUMNS,
        *DERIVATIVE_REFERENCE_COLUMNS,
    ]
```

In `data_preprocess/operator_futures/commodity/downscale.py`, update `downscale_orderbook()` after the depth loop:

```python
    for column in ("LowerLimitPrice", "UpperLimitPrice"):
        if column in second_df.columns:
            expressions.append(pl.col(column))
            output_columns.append(column)
```

Keep the existing resample aggregation:

```python
    result = _resample(
        renamed,
        target_freq,
        [pl.col(column).last().alias(column) for column in output_columns],
    )
```

- [x] **Step 4: Run tests and verify they pass**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate finetf
pytest data_preprocess/tests/test_commodity_config_schema.py::test_reward_execution_manifest_for_depth_five data_preprocess/tests/test_commodity_feature_pipeline.py::test_manifest_replaces_first_106_reward_columns data_preprocess/tests/test_commodity_downscale.py::test_downscale_orderbook_preserves_price_limit_columns -q
```

Expected: PASS.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 2: 增强单边盘口 snapshot 特征计算

> **trace:** plan-ready.md → `### Task 2: 增强单边盘口 snapshot 特征计算` | tasks.md → `- [ ] 1.2 Enhance snapshot feature generation for single-sided books, including `ask_side_empty` and `bid_side_empty`.`
> **sync:** tasks.md → `- [ ] 1.2 Enhance snapshot feature generation for single-sided books, including `ask_side_empty` and `bid_side_empty`.` | plan-ready.md → `### Task 2: 增强单边盘口 snapshot 特征计算`

**Files:**
- Modify: `data_preprocess/operator_futures/cross_section/base_feature_util.py`
- Test: `data_preprocess/tests/test_commodity_feature_pipeline.py`
- Test: `data_preprocess/tests/test_polars_feature_generation.py`

- [x] **Step 1: Write failing snapshot tests**

Add these helpers and tests to `data_preprocess/tests/test_commodity_feature_pipeline.py` near `_snapshot()`:

```python
def _single_sided_snapshot(empty_side: str):
    frame = _snapshot()
    if empty_side == "ask":
        for level in range(1, 6):
            frame[f"ask{level}_price"] = 3050.0
            frame[f"ask{level}_size"] = 0
    elif empty_side == "bid":
        for level in range(1, 6):
            frame[f"bid{level}_price"] = 2600.0
            frame[f"bid{level}_size"] = 0
    else:
        raise ValueError(empty_side)
    return frame


def test_snapshot_features_handle_empty_ask_side_without_nan():
    features = process_snapshot_features(_single_sided_snapshot("ask"), topk=3, depth=5)

    row = features.row(0, named=True)
    assert bool(row["ask_side_empty"]) is True
    assert bool(row["bid_side_empty"]) is False
    assert row["sell_wap"] == 3050.0
    assert row["buy_sell_wap_spread"] == row["buy_wap"] - row["sell_wap"]
    for level in range(1, 6):
        assert row[f"ask{level}_size_n"] == 0.0
    assert not features.select(pl.any_horizontal(pl.selectors.float().is_nan())).item()
    assert not features.select(pl.any_horizontal(pl.selectors.float().is_infinite())).item()


def test_snapshot_features_handle_empty_bid_side_without_nan():
    features = process_snapshot_features(_single_sided_snapshot("bid"), topk=3, depth=5)

    row = features.row(0, named=True)
    assert bool(row["ask_side_empty"]) is False
    assert bool(row["bid_side_empty"]) is True
    assert row["buy_wap"] == 2600.0
    assert row["buy_sell_wap_spread"] == row["buy_wap"] - row["sell_wap"]
    for level in range(1, 6):
        assert row[f"bid{level}_size_n"] == 0.0
    assert not features.select(pl.any_horizontal(pl.selectors.float().is_nan())).item()
    assert not features.select(pl.any_horizontal(pl.selectors.float().is_infinite())).item()


def test_snapshot_features_flag_normal_two_sided_book_as_not_empty():
    features = process_snapshot_features(_snapshot(), topk=3, depth=5)
    row = features.row(0, named=True)

    assert bool(row["ask_side_empty"]) is False
    assert bool(row["bid_side_empty"]) is False
    assert row["ask1_size_n"] == 1 / sum(range(1, 6))
    assert row["bid1_size_n"] == 2 / sum(range(2, 7))


def test_snapshot_features_reject_both_sides_empty():
    frame = _snapshot()
    for side in ("ask", "bid"):
        for level in range(1, 6):
            frame[f"{side}{level}_size"] = 0

    with pytest.raises(ValueError, match="both sides have zero total size"):
        process_snapshot_features(frame, topk=3, depth=5)
```

In `data_preprocess/tests/test_polars_feature_generation.py`, extend `test_cross_section_features_return_polars_with_timestamp`:

```python
    assert "ask_side_empty" in snapshot_features.columns
    assert "bid_side_empty" in snapshot_features.columns
```

- [x] **Step 2: Run tests and verify they fail**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate finetf
pytest data_preprocess/tests/test_commodity_feature_pipeline.py::test_snapshot_features_handle_empty_ask_side_without_nan data_preprocess/tests/test_commodity_feature_pipeline.py::test_snapshot_features_handle_empty_bid_side_without_nan data_preprocess/tests/test_commodity_feature_pipeline.py::test_snapshot_features_flag_normal_two_sided_book_as_not_empty data_preprocess/tests/test_commodity_feature_pipeline.py::test_snapshot_features_reject_both_sides_empty data_preprocess/tests/test_polars_feature_generation.py::test_cross_section_features_return_polars_with_timestamp -q
```

Expected: FAIL because side-empty columns do not exist and current empty-side math produces NaN.

- [x] **Step 3: Implement total-size-aware snapshot math**

In `data_preprocess/operator_futures/cross_section/base_feature_util.py`, modify the Polars `process_snapshot_features()` implementation. Use NumPy arrays already present in the function:

```python
    ask_total = np.sum(ask_size_array, axis=1)
    bid_total = np.sum(bid_size_array, axis=1)
    ask_side_empty = ask_total <= 0
    bid_side_empty = bid_total <= 0
    both_side_empty = ask_side_empty & bid_side_empty
    if np.any(both_side_empty):
        raise ValueError("both sides have zero total size")

    normalized_ask_size_array = np.divide(
        ask_size_array,
        ask_total.reshape(-1, 1),
        out=np.zeros_like(ask_size_array, dtype=float),
        where=ask_total.reshape(-1, 1) > 0,
    )
    normalized_bid_size_array = np.divide(
        bid_size_array,
        bid_total.reshape(-1, 1),
        out=np.zeros_like(bid_size_array, dtype=float),
        where=bid_total.reshape(-1, 1) > 0,
    )
```

Replace direct `wap_1` and `wap_2` divisions with guarded division:

```python
    wap_1_denominator = best_ask_size_array + best_bid_size_array
    data["wap_1"] = np.divide(
        best_ask_size_array * best_bid_price_array
        + best_bid_size_array * best_ask_price_array,
        wap_1_denominator,
        out=(best_ask_price_array + best_bid_price_array) / 2,
        where=wap_1_denominator > 0,
    )
    wap_2_denominator = bid_size_array[:, 1] + ask_size_array[:, 1]
    data["wap_2"] = np.divide(
        ask_size_array[:, 1] * bid_price_array[:, 1]
        + bid_size_array[:, 1] * ask_price_array[:, 1],
        wap_2_denominator,
        out=(ask_price_array[:, 1] + bid_price_array[:, 1]) / 2,
        where=wap_2_denominator > 0,
    )
```

After computing raw side WAPs, apply empty-side fallback:

```python
    sell_wap = np.sum(normalized_ask_size_array * ask_price_array, axis=1)
    buy_wap = np.sum(normalized_bid_size_array * bid_price_array, axis=1)
    data["sell_wap"] = np.where(ask_side_empty, best_ask_price_array, sell_wap)
    data["buy_wap"] = np.where(bid_side_empty, best_bid_price_array, buy_wap)
    data["buy_sell_wap_spread"] = data["buy_wap"] - data["sell_wap"]
```

Set totals and flags:

```python
    data["buy_volume_oe"] = bid_total
    data["sell_volume_oe"] = ask_total
    data["ask_side_empty"] = ask_side_empty
    data["bid_side_empty"] = bid_side_empty
```

Replace normalized size divisions:

```python
    for i in range(1, depth + 1):
        data[f"ask{i}_size_n"] = normalized_ask_size_array[:, i - 1]
        data[f"bid{i}_size_n"] = normalized_bid_size_array[:, i - 1]
```

Apply the same total-size logic to the earlier pandas compatibility implementation in the same file if tests or reference adapters still call it.

- [x] **Step 4: Run tests and verify they pass**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate finetf
pytest data_preprocess/tests/test_commodity_feature_pipeline.py::test_snapshot_features_handle_empty_ask_side_without_nan data_preprocess/tests/test_commodity_feature_pipeline.py::test_snapshot_features_handle_empty_bid_side_without_nan data_preprocess/tests/test_commodity_feature_pipeline.py::test_snapshot_features_flag_normal_two_sided_book_as_not_empty data_preprocess/tests/test_commodity_feature_pipeline.py::test_snapshot_features_reject_both_sides_empty data_preprocess/tests/test_polars_feature_generation.py::test_cross_section_features_return_polars_with_timestamp -q
```

Expected: PASS.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 3: 更新列合同文档和 expected columns

> **trace:** plan-ready.md → `### Task 3: 更新列合同文档和 expected columns` | tasks.md → `- [ ] 1.3 Update expected-column helpers and documentation for the expanded reward and snapshot feature contracts.`
> **sync:** tasks.md → `- [ ] 1.3 Update expected-column helpers and documentation for the expanded reward and snapshot feature contracts.` | plan-ready.md → `### Task 3: 更新列合同文档和 expected columns`

**Files:**
- Modify: `data_preprocess/operator_futures/feature_validation/expected_columns.py`
- Modify: `docs/datapreprocess/5.SNAPSHOT_FEATURE_82_COLUMNS.md`
- Modify: `docs/datapreprocess/6.TIME_FEATURE_3375_COLUMNS.md`
- Modify: `docs/datapreprocess/1.DATA_PREPROCESS_REWARD_ENVIRONMENT_106_COLUMNS(挂单).md`
- Test: `data_preprocess/tests/test_feature_validation_compare_report.py`

- [x] **Step 1: Update expected-column tests or assertions**

Inspect current expected-column tests:

```bash
rg -n "82|106|3375|SNAPSHOT_FEATURE|reward_environment|BASE_FEATURE_COLUMNS|SNAPSHOT" data_preprocess/tests data_preprocess/operator_futures/feature_validation -S
```

Where tests assert old snapshot column counts, change expected count from 82 to 84. Where commodity reward manifest count is asserted, use 29 for depth=5.

- [x] **Step 2: Update expected columns helper**

In `data_preprocess/operator_futures/feature_validation/expected_columns.py`, add:

```python
"ask_side_empty",
"bid_side_empty",
```

to the snapshot feature column list after volume-side aggregate features or immediately before normalized size columns. Keep ordering aligned with `process_snapshot_features()`.

If the file has a reward/environment list for commodity or generic 25-depth orderbook columns, add:

```python
"LowerLimitPrice",
"UpperLimitPrice",
```

after depth-aware orderbook columns and before derivative reference columns.

- [x] **Step 3: Update snapshot feature docs**

In `docs/datapreprocess/5.SNAPSHOT_FEATURE_82_COLUMNS.md`, change the title and summary from 82 columns to 84 columns. Add a section:

```markdown
## 五、单边盘口标志（2 列）

| # | 列名 | 含义 | 计算公式 |
|---|------|------|----------|
| 83 | `ask_side_empty` | ask 侧是否无挂单量 | `ΣAS_l <= 0` |
| 84 | `bid_side_empty` | bid 侧是否无挂单量 | `ΣBS_l <= 0` |
```

Update formula notes for `sell_wap`, `buy_wap`, `buy_sell_wap_spread`, `ask{i}_size_n`, and `bid{i}_size_n` to describe empty-side fallback:

```markdown
当 `ΣAS_l <= 0` 时，`AS_l_norm = 0` 且 `sell_wap = AP_1`。
当 `ΣBS_l <= 0` 时，`BS_l_norm = 0` 且 `buy_wap = BP_1`。
```

- [x] **Step 4: Update reward/time docs**

In `docs/datapreprocess/1.DATA_PREPROCESS_REWARD_ENVIRONMENT_106_COLUMNS(挂单).md`, add a note that commodity depth=5 now uses explicit manifest columns rather than the old crypto 106-column positional convention. Include:

```markdown
商品期货 depth=5 reward/execution manifest 为：
`timestamp`、`contract`、20 个五档 orderbook 列、`LowerLimitPrice`、`UpperLimitPrice`、5 个 derivative reference 兼容列，共 29 列。
```

In `docs/datapreprocess/6.TIME_FEATURE_3375_COLUMNS.md`, add `ask_side_empty` and `bid_side_empty` only if time feature generation includes them. If implementation does not include side flags in time rolling price features, document that they are snapshot/state candidate columns but not part of the configured time price feature list.

- [x] **Step 5: Run expected-column tests**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate finetf
pytest data_preprocess/tests/test_feature_validation_compare_report.py data_preprocess/tests/test_commodity_config_schema.py data_preprocess/tests/test_commodity_feature_pipeline.py -q
```

Expected: PASS.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 4: 端到端聚焦验证与 OpenSpec 校验

> **trace:** plan-ready.md → `### Task 4: 端到端聚焦验证与 OpenSpec 校验` | tasks.md → `- [ ] 1.4 Add and run focused validation for reward columns, single-sided snapshot behavior, time feature input legality, and OpenSpec strict validation.`
> **sync:** tasks.md → `- [ ] 1.4 Add and run focused validation for reward columns, single-sided snapshot behavior, time feature input legality, and OpenSpec strict validation.` | plan-ready.md → `### Task 4: 端到端聚焦验证与 OpenSpec 校验`

**Files:**
- Modify: `data_preprocess/tests/test_time_operator_polars.py`
- Modify: `data_preprocess/tests/test_time_operator_pandas_create_feature.py`
- Modify: `openspec/changes/enhance-limit-single-sided-features/tasks.md`

- [x] **Step 1: Add time feature input validation regression**

In `data_preprocess/tests/test_time_operator_polars.py`, add a fixture row that includes enhanced single-sided columns and assert validation no longer fails for legal single-sided data. If the existing tests construct an invalid frame directly, add:

```python
def test_time_feature_input_accepts_enhanced_single_sided_snapshot():
    from operator_futures.data_quality import DataQualityValidator

    frame = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 10, 8, 9, 0)],
            "sell_wap": [3050.0],
            "buy_wap": [3049.98],
            "buy_sell_wap_spread": [-0.02],
            "ask1_size_n": [0.0],
            "bid1_size_n": [1.0],
            "ask_side_empty": [True],
            "bid_side_empty": [False],
            "LowerLimitPrice": [2500.0],
            "UpperLimitPrice": [3050.0],
        }
    )

    DataQualityValidator.validate_no_illegal_values(
        frame,
        stage="time_feature_input",
        contract="fu2411",
        trading_day="2023-01-01-2026-03-01",
    )
```

Mirror the same intent in `data_preprocess/tests/test_time_operator_pandas_create_feature.py` if that file has a pandas-wrapper validation path with a small fixture.

- [x] **Step 2: Run focused time validation tests**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate finetf
pytest data_preprocess/tests/test_time_operator_polars.py data_preprocess/tests/test_time_operator_pandas_create_feature.py -q
```

Expected: PASS.

- [x] **Step 3: Run combined focused preprocessing tests**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate finetf
pytest data_preprocess/tests/test_commodity_downscale.py data_preprocess/tests/test_commodity_config_schema.py data_preprocess/tests/test_commodity_feature_pipeline.py data_preprocess/tests/test_polars_feature_generation.py data_preprocess/tests/test_time_operator_polars.py data_preprocess/tests/test_time_operator_pandas_create_feature.py -q
```

Expected: PASS.

- [x] **Step 4: Validate OpenSpec**

Run:

```bash
openspec validate enhance-limit-single-sided-features --strict
```

Expected:

```text
Change 'enhance-limit-single-sided-features' is valid
```

- [x] **Step 5: Update task checkboxes after implementation**

After all implementation tests pass, update `openspec/changes/enhance-limit-single-sided-features/tasks.md` by changing these lines to checked:

```markdown
- [x] 1.1 Extend commodity orderbook downscale outputs and reward/execution manifest to include `LowerLimitPrice` and `UpperLimitPrice`.
- [x] 1.2 Enhance snapshot feature generation for single-sided books, including `ask_side_empty` and `bid_side_empty`.
- [x] 1.3 Update expected-column helpers and documentation for the expanded reward and snapshot feature contracts.
- [x] 1.4 Add and run focused validation for reward columns, single-sided snapshot behavior, time feature input legality, and OpenSpec strict validation.
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）
