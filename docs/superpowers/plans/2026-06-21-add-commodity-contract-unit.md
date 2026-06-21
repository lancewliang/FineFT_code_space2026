# Add Commodity Contract Unit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add commodity-level contract units so fuel-oil (`fu`) trade-derived prices use `Turnover / Volume / 10` while raw `tradeval` remains unchanged.

**Architecture:** Store the contract unit in `CommodityConfig` and validate it at construction time. Commodity base-feature downscaling reads the unit through the existing `symbol` flow, applies it only to price calculations, and leaves raw turnover deltas intact.

**Tech Stack:** Python 3.10, Polars, pytest, OpenSpec, existing `data_preprocess/operator_futures/commodity` modules.

**Traceability (sddflow):**
- plan-ready: `openspec/changes/add-commodity-contract-unit/plan-ready.md`
- tasks: `openspec/changes/add-commodity-contract-unit/tasks.md`
- plan: `docs/superpowers/plans/2026-06-21-add-commodity-contract-unit.md`

---

### Task 1: 商品配置合约交易单位

> **trace:** plan-ready.md → `### Task 1: 商品配置合约交易单位` | tasks.md → ``- [ ] 1.0 商品配置合约交易单位完成（与 `plan-ready.md` Task 1 和 superpowers plan Task 1 同步）``
> **sync:** tasks.md → ``- [ ] 1.0 商品配置合约交易单位完成（与 `plan-ready.md` Task 1 和 superpowers plan Task 1 同步）`` | plan-ready.md → `### Task 1: 商品配置合约交易单位`

**Files:**
- Modify: `data_preprocess/operator_futures/commodity/config.py`
- Modify: `data_preprocess/tests/test_commodity_config_schema.py`

- [x] **Step 1: Write the failing config tests**

In `data_preprocess/tests/test_commodity_config_schema.py`, update the imports and config test, then add the invalid-unit test:

```python
import pytest

from operator_futures.commodity.config import CommodityConfig, get_commodity_config
```

```python
def test_fu_config_contract():
    config = get_commodity_config("fu")

    assert config.symbol == "fu"
    assert config.dataset_name == "fu"
    assert config.orderbook_depth == 5
    assert config.funding_enabled is False
    assert config.buy_fee_rate == 0.0001
    assert config.sell_fee_rate == 0.0003
    assert config.main_contract_months == tuple(range(1, 13))
    assert config.contract_unit == 10
    assert config.use_contract_multiplier is False
```

```python
def test_commodity_config_rejects_non_positive_contract_unit():
    with pytest.raises(ValueError, match="contract_unit must be positive"):
        CommodityConfig(
            symbol="bad",
            display_name="bad",
            dataset_name="bad",
            orderbook_depth=5,
            funding_enabled=False,
            buy_fee_rate=0.0001,
            sell_fee_rate=0.0003,
            main_contract_months=(1,),
            contract_unit=0,
            use_contract_multiplier=False,
        )
```

- [x] **Step 2: Run config tests to verify they fail**

Run:

```bash
conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_config_schema.py::test_fu_config_contract data_preprocess/tests/test_commodity_config_schema.py::test_commodity_config_rejects_non_positive_contract_unit -q
```

Expected: FAIL. `test_fu_config_contract` fails because `CommodityConfig` has no `contract_unit`, and the invalid-unit test fails because the dataclass does not accept or validate that field yet.

- [x] **Step 3: Add `contract_unit` to commodity config**

In `data_preprocess/operator_futures/commodity/config.py`, update `CommodityConfig` and the `fu` config:

```python
@dataclass(frozen=True)
class CommodityConfig:
    symbol: str
    display_name: str
    dataset_name: str
    orderbook_depth: int
    funding_enabled: bool
    buy_fee_rate: float
    sell_fee_rate: float
    main_contract_months: Tuple[int, ...]
    contract_unit: float
    use_contract_multiplier: bool

    def __post_init__(self) -> None:
        if self.contract_unit <= 0:
            raise ValueError("contract_unit must be positive")
```

```python
COMMODITY_CONFIGS: Dict[str, CommodityConfig] = {
    "fu": CommodityConfig(
        symbol="fu",
        display_name="燃料油",
        dataset_name="fu",
        orderbook_depth=5,
        funding_enabled=False,
        buy_fee_rate=0.0001,
        sell_fee_rate=0.0003,
        main_contract_months=tuple(range(1, 13)),
        contract_unit=10,
        use_contract_multiplier=False,
    )
}
```

- [x] **Step 4: Run config tests to verify they pass**

Run:

```bash
conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_config_schema.py -q
```

Expected: PASS. The test file reports all tests passing, including `contract_unit == 10` and the `ValueError` path.

- [x] **Step 5: Commit Task 1 changes**

Run:

```bash
git add data_preprocess/operator_futures/commodity/config.py data_preprocess/tests/test_commodity_config_schema.py
git commit -m "feat: add commodity contract unit config"
```

Expected: commit succeeds with only the config and config-schema test changes staged.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 2: 商品成交价格口径修正

> **trace:** plan-ready.md → `### Task 2: 商品成交价格口径修正` | tasks.md → ``- [ ] 2.0 商品成交价格口径修正完成（与 `plan-ready.md` Task 2 和 superpowers plan Task 2 同步）``
> **sync:** tasks.md → ``- [ ] 2.0 商品成交价格口径修正完成（与 `plan-ready.md` Task 2 和 superpowers plan Task 2 同步）`` | plan-ready.md → `### Task 2: 商品成交价格口径修正`

**Files:**
- Modify: `data_preprocess/operator_futures/commodity/downscale.py`
- Modify: `data_preprocess/operator_futures/commodity/downscale_single_day.py`
- Modify: `data_preprocess/operator_futures/commodity/downscale_continuous_by_trading_day.py`
- Modify: `data_preprocess/tests/test_commodity_downscale.py`

- [x] **Step 1: Write the failing downscale test**

In `data_preprocess/tests/test_commodity_downscale.py`, add this test after `test_derivative_reference_falls_back_to_midprice_for_invalid_lastprice`:

```python
def test_base_features_use_contract_unit_for_prices_but_keep_raw_tradeval():
    second = pl.DataFrame(
        {
            "timestamp": [
                datetime(2023, 1, 3, 9, 0, 0),
                datetime(2023, 1, 3, 9, 0, 1),
                datetime(2023, 1, 3, 9, 0, 2),
            ],
            "InstrumentID": ["fu2302", "fu2302", "fu2302"],
            "BidPrice1": [2599.0, 2599.0, 2600.0],
            "AskPrice1": [2601.0, 2601.0, 2602.0],
            "LastPrice": [2600.0, 2600.0, 2601.0],
            "Volume": [0, 1, 2],
            "Turnover": [0.0, 26000.0, 52010.0],
        }
    )

    base = downscale_base_features(second, "5min", "fu").filter(pl.col("volume") > 0)

    assert base.item(0, "open") == 2600.0
    assert base.item(0, "close") == 2601.0
    assert base.item(0, "volume") == 2
    assert base.item(0, "tradeval") == 52010.0
    assert base.item(0, "vwap") == 2600.5
```

- [x] **Step 2: Run the new downscale test to verify it fails**

Run:

```bash
conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_downscale.py::test_base_features_use_contract_unit_for_prices_but_keep_raw_tradeval -q
```

Expected: FAIL with `TypeError` because `downscale_base_features` currently accepts only `second_df` and `target_freq`, not `symbol`.

- [x] **Step 3: Read commodity config in downscale code**

In `data_preprocess/operator_futures/commodity/downscale.py`, add the config import below the Polars import:

```python
from .config import get_commodity_config
from .main_contract import with_normalized_timestamp
```

Change `_second_trade_frame` and `downscale_base_features` to accept a contract unit:

```python
def _second_trade_frame(second_df: pl.DataFrame, contract_unit: float) -> pl.DataFrame:
    frame = second_df.sort("timestamp").with_columns(
        pl.col("Volume")
        .cast(pl.Float64, strict=False)
        .diff()
        .alias("second_volume"),
        pl.col("Turnover")
        .cast(pl.Float64, strict=False)
        .diff()
        .alias("second_tradeval"),
    )
    invalid_rows = frame.filter(
        (pl.col("second_volume") > 0)
        & (pl.col("second_tradeval").is_null() | (pl.col("second_tradeval") <= 0))
    )
    if invalid_rows.height:
        row = invalid_rows.row(0, named=True)
        raise ValueError(
            "Invalid turnover delta with positive volume: "
            f"timestamp={row.get('timestamp')}, contract={row.get('InstrumentID')}, "
            f"second_volume={row['second_volume']}, "
            f"second_tradeval={row['second_tradeval']}"
        )

    frame = frame.with_columns(
        pl.when(pl.col("second_volume") > 0)
        .then(pl.col("second_tradeval") / pl.col("second_volume") / contract_unit)
        .otherwise(None)
        .alias("second_avg_price")
    ).with_row_index("_row_nr")
```

Keep the existing direction calculation and return block after that snippet unchanged.

Replace the `downscale_base_features` function header and first lines with:

```python
def downscale_base_features(
    second_df: pl.DataFrame, target_freq: str, symbol: str = "fu"
) -> pl.DataFrame:
    contract_unit = get_commodity_config(symbol).contract_unit
    frame = _with_reference_price(
        _second_trade_frame(second_df, contract_unit)
    ).with_columns(
```

Replace the `vwap` expression with:

```python
    return grouped.with_columns(
        pl.when(pl.col("volume") > 0)
        .then(pl.col("tradeval") / pl.col("volume") / contract_unit)
        .otherwise(pl.col("close"))
        .alias("vwap"),
        pl.col("awap").alias("twap"),
    ).select(
```

- [x] **Step 4: Pass `symbol` from commodity downscale entry points**

In `data_preprocess/operator_futures/commodity/downscale_single_day.py`, replace the base-feature call:

```python
    base = downscale_base_features(second, args.target_freq, args.symbol)
```

In `data_preprocess/operator_futures/commodity/downscale_continuous_by_trading_day.py`, replace the `BASE_FEATURE` output expression:

```python
            "BASE_FEATURE": downscale_base_features(second, target_freq, symbol),
```

- [x] **Step 5: Run the new downscale test to verify it passes**

Run:

```bash
conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_downscale.py::test_base_features_use_contract_unit_for_prices_but_keep_raw_tradeval -q
```

Expected: PASS. The test confirms `open=2600.0`, `close=2601.0`, raw `tradeval=52010.0`, and `vwap=2600.5`.

- [x] **Step 6: Run commodity downscale regression tests**

Run:

```bash
conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_downscale.py data_preprocess/tests/test_commodity_main_contract_cli.py::test_downscale_single_day_cli_accepts_output_root_alias -q
```

Expected: PASS. Existing sample-file and CLI tests still create base features successfully with `symbol=fu`.

- [x] **Step 7: Commit Task 2 changes**

Run:

```bash
git add data_preprocess/operator_futures/commodity/downscale.py data_preprocess/operator_futures/commodity/downscale_single_day.py data_preprocess/operator_futures/commodity/downscale_continuous_by_trading_day.py data_preprocess/tests/test_commodity_downscale.py
git commit -m "fix: apply commodity contract unit to trade prices"
```

Expected: commit succeeds with only downscale code and downscale tests staged.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 3: 规格与回归验证

> **trace:** plan-ready.md → `### Task 3: 规格与回归验证` | tasks.md → ``- [ ] 3.0 规格与回归验证完成（与 `plan-ready.md` Task 3 和 superpowers plan Task 3 同步）``
> **sync:** tasks.md → ``- [ ] 3.0 规格与回归验证完成（与 `plan-ready.md` Task 3 和 superpowers plan Task 3 同步）`` | plan-ready.md → `### Task 3: 规格与回归验证`

**Files:**
- Modify: `docs/上海商品交易所/commodity_futures_preprocess.md`
- Modify: `openspec/changes/add-commodity-contract-unit/tasks.md`
- Modify: `openspec/changes/add-commodity-contract-unit/plan-ready.md`
- Modify: `docs/superpowers/plans/2026-06-21-add-commodity-contract-unit.md`

- [ ] **Step 1: Update the commodity futures preprocessing doc**

In `docs/上海商品交易所/commodity_futures_preprocess.md`, replace the trades-feature bullet with:

```markdown
- trades 特征由每秒累计 `Volume` 与 `Turnover` 差分估计，`second_avg_price = Turnover.diff() / Volume.diff() / contract_unit`；`tradeval` 保留原始成交额差分。
```

Replace the fee/config bullets with:

```markdown
- 买入费率：`0.0001`
- 卖出费率：`0.0003`
- 合约交易单位：`10`
- `contract_unit` 仅用于商品成交均价和 `vwap` 的价格口径修正，不用于 PnL、保证金或手续费。
```

- [ ] **Step 2: Run OpenSpec strict validation**

Run:

```bash
openspec validate add-commodity-contract-unit --strict
```

Expected: PASS with `Change 'add-commodity-contract-unit' is valid`.

- [ ] **Step 3: Run focused commodity tests**

Run:

```bash
conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_config_schema.py data_preprocess/tests/test_commodity_downscale.py -q
```

Expected: PASS. Config schema and downscale tests confirm `contract_unit=10`, price correction, and raw `tradeval` preservation.

- [ ] **Step 4: Run diff formatting check**

Run:

```bash
git diff --check
```

Expected: no output and exit code `0`.

- [ ] **Step 5: Mark implementation task checkboxes after verification**

After Steps 2-4 pass, change the task-level checkbox for Task 3 from unchecked to checked in `openspec/changes/add-commodity-contract-unit/tasks.md`, `openspec/changes/add-commodity-contract-unit/plan-ready.md`, and this plan's Task 3 `Task complete` line. Do this only after every Step in Task 3 is checked.

- [ ] **Step 6: Commit Task 3 changes**

Run:

```bash
git add docs/上海商品交易所/commodity_futures_preprocess.md openspec/changes/add-commodity-contract-unit/tasks.md openspec/changes/add-commodity-contract-unit/plan-ready.md docs/superpowers/plans/2026-06-21-add-commodity-contract-unit.md
git commit -m "docs: record commodity contract unit plan"
```

Expected: commit succeeds with docs, OpenSpec task status, plan-ready status, and plan status changes staged.

- [ ] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）
