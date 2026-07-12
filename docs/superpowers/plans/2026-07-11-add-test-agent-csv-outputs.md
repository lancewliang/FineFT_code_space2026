# Add Test Agent CSV Outputs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add readable aggregate CSV output and optional per-step trading detail CSV output for `test_agent_index.py`, backed by true execution metrics from the trading environment.

**Architecture:** The implementation keeps `test_agent_index.py` as the output orchestrator and extends the environment layer only where true execution metrics are produced. `futures_util.py` exposes a tuple-compatible result object so existing six-value consumers can be migrated deliberately while new code reads named execution metric fields.

**Tech Stack:** Python, pandas, numpy, pytest, OpenSpec, FineFT environment classes.

**Traceability (sddflow):**
- plan-ready: `openspec/changes/add-test-agent-csv-outputs/plan-ready.md`
- tasks: `openspec/changes/add-test-agent-csv-outputs/tasks.md`
- plan: `docs/superpowers/plans/2026-07-11-add-test-agent-csv-outputs.md`

---

### Task 1: Extend trading wallet-change results to expose true `commission_fee_step`, `realized_pnl_step`, and `slippage_step` while preserving existing fee semantics.

> **trace:** plan-ready.md → `### Task 1: Extend trading wallet-change results to expose true `commission_fee_step`, `realized_pnl_step`, and `slippage_step` while preserving existing fee semantics.` | tasks.md → `- [ ] 1.1 Extend trading wallet-change results to expose true `commission_fee_step`, `realized_pnl_step`, and `slippage_step` while preserving existing fee semantics.`
> **sync:** tasks.md → `- [ ] 1.1 Extend trading wallet-change results to expose true `commission_fee_step`, `realized_pnl_step`, and `slippage_step` while preserving existing fee semantics.` | plan-ready.md → `### Task 1: Extend trading wallet-change results to expose true `commission_fee_step`, `realized_pnl_step`, and `slippage_step` while preserving existing fee semantics.`

**Files:**
- Modify: `FineFT/env/env_class/futures_util.py`
- Modify: `FineFT/tests/env/test_commodity_env.py`

- [x] **Step 1: Add a failing wallet-change test for named execution metrics**

Add this assertion block to `FineFT/tests/env/test_commodity_env.py::test_wallet_change_can_use_buy_and_sell_fee_rates` after the existing `opened` call:

```python
    assert opened.commission_fee_step == pytest.approx(101.0 * 0.1)
    assert opened.realized_pnl_step == 0
    assert opened.slippage_step == pytest.approx(101.0 - 100.0 / 5)
```

Add this assertion block after the existing `closed` call:

```python
    assert closed.commission_fee_step == pytest.approx(99.0 * 0.3)
    assert closed.realized_pnl_step == pytest.approx(-1.0 + 99.0 - 100.0)
    assert closed.slippage_step == pytest.approx(100.0 - 99.0)
```

- [x] **Step 2: Run the focused test and verify it fails for missing attributes**

Run:

```bash
conda activate finetf && pytest FineFT/tests/env/test_commodity_env.py::test_wallet_change_can_use_buy_and_sell_fee_rates -q
```

Expected: failure mentioning that the returned tuple has no attribute `commission_fee_step`.

- [x] **Step 3: Add a tuple-compatible result class in `futures_util.py`**

Add near the top of `FineFT/env/env_class/futures_util.py` after imports:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class WalletChangeResult:
    leverage: float
    position: float
    initial_margin: float
    unrealized_pnl: float
    wallet_balance: float
    slippage_step: float
    commission_fee_step: float = 0.0
    realized_pnl_step: float = 0.0

    def legacy_tuple(self):
        return (
            self.leverage,
            self.position,
            self.initial_margin,
            self.unrealized_pnl,
            self.wallet_balance,
            self.slippage_step,
        )

    def __iter__(self):
        return iter(self.legacy_tuple())

    def __getitem__(self, index):
        return self.legacy_tuple()[index]
```

- [x] **Step 4: Return `WalletChangeResult` from no-trade and leverage-only paths**

In `change_of_wallet(...)`, replace the direct tuple returned for rejected side flips with:

```python
            return WalletChangeResult(
                leverage=previous_leverage,
                position=previous_position,
                initial_margin=previous_initial_margine,
                unrealized_pnl=previous_unrealized_pnL,
                wallet_balance=previous_wallet_balance,
                slippage_step=0,
                commission_fee_step=0,
                realized_pnl_step=0,
            )
```

In `change_of_leverage(...)`, replace the final tuple return with:

```python
    return WalletChangeResult(
        leverage=current_leverage,
        position=current_position,
        initial_margin=current_initial_margine,
        unrealized_pnl=current_unrealized_pnL,
        wallet_balance=current_wallet_balance,
        slippage_step=slippage,
        commission_fee_step=0,
        realized_pnl_step=0,
    )
```

- [x] **Step 5: Return true execution metrics from open/close helpers**

In each helper final return, use named values. For `open_short_position(...)`, successful open returns `commission_fee` and zero realized PnL:

```python
    return WalletChangeResult(
        leverage=current_leverage,
        position=current_position,
        initial_margin=current_initial_margine,
        unrealized_pnl=current_unrealized_pnL,
        wallet_balance=current_wallet_balance,
        slippage_step=slippage,
        commission_fee_step=commission_fee if current_position != previous_position else 0,
        realized_pnl_step=0,
    )
```

Apply the same shape to `open_long_position(...)`. For `close_short_position(...)` and `close_long_position(...)`, return:

```python
    return WalletChangeResult(
        leverage=current_leverage,
        position=current_position,
        initial_margin=current_initial_margine,
        unrealized_pnl=current_unrealized_pnL,
        wallet_balance=current_wallet_balance,
        slippage_step=slippage,
        commission_fee_step=commission_fee,
        realized_pnl_step=realized_pnL,
    )
```

- [x] **Step 6: Preserve close-then-leverage metrics in `change_of_wallet(...)`**

In the close-long and close-short branches where code currently returns `change_of_leverage(...)` after closing, assign the close result first, then combine it with the leverage result:

```python
                close_result = close_long_position(
                    markprice,
                    bid_prices,
                    bid_qtys,
                    sell_fee_rate,
                    previous_leverage,
                    previous_position,
                    previous_initial_margine,
                    previous_unrealized_pnL,
                    previous_wallet_balance,
                    previous_leverage,
                    current_position,
                    silent=silent,
                )
                leverage_result = change_of_leverage(
                    markprice,
                    close_result.leverage,
                    close_result.position,
                    close_result.initial_margin,
                    close_result.unrealized_pnl,
                    close_result.wallet_balance,
                    current_leverage,
                    silent=silent,
                )
                return WalletChangeResult(
                    leverage=leverage_result.leverage,
                    position=leverage_result.position,
                    initial_margin=leverage_result.initial_margin,
                    unrealized_pnl=leverage_result.unrealized_pnl,
                    wallet_balance=leverage_result.wallet_balance,
                    slippage_step=close_result.slippage_step + leverage_result.slippage_step,
                    commission_fee_step=close_result.commission_fee_step,
                    realized_pnl_step=close_result.realized_pnl_step,
                )
```

Use the same pattern for the close-short branch with `close_short_position(...)`.

- [x] **Step 7: Run the focused wallet-change test**

Run:

```bash
conda activate finetf && pytest FineFT/tests/env/test_commodity_env.py::test_wallet_change_can_use_buy_and_sell_fee_rates -q
```

Expected: pass.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 2: Update `Base_Env` and `Simple_Env` callers to consume the explicit wallet-change result and expose/reset per-step and cumulative execution metrics.

> **trace:** plan-ready.md → `### Task 2: Update `Base_Env` and `Simple_Env` callers to consume the explicit wallet-change result and expose/reset per-step and cumulative execution metrics.` | tasks.md → `- [ ] 1.2 Update `Base_Env` and `Simple_Env` callers to consume the explicit wallet-change result and expose/reset per-step and cumulative execution metrics.`
> **sync:** tasks.md → `- [ ] 1.2 Update `Base_Env` and `Simple_Env` callers to consume the explicit wallet-change result and expose/reset per-step and cumulative execution metrics.` | plan-ready.md → `### Task 2: Update `Base_Env` and `Simple_Env` callers to consume the explicit wallet-change result and expose/reset per-step and cumulative execution metrics.`

**Files:**
- Modify: `FineFT/env/env_class/base_env.py`
- Modify: `FineFT/env/env_class/simple_env.py`
- Modify: `FineFT/tests/env/test_commodity_env.py`

- [x] **Step 1: Add a failing env info test for execution metrics**

Add to `FineFT/tests/env/test_commodity_env.py`:

```python
def test_commodity_env_step_exposes_execution_metrics():
    env = initiate_commodity_env(
        _df(),
        ["feature_a"],
        max_holding_number=1,
        position_choices=3,
        buy_fee_rate=0.0001,
        sell_fee_rate=0.0003,
    )
    env.reset()
    _, _, _, info = env.step(
        env.env_map_position_leverage_to_action(1, env.leverage_choices[0])
    )

    assert "commission_fee_step" in info
    assert "realized_pnl_step" in info
    assert "slippage_step" in info
    assert "cumulative_commission_fee" in info
    assert "cumulative_realized_pnl" in info
    assert "cumulative_slippage" in info
    assert info["commission_fee_step"] == env.commission_fee_step
    assert info["cumulative_commission_fee"] == env.cumulative_commission_fee
```

- [x] **Step 2: Run the env info test and verify it fails**

Run:

```bash
conda activate finetf && pytest FineFT/tests/env/test_commodity_env.py::test_commodity_env_step_exposes_execution_metrics -q
```

Expected: failure because execution metric keys are missing from `info`.

- [x] **Step 3: Add metric reset and info helpers to `Base_Env`**

In `FineFT/env/env_class/base_env.py`, add methods inside `Base_Env`:

```python
    def _reset_execution_metrics(self):
        self.commission_fee_step = 0
        self.realized_pnl_step = 0
        self.slippage_step = 0
        self.cumulative_commission_fee = 0
        self.cumulative_realized_pnl = 0
        self.cumulative_slippage = 0

    def _update_execution_metrics(self, wallet_change):
        self.commission_fee_step = wallet_change.commission_fee_step
        self.realized_pnl_step = wallet_change.realized_pnl_step
        self.slippage_step = wallet_change.slippage_step
        self.cumulative_commission_fee += self.commission_fee_step
        self.cumulative_realized_pnl += self.realized_pnl_step
        self.cumulative_slippage += self.slippage_step

    def _execution_metric_info(self):
        return {
            "commission_fee_step": self.commission_fee_step,
            "realized_pnl_step": self.realized_pnl_step,
            "slippage_step": self.slippage_step,
            "cumulative_commission_fee": self.cumulative_commission_fee,
            "cumulative_realized_pnl": self.cumulative_realized_pnl,
            "cumulative_slippage": self.cumulative_slippage,
        }
```

Call `self._reset_execution_metrics()` in `__init__` after `self.slippage_sum = 0` and in `reset()` after resetting `self.slippage_sum`.

- [x] **Step 4: Use named wallet-change result in `Base_Env.step()`**

Replace the tuple unpack around `change_of_wallet(...)` with:

```python
        wallet_change = change_of_wallet(
            markprice=self.current_markprice,
            ask_prices=self.ask_prices,
            ask_qtys=self.ask_qtys,
            bid_prices=self.bid_prices,
            bid_qtys=self.bid_qtys,
            long_estimated_rate=self.long_estimated_rate,
            short_estimated_rate=self.short_estimated_rate,
            commission_rate=self.commission_rate,
            previous_leverage=self.leverage,
            previous_position=self.position,
            previous_initial_margine=self.initial_margin,
            previous_unrealized_pnL=self.unrealized_pnl,
            previous_wallet_balance=self.wallet_balance,
            current_leverage=target_leverage,
            current_position=target_position,
            silent=False,
            buy_fee_rate=self.buy_fee_rate,
            sell_fee_rate=self.sell_fee_rate,
        )
        leverage = wallet_change.leverage
        position = wallet_change.position
        initial_margin = wallet_change.initial_margin
        unrealized_pnL = wallet_change.unrealized_pnl
        wallet_balance = wallet_change.wallet_balance
        slippage = wallet_change.slippage_step
        self._update_execution_metrics(wallet_change)
```

In every `info` dictionary returned by `Base_Env.step()`, add:

```python
                    **self._execution_metric_info(),
```

In the `reset()` info dictionary, add the same expansion so initial info is consistent.

- [x] **Step 5: Apply the same wallet-change consumption pattern to `Simple_Env`**

In `FineFT/env/env_class/simple_env.py`, add `_reset_execution_metrics`, `_update_execution_metrics`, and `_execution_metric_info` with the same bodies as `Base_Env`. Replace the tuple unpack around `change_of_wallet(...)` with named attributes:

```python
        wallet_change = change_of_wallet(
            markprice=self.current_markprice,
            ask_prices=self.ask_prices,
            ask_qtys=self.ask_qtys,
            bid_prices=self.bid_prices,
            bid_qtys=self.bid_qtys,
            long_estimated_rate=self.long_estimated_rate,
            short_estimated_rate=self.short_estimated_rate,
            commission_rate=self.commission_rate,
            previous_leverage=self.leverage,
            previous_position=self.position,
            previous_initial_margine=self.initial_margin,
            previous_unrealized_pnL=self.unrealized_pnl,
            previous_wallet_balance=self.wallet_balance,
            current_leverage=target_leverage,
            current_position=target_position,
            silent=False,
        )
        leverage = wallet_change.leverage
        position = wallet_change.position
        initial_margin = wallet_change.initial_margin
        unrealized_pnL = wallet_change.unrealized_pnl
        wallet_balance = wallet_change.wallet_balance
        slippage = wallet_change.slippage_step
        self._update_execution_metrics(wallet_change)
```

Add `**self._execution_metric_info()` to every returned info dictionary in `Simple_Env`.

- [x] **Step 6: Run relevant env tests**

Run:

```bash
conda activate finetf && pytest FineFT/tests/env/test_commodity_env.py -q
```

Expected: pass.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 3: Update `test_agent_index.py` aggregate result collection to include `df_path` and write default `analysis_result.csv` with JSON array fields while preserving `analysis_result.npy`.

> **trace:** plan-ready.md → `### Task 3: Update `test_agent_index.py` aggregate result collection to include `df_path` and write default `analysis_result.csv` with JSON array fields while preserving `analysis_result.npy`.` | tasks.md → `- [ ] 1.3 Update `test_agent_index.py` aggregate result collection to include `df_path` and write default `analysis_result.csv` with JSON array fields while preserving `analysis_result.npy`.`
> **sync:** tasks.md → `- [ ] 1.3 Update `test_agent_index.py` aggregate result collection to include `df_path` and write default `analysis_result.csv` with JSON array fields while preserving `analysis_result.npy`.` | plan-ready.md → `### Task 3: Update `test_agent_index.py` aggregate result collection to include `df_path` and write default `analysis_result.csv` with JSON array fields while preserving `analysis_result.npy`.`

**Files:**
- Modify: `FineFT/RL/DiHFT/low_level/test_agent_index.py`
- Modify: `FineFT/tests/rl/test_test_agent_index.py`

- [x] **Step 1: Add a failing aggregate CSV test**

Add imports to `FineFT/tests/rl/test_test_agent_index.py`:

```python
import json
import numpy as np
```

Extend `test_weighted_trader_passes_order_book_depth_to_base_env` after `trader.test()`:

```python
    npy_path = tmp_path / "analysis_result.npy"
    csv_path = tmp_path / "analysis_result.csv"

    assert npy_path.exists()
    assert csv_path.exists()

    result = np.load(npy_path, allow_pickle=True).tolist()
    assert result[0]["df_path"] == ["df_0.feather"]

    csv_df = pd.read_csv(csv_path)
    assert list(csv_df.columns) == [
        "label",
        "initial_action",
        "bin_index",
        "df_path",
        "reward_sum",
        "df_length",
        "turnover",
    ]
    assert json.loads(csv_df.loc[0, "df_path"]) == ["df_0.feather"]
    assert json.loads(csv_df.loc[0, "reward_sum"]) == [1.0]
    assert json.loads(csv_df.loc[0, "df_length"]) == [1]
    assert json.loads(csv_df.loc[0, "turnover"]) == [0.0]
```

- [x] **Step 2: Run the aggregate CSV test and verify it fails**

Run:

```bash
conda activate finetf && pytest FineFT/tests/rl/test_test_agent_index.py::test_weighted_trader_passes_order_book_depth_to_base_env -q
```

Expected: failure because `analysis_result.csv` does not exist or `df_path` is absent.

- [x] **Step 3: Add JSON CSV helpers to `test_agent_index.py`**

In `FineFT/RL/DiHFT/low_level/test_agent_index.py`, add imports:

```python
import json
```

Add helper functions near `build_serial_model_path`:

```python
AGGREGATE_JSON_COLUMNS = ["df_path", "reward_sum", "df_length", "turnover"]


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_array(value):
    return json.dumps(list(value), default=_json_default)


def write_analysis_csv(overall_result, csv_path):
    analysis_df = pd.DataFrame(overall_result)
    for column in AGGREGATE_JSON_COLUMNS:
        analysis_df[column] = analysis_df[column].apply(_json_array)
    analysis_df.to_csv(csv_path, index=False)
```

- [x] **Step 4: Collect `df_path` alongside aggregate metrics**

Inside `weighted_trader.test()`, add the list before the `for df_path in df_list:` loop:

```python
                    single_label_initial_action_bin_index_df_path_result = []
```

Inside the `for df_path in df_list:` loop, append:

```python
                        single_label_initial_action_bin_index_df_path_result.append(
                            df_path
                        )
```

Add the field to `_overall_result`:

```python
                            "df_path": single_label_initial_action_bin_index_df_path_result,
```

- [x] **Step 5: Write aggregate CSV after saving npy**

At the end of `weighted_trader.test()`, replace the single `np.save(...)` with:

```python
        np.save(os.path.join(self.epoch_path, "analysis_result.npy"), overall_result)
        write_analysis_csv(
            overall_result,
            os.path.join(self.epoch_path, "analysis_result.csv"),
        )
```

- [x] **Step 6: Run the aggregate CSV test**

Run:

```bash
conda activate finetf && pytest FineFT/tests/rl/test_test_agent_index.py::test_weighted_trader_passes_order_book_depth_to_base_env -q
```

Expected: pass.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 4: Add `--save_trading_detail_csv` to `test_agent_index.py` and write `trading_action_detail_epoch_<epoch_num>.csv` only when the flag is provided.

> **trace:** plan-ready.md → `### Task 4: Add `--save_trading_detail_csv` to `test_agent_index.py` and write `trading_action_detail_epoch_<epoch_num>.csv` only when the flag is provided.` | tasks.md → `- [ ] 1.4 Add `--save_trading_detail_csv` to `test_agent_index.py` and write `trading_action_detail_epoch_<epoch_num>.csv` only when the flag is provided.`
> **sync:** tasks.md → `- [ ] 1.4 Add `--save_trading_detail_csv` to `test_agent_index.py` and write `trading_action_detail_epoch_<epoch_num>.csv` only when the flag is provided.` | plan-ready.md → `### Task 4: Add `--save_trading_detail_csv` to `test_agent_index.py` and write `trading_action_detail_epoch_<epoch_num>.csv` only when the flag is provided.`

**Files:**
- Modify: `FineFT/RL/DiHFT/low_level/test_agent_index.py`
- Modify: `FineFT/tests/rl/test_test_agent_index.py`

- [x] **Step 1: Add failing tests for the detail CSV switch**

Add a helper factory in `FineFT/tests/rl/test_test_agent_index.py` so both detail tests can create a trader:

```python
def _make_test_trader(tai, tmp_path, save_trading_detail_csv=False):
    valid_dir = tmp_path / "valid" / "label"
    valid_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"mark_price": [100.0]}).to_feather(valid_dir / "df_0.feather")

    trader = tai.weighted_trader.__new__(tai.weighted_trader)
    trader.eval_net = FakeNet()
    trader.valid_data_path = str(tmp_path / "valid")
    trader.initial_action_list = [0]
    trader.N = 1
    trader.leverage_choices = [1]
    trader.position_list = [0]
    trader.initial_wallet_balance = 100000
    trader.initial_unrealized_pnL = 0
    trader.max_holding_number = 1
    trader.position_choices = 3
    trader.order_book_depth = 5
    trader.long_estimated_rate = 0
    trader.short_estimated_rate = 0
    trader.transcation_cost = 0
    trader.maintenance_margin_ratio_dict = {}
    trader.tech_indicator_list = []
    trader.epoch_path = str(tmp_path)
    trader.epoch_num = 1
    trader.save_trading_detail_csv = save_trading_detail_csv
    trader.act_test = lambda state, info, bin_index: 0
    return trader
```

Add tests:

```python
def test_trading_detail_csv_is_disabled_by_default(monkeypatch, tmp_path):
    from RL.DiHFT.low_level import test_agent_index as tai

    monkeypatch.setattr(tai, "initiate_base_env", lambda **kwargs: FakeEnv())
    monkeypatch.setattr(tai, "map_action_to_position_leverage", lambda *args: (0, 1))

    trader = _make_test_trader(tai, tmp_path, save_trading_detail_csv=False)
    trader.test()

    assert not (tmp_path / "trading_action_detail_epoch_1.csv").exists()


def test_trading_detail_csv_is_written_when_enabled(monkeypatch, tmp_path):
    from RL.DiHFT.low_level import test_agent_index as tai

    monkeypatch.setattr(tai, "initiate_base_env", lambda **kwargs: FakeEnv())
    monkeypatch.setattr(tai, "map_action_to_position_leverage", lambda *args: (0, 1))

    trader = _make_test_trader(tai, tmp_path, save_trading_detail_csv=True)
    trader.test()

    assert (tmp_path / "trading_action_detail_epoch_1.csv").exists()
```

- [x] **Step 2: Run the detail switch tests and verify the enabled case fails**

Run:

```bash
conda activate finetf && pytest FineFT/tests/rl/test_test_agent_index.py::test_trading_detail_csv_is_disabled_by_default FineFT/tests/rl/test_test_agent_index.py::test_trading_detail_csv_is_written_when_enabled -q
```

Expected: enabled case fails because the flag and file writer are not implemented.

- [x] **Step 3: Add the CLI flag and trader attribute**

In `test_agent_index.py`, add parser argument near result options:

```python
parser.add_argument(
    "--save_trading_detail_csv",
    action="store_true",
    help="write per-step trading detail CSV for the tested epoch",
)
```

In `weighted_trader.__init__`, add:

```python
        self.save_trading_detail_csv = args.save_trading_detail_csv
```

- [x] **Step 4: Add a detail CSV file helper**

Add near the aggregate CSV helper:

```python
def trading_detail_csv_path(epoch_path, epoch_num):
    return os.path.join(epoch_path, f"trading_action_detail_epoch_{epoch_num}.csv")


def write_trading_detail_csv(detail_rows, csv_path):
    pd.DataFrame(detail_rows).to_csv(csv_path, index=False)
```

- [x] **Step 5: Initialize and conditionally write detail rows**

At the start of `weighted_trader.test()`, after `overall_result = []`, add:

```python
        trading_detail_rows = []
```

At the end of `weighted_trader.test()`, after `write_analysis_csv(...)`, add:

```python
        if self.save_trading_detail_csv:
            write_trading_detail_csv(
                trading_detail_rows,
                trading_detail_csv_path(self.epoch_path, self.epoch_num),
            )
```

This task only establishes the switch and file creation. Task 5 fills the row payload.

- [x] **Step 6: Run the detail switch tests**

Run:

```bash
conda activate finetf && pytest FineFT/tests/rl/test_test_agent_index.py::test_trading_detail_csv_is_disabled_by_default FineFT/tests/rl/test_test_agent_index.py::test_trading_detail_csv_is_written_when_enabled -q
```

Expected: pass. The enabled file may be empty until Task 5.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 5: Build detail CSV rows with context, optional OHLCV fields, action target state, actual pre/post execution state, action-change counts, trade counts, execution economics, and account value columns.

> **trace:** plan-ready.md → `### Task 5: Build detail CSV rows with context, optional OHLCV fields, action target state, actual pre/post execution state, action-change counts, trade counts, execution economics, and account value columns.` | tasks.md → `- [ ] 1.5 Build detail CSV rows with context, optional OHLCV fields, action target state, actual pre/post execution state, action-change counts, trade counts, execution economics, and account value columns.`
> **sync:** tasks.md → `- [ ] 1.5 Build detail CSV rows with context, optional OHLCV fields, action target state, actual pre/post execution state, action-change counts, trade counts, execution economics, and account value columns.` | plan-ready.md → `### Task 5: Build detail CSV rows with context, optional OHLCV fields, action target state, actual pre/post execution state, action-change counts, trade counts, execution economics, and account value columns.`

**Files:**
- Modify: `FineFT/RL/DiHFT/low_level/test_agent_index.py`
- Modify: `FineFT/tests/rl/test_test_agent_index.py`

- [x] **Step 1: Replace the simple fake env with a multi-step detail fake**

Add to `FineFT/tests/rl/test_test_agent_index.py`:

```python
class DetailFakeEnv:
    initial_margin_history = []
    wallet_balance_history = []
    unrealized_pnl_history = []
    maintain_marigine_history = []
    new_position_required_money_history = []

    def __init__(self):
        self.step_index = 0
        self.position = 0
        self.leverage = 1
        self.wallet_balance = 1000.0
        self.unrealized_pnl = 0.0

    def reset(self):
        return [0.0], {
            "previous_action": 0,
            "avaliable_action": [1, 1, 1],
            "funding_count_down_hour": 0,
            "funding_count_down_minute": 0,
        }

    def step(self, action):
        if self.step_index == 0:
            self.step_index += 1
            return [0.0], 2.0, False, {
                "previous_action": action,
                "avaliable_action": [1, 1, 1],
                "funding_count_down_hour": 0,
                "funding_count_down_minute": 0,
                "commission_fee_step": 0.5,
                "realized_pnl_step": 0.0,
                "slippage_step": 0.1,
                "cumulative_commission_fee": 0.5,
                "cumulative_realized_pnl": 0.0,
                "cumulative_slippage": 0.1,
            }
        self.position = 1
        self.wallet_balance = 1001.0
        self.unrealized_pnl = 3.0
        self.step_index += 1
        return [0.0], 4.0, True, {
            "previous_action": action,
            "avaliable_action": [1, 1, 1],
            "funding_count_down_hour": 0,
            "funding_count_down_minute": 0,
            "commission_fee_step": 0.7,
            "realized_pnl_step": 5.0,
            "slippage_step": 0.2,
            "cumulative_commission_fee": 1.2,
            "cumulative_realized_pnl": 5.0,
            "cumulative_slippage": 0.3,
        }
```

- [x] **Step 2: Add a failing detail row content test**

Add to `FineFT/tests/rl/test_test_agent_index.py`:

```python
def test_trading_detail_csv_records_actions_trades_and_execution_metrics(monkeypatch, tmp_path):
    from RL.DiHFT.low_level import test_agent_index as tai

    valid_dir = tmp_path / "valid" / "label"
    valid_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="min"),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [10.0, 11.0],
            "mark_price": [100.0, 101.0],
        }
    ).to_feather(valid_dir / "df_0.feather")

    monkeypatch.setattr(tai, "initiate_base_env", lambda **kwargs: DetailFakeEnv())
    monkeypatch.setattr(
        tai,
        "map_action_to_position_leverage",
        lambda action, leverage_choices, position_list: (1, 1) if action == 1 else (0, 1),
    )

    trader = _make_test_trader(tai, tmp_path, save_trading_detail_csv=True)
    trader.act_test = lambda state, info, bin_index: 1
    trader.test()

    detail_df = pd.read_csv(tmp_path / "trading_action_detail_epoch_1.csv")
    assert len(detail_df) == 2
    assert detail_df.loc[0, "action_change_step"] == 1
    assert detail_df.loc[0, "trade_count_step"] == 0
    assert detail_df.loc[1, "action_change_step"] == 0
    assert detail_df.loc[1, "trade_count_step"] == 1
    assert detail_df.loc[1, "cumulative_action_change_count"] == 1
    assert detail_df.loc[1, "cumulative_trade_count"] == 1
    assert detail_df.loc[1, "commission_fee_step"] == 0.7
    assert detail_df.loc[1, "cumulative_commission_fee"] == 1.2
    assert detail_df.loc[1, "realized_pnl_step"] == 5.0
    assert detail_df.loc[1, "cumulative_realized_pnl"] == 5.0
    assert detail_df.loc[1, "slippage_step"] == 0.2
    assert detail_df.loc[1, "cumulative_slippage"] == 0.3
    assert detail_df.loc[1, "margin_balance"] == 1004.0
    assert detail_df.loc[1, "total_value"] == 1004.0
    assert detail_df.loc[1, "notional_asset_value"] == 101.0
```

- [x] **Step 3: Run the detail row content test and verify it fails**

Run:

```bash
conda activate finetf && pytest FineFT/tests/rl/test_test_agent_index.py::test_trading_detail_csv_records_actions_trades_and_execution_metrics -q
```

Expected: failure because detail rows are empty or required columns are absent.

- [x] **Step 4: Add market and account helper functions**

Add to `test_agent_index.py`:

```python
DETAIL_MARKET_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "mark_price"]


def _market_fields(test_df, timestep):
    row = test_df.iloc[timestep]
    return {
        column: row[column]
        for column in DETAIL_MARKET_COLUMNS
        if column in test_df.columns
    }


def _personal_state_from_env(test_env):
    return {
        "wallet_balance": getattr(test_env, "wallet_balance", np.nan),
        "unrealized_pnl": getattr(test_env, "unrealized_pnl", np.nan),
        "position": getattr(test_env, "position", np.nan),
        "leverage": getattr(test_env, "leverage", np.nan),
    }
```

- [x] **Step 5: Add a detail row builder**

Add to `test_agent_index.py`:

```python
def build_trading_detail_row(
    *,
    label,
    df_path,
    initial_action,
    bin_index,
    timestep,
    test_df,
    action,
    target_position,
    target_leverage,
    position_before,
    leverage_before,
    test_env,
    info,
    step_reward,
    action_change_step,
    trade_count_step,
    cumulative_action_change_count,
    cumulative_trade_count,
):
    state_after = _personal_state_from_env(test_env)
    mark_price = _market_fields(test_df, timestep).get("mark_price", np.nan)
    wallet_balance = state_after["wallet_balance"]
    unrealized_pnl = state_after["unrealized_pnl"]
    position_after = state_after["position"]
    margin_balance = wallet_balance + unrealized_pnl
    row = {
        "label": label,
        "df_path": df_path,
        "initial_action": initial_action,
        "bin_index": bin_index,
        "timestep": timestep,
        **_market_fields(test_df, timestep),
        "action": action,
        "target_position": target_position,
        "target_leverage": target_leverage,
        "position_before": position_before,
        "leverage_before": leverage_before,
        "position_after": position_after,
        "leverage_after": state_after["leverage"],
        "action_change_step": action_change_step,
        "trade_count_step": trade_count_step,
        "cumulative_action_change_count": cumulative_action_change_count,
        "cumulative_trade_count": cumulative_trade_count,
        "step_reward": step_reward,
        "realized_pnl_step": info["realized_pnl_step"],
        "cumulative_realized_pnl": info["cumulative_realized_pnl"],
        "commission_fee_step": info["commission_fee_step"],
        "cumulative_commission_fee": info["cumulative_commission_fee"],
        "slippage_step": info["slippage_step"],
        "cumulative_slippage": info["cumulative_slippage"],
        "wallet_balance": wallet_balance,
        "unrealized_pnl": unrealized_pnl,
        "margin_balance": margin_balance,
        "notional_asset_value": mark_price * position_after,
        "cash_balance": wallet_balance,
        "total_value": margin_balance,
    }
    return row
```

- [x] **Step 6: Append detail rows inside the step loop**

Inside the `while not done:` loop in `weighted_trader.test()`, capture before-state before `self.act_test(...)` and append after `test_env.step(a)`:

```python
                            timestep = len(action_list)
                            position_before = getattr(test_env, "position", initial_position)
                            leverage_before = getattr(test_env, "leverage", initial_leverage)
                            a = self.act_test(s, info, bin_index)
                            target_position, target_leverage = map_action_to_position_leverage(
                                a,
                                self.leverage_choices,
                                self.position_list,
                            )
                            action_change_step = int(a != previous_action)
                            turn_over += np.abs(a - previous_action) / 4
                            s_, r, done, info = test_env.step(a)
                            position_after = getattr(test_env, "position", position_before)
                            leverage_after = getattr(test_env, "leverage", leverage_before)
                            trade_count_step = int(
                                position_after != position_before
                                or leverage_after != leverage_before
                            )
                            cumulative_action_change_count += action_change_step
                            cumulative_trade_count += trade_count_step
                            if self.save_trading_detail_csv:
                                trading_detail_rows.append(
                                    build_trading_detail_row(
                                        label=label,
                                        df_path=df_path,
                                        initial_action=initial_action,
                                        bin_index=bin_index,
                                        timestep=timestep,
                                        test_df=self.test_df,
                                        action=a,
                                        target_position=target_position,
                                        target_leverage=target_leverage,
                                        position_before=position_before,
                                        leverage_before=leverage_before,
                                        test_env=test_env,
                                        info=info,
                                        step_reward=r,
                                        action_change_step=action_change_step,
                                        trade_count_step=trade_count_step,
                                        cumulative_action_change_count=cumulative_action_change_count,
                                        cumulative_trade_count=cumulative_trade_count,
                                    )
                                )
```

Initialize the trajectory-local counters just before the `while not done:` loop:

```python
                        cumulative_action_change_count = 0
                        cumulative_trade_count = 0
```

- [x] **Step 7: Run the detail row content test**

Run:

```bash
conda activate finetf && pytest FineFT/tests/rl/test_test_agent_index.py::test_trading_detail_csv_records_actions_trades_and_execution_metrics -q
```

Expected: pass.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 6: Add focused unit tests for aggregate CSV output, detail CSV opt-in behavior, execution metric exposure, and action-change versus actual-trade counting.

> **trace:** plan-ready.md → `### Task 6: Add focused unit tests for aggregate CSV output, detail CSV opt-in behavior, execution metric exposure, and action-change versus actual-trade counting.` | tasks.md → `- [ ] 1.6 Add focused unit tests for aggregate CSV output, detail CSV opt-in behavior, execution metric exposure, and action-change versus actual-trade counting.`
> **sync:** tasks.md → `- [ ] 1.6 Add focused unit tests for aggregate CSV output, detail CSV opt-in behavior, execution metric exposure, and action-change versus actual-trade counting.` | plan-ready.md → `### Task 6: Add focused unit tests for aggregate CSV output, detail CSV opt-in behavior, execution metric exposure, and action-change versus actual-trade counting.`

**Files:**
- Modify: `FineFT/tests/rl/test_test_agent_index.py`
- Modify: `FineFT/tests/env/test_commodity_env.py`

- [x] **Step 1: Run the full focused RL test file**

Run:

```bash
conda activate finetf && pytest FineFT/tests/rl/test_test_agent_index.py -q
```

Expected: pass. Failures should identify only issues in the CSV helpers or fake env setup.

- [x] **Step 2: Run the full focused environment test file**

Run:

```bash
conda activate finetf && pytest FineFT/tests/env/test_commodity_env.py -q
```

Expected: pass. Failures should identify only metric exposure or wallet-change result shape issues.

- [x] **Step 3: Confirm tests cover all requested boundaries**

Run:

```bash
conda activate finetf && pytest FineFT/tests/rl/test_test_agent_index.py FineFT/tests/env/test_commodity_env.py -q
```

Expected: pass. Confirm these behaviors are asserted in the tests:

```text
analysis_result.npy exists
analysis_result.csv exists
aggregate JSON columns parse with json.loads
detail CSV is absent without --save_trading_detail_csv
detail CSV is present with --save_trading_detail_csv
action_change_step can increment without trade_count_step
trade_count_step increments only when actual position or leverage changes
execution economics come from explicit env info fields
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 7: Run focused pytest for `FineFT/tests/rl/test_test_agent_index.py` and relevant environment tests.

> **trace:** plan-ready.md → `### Task 7: Run focused pytest for `FineFT/tests/rl/test_test_agent_index.py` and relevant environment tests.` | tasks.md → `- [ ] 2.1 Run focused pytest for `FineFT/tests/rl/test_test_agent_index.py` and relevant environment tests.`
> **sync:** tasks.md → `- [ ] 2.1 Run focused pytest for `FineFT/tests/rl/test_test_agent_index.py` and relevant environment tests.` | plan-ready.md → `### Task 7: Run focused pytest for `FineFT/tests/rl/test_test_agent_index.py` and relevant environment tests.`

**Files:**
- Verify: `FineFT/tests/rl/test_test_agent_index.py`
- Verify: `FineFT/tests/env/test_commodity_env.py`

- [x] **Step 1: Run focused pytest**

Run:

```bash
conda activate finetf && pytest FineFT/tests/rl/test_test_agent_index.py FineFT/tests/env/test_commodity_env.py -q
```

Expected: all tests pass.

- [x] **Step 2: Record any environment limitation in the final implementation response**

If the command cannot run because `conda` or the `finetf` environment is unavailable, run the nearest available project Python command and record the exact limitation and fallback command in the final response.

Expected fallback shape:

```bash
python -m pytest FineFT/tests/rl/test_test_agent_index.py FineFT/tests/env/test_commodity_env.py -q
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 8: Run `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/test_agent_index.py FineFT/env/env_class/base_env.py FineFT/env/env_class/simple_env.py FineFT/env/env_class/futures_util.py`.

> **trace:** plan-ready.md → `### Task 8: Run `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/test_agent_index.py FineFT/env/env_class/base_env.py FineFT/env/env_class/simple_env.py FineFT/env/env_class/futures_util.py`.` | tasks.md → `- [ ] 2.2 Run `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/test_agent_index.py FineFT/env/env_class/base_env.py FineFT/env/env_class/simple_env.py FineFT/env/env_class/futures_util.py`.`
> **sync:** tasks.md → `- [ ] 2.2 Run `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/test_agent_index.py FineFT/env/env_class/base_env.py FineFT/env/env_class/simple_env.py FineFT/env/env_class/futures_util.py`.` | plan-ready.md → `### Task 8: Run `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/test_agent_index.py FineFT/env/env_class/base_env.py FineFT/env/env_class/simple_env.py FineFT/env/env_class/futures_util.py`.`

**Files:**
- Verify: `FineFT/RL/DiHFT/low_level/test_agent_index.py`
- Verify: `FineFT/env/env_class/base_env.py`
- Verify: `FineFT/env/env_class/simple_env.py`
- Verify: `FineFT/env/env_class/futures_util.py`

- [x] **Step 1: Run py_compile**

Run:

```bash
conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/test_agent_index.py FineFT/env/env_class/base_env.py FineFT/env/env_class/simple_env.py FineFT/env/env_class/futures_util.py
```

Expected: command exits 0 with no syntax errors.

- [x] **Step 2: Record any environment limitation in the final implementation response**

If the conda environment cannot be activated, run:

```bash
python -m py_compile FineFT/RL/DiHFT/low_level/test_agent_index.py FineFT/env/env_class/base_env.py FineFT/env/env_class/simple_env.py FineFT/env/env_class/futures_util.py
```

Expected: command exits 0, or the final response reports the exact import/syntax failure.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 9: Run `openspec validate add-test-agent-csv-outputs --strict`.

> **trace:** plan-ready.md → `### Task 9: Run `openspec validate add-test-agent-csv-outputs --strict`.` | tasks.md → `- [ ] 2.3 Run `openspec validate add-test-agent-csv-outputs --strict`.`
> **sync:** tasks.md → `- [ ] 2.3 Run `openspec validate add-test-agent-csv-outputs --strict`.` | plan-ready.md → `### Task 9: Run `openspec validate add-test-agent-csv-outputs --strict`.`

**Files:**
- Verify: `openspec/changes/add-test-agent-csv-outputs/proposal.md`
- Verify: `openspec/changes/add-test-agent-csv-outputs/design.md`
- Verify: `openspec/changes/add-test-agent-csv-outputs/specs/fineft-low-level-test-results/spec.md`
- Verify: `openspec/changes/add-test-agent-csv-outputs/tasks.md`
- Verify: `openspec/changes/add-test-agent-csv-outputs/plan-ready.md`

- [x] **Step 1: Run strict OpenSpec validation**

Run:

```bash
openspec validate add-test-agent-csv-outputs --strict
```

Expected:

```text
Change 'add-test-agent-csv-outputs' is valid
```

- [x] **Step 2: Check changed files before handoff**

Run:

```bash
git status --short
```

Expected: changed files are limited to the implementation files, tests, and the sddflow tracking documents for `add-test-agent-csv-outputs`.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 10: Change aggregate and detail CSV headers to English/Chinese bilingual names while preserving row values and JSON array cells.

> **trace:** plan-ready.md → `### Task 10: Change aggregate and detail CSV headers to English/Chinese bilingual names while preserving row values and JSON array cells.` | tasks.md → `- [ ] 1.7 Change aggregate and detail CSV headers to English/Chinese bilingual names while preserving row values and JSON array cells.`
> **sync:** tasks.md → `- [ ] 1.7 Change aggregate and detail CSV headers to English/Chinese bilingual names while preserving row values and JSON array cells.` | plan-ready.md → `### Task 10: Change aggregate and detail CSV headers to English/Chinese bilingual names while preserving row values and JSON array cells.`

**Files:**
- Modify: `FineFT/RL/DiHFT/low_level/test_agent_index.py`
- Modify: `FineFT/tests/rl/test_test_agent_index.py`

- [x] **Step 1: Add failing aggregate CSV bilingual-header assertions**

In `FineFT/tests/rl/test_test_agent_index.py::test_weighted_trader_passes_order_book_depth_to_base_env`, replace the aggregate CSV column assertion with:

```python
    assert list(csv_df.columns) == [
        "label/标签",
        "initial_action/初始动作",
        "bin_index/分箱索引",
        "df_path/数据文件",
        "reward_sum/奖励总和",
        "df_length/数据长度",
        "turnover/换手率",
    ]
    assert json.loads(csv_df.loc[0, "df_path/数据文件"]) == ["df_0.feather"]
    assert json.loads(csv_df.loc[0, "reward_sum/奖励总和"]) == [1.0]
    assert json.loads(csv_df.loc[0, "df_length/数据长度"]) == [1]
    assert json.loads(csv_df.loc[0, "turnover/换手率"]) == [0.0]
```

- [x] **Step 2: Add failing detail CSV bilingual-header assertions**

In `FineFT/tests/rl/test_test_agent_index.py::test_trading_detail_csv_records_actions_trades_and_execution_metrics`, replace the required-column loop and detail column reads with bilingual names:

```python
    for column in [
        "label/标签",
        "df_path/数据文件",
        "timestamp/时间戳",
        "target_position/目标仓位",
        "target_leverage/目标杠杆",
        "realized_pnl_step/单步已实现盈亏",
        "cumulative_realized_pnl/累计已实现盈亏",
        "commission_fee_step/单步手续费",
        "cumulative_commission_fee/累计手续费",
        "slippage_step/单步滑点",
        "cumulative_slippage/累计滑点",
    ]:
        assert column in detail_df.columns
    assert detail_df.loc[0, "label/标签"] == "label"
    assert detail_df.loc[0, "df_path/数据文件"] == "df_0.feather"
    assert detail_df.loc[0, "target_position/目标仓位"] == 1
    assert detail_df.loc[0, "target_leverage/目标杠杆"] == 1
    assert detail_df.loc[0, "action_change_step/动作变化"] == 1
    assert detail_df.loc[0, "trade_count_step/交易计数"] == 0
    assert detail_df.loc[1, "action_change_step/动作变化"] == 0
    assert detail_df.loc[1, "trade_count_step/交易计数"] == 1
    assert detail_df.loc[1, "cumulative_action_change_count/累计动作变化次数"] == 1
    assert detail_df.loc[1, "cumulative_trade_count/累计交易次数"] == 1
    assert detail_df.loc[1, "commission_fee_step/单步手续费"] == 0.7
    assert detail_df.loc[1, "cumulative_commission_fee/累计手续费"] == 1.2
    assert detail_df.loc[1, "realized_pnl_step/单步已实现盈亏"] == 5.0
    assert detail_df.loc[1, "cumulative_realized_pnl/累计已实现盈亏"] == 5.0
    assert detail_df.loc[1, "slippage_step/单步滑点"] == 0.2
    assert detail_df.loc[1, "cumulative_slippage/累计滑点"] == 0.3
    assert detail_df.loc[1, "margin_balance/保证金余额"] == 1004.0
    assert detail_df.loc[1, "total_value/总价值"] == 1004.0
    assert detail_df.loc[1, "notional_asset_value/名义资产价值"] == 101.0
```

- [x] **Step 3: Run the RL CSV tests and verify they fail for old English-only headers**

Run:

```bash
conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_test_agent_index.py::test_weighted_trader_passes_order_book_depth_to_base_env FineFT/tests/rl/test_test_agent_index.py::test_trading_detail_csv_records_actions_trades_and_execution_metrics -q
```

Expected: fail because `analysis_result.csv` and detail CSV currently emit English-only headers.

- [x] **Step 4: Add a bilingual CSV header mapping helper**

In `FineFT/RL/DiHFT/low_level/test_agent_index.py`, add near the CSV helper constants:

```python
CSV_HEADER_LABELS = {
    "label": "label/标签",
    "initial_action": "initial_action/初始动作",
    "bin_index": "bin_index/分箱索引",
    "df_path": "df_path/数据文件",
    "reward_sum": "reward_sum/奖励总和",
    "df_length": "df_length/数据长度",
    "turnover": "turnover/换手率",
    "timestep": "timestep/时间步",
    "timestamp": "timestamp/时间戳",
    "open": "open/开盘价",
    "high": "high/最高价",
    "low": "low/最低价",
    "close": "close/收盘价",
    "volume": "volume/成交量",
    "mark_price": "mark_price/标记价格",
    "action": "action/动作",
    "target_position": "target_position/目标仓位",
    "target_leverage": "target_leverage/目标杠杆",
    "position_before": "position_before/执行前仓位",
    "leverage_before": "leverage_before/执行前杠杆",
    "position_after": "position_after/执行后仓位",
    "leverage_after": "leverage_after/执行后杠杆",
    "action_change_step": "action_change_step/动作变化",
    "trade_count_step": "trade_count_step/交易计数",
    "cumulative_action_change_count": "cumulative_action_change_count/累计动作变化次数",
    "cumulative_trade_count": "cumulative_trade_count/累计交易次数",
    "step_reward": "step_reward/单步奖励",
    "realized_pnl_step": "realized_pnl_step/单步已实现盈亏",
    "cumulative_realized_pnl": "cumulative_realized_pnl/累计已实现盈亏",
    "commission_fee_step": "commission_fee_step/单步手续费",
    "cumulative_commission_fee": "cumulative_commission_fee/累计手续费",
    "slippage_step": "slippage_step/单步滑点",
    "cumulative_slippage": "cumulative_slippage/累计滑点",
    "wallet_balance": "wallet_balance/钱包余额",
    "unrealized_pnl": "unrealized_pnl/未实现盈亏",
    "margin_balance": "margin_balance/保证金余额",
    "notional_asset_value": "notional_asset_value/名义资产价值",
    "cash_balance": "cash_balance/现金余额",
    "total_value": "total_value/总价值",
}


def _bilingual_csv_columns(df):
    return df.rename(columns=CSV_HEADER_LABELS)
```

- [x] **Step 5: Apply bilingual headers to both CSV writers**

In `write_analysis_csv(...)`, change the final write to:

```python
    _bilingual_csv_columns(analysis_df).to_csv(csv_path, index=False)
```

In `write_trading_detail_csv(...)`, change the final write to:

```python
    _bilingual_csv_columns(pd.DataFrame(detail_rows)).to_csv(csv_path, index=False)
```

- [x] **Step 6: Run the RL CSV tests**

Run:

```bash
conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_test_agent_index.py::test_weighted_trader_passes_order_book_depth_to_base_env FineFT/tests/rl/test_test_agent_index.py::test_trading_detail_csv_records_actions_trades_and_execution_metrics -q
```

Expected: pass.

- [x] **Step 7: Run the full focused verification**

Run:

```bash
conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_test_agent_index.py FineFT/tests/env/test_commodity_env.py -q
conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/test_agent_index.py
openspec validate add-test-agent-csv-outputs --strict
```

Expected: all commands pass.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 11: Align CSV header tests and close verification with the current Chinese semantic headers.

> **trace:** plan-ready.md → `### Task 11: Align CSV header tests and close verification with the current Chinese semantic headers.` | tasks.md → `- [ ] 1.8 Align CSV header tests and close verification with the current Chinese semantic headers.`
> **sync:** tasks.md → `- [ ] 1.8 Align CSV header tests and close verification with the current Chinese semantic headers.` | plan-ready.md → `### Task 11: Align CSV header tests and close verification with the current Chinese semantic headers.`

**Files:**
- Modify: `FineFT/tests/rl/test_test_agent_index.py`
- Modify: `openspec/changes/add-test-agent-csv-outputs/close-issues.md`

- [x] **Step 1: Update aggregate CSV header assertions to Chinese semantic names**

In `FineFT/tests/rl/test_test_agent_index.py::test_weighted_trader_passes_order_book_depth_to_base_env`, replace the aggregate header assertion and JSON column reads with:

```python
    assert list(csv_df.columns) == [
        "标签",
        "初始动作",
        "分箱索引",
        "数据文件",
        "奖励总和",
        "数据长度",
        "换手率",
    ]
    assert json.loads(csv_df.loc[0, "数据文件"]) == ["df_0.feather"]
    assert json.loads(csv_df.loc[0, "奖励总和"]) == [1.0]
    assert json.loads(csv_df.loc[0, "数据长度"]) == [1]
    assert json.loads(csv_df.loc[0, "换手率"]) == [0.0]
```

- [x] **Step 2: Update detail CSV required-column assertions to Chinese semantic names**

In `FineFT/tests/rl/test_test_agent_index.py::test_trading_detail_csv_records_actions_trades_and_execution_metrics`, replace the required-column loop and detail column reads with:

```python
    for column in [
        "标签",
        "数据文件",
        "时间戳",
        "目标仓位",
        "目标杠杆",
        "单步实现盈亏",
        "累计已实现盈亏",
        "单步手续费",
        "累计手续费",
        "单步滑点",
        "累计滑点",
    ]:
        assert column in detail_df.columns
    assert detail_df.loc[0, "标签"] == "label"
    assert detail_df.loc[0, "数据文件"] == "df_0.feather"
    assert detail_df.loc[0, "目标仓位"] == 1
    assert detail_df.loc[0, "目标杠杆"] == 1
    assert detail_df.loc[0, "动作变化"] == 1
    assert detail_df.loc[0, "交易计数"] == 0
    assert detail_df.loc[1, "动作变化"] == 0
    assert detail_df.loc[1, "交易计数"] == 1
    assert detail_df.loc[1, "累计动作变化次数"] == 1
    assert detail_df.loc[1, "累计交易次数"] == 1
    assert detail_df.loc[1, "单步手续费"] == 0.7
    assert detail_df.loc[1, "累计手续费"] == 1.2
    assert detail_df.loc[1, "单步实现盈亏"] == 5.0
    assert detail_df.loc[1, "累计已实现盈亏"] == 5.0
    assert detail_df.loc[1, "单步滑点"] == 0.2
    assert detail_df.loc[1, "累计滑点"] == 0.3
    assert detail_df.loc[1, "保证金余额"] == 1004.0
    assert detail_df.loc[1, "浮动总价值"] == 1004.0
    assert detail_df.loc[1, "持仓资产"] == 101.0
```

- [x] **Step 3: Run the focused RL CSV tests**

Run:

```bash
source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_test_agent_index.py -q
```

Expected: all tests in `FineFT/tests/rl/test_test_agent_index.py` pass.

- [x] **Step 4: Re-run combined focused close verification**

Run:

```bash
source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_test_agent_index.py FineFT/tests/env/test_commodity_env.py -q
```

Expected: all focused RL/env tests pass.

- [x] **Step 5: Update close issue status after tests pass**

In `openspec/changes/add-test-agent-csv-outputs/close-issues.md`, append:

```markdown
## Resolution

- Updated tests to match the amended Chinese semantic header requirement.
- Re-ran focused pytest; result: pass.
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

---

## Self-Review

Spec coverage:
- Aggregate CSV default output is covered by Task 3.
- Detail CSV opt-in behavior and epoch file naming are covered by Task 4.
- Detail row fields, action-change counting, trade-count counting, and account value fields are covered by Task 5.
- True commission fee, realized PnL, and slippage exposure are covered by Tasks 1 and 2.
- Verification requirements are covered by Tasks 6 through 9.
- The earlier bilingual CSV header requirement is historical context from Task 10 and is superseded by the 2026-07-12 Chinese semantic header amendment in Task 11.

Placeholder scan:
- The plan contains concrete file paths, commands, expected results, and code snippets for each implementation task.

Type consistency:
- The execution metric field names are consistently `commission_fee_step`, `realized_pnl_step`, `slippage_step`, `cumulative_commission_fee`, `cumulative_realized_pnl`, and `cumulative_slippage`.
- Header requirements now use the current Chinese semantic labels confirmed by the user on 2026-07-12.
