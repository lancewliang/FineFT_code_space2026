# Design: add-test-agent-csv-outputs

## Context

`FineFT/RL/DiHFT/low_level/test_agent_index.py` currently writes aggregate validation results to `analysis_result.npy`. The aggregate payload is a list of dictionaries keyed by `label`, `initial_action`, and `bin_index`, with list-valued metrics for each validation feather file. The payload does not include `df_path`, so readers cannot reliably map each list element back to the source file without reconstructing the directory traversal order.

The requested trading detail CSV also needs values that are not currently exposed as first-class step outputs. `Base_Env.step()` receives a tuple from `change_of_wallet(...)` that includes slippage, but `commission_fee` and `realized_pnl` are computed inside lower-level helpers and are not returned to the caller. Inferring those values from wallet balance deltas would mix realized PnL, fees, opening losses, funding effects, unrealized PnL changes, and slippage.

## Goals

- Keep `analysis_result.npy` compatible while adding a readable aggregate CSV.
- Make aggregate CSV rows self-describing by including `df_path` aligned with list-valued metrics.
- Add optional per-step trading detail output without making every test run emit a large CSV.
- Record true step-level commission fee, realized PnL, and slippage from the trading calculation path.
- Distinguish model action changes from actual executed position/leverage changes.

## Decisions

### Aggregate CSV

`test_agent_index.py` will continue to build `overall_result` and save `analysis_result.npy`. Each aggregate record will also include `df_path`, a list aligned with `reward_sum`, `df_length`, and `turnover`. The script will write `analysis_result.csv` in the same `epoch_path`; list-valued columns will be serialized as JSON array strings.

### Trading detail toggle

`test_agent_index.py` will add a `--save_trading_detail_csv` flag. When absent, the detailed per-step CSV is not generated. When present, the script writes one file for the tested epoch: `trading_action_detail_epoch_<epoch_num>.csv`.

### Trade execution metrics

The environment layer will expose explicit step execution metrics. The preferred shape is a small structured value or dictionary with:

- `realized_pnl_step`
- `commission_fee_step`
- `slippage_step`

`Base_Env.step()` will reset/update those fields for each action and include them in returned `info`, so callers can read the values without depending on private state. `change_of_wallet(...)` and its open/close helpers will be extended to return the execution metrics while preserving existing callers through a minimal compatibility strategy.

`Simple_Env` and existing tests that call `change_of_wallet(...)` directly must be updated consistently so the return shape is handled deliberately.

### Detail row construction

`test_agent_index.py` will capture position/leverage before `env.step(action)`, derive target position/leverage from the action, then read position/leverage and execution metrics after the step. `trade_count_step` is based on the actual post-step environment state, not on whether the selected action differs from the previous action.

The CSV will keep both environment equity (`margin_balance` / `total_value`) and notional asset value (`mark_price * position_after`) as separate columns.

## Risks

- Changing `change_of_wallet(...)` return values can break existing callers if not updated together.
- Per-step detail output can be large, so it must remain opt-in.
- OHLCV columns vary by dataset; CSV generation must write existing market columns without making optional columns mandatory.

