# fineft-low-level-test-results Specification

## Purpose
TBD - created by archiving change add-test-agent-csv-outputs. Update Purpose after archive.
## Requirements
### Requirement: Low-level test agent SHALL export readable aggregate CSV results
系统 SHALL 在 `FineFT/RL/DiHFT/low_level/test_agent_index.py` 完成低层 agent index 测试后，保留现有 `analysis_result.npy`，并在同一 epoch 输出目录中生成可读的 `analysis_result.csv`。

#### Scenario: Aggregate CSV is written next to the existing npy result
- **WHEN** `weighted_trader.test()` completes the validation loops for an epoch
- **THEN** the system SHALL save the existing `analysis_result.npy` under `epoch_path`
- **AND** the system SHALL save `analysis_result.csv` under the same `epoch_path`
- **AND** the CSV write SHALL fail fast if the file cannot be written

#### Scenario: Aggregate CSV keeps one row per label-action-bin group
- **WHEN** the system writes `analysis_result.csv`
- **THEN** each CSV row SHALL correspond to one `(label, initial_action, bin_index)` group
- **AND** the CSV SHALL include the columns `label`, `initial_action`, `bin_index`, `df_path`, `reward_sum`, `df_length`, and `turnover`
- **AND** the system SHALL NOT expand the aggregate CSV to one row per validation feather file

#### Scenario: Aggregate CSV preserves df path alignment
- **WHEN** the system records aggregate metrics for a group
- **THEN** `df_path` SHALL contain the validation feather file names processed for that group
- **AND** `reward_sum`, `df_length`, and `turnover` SHALL remain aligned with `df_path` by list index
- **AND** `df_path`, `reward_sum`, `df_length`, and `turnover` SHALL be serialized in CSV cells as JSON array strings
- **AND** those JSON array strings SHALL be parseable by `json.loads`

### Requirement: Low-level test agent SHALL optionally export per-step trading detail CSV results
系统 SHALL 支持通过显式参数为 `test_agent_index.py` 生成每个 epoch 的逐时间步交易动作明细 CSV，并在默认情况下不生成该大文件。

#### Scenario: Detail CSV is disabled by default
- **WHEN** `test_agent_index.py` runs without `--save_trading_detail_csv`
- **THEN** the system SHALL NOT write a trading detail CSV
- **AND** the aggregate `analysis_result.npy` and `analysis_result.csv` outputs SHALL still be written

#### Scenario: Detail CSV is written when enabled
- **WHEN** `test_agent_index.py` runs with `--save_trading_detail_csv`
- **THEN** the system SHALL write `trading_action_detail_epoch_<epoch_num>.csv` under `epoch_path`
- **AND** the detail CSV SHALL contain rows for all tested `label`, `df_path`, `initial_action`, and `bin_index` combinations
- **AND** each row SHALL correspond to one environment time step
- **AND** detail CSV write failures SHALL be surfaced as errors

#### Scenario: Detail CSV includes market and test context fields
- **WHEN** the system writes a detail CSV row
- **THEN** the row SHALL include `label`, `df_path`, `initial_action`, `bin_index`, `timestep`, and `timestamp`
- **AND** the row SHALL include any of `open`, `high`, `low`, `close`, `volume`, and `mark_price` that are present in the source validation dataframe
- **AND** missing optional OHLCV columns SHALL NOT cause the test run to fail

#### Scenario: Detail CSV records requested and executed action state
- **WHEN** the system writes a detail CSV row after an environment step
- **THEN** the row SHALL include `action`, `target_position`, `target_leverage`, `position_before`, `leverage_before`, `position_after`, and `leverage_after`
- **AND** `target_position` and `target_leverage` SHALL be derived from the selected action using the same action mapping semantics as the environment
- **AND** `position_after` and `leverage_after` SHALL reflect the actual environment state after executing the action
- **AND** `position_after` and `leverage_after` MAY differ from the target values when the environment rejects or adjusts the requested transition

#### Scenario: Detail CSV distinguishes action changes from actual trades
- **WHEN** the system writes sequential detail CSV rows for one trajectory
- **THEN** `action_change_step` SHALL be `1` when the current action id differs from the previous action id for that trajectory, otherwise `0`
- **AND** `cumulative_action_change_count` SHALL be the trajectory-local cumulative sum of `action_change_step`
- **AND** `trade_count_step` SHALL be `1` when actual `position_after` differs from `position_before` or actual `leverage_after` differs from `leverage_before`, otherwise `0`
- **AND** `cumulative_trade_count` SHALL be the trajectory-local cumulative sum of `trade_count_step`
- **AND** an action id change without an actual position or leverage change SHALL increase only the action-change count

#### Scenario: Detail CSV records true execution economics and account values
- **WHEN** the system writes a detail CSV row after an environment step
- **THEN** the row SHALL include `step_reward`, `realized_pnl_step`, `cumulative_realized_pnl`, `commission_fee_step`, `cumulative_commission_fee`, `slippage_step`, and `cumulative_slippage`
- **AND** `commission_fee_step` SHALL be the true commission fee produced by the trade calculation for that step
- **AND** `realized_pnl_step` SHALL be the true realized PnL produced by the trade calculation for that step
- **AND** the system SHALL NOT infer `commission_fee_step` or `realized_pnl_step` from wallet balance deltas
- **AND** the row SHALL include `wallet_balance`, `unrealized_pnl`, `margin_balance`, `notional_asset_value`, `cash_balance`, and `total_value`
- **AND** `margin_balance` SHALL equal `wallet_balance + unrealized_pnl`
- **AND** `total_value` SHALL equal `margin_balance`
- **AND** `notional_asset_value` SHALL equal `mark_price * position_after`
- **AND** `cash_balance` SHALL equal `wallet_balance`

### Requirement: Trading environment SHALL expose true per-step execution metrics
系统 SHALL 在低层测试使用的交易环境中显式暴露每步真实手续费、真实已实现利润和真实滑点，供测试明细 CSV 使用。

#### Scenario: Wallet change helpers return explicit execution metrics
- **WHEN** `change_of_wallet(...)` or its open/close helper functions process a requested position or leverage transition
- **THEN** the result SHALL expose `commission_fee_step`, `realized_pnl_step`, and `slippage_step`
- **AND** no-trade or leverage-only transitions SHALL expose zero commission fee and zero realized PnL unless an existing trade helper actually realizes PnL
- **AND** failed or rejected position transitions SHALL expose zero commission fee, zero realized PnL, and zero slippage for the rejected trade portion

#### Scenario: Base environment info exposes execution metrics
- **WHEN** `Base_Env.step(action)` returns
- **THEN** the returned `info` SHALL include the execution metrics for that step
- **AND** the environment SHALL maintain cumulative commission fee, cumulative realized PnL, and cumulative slippage values that match the sum of step metrics since the last reset
- **AND** `Base_Env.reset()` SHALL reset those step and cumulative execution metrics

#### Scenario: Existing environment callers remain compatible
- **WHEN** the execution metric exposure is implemented
- **THEN** existing direct callers of `change_of_wallet(...)` in environment code and tests SHALL be updated to use the new explicit result shape
- **AND** `Simple_Env` SHALL continue to run with the updated wallet-change result
- **AND** existing fee-rate behavior for buy and sell fees SHALL remain unchanged

#### Scenario: Lightweight verification covers CSV and execution metric behavior
- **WHEN** this change is implemented
- **THEN** focused tests SHALL verify aggregate CSV generation and JSON list parsing
- **AND** focused tests SHALL verify detail CSV disabled-by-default behavior
- **AND** focused tests SHALL verify detail CSV file naming and required columns when enabled
- **AND** focused tests SHALL cover a trajectory where the action id changes but the actual position and leverage do not change
- **AND** focused tests SHALL verify that commission fee, realized PnL, and slippage values used in detail CSV rows come from explicit environment metrics
- **AND** `openspec validate add-test-agent-csv-outputs --strict` SHALL pass

### Requirement: Low-level test agent CSV headers SHALL use the current Chinese semantic labels
系统 SHALL 为低层测试导出的 CSV 使用当前实现中的中文语义表头，并保持 CSV 单元格内容、字段语义和行粒度不变。

#### Scenario: Aggregate CSV uses Chinese semantic headers
- **WHEN** the system writes `analysis_result.csv`
- **THEN** the CSV header row SHALL use these exact columns in order: `标签`, `初始动作`, `分箱索引`, `数据文件`, `奖励总和`, `数据长度`, and `换手率`
- **AND** the JSON array cells under `数据文件`, `奖励总和`, `数据长度`, and `换手率` SHALL remain parseable by `json.loads`

#### Scenario: Detail CSV uses Chinese semantic headers
- **WHEN** the system writes `trading_action_detail_epoch_<epoch_num>.csv`
- **THEN** every detail CSV column emitted by the implementation SHALL use the current Chinese semantic label from `CSV_HEADER_LABELS`
- **AND** the required context headers SHALL include `标签`, `数据文件`, `初始动作`, `分箱索引`, `时间步`, and `时间戳`
- **AND** the required action-state headers SHALL include `动作`, `目标仓位`, `目标杠杆`, `执行前仓位`, `执行前杠杆`, `执行后仓位`, and `执行后杠杆`
- **AND** the required count headers SHALL include `动作变化`, `交易计数`, `累计动作变化次数`, and `累计交易次数`
- **AND** the required execution metric headers SHALL include `单步实现盈亏`, `累计已实现盈亏`, `单步手续费`, `累计手续费`, `单步滑点`, and `累计滑点`
- **AND** the required account value headers SHALL include `结算总价值`, `浮动盈亏`, `保证金余额`, `持仓资产`, and `浮动总价值`
- **AND** existing row values for action state, trade counts, execution economics, and account values SHALL NOT be changed by the header translation

