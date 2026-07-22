## MODIFIED Requirements

### Requirement: Low-level test agent SHALL export readable aggregate CSV results
系统 SHALL 在 `FineFT/RL/DiHFT/low_level/test_agent_index.py` 完成低层 agent index 测试后，保留现有 `analysis_result.npy`，并在同一 epoch 输出目录中生成可读的 `analysis_result.csv`。

#### Scenario: Aggregate CSV is written next to the existing npy result
- **WHEN** `weighted_trader.test()` completes the validation loops for an epoch
- **THEN** the system SHALL save the existing `analysis_result.npy` under `epoch_path`
- **AND** the system SHALL save `analysis_result.csv` under the same `epoch_path`
- **AND** the CSV write SHALL fail fast if the file cannot be written

#### Scenario: Aggregate result reads commodity contract label slices only
- **WHEN** `weighted_trader.test()` discovers validation feather files
- **THEN** the system SHALL read only files matching `valid/<contract>/label_*/df_*.feather`
- **AND** the system SHALL ignore `valid/processed/*` and root-level `valid/*.feather` files
- **AND** the system SHALL fail fast when no matching label slice files exist
- **AND** the system SHALL parse `<contract>` into a `contract` field and `<label>` into a pure `label_<integer>` field

#### Scenario: Aggregate CSV keeps one row per pure label-action-bin group
- **WHEN** the system writes `analysis_result.csv`
- **THEN** each CSV row SHALL correspond to one `(label, initial_action, bin_index)` group
- **AND** `label` SHALL be the pure label name such as `label_0`, without any contract path prefix
- **AND** each row SHALL aggregate all validation slices for that pure label across all discovered contracts
- **AND** the CSV SHALL include the columns `label`, `initial_action`, `bin_index`, `contract`, `df_path`, `reward_sum`, `df_length`, and `turnover`
- **AND** the system SHALL NOT expand the aggregate CSV to one row per validation feather file

#### Scenario: Aggregate CSV preserves contract and df path alignment
- **WHEN** the system records aggregate metrics for a group
- **THEN** `contract` SHALL contain the contract name for each processed validation slice
- **AND** `df_path` SHALL contain the contract-relative validation feather path for each processed slice, such as `fu2409/label_0/df_0.feather`
- **AND** `contract`, `df_path`, `reward_sum`, `df_length`, and `turnover` SHALL remain aligned by list index
- **AND** `contract`, `df_path`, `reward_sum`, `df_length`, and `turnover` SHALL be serialized in CSV cells as JSON array strings
- **AND** those JSON array strings SHALL be parseable by `json.loads`

### Requirement: Low-level test agent CSV headers SHALL use the current Chinese semantic labels
系统 SHALL 为低层测试导出的 CSV 使用当前实现中的中文语义表头，并保持 CSV 单元格内容、字段语义和行粒度不变。

#### Scenario: Aggregate CSV uses Chinese semantic headers
- **WHEN** the system writes `analysis_result.csv`
- **THEN** the CSV header row SHALL use these exact columns in order: `标签`, `初始动作`, `分箱索引`, `合约`, `数据文件`, `奖励总和`, `数据长度`, and `换手率`
- **AND** the JSON array cells under `合约`, `数据文件`, `奖励总和`, `数据长度`, and `换手率` SHALL remain parseable by `json.loads`

#### Scenario: Detail CSV uses Chinese semantic headers
- **WHEN** the system writes `trading_action_detail_epoch_<epoch_num>.csv`
- **THEN** every detail CSV column emitted by the implementation SHALL use the current Chinese semantic label from `CSV_HEADER_LABELS`
- **AND** the required context headers SHALL include `标签`, `数据文件`, `初始动作`, `分箱索引`, `时间步`, and `时间戳`
- **AND** if the detail CSV emits contract context, the contract header SHALL be `合约`
- **AND** the required action-state headers SHALL include `动作`, `目标仓位`, `目标杠杆`, `执行前仓位`, `执行前杠杆`, `执行后仓位`, and `执行后杠杆`
- **AND** the required count headers SHALL include `动作变化`, `交易计数`, `累计动作变化次数`, and `累计交易次数`
- **AND** the required execution metric headers SHALL include `单步实现盈亏`, `累计已实现盈亏`, `单步手续费`, `累计手续费`, `单步滑点`, and `累计滑点`
- **AND** the required account value headers SHALL include `结算总价值`, `浮动盈亏`, `保证金余额`, `持仓资产`, and `浮动总价值`
- **AND** existing row values for action state, trade counts, execution economics, and account values SHALL NOT be changed by the header translation

