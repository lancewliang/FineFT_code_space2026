# commodity-futures-support Specification

## Requirements

### Requirement: 商品期货 BASE_TIME_FEATURE 时间编码特征
系统 SHALL 生成 9 个非绝对 BASE_TIME_FEATURE 时间编码特征，强制作为 State Feature 保留并跳过 Robust Scaling。

#### Scenario: 9 个 BASE_TIME_FEATURE 列名与语义
- **WHEN** 生成商品期货 `BASE_TIME_FEATURE`
- **THEN** 输出包含 9 个列：`trading_minute_progress`、`morning_session`、`afternoon_session`、`night_session`、`is_opening_30m`、`is_closing_30m`、`contract_month_sin`、`contract_month_cos` 和 `contract_life_remaining_ratio`
- **AND** `trading_minute_progress` 为当前 timestamp 在所属 Trading Session 内的归一化进度
- **AND** `morning_session` / `afternoon_session` / `night_session` 为互斥 one-hot 标记
- **AND** `is_opening_30m` / `is_closing_30m` 为 session 独立首尾半小时标记
- **AND** `contract_month_sin` / `contract_month_cos` 为合约交割月份的 sin/cos 周期编码
- **AND** `contract_life_remaining_ratio` 为合约剩余生命周期比例

#### Scenario: Daily Merge join BASE_TIME_FEATURE
- **WHEN** 运行 daily merge
- **THEN** 按 `timestamp` 将 `BASE_TIME_FEATURE` join 到 `FUTURE_FEATURE`
- **AND** timestamp 不一致时 fail-fast

#### Scenario: Feature Selection 强制保留 BASE_TIME_FEATURE
- **WHEN** 运行 Feature Selection
- **THEN** 传入 `--mandatory_state_features` 保护 `BASE_TIME_FEATURE_COLUMNS` 9 个特征列
- **AND** 这些特征不参与 Hard Filter、Stability Filter、Composite Score、Correlation Filter 或 Blacklist 过滤，强制保留在 `state_features.npy` 中

#### Scenario: Scale Save 跳过 BASE_TIME_FEATURE 缩放
- **WHEN** 运行 Scale Save
- **THEN** 传入 `--passthrough_features` 包含 `BASE_TIME_FEATURE_COLUMNS` 9 个特征列
- **AND** 这些特征列直接保存原始编码值，不参与 robust scaler 的 fit、transform 或 clip
