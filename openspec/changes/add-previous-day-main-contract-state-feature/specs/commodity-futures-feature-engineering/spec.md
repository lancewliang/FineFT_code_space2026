# commodity-futures-feature-engineering Delta Specification

## ADDED: 3.6 基础时间与合约状态编码特征 - 前一交易日合约角色档位

### Requirement: 前一交易日合约角色档位特征 (prev_day_contract_role_tier)
系统 SHALL 在 `BASE_TIME_FEATURE` 中生成 `prev_day_contract_role_tier` 特征列，指示当前合约在上一交易日的市场流动性地位（三档值）。

#### Scenario: 判定上一交易日合约角色档位
- **WHEN** 计算指定 `contract` 在当前 `trading_day` 的 `BASE_TIME_FEATURE`
- **AND** `main_contract_summary.json` 中的 `main_sub_roles` 存在早于 `trading_day` 的历史交易日记录
- **THEN** 系统定位满足 `day < trading_day` 的最大交易日 `role_trading_day`
- **AND** 当 `main_sub_roles[role_trading_day].get(contract) == "main"` 时，整日所有 Bar 的 `prev_day_contract_role_tier` 输出 `1.0`
- **AND** 当 `main_sub_roles[role_trading_day].get(contract) == "sub"` 时，整日所有 Bar 的 `prev_day_contract_role_tier` 输出 `0.5`
- **AND** 当 `main_sub_roles[role_trading_day].get(contract) == "other"` 或未记录时，整日所有 Bar 的 `prev_day_contract_role_tier` 输出 `0.0`

#### Scenario: 历史首日与边界冷启动
- **WHEN** 计算指定 `contract` 在当前 `trading_day` 的 `BASE_TIME_FEATURE`
- **AND** `main_sub_roles` 中不存在早于 `trading_day` 的历史交易日记录
- **THEN** 系统确定性为所有 Bar 输出 `prev_day_contract_role_tier = 0.0`
- **AND** 输出 NOT 包含 `NaN`、`None` 或抛出未捕获异常

#### Scenario: 强制保留与直通不缩放
- **WHEN** 执行特征选择与特征缩放
- **THEN** `prev_day_contract_role_tier` 作为 `BASE_TIME_FEATURE_COLUMNS` 成员传入 `--mandatory_state_features`
- **AND** `prev_day_contract_role_tier` 作为 `BASE_TIME_FEATURE_COLUMNS` 成员传入 `--passthrough_features`
- **AND** 输出数据中该列数值严格在 `{0.0, 0.5, 1.0}` 集合内
