# non-main-contract-defensive-gating Delta Specification

## ADDED Requirements

### Requirement: 非主力合约门控防御策略 (non-main contract defensive gating)
当门控防御开关启用且状态中包含 `prev_day_contract_role_tier` 时，门控决策层 SHALL 在该特征值指示非主力/次主力合约状态（值小于 0.5）时自动拦截常规 VAE 路由与底层 Agent 调度，强制执行规则平仓动作（`rule_based_close`）。

#### Scenario: 非主力合约状态触发逐步平仓防御
- **GIVEN** `enable_non_main_contract_defense` 处于启用状态
- **AND** 状态特征列表中包含 `prev_day_contract_role_tier`
- **WHEN** 门控执行 `get_action` 且当前状态中 `prev_day_contract_role_tier < 0.5`
- **THEN** 门控直接返回 `rule_based_close` 计算的平仓/空仓动作
- **AND** `macro_action_history` 追加记录防御槽位 `slot_count`
- **AND** 门控底层的 Q-Network `agent_act` NOT 被调用

#### Scenario: 主力与次主力合约正常执行 VAE 门控
- **GIVEN** `enable_non_main_contract_defense` 处于启用状态
- **AND** 状态特征列表中包含 `prev_day_contract_role_tier`
- **WHEN** 门控执行 `get_action` 且当前状态中 `prev_day_contract_role_tier >= 0.5`（1.0 主力或 0.5 次主力）
- **THEN** 门控按常规双轴 VAE 似然分位数进行阈值比较与 Slot Agent 选择
- **AND** 若未触发 OOD 则调度选定 Agent 的 Q-Network 输出动作

#### Scenario: 缺失合约角色特征时保持安全回退
- **GIVEN** `enable_non_main_contract_defense` 处于启用状态
- **AND** 数据集特征列表中不包含 `prev_day_contract_role_tier`
- **WHEN** 门控执行 `get_action`
- **THEN** 门控回退到常规 VAE 路由调度，不产生未捕获异常

### Requirement: 消融与退化测试开关 (enable_non_main_contract_defense)
系统 SHALL 提供 CLI 与配置参数 `--enable_non_main_contract_defense`，默认为 `False`，支持在开启与关闭状态之间无缝切换以进行消融和退化测试。

#### Scenario: 开关处于默认关闭状态（消融基准）
- **GIVEN** 未传递 `--enable_non_main_contract_defense`（即值为 `False`）
- **WHEN** 门控在 `prev_day_contract_role_tier < 0.5` 的非主力合约状态下执行 `get_action`
- **THEN** 门控忽略合约角色，正常执行常规 VAE 门控与 Agent 调度
- **AND** 宏观动作时序记录与输出动作与既有基线保持完全一致

#### Scenario: Optuna 搜参跨 Trial 配置继承
- **GIVEN** 在 Optuna 寻优主进程中配置了 `--enable_non_main_contract_defense`
- **WHEN** 每一个 Worker 进程通过 `reconfigure_routing` 切换不同超参数 Trial
- **THEN** 该防御开关配置在复用 router 实例时保持不变
