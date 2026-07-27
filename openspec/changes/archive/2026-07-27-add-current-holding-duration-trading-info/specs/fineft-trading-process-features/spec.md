## ADDED Requirements

### Requirement: Trading Process Feature SHALL include normalized Current Holding Duration
系统 SHALL 通过 Futures Trading Environment 暴露四字段 `trading_info`，其中第四个字段为 `current_holding_duration_norm`。

#### Scenario: Trading info keys define the four-field contract
- **WHEN** 调用方读取 Trading Process Feature 的字段顺序
- **THEN** 系统 SHALL 提供字段 `position_exposure`
- **AND** 系统 SHALL 提供字段 `single_holding_return_rate`
- **AND** 系统 SHALL 提供字段 `single_holding_max_drawdown`
- **AND** 系统 SHALL 提供字段 `current_holding_duration_norm`
- **AND** 字段顺序 SHALL 与环境返回的 `trading_info` 数组顺序一致

#### Scenario: Flat reset exposes zero duration
- **WHEN** Futures Trading Environment 使用 position 为 0 的 `initial_state` 执行 `reset()`
- **THEN** 返回的 `info["trading_info"]` SHALL 是四维数组
- **AND** `current_holding_duration_norm` SHALL 等于 0

#### Scenario: Non-zero reset starts an existing Current Holding
- **WHEN** Futures Trading Environment 使用非零 position 的 `initial_state` 执行 `reset()`
- **THEN** 返回的 `info["trading_info"]` SHALL 是四维数组
- **AND** `current_holding_duration_norm` SHALL 等于 `1 / holding_duration_norm_steps`

#### Scenario: Opening a position starts duration at one step
- **WHEN** 环境从空仓执行动作后实际 position 变为非零
- **THEN** 下一状态的 `current_holding_duration_norm` SHALL 等于 `1 / holding_duration_norm_steps`

#### Scenario: Same-direction holding increments duration
- **WHEN** 环境在非零 position 下执行动作后实际 position 与旧 position 同方向
- **AND** 实际 position 未归零
- **THEN** 下一状态的 Current Holding Duration SHALL 比上一个可观测状态增加 1 个 env step
- **AND** `current_holding_duration_norm` SHALL 使用增加后的 Current Holding Duration 计算

#### Scenario: Same-direction add or reduce does not reset duration
- **WHEN** 环境执行同方向加仓或同方向减仓
- **THEN** 系统 SHALL 保留同一个 Current Holding
- **AND** 系统 SHALL NOT 将 `current_holding_duration_norm` 重置为 `1 / holding_duration_norm_steps`

#### Scenario: Closing to flat resets duration
- **WHEN** 环境执行动作后实际 position 变为 0
- **THEN** 下一状态的 `current_holding_duration_norm` SHALL 等于 0

#### Scenario: Reverse Position starts a new direction at one step
- **WHEN** 环境执行动作后实际 position 与旧 position 方向相反
- **AND** 实际 position 非零
- **THEN** 系统 SHALL 结束旧 Current Holding
- **AND** 新方向的 `current_holding_duration_norm` SHALL 等于 `1 / holding_duration_norm_steps`

#### Scenario: Duration normalization clips long holdings
- **WHEN** Current Holding Duration 大于或等于 `holding_duration_norm_steps`
- **THEN** `current_holding_duration_norm` SHALL 等于 1.0
- **AND** `current_holding_duration_norm` SHALL NOT 超过 1.0

#### Scenario: Invalid normalization window fails fast
- **WHEN** 调用方用小于等于 0 的 `holding_duration_norm_steps` 构造环境
- **THEN** 系统 SHALL 抛出错误
- **AND** 系统 SHALL NOT 创建会产生非有限 `current_holding_duration_norm` 的环境

### Requirement: Low-level Q networks SHALL consume the four-field Trading Process Feature
系统 SHALL 将低层 Q 网络默认 `trading_info` 输入维度与 Futures Trading Environment 暴露的四字段 `trading_info` 契约保持一致。

#### Scenario: Qnet accepts four-field trading info by default
- **WHEN** 调用方用默认参数构造 Qnet
- **AND** 传入 shape 为 `(batch_size, 4)` 的 `trading_info`
- **THEN** Qnet forward SHALL 成功返回 action value

#### Scenario: Ensemble Qnet accepts four-field trading info by default
- **WHEN** 调用方用默认参数构造 ensemble_Qnet
- **AND** 传入 shape 为 `(batch_size, 4)` 的 `trading_info`
- **THEN** ensemble_Qnet forward SHALL 成功返回 ensemble action value

#### Scenario: Stage I low-level paths use the upgraded model contract
- **WHEN** Stage I low-level training、parallel training、low-level testing 或依赖低层模型的 routing 路径构造低层 Q 网络
- **THEN** 构造出的低层 Q 网络 SHALL 与四字段 `trading_info` 兼容
- **AND** 这些路径 SHALL NOT 继续硬编码三字段 `TRADING_INFO_DIM`

#### Scenario: Old three-field checkpoints are not compatibility-loaded
- **WHEN** 用户尝试把旧三字段低层 checkpoint 加载到四字段低层模型中
- **THEN** 系统 MAY 因模型权重 shape 不匹配而失败
- **AND** 系统 SHALL NOT 自动补零或静默迁移旧 checkpoint

### Requirement: Replay buffers SHALL preserve the four-field Trading Process Feature
系统 SHALL 在 replay buffer 存储和采样 `info["trading_info"]` 时保持四字段 Trading Process Feature 的 shape 和数值。

#### Scenario: Multi-step replay buffer samples four-field trading info
- **WHEN** replay buffer 存入包含四字段 `trading_info` 的 transition
- **AND** 调用方执行 sample
- **THEN** 返回的当前 `infos["trading_info"]` SHALL 是 tensor
- **AND** 返回的下一状态 `next_infos["trading_info"]` SHALL 是 tensor
- **AND** 两者最后一维 SHALL 等于 4

#### Scenario: Evaluate sample preserves four-field trading info
- **WHEN** replay buffer 存入包含四字段 `trading_info` 的 transition
- **AND** 调用方执行 sample_evaluate
- **THEN** 返回的当前和下一状态 `trading_info` SHALL 保持四字段 shape
