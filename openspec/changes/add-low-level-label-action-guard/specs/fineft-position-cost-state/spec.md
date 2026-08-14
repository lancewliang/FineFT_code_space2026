## ADDED Requirements

### Requirement: Futures Trading Environment SHALL 暴露当前持仓开仓价与持仓均价

系统 SHALL 在 `FineFT/env/env_class/base_env.py` 维护 `current_holding_opening_price` 和 `current_holding_average_price`，并通过环境属性、`reset()` 返回的 `info` 和每次 `step()` 返回的 `info` 对外暴露。

#### Scenario: 空仓 reset 暴露零成本价
- **WHEN** Futures Trading Environment 使用 position 为 0 的 `initial_state` 执行 `reset()`
- **THEN** 环境属性 `current_holding_opening_price` SHALL 为 `0.0`
- **AND** 环境属性 `current_holding_average_price` SHALL 为 `0.0`
- **AND** `info` 中的同名字段 SHALL 均为 `0.0`

#### Scenario: 成本价不改变四维 Trading Process Feature
- **WHEN** 环境在 `reset()` 或 `step()` 中暴露当前持仓成本价
- **THEN** `info["trading_info"]` SHALL 仍为四字段契约
- **AND** 开仓价与持仓均价 SHALL NOT 加入 Q 网络输入
- **AND** 现有四字段 checkpoint 结构 SHALL 不因本变更而需要迁移

### Requirement: 开仓成本 SHALL 基于真实订单簿成交与已发生开仓税费

系统 SHALL 从 `FineFT/env/env_class/futures_util.py` 的交易结算结果获取开仓腿实际数量、成交额和开仓税费，以支持普通开仓、加仓、部分成交和 Reverse Position 的成本维护。

#### Scenario: 多头有效开仓价包含滑点与开仓税费
- **WHEN** 环境从空仓通过 ask 侧订单簿成交数量 `q > 0`
- **AND** 实际开仓成交额为 `open_value`，已发生开仓税费为 `open_fee`
- **THEN** `current_holding_opening_price` SHALL 等于 `(open_value + open_fee) / q`
- **AND** `current_holding_average_price` SHALL 等于该值
- **AND** `open_value` SHALL 来自真实订单簿成交以保留滑点

#### Scenario: 空头有效开仓价包含滑点与开仓税费
- **WHEN** 环境从空仓通过 bid 侧订单簿成交数量 `q > 0`
- **AND** 实际开仓成交额为 `open_value`，已发生开仓税费为 `open_fee`
- **THEN** `current_holding_opening_price` SHALL 等于 `(open_value - open_fee) / q`
- **AND** `current_holding_average_price` SHALL 等于该值
- **AND** `open_value` SHALL 来自真实订单簿成交以保留滑点

#### Scenario: 成本价不预估平仓税费
- **WHEN** 环境建立或增加当前持仓
- **THEN** 开仓价和持仓均价 SHALL 只包含已发生的开仓税费
- **AND** 系统 SHALL NOT 使用预计平仓成交价或未来平仓税费调整成本价

### Requirement: 当前持仓成本状态 SHALL 遵守持仓回合生命周期

系统 SHALL 以从非零仓位开始到平仓、换向或轨迹结束的当前持仓为成本状态边界。

#### Scenario: 同向加仓保留开仓价并更新持仓均价
- **WHEN** 当前仓位绝对数为 `q_old > 0`，持仓均价为 `p_old`
- **AND** 同方向实际新增仓位为 `q_add > 0`，有效开仓价为 `p_add`
- **THEN** `current_holding_opening_price` SHALL 保持当前持仓首笔开仓值不变
- **AND** `current_holding_average_price` SHALL 等于 `(q_old * p_old + q_add * p_add) / (q_old + q_add)`
- **AND** 计算 SHALL 使用实际成交数量，不使用未完全成交的目标数量

#### Scenario: 部分减仓不改变开仓价与持仓均价
- **WHEN** 环境实际减少当前持仓绝对数但仍保持原方向非零仓位
- **THEN** `current_holding_opening_price` SHALL 保持不变
- **AND** `current_holding_average_price` SHALL 保持不变

#### Scenario: 完全平仓清空成本价
- **WHEN** 环境实际 position 从非零变为 0
- **THEN** `current_holding_opening_price` SHALL 变为 `0.0`
- **AND** `current_holding_average_price` SHALL 变为 `0.0`
- **AND** 返回 `info` SHALL 反映零成本价

#### Scenario: Reverse Position 使用新开仓腿重置成本价
- **WHEN** `allow_reverse_position=True`
- **AND** 环境实际先平旧仓后成功建立新方向仓位
- **THEN** 系统 SHALL 结束旧当前持仓成本状态
- **AND** 新 `current_holding_opening_price` SHALL 仅使用新方向开仓腿的实际成交与开仓税费
- **AND** 新 `current_holding_average_price` SHALL 等于该新开仓价
- **AND** 旧方向平仓腿的成交额与税费 SHALL NOT 混入新成本价

#### Scenario: Reverse Position 只平仓未开出新仓时成本价归零
- **WHEN** Reverse Position 因保证金或深度限制只完成旧仓平仓
- **AND** 实际 position 为 0
- **THEN** 两个成本价 SHALL 均为 `0.0`

### Requirement: Initial-action 情景 SHALL 使用首行 mark price 初始化非零持仓成本

系统 SHALL 保留现有 Initial-action 反事实回测语义，不在 reset 时虚构真实订单成交。

#### Scenario: 非零 initial position 使用 validation 片段首行 mark price
- **WHEN** `Base_Env` 使用非零 initial position 执行 `reset()`
- **AND** validation Market Dynamic Segment 首行 `mark_price` 为 `p0`
- **THEN** `current_holding_opening_price` SHALL 等于 `p0`
- **AND** `current_holding_average_price` SHALL 等于 `p0`
- **AND** reset SHALL NOT 扣除虚构开仓税费
- **AND** reset SHALL NOT 根据订单簿虚构开仓滑点

### Requirement: 交易结算结果扩展 SHALL 保持现有调用契约

系统 SHALL 以向后兼容方式扩展 `WalletChangeResult` 类似的命名结果，保留现有六值 tuple-compatible 迭代与索引行为。

#### Scenario: 旧六值解包不因开仓腿元数据失效
- **WHEN** 现有调用方继续将交易结算结果解包为 leverage、position、initial margin、unrealized PnL、wallet balance 和 slippage
- **THEN** 解包 SHALL 保持成功
- **AND** 新开仓腿元数据 SHALL 通过命名字段读取
