## ADDED Requirements

### Requirement: Base Futures Trading Environment SHALL 使用显式涨跌停状态实施成交约束

系统 SHALL 在 `FineFT/env/env_class/base_env.py` 根据当前时间步的 `is_limit_down` / `is_limit_up` 判定涨跌停硬成交限制。

#### Scenario: 跌停状态只依赖 is_limit_down
- **WHEN** 当前时间步 `is_limit_down=True`
- **THEN** 环境 SHALL 启用跌停卖出限制
- **AND** 限制 SHALL 不依赖当前 validation Label 名称或 CLI Label 语义

#### Scenario: 涨停状态只依赖 is_limit_up
- **WHEN** 当前时间步 `is_limit_up=True`
- **THEN** 环境 SHALL 启用涨停买入限制
- **AND** 限制 SHALL 不依赖当前 validation Label 名称或 CLI Label 语义

#### Scenario: near-limit 不自动成为硬成交限制
- **WHEN** 当前行情只显示接近 LowerLimitPrice 或 UpperLimitPrice
- **AND** `is_limit_down=False` 且 `is_limit_up=False`
- **THEN** 环境 SHALL NOT 因 near-limit 或 Label 语义禁止买入或卖出
- **AND** 动作可用性 SHALL 继续由订单簿深度、保证金与现有环境约束决定

### Requirement: 跌停时 Futures Trading Environment SHALL 禁止所有卖出数量

系统 SHALL 在 `is_limit_down=True` 时禁止减平多头以及开出或增加空头。

#### Scenario: 跌停多头不能减仓或平仓
- **WHEN** 当前 position 为正且 `is_limit_down=True`
- **THEN** `reset()` 或上一步 `step()` 返回的 `avaliable_action` SHALL NOT 包含任何较小多仓或 position 0 目标
- **AND** 保持当前多仓的动作 SHALL 仍可用

#### Scenario: 跌停空仓或空头不能开加空仓
- **WHEN** `is_limit_down=True`
- **AND** 目标动作要求从空仓开空或增加当前空仓绝对值
- **THEN** `avaliable_action` SHALL NOT 包含该目标

#### Scenario: 跌停空头仍可买入减仓或平仓
- **WHEN** 当前 position 为负且 `is_limit_down=True`
- **AND** ask 侧订单簿深度和保证金等现有约束允许
- **THEN** 减少空仓绝对值或平到 0 的买入动作 SHALL 保持可用

### Requirement: 涨停时 Futures Trading Environment SHALL 禁止所有买入数量

系统 SHALL 在 `is_limit_up=True` 时禁止减平空头以及开出或增加多头。

#### Scenario: 涨停空头不能减仓或平仓
- **WHEN** 当前 position 为负且 `is_limit_up=True`
- **THEN** `reset()` 或上一步 `step()` 返回的 `avaliable_action` SHALL NOT 包含任何较小空仓绝对值或 position 0 目标
- **AND** 保持当前空仓的动作 SHALL 仍可用

#### Scenario: 涨停空仓或多头不能开加多仓
- **WHEN** `is_limit_up=True`
- **AND** 目标动作要求从空仓开多或增加当前多仓绝对值
- **THEN** `avaliable_action` SHALL NOT 包含该目标

#### Scenario: 涨停多头仍可卖出减仓或平仓
- **WHEN** 当前 position 为正且 `is_limit_up=True`
- **AND** bid 侧订单簿深度和保证金等现有约束允许
- **THEN** 减少多仓绝对值或平到 0 的卖出动作 SHALL 保持可用

### Requirement: Futures Trading Environment SHALL 在 step 内再次拒绝涨跌停不可成交动作

系统 SHALL 不依赖调用方遵守 `avaliable_action`；直接传入涨跌停不可成交动作时，`Base_Env.step()` SHALL 拒绝仓位变化且不中断行为轨迹。

#### Scenario: 绕过掩码直接在跌停卖出时保持实际仓位
- **WHEN** `is_limit_down=True`
- **AND** 调用方绕过 `avaliable_action` 直接向 `step()` 传入减平多头或开加空头动作
- **THEN** 环境实际 position SHALL 保持调整前值
- **AND** 环境 SHALL NOT 产生该禁止交易的开平仓税费、已实现盈亏或滑点
- **AND** `step()` SHALL 继续返回下一状态而不抛出动作不可用异常

#### Scenario: 绕过掩码直接在涨停买入时保持实际仓位
- **WHEN** `is_limit_up=True`
- **AND** 调用方绕过 `avaliable_action` 直接向 `step()` 传入减平空头或开加多头动作
- **THEN** 环境实际 position SHALL 保持调整前值
- **AND** 环境 SHALL NOT 产生该禁止交易的开平仓税费、已实现盈亏或滑点
- **AND** `step()` SHALL 继续返回下一状态而不抛出动作不可用异常

### Requirement: 涨跌停成交约束 SHALL 优先于 Reverse Position 与守卫平仓意图

系统 SHALL 在涨跌停禁止的买卖方向不虚构成交，即使目标来自 Reverse Position 或逐步 Label 动作守卫的止损平仓。

#### Scenario: 跌停阻止逆向多头的守卫止损平仓
- **WHEN** 下跌 Label 下的逆向多头已达到止损阈值
- **AND** 守卫生成 position 0 的最终动作
- **AND** 当前 `is_limit_down=True`
- **THEN** 环境 SHALL 拒绝卖出平多
- **AND** 实际多头 position SHALL 保持不变
- **AND** 行为明细仍 SHALL 记录守卫交给环境的平仓动作与执行后仓位差异

#### Scenario: 涨停阻止逆向空头的守卫止损平仓
- **WHEN** 上涨 Label 下的逆向空头已达到止损阈值
- **AND** 守卫生成 position 0 的最终动作
- **AND** 当前 `is_limit_up=True`
- **THEN** 环境 SHALL 拒绝买入平空
- **AND** 实际空头 position SHALL 保持不变

#### Scenario: Reverse Position 包含禁止交易方向时整个仓位调整被拒绝
- **WHEN** `allow_reverse_position=True`
- **AND** Reverse Position 目标的平旧仓或开新仓腿需要在跌停时卖出或在涨停时买入
- **THEN** 该目标 SHALL 从 `avaliable_action` 排除
- **AND** 直接传入 `step()` 时实际 position SHALL 保持原值
- **AND** 环境 SHALL NOT 通过部分执行虚构不可成交腿

### Requirement: 涨跌停双重防护 SHALL 保持其他高保真 Futures Trading Environment 契约

系统 SHALL 在普通非涨跌停时继续使用订单簿深度、mark price、交易税费、滑点/订单损失、杠杆、资金费、维持保证金和强平约束。

#### Scenario: 非涨跌停时不改变现有成交行为
- **WHEN** `is_limit_down=False` 且 `is_limit_up=False`
- **THEN** 可用动作与 `step()` 成交 SHALL 由现有订单簿深度、保证金、Reverse Position 开关和其他环境规则决定
- **AND** 本变更 SHALL NOT 改变资金费、维持保证金或强平计算

### Requirement: 本涨跌停成交能力 SHALL 限于 Base Futures Trading Environment

系统 SHALL 仅修改 Stage II validation 实际使用的 Base Futures Trading Environment，不要求 `Simple_Env` 同步实现本次双重防护。

#### Scenario: Simple Environment 不在本变更验收范围
- **WHEN** 实现与验证本变更
- **THEN** 定向涨跌停可用动作与 `step` 双重防护测试 SHALL 面向 `Base_Env`
- **AND** 实现 SHALL NOT 为了本变更扩展 `Simple_Env`
