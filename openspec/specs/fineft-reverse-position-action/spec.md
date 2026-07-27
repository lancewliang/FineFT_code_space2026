# fineft-reverse-position-action Specification

## Purpose
定义 Futures Trading Environment 中的反手操作 (Reverse Position Action) 语义、可用动作计算、DP Q 表响应和功能开关规则。

## Requirements

### Requirement: Futures Trading Environment 反手操作
系统 SHALL 在 `FineFT/env/` 中提供反手操作支持，允许 Agent 在一步内平掉当前仓位并反向开仓。

#### Scenario: 默认关闭反手操作
- **WHEN** 环境初始化 `allow_reverse_position=False`（默认值）
- **AND** 当前持仓为多头 (position > 0) 或空头 (position < 0)
- **AND** Agent 选择反向仓位 (target_position * current_position < 0)
- **THEN** 系统 SHALL 打印 warning
- **AND** 系统 SHALL 拒绝反手操作，返回原始状态不变
- **AND** `calculate_avaiable_action()` SHALL NOT 包含反方向仓位选项

#### Scenario: 开关打开时执行反手操作
- **WHEN** 环境初始化 `allow_reverse_position=True`
- **AND** Agent 从当前仓位选择反向仓位 (target_position * current_position < 0)
- **THEN** 系统 SHALL 先平仓（调用 `close_long_position` 或 `close_short_position`），结算已实现盈亏与平仓手续费/滑点
- **AND** 系统 SHALL 用平仓后的 `wallet_balance` 检查反向开仓保证金
- **AND** 保证金充足时，系统 SHALL 用 `target_leverage` 执行反向开仓
- **AND** 深度不足时，系统 SHALL 截断到 `position_list` 中最大可行仓位
- **AND** 保证金不足时，系统 SHALL 只保留平仓，position 归零
- **AND** 返回的 `WalletChangeResult` SHALL 包含两步的累计滑点和手续费

#### Scenario: 反手时重置单次持仓收益与历史
- **WHEN** `allow_reverse_position=True`
- **AND** `step()` 检测到 `position * new_position < 0`
- **THEN** 系统 SHALL 重置 `single_holding_return = 0` 和 `single_holding_history = [0]`
- **AND** 重置行为 SHALL 与 position 归零时的逻辑一致

#### Scenario: 可用动作精确模拟反手两步
- **WHEN** `allow_reverse_position=True`
- **AND** 调用 `calculate_avaiable_action()`
- **THEN** 系统 SHALL 计算平仓后的预估 `wallet_balance_new`
- **AND** 系统 SHALL 检查反向开仓所需的初始保证金与开仓损失
- **AND** 系统 SHALL 将满足两步保证金要求的反向仓位加入可用动作

#### Scenario: DP Q 表同步支持反手开关
- **WHEN** `create_optimal_q_table` 计算 DP 专家 Q 表
- **AND** `allow_reverse_position=False`
- **THEN** 对 `future_position * current_position < 0` 的动作赋 `-max_punishment` 惩罚
- **WHEN** `allow_reverse_position=True`
- **THEN** 系统 SHALL 调用 `change_of_wallet()` 计算反手后的实际 reward
