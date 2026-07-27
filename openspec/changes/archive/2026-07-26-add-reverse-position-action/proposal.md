# add-reverse-position-action

## 背景与目标
在期货交易中，当市场发生趋势反转时，Agent 需要快速调整仓位。原先环境需要两步（先平仓到 0，再反向开仓），增加了交易延迟和额外的滑点风险。
本变更在 `FineFT/env/` 中引入一步反手平仓开仓 (Reverse Position Action) 功能，由 `allow_reverse_position` 开关控制，默认关闭以保持向后兼容。

## 关键决策
- **Best-effort 语义**：平仓一定成功，反向开仓可能受保证金或深度限制而失败（归零）。
- **跳过杠杆调整**：平仓过程不调整杠杆，只在开反向仓时使用 `target_leverage`。
- **精确模拟两步保证金**：可用动作计算先模拟平仓后的 wallet_balance，再检查开仓保证金。
- **离散不变量截断**：深度不足时截断到 `position_list` 中的有效值。
- **单次持仓收益重置**：反手表示结束旧持仓并开始新持仓，重置 `single_holding_return`。
- **DP Q 表同步**：`create_optimal_q_table` 增加开关支持，用于 pretrain 专家路径生成。

## 验收标准
- 开关关闭时，反手行为触发 warning 并拒绝，与旧代码完全兼容。
- 开关打开时，支持持多单反手开空、持空单反手开多。
- 保证金或深度不足时优雅截断或归零，不抛出异常或违反 `position_list` 不变量。
- 可用动作列表和 DP Q 表生成正确支持反手策略。
