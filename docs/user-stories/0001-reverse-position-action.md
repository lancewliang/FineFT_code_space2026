---
status: accepted
source: user-request
---

# Reverse Position Action

## User Story

As a RL agent
I want to reverse my position in a single step
So that I can quickly respond to market reversal signals without wasting two steps to close then re-open

## Background

当前 `change_of_wallet()` 在 `current_position * previous_position < 0` 时直接拒绝反手操作，返回原始状态不变。`calculate_avaiable_action()` 也通过 `position_lower = max(0, position_lower)` 和 `position_upper = min(0, position_upper)` 限制可用仓位只能在当前方向变动。

这意味着 agent 想从多头翻到空头（或反之），需要两步：先平仓到 0，再反向开仓。在快速行情中，两步延迟可能导致错过最佳入场价格。

## Decisions

| # | 问题 | 决策 | 理由 |
|---|------|------|------|
| 1 | 反手语义 | **Best-effort**：平仓一定成功，反向开仓可能失败，position 归零 | 真实交易中先平后开是两笔独立订单，不存在原子保证 |
| 2 | 杠杆调整时序 | **跳过杠杆调整**，直接平仓，开仓时用 target_leverage | 平仓不需要杠杆变更，反手时杠杆只在开反向仓时才有意义 |
| 3 | 可用动作检查 | **精确模拟两步**（先平后开）的保证金检查 | 与实际执行逻辑一致，agent 看到的可用动作和执行结果完全匹配 |
| 4 | 深度不足处理 | **截断到 position_list 中最大可行值**，保持离散不变量 | 避免 position 不在 position_list 中导致下游 assert 失败和动作空间不匹配 |
| 5 | 术语 | **Reverse Position** | 与现有动词短语命名风格一致（change_of_wallet, open_long_position） |
| 6 | 实现结构 | **内联在 `change_of_wallet()` 中** | 与现有 close+change_of_leverage 内联风格一致，反手逻辑不会被复用 |
| 7 | 可用动作分支 | **新增第四个分支**处理反手 | 反手的保证金检查逻辑与现有三个分支都不同，强行塞入会导致逻辑混乱 |
| 8 | 开仓函数修改 | **无需修改**，从 position=0 开仓兼容现有 assert | `open_long_position` assert `previous_position >= 0`，`open_short_position` assert `previous_position <= 0`，0 均通过 |
| 9 | single_holding_return | **反手时重置**，需在 `base_env.py` 的 `step()` 中检测 | 反手 = 结束旧持仓 + 开始新持仓，收益应分开统计 |
| 10 | 功能开关 | **默认关闭**，通过 `allow_reverse_position` 参数控制；关闭时行为与原来一致（warning + 拒绝） | 向后兼容，不影响已有训练结果和 Q 表；开关打开时才启用反手功能 |
| 11 | DP Q 表影响 | **`create_optimal_q_table` 需同步支持开关**；关闭时维持 `-max_punishment` 惩罚反手，打开时改为计算反手后的实际 reward | DP Q 表的动作约束必须与环境一致，否则 pretrain warmup 会产生不匹配的专家路径 |

## Scenarios

### 开关关闭时反手被拒绝（默认行为）

Given:
- `allow_reverse_position=False`（默认）
- 当前持仓为多头仓位（position > 0）
- agent 选择的目标仓位为空头仓位（target_position < 0）

When:
- 执行 `change_of_wallet()`

Then:
- 打印 warning："You can not turn over the position in just one step..."
- 返回原始状态不变（与当前行为完全一致）
- `calculate_avaiable_action()` 不包含反方向仓位选项
- `create_optimal_q_table` 对反手动作赋 `-max_punishment`

### 开关打开时持空单反手开多

Given:
- `allow_reverse_position=True`
- 当前持仓为空头仓位（position < 0）
- agent 选择的目标仓位为多头仓位（target_position > 0）

When:
- 执行 Reverse Position

Then:
- 先平空仓：调用 `close_short_position()` 结算已实现盈亏、手续费和滑点
- 再开多仓：用平仓后的 wallet_balance 检查保证金是否充足，充足则调用 `open_long_position()` 开多
- 保证金不足时：只完成平仓，不开反向仓（position 归零）
- 深度不足时：截断到 position_list 中最大可行多头仓位
- 返回的 `WalletChangeResult` 包含两步的累计滑点和手续费

### 开关打开时持多单反手开空

Given:
- `allow_reverse_position=True`
- 当前持仓为多头仓位（position > 0）
- agent 选择的目标仓位为空头仓位（target_position < 0）

When:
- 执行 Reverse Position

Then:
- 先平多仓：调用 `close_long_position()` 结算已实现盈亏、手续费和滑点
- 再开空仓：用平仓后的 wallet_balance 检查保证金是否充足，充足则调用 `open_short_position()` 开空
- 保证金不足时：只完成平仓，不开反向仓（position 归零）
- 深度不足时：截断到 position_list 中最大可行空头仓位
- 返回的 `WalletChangeResult` 包含两步的累计滑点和手续费

### 反手时保证金不足只能平仓

Given:
- `allow_reverse_position=True`
- 当前持仓为多头仓位（position > 0）
- wallet_balance 较低，平仓后不足以开反向空头仓位的初始保证金 + 开仓损失

When:
- 执行 Reverse Position

Then:
- 平多仓成功，已实现盈亏结算
- 开空仓因保证金不足被拒绝
- 最终 position = 0，leverage 保持平仓后的值
- `WalletChangeResult` 只包含平仓步骤的滑点和手续费

### 反手时 orderbook 深度不足

Given:
- `allow_reverse_position=True`
- 当前持仓为空头仓位（position < 0）
- ask 档位总量不足以支撑目标多头仓位

When:
- 执行 Reverse Position

Then:
- 平空仓成功
- 开多仓受 orderbook 深度限制，截断到 position_list 中最大可行多头仓位
- 最终 position 为截断后的多头仓位

### 空仓时不能反手

Given:
- 当前持仓为 0（position = 0）

When:
- agent 选择任意目标仓位

Then:
- 不触发反手逻辑，按现有开仓逻辑处理（直接开多或开空）

### 可用动作计算包含反手选项

Given:
- `allow_reverse_position=True`
- 当前持仓为多头仓位（position > 0）
- wallet_balance 和 orderbook 深度允许反手到空头仓位

When:
- 调用 `calculate_avaiable_action()`

Then:
- 返回的可用仓位列表包含空头仓位选项
- 反手开仓的保证金检查精确模拟两步：先计算平仓后的 wallet_balance_new，再检查是否足以覆盖反向开仓的初始保证金 + 开仓损失

### 反手后强平检查

Given:
- `allow_reverse_position=True`
- Reverse Position 执行完成（先平仓再反向开仓）

When:
- 执行强平判断 `judge_liquidation()`

Then:
- 使用反手后的 position、unrealized_pnl、wallet_balance 进行强平检查
- 如果反手后保证金余额不足以维持新仓位，触发强平

### 反手时重置单次持仓收益

Given:
- `allow_reverse_position=True`
- 当前持仓为多头仓位（position > 0）
- 执行 Reverse Position 后持仓变为空头仓位

When:
- `step()` 检测到 `self.position * new_position < 0`

Then:
- 重置 `single_holding_return = 0` 和 `single_holding_history = [0]`
- 与 position 归零时的重置逻辑一致

### DP Q 表开关打开时计算反手 reward

Given:
- `allow_reverse_position=True`
- `create_optimal_q_table` 计算 Q 表

When:
- `future_position * current_position < 0`

Then:
- 不再赋 `-max_punishment`，改为调用 `change_of_wallet()` 计算反手后的实际 reward
- 反手失败（position 归零）时赋 `-max_punishment`（与 `changed_position != future_position` 的处理一致）

## Implementation Notes

### 功能开关传递链

`allow_reverse_position` 参数需要从环境构造函数一路传递到所有使用点：

```
Base_Env.__init__(allow_reverse_position=False)
  → self.allow_reverse_position = allow_reverse_position
  → change_of_wallet(..., allow_reverse_position=self.allow_reverse_position)
  → calculate_avaiable_action(..., allow_reverse_position=self.allow_reverse_position)

Demo_Env.__init__(allow_reverse_position=False)
  → super().__init__(allow_reverse_position=allow_reverse_position)
  → create_optimal_q_table(..., allow_reverse_position=allow_reverse_position)

Commodity_Env / Agg_Env
  → 继承 Base_Env，自动获得开关

create_optimal_q_table(allow_reverse_position=False)
  → 当 allow_reverse_position=False 时，维持 -max_punishment 惩罚
  → 当 allow_reverse_position=True 时，调用 change_of_wallet 计算反手 reward

create_optimal_q_table_from_df(allow_reverse_position=False)
  → 透传给 create_optimal_q_table
```

### 涉及文件

1. **`FineFT/env/env_class/futures_util.py`**:
   - `change_of_wallet()`: 增加 `allow_reverse_position=False` 和 `position_list=None` 参数；关闭时维持原 warning + 拒绝行为，打开时内联执行先平后开（跳过杠杆调整）；深度截断时重新调用开仓函数
   - `calculate_avaiable_action()`: 增加 `allow_reverse_position=False` 参数；关闭时维持 `position_lower = max(0, ...)` 和 `position_upper = min(0, ...)` 限制，打开时新增第四个分支精确模拟两步
   - `create_optimal_q_table()`: 增加 `allow_reverse_position=False` 参数；关闭时维持 `-max_punishment`，打开时改为计算反手 reward
   - `create_optimal_q_table_from_df()`: 增加 `allow_reverse_position=False` 参数，透传给 `create_optimal_q_table`

2. **`FineFT/env/env_class/base_env.py`**:
   - `__init__()`: 增加 `allow_reverse_position=False` 参数，存储为实例属性
   - `step()`: 调用 `change_of_wallet` 和 `calculate_avaiable_action` 时传入 `allow_reverse_position`
   - `single_holding_return` 重置条件增加反手检测：`if self.position == 0 or (self.allow_reverse_position and self.position * position < 0)`

3. **`FineFT/env/env_class/demo_env.py`**:
   - `__init__()`: 增加 `allow_reverse_position=False` 参数，透传给 `super().__init__()` 和 `create_optimal_q_table()`

4. **`FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py`**:
   - 调用 `create_optimal_q_table_from_df` 时透传 `allow_reverse_position`

5. **`FineFT/RL/DiHFT/low_level/trace_udpate_index.py`**:
   - 调用 `create_optimal_q_table_from_df` 时透传 `allow_reverse_position`

6. **`FineFT/RL/DiHFT/ablation/converge_steps_sun/FineFT.py`**:
   - 调用 `create_optimal_q_table_from_df` 时透传 `allow_reverse_position`

### 反手执行顺序

反手动作的执行顺序为：
1. 平掉当前仓位（`close_long_position` 或 `close_short_position`）— 结算已实现盈亏
2. 用平仓后的 wallet_balance 检查保证金
3. 保证金充足则反向开仓（`open_long_position` 或 `open_short_position`），使用 target_leverage
4. 深度不足时截断到 position_list 中最大可行值
5. 保证金不足则 position 归零

### WalletChangeResult 汇总

`slippage_step` = 平仓滑点 + 开仓滑点
`commission_fee_step` = 平仓手续费 + 开仓手续费
`realized_pnl_step` = 平仓步骤的已实现盈亏（开仓步骤 realized_pnl = 0）
