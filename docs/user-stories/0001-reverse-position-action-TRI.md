---
status: draft
owner: FineFT
---

# Reverse Position Action — TRI

## Context

- User story: [0001-reverse-position-action.md](./0001-reverse-position-action.md)

### Key domain terms

| Term | Definition |
|------|------------|
| Reverse Position (反手) | 一步内先平仓再反向开仓；best-effort 语义；通过 `allow_reverse_position` 开关控制，默认关闭 |
| position_list | 离散仓位值列表（如 [-8, -6, -4, -2, 0, 2, 4, 6, 8]），position 必须始终在其中 |

_Full glossary: [LANGUAGE.md](../../LANGUAGE.md)_

### Components involved

| Component | Role in this feature |
|-----------|---------------------|
| Futures Trading Environment (`FineFT/env/`) | `futures_util.py` 提供 `change_of_wallet` 和 `calculate_avaiable_action` 的反手路由与可用性计算；`base_env.py` 提供开关传递和 `single_holding_return` 重置 |
| Demo_Env (`FineFT/env/env_class/demo_env.py`) | 透传开关给 `Base_Env` 和 `create_optimal_q_table` |
| Stage I Low-level Training (`FineFT/RL/DiHFT/low_level/`) | `pretrain_qtable_diagnostics.py` 和 `trace_udpate_index.py` 调用 `create_optimal_q_table_from_df`，需透传开关 |

_Full system map: [MAP.md](../../MAP.md)_

### Relevant decisions

| Decision | Summary |
|----------|---------|
| [0001-reverse-position-semantics.md](../decisions/0001-reverse-position-semantics.md) | best-effort 语义；深度不足截断到 position_list；`allow_reverse_position` 开关默认关闭；DP Q 表必须与环境一致 |

---

## What is being built

为 `Base_Env` 增加 Reverse Position 功能：当 `allow_reverse_position=True` 时，`change_of_wallet()` 在检测到仓位方向反转时执行先平仓后反向开仓，`calculate_avaiable_action()` 精确模拟两步保证金检查，`create_optimal_q_table()` 计算反手后的实际 reward 而非直接惩罚。

## Functional requirements

1. `change_of_wallet()` 增加 `allow_reverse_position=False` 和 `position_list=None` 参数；关闭时维持原 warning + 拒绝行为，打开时内联执行先平仓后反向开仓（跳过杠杆调整，开仓时用 target_leverage）；深度截断时重新调用开仓函数以获取正确的保证金和盈亏
2. `calculate_avaiable_action()` 增加 `allow_reverse_position=False` 参数；关闭时维持方向限制，打开时新增第四个分支 `available_position * position < 0` 精确模拟两步保证金检查
3. `create_optimal_q_table()` 和 `create_optimal_q_table_from_df()` 增加 `allow_reverse_position=False` 参数；关闭时维持 `-max_punishment`，打开时调用 `change_of_wallet()` 计算反手 reward
4. `Base_Env.__init__()` 增加 `allow_reverse_position=False` 参数，存储为实例属性，传递给 `change_of_wallet` 和 `calculate_avaiable_action`
5. `Demo_Env.__init__()` 增加 `allow_reverse_position=False` 参数，透传给 `super().__init__()` 和 `create_optimal_q_table()`
6. `Base_Env.step()` 中 `single_holding_return` 重置条件增加反手检测：`if self.position == 0 or (self.allow_reverse_position and self.position * position < 0)`
7. `pretrain_qtable_diagnostics.py`、`trace_udpate_index.py`、ablation `FineFT.py` 调用 `create_optimal_q_table_from_df` 时透传 `allow_reverse_position`

## Non-Functional Requirements

- **Performance**: 反手分支在 `calculate_avaiable_action()` 中增加一轮循环迭代（反方向仓位），对 O(position_choices × leverage_choices) 的复杂度影响可忽略
- **Backward Compatibility**: `allow_reverse_position=False`（默认）时行为与修改前完全一致，已有训练结果、Q 表和专家路径不受影响
- **Reliability**: 反手 best-effort 语义确保平仓步骤不会失败，反向开仓失败时 position 安全归零

---

## Design

### Architecture

修改仅涉及现有组件内部逻辑，不引入新组件。开关参数沿调用链透传：

```
Base_Env.__init__(allow_reverse_position=False)
  │
  ├─→ step()
  │     ├─→ change_of_wallet(..., allow_reverse_position)
  │     └─→ calculate_avaiable_action(..., allow_reverse_position)
  │
  └─→ single_holding_return reset condition

Demo_Env.__init__(allow_reverse_position=False)
  ├─→ super().__init__(allow_reverse_position)
  └─→ create_optimal_q_table(..., allow_reverse_position)

create_optimal_q_table(allow_reverse_position=False)
  └─→ change_of_wallet(..., allow_reverse_position)  [反手分支]

create_optimal_q_table_from_df(allow_reverse_position=False)
  └─→ create_optimal_q_table(..., allow_reverse_position)
```

### Data Model

`WalletChangeResult` 无变更。反手成功时汇总两步结果：

| 字段 | 来源 |
|------|------|
| `leverage` | `open_result.leverage` |
| `position` | `open_result.position` |
| `initial_margin` | `open_result.initial_margin` |
| `unrealized_pnl` | `open_result.unrealized_pnl` |
| `wallet_balance` | `open_result.wallet_balance` |
| `slippage_step` | `close_result.slippage_step + open_result.slippage_step` |
| `commission_fee_step` | `close_result.commission_fee_step + open_result.commission_fee_step` |
| `realized_pnl_step` | `close_result.realized_pnl_step`（开仓步骤 realized_pnl = 0） |

反手失败（只平仓）时直接返回 `close_result`。

### API Design

无新 API。所有变更为现有函数签名增加可选参数 `allow_reverse_position=False`，默认值确保向后兼容。

---

## Implementation Plan

### Phase 1: futures_util.py 核心逻辑

- [ ] `change_of_wallet()`: 增加 `allow_reverse_position=False` 和 `position_list=None` 参数；`current_position * previous_position < 0` 分支改为条件判断：关闭时维持原行为，打开时内联先平后开
- [ ] `change_of_wallet()` 反手分支：持多→平多→开空；持空→平空→开多；保证金不足时 position 归零；深度截断时重新调用开仓函数以获取正确的保证金和盈亏
- [ ] `calculate_avaiable_action()`: 增加 `allow_reverse_position=False` 参数；关闭时维持方向限制；打开时移除方向限制并新增第四分支精确模拟两步
- [ ] `create_optimal_q_table()`: 增加 `allow_reverse_position=False` 参数；修复 `change_of_wallet` 返回值 tuple 解包 bug（改为属性访问）；关闭时维持 `-max_punishment`，打开时调用 `change_of_wallet` 计算反手 reward
- [ ] `create_optimal_q_table_from_df()`: 增加 `allow_reverse_position=False` 参数，透传给 `create_optimal_q_table`

### Phase 2: base_env.py 和 demo_env.py

- [ ] `Base_Env.__init__()`: 增加 `allow_reverse_position=False` 参数，存储为实例属性
- [ ] `Base_Env.step()`: 调用 `change_of_wallet` 和 `calculate_avaiable_action` 时传入 `allow_reverse_position`
- [ ] `Base_Env.step()`: `single_holding_return` 重置条件增加 `or (self.allow_reverse_position and self.position * position < 0)`
- [ ] `Demo_Env.__init__()`: 增加 `allow_reverse_position=False` 参数，透传给 `super().__init__()` 和 `create_optimal_q_table()`

### Phase 3: DP 调用方透传

- [ ] `pretrain_qtable_diagnostics.py`: 调用 `create_optimal_q_table_from_df` 时透传 `allow_reverse_position`
- [ ] `trace_udpate_index.py`: 调用 `create_optimal_q_table_from_df` 时透传 `allow_reverse_position`
- [ ] ablation `FineFT.py`: 调用 `create_optimal_q_table_from_df` 时透传 `allow_reverse_position`

### Phase 4: 测试

- [ ] 单元测试：`change_of_wallet` 反手分支（开关开/关、保证金不足、深度不足）
- [ ] 单元测试：`calculate_avaiable_action` 反手分支（开关开/关、精确两步模拟）
- [ ] 单元测试：`create_optimal_q_table` 反手 reward 计算
- [ ] 集成测试：`Base_Env.step()` 反手后 `single_holding_return` 重置

---

## New Dependencies

无新依赖。

---

## Testing Strategy

### Unit Tests

- `test_change_of_wallet_reverse_disabled`: `allow_reverse_position=False` 时反手返回原状态 + warning
- `test_change_of_wallet_reverse_long_to_short`: 持多反手开空，两步成功
- `test_change_of_wallet_reverse_short_to_long`: 持空反手开多，两步成功
- `test_change_of_wallet_reverse_insufficient_margin`: 反手开仓保证金不足，position 归零
- `test_change_of_wallet_reverse_insufficient_depth`: 反手开仓深度不足，截断到 position_list 最大可行值
- `test_change_of_wallet_reverse_slippage_commission`: 两步滑点和手续费累计正确
- `test_calculate_available_action_reverse_disabled`: 开关关闭时不包含反方向仓位
- `test_calculate_available_action_reverse_enabled`: 开关打开时包含反方向仓位（保证金充足）
- `test_calculate_available_action_reverse_insufficient_margin`: 开关打开但保证金不足时不包含反方向仓位
- `test_create_optimal_q_table_reverse_disabled`: 开关关闭时反手动作赋 `-max_punishment`
- `test_create_optimal_q_table_reverse_enabled`: 开关打开时反手动作计算实际 reward

### Integration Tests

- `test_base_env_step_reverse_position`: `Base_Env.step()` 反手后 position、wallet_balance、single_holding_return 正确
- `test_base_env_step_reverse_position_reset_holding_return`: 反手时 `single_holding_return` 重置

### End-to-End Tests

- `test_demo_env_reverse_position_training`: `Demo_Env` 开关打开时 Q 表包含反手路径，pretrain warmup 可正常完成

---

## Security Considerations

无新增安全考量。

## Performance Considerations

- `calculate_avaiable_action()` 反手分支增加一轮循环，复杂度从 O(position_choices × leverage_choices) 不变（只是 position_choices 的可用范围扩大），对性能影响可忽略
- `create_optimal_q_table()` 反手分支调用 `change_of_wallet()` 计算实际 reward，替代原来的 `-max_punishment` 赋值，单次调用开销增加但仍在 O(1) 级别

---

## Known risks

- `calculate_avaiable_action()` 不处理 orderbook 深度截断对可用动作的影响（与现有开仓分支行为一致），可能导致 agent 看到一些深度不足但保证金充足的反手动作，执行时会被截断
- `pretrain_qtable_diagnostics.py`、`trace_udpate_index.py`、ablation `FineFT.py` 尚未透传 `allow_reverse_position` 参数（Phase 3 未实现）

## Open Questions

无。

## References

- [0001-reverse-position-semantics.md](../decisions/0001-reverse-position-semantics.md)
- [0001-reverse-position-action.md](./0001-reverse-position-action.md)
