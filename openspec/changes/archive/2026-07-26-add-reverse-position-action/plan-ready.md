# 实现计划：add-reverse-position-action

## 来源
- 需求: docs/user-stories/0001-reverse-position-action.md
- 规格: openspec/changes/archive/2026-07-26-add-reverse-position-action/specs/fineft-reverse-position-action/spec.md

## 实现步骤
1. 在 `futures_util.py` 中增加 `allow_reverse_position` 开关控制与反手平仓开仓逻辑。
2. 在 `base_env.py` 和 `demo_env.py` 中透传 `allow_reverse_position` 开关，并在 `step()` 中处理持仓反转时的收益率重置。
3. 在 `create_optimal_q_table` 和 `pretrain_qtable_diagnostics.py` 中同步反手开关支持。
4. 编写并运行回归测试 `test_futures_reverse_position.py` 验证功能。
