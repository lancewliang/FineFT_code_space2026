# 实现计划：add-risk-liquidity-state-features

## 来源
- 需求: docs/user-stories/0004-risk-liquidity-state-features.md
- 规格: openspec/changes/archive/2026-07-26-add-risk-liquidity-state-features/specs/commodity-futures-support/spec.md

## 实现步骤
1. 在 `downscale.py` 中导出 `open_interest` 基础列并补充校验。
2. 在 `time_operator` 中扩展风险特征（6 个）与流动性特征（4 个）计算函数。
3. 日化系数由 `CommodityConfig.trading_sessions` 与 `target_freq` 动态推导。
4. 编写并运行回归测试。
