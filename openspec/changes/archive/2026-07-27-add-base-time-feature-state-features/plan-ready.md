# 实现计划：add-base-time-feature-state-features

## 来源
- 需求: docs/user-stories/0003-time-context-state-features.md
- 规格: openspec/changes/archive/2026-07-27-add-base-time-feature-state-features/specs/commodity-futures-support/spec.md

## 实现步骤
1. 新增 `BASE_TIME_FEATURE` 生成器，按 Trading Session 和合约生命周期计算 9 个非绝对特征。
2. 在 daily merge 中按 timestamp 关联 `BASE_TIME_FEATURE` 到 `FUTURE_FEATURE`。
3. 修改 Feature Selection 增加 `--mandatory_state_features` 保护机制。
4. 修改 Scale Save 增加 `--passthrough_features` 跳过缩放机制。
5. 编写并运行回归测试。
