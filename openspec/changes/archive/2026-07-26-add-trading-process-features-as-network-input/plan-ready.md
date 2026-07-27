# 实现计划：add-trading-process-features-as-network-input

## 来源
- 需求: docs/user-stories/0002-trading-process-features-as-network-input.md
- 规格: openspec/changes/archive/2026-07-26-add-trading-process-features-as-network-input/specs/fineft-trading-process-features/spec.md

## 实现步骤
1. 修改 `base_env.py`，在 `reset()` 和 `step()` 的 `info` 中暴露 `trading_info` 数组。
2. 修改 `replay_buffer_DQN.py`，在采样白名单常量中新增 `trading_info`。
3. 修改 `low_level.py`，为 `Qnet` / `ensemble_Qnet` 新增 `fc_trading` 层及 `trading_info` 输入支持。
4. 全量更新 Stage I/II/III 训练、测试、Ablation 及 Baselines 调用点。
5. 编写并运行单元与集成测试。
