# 01: 集中观测与行情辅助数据契约

**What to build:** 集中交易观测和行情辅助数据的字段、类型、版本与校验入口，为各生产者和消费者提供同一份轻量契约；本票保持现有四维运行行为。

**Blocked by:** None (can start immediately).

**Status:** resolved

**Implementation:** 未实施。此前代码改动已全部回退；本轮只重切任务，不启动代码修改。ready-for-agent 表示规格可实施，不表示依赖已完成。

## 文件范围

轻量契约模块；现有环境、模型、CPU/GPU replay 引用常量和校验的入口。

## 验收标准

- [ ] 集中当前四维有序字段与维度来源，保持现有运行宽度；声明目标十一维字段与 schema，但不在本票提前切换全局运行宽度。
- [ ] 固定行情辅助列的正式名称、数值类型和布尔有效位，记录窗口 18、来源 mark_price、原价格单位、同合约同分区及行坐标契约；生产和消费端引用同一来源。
- [ ] 提供观测宽度、有限值和辅助数据形状/有效性校验；辅助值有限非负，无效预热行严格为 0/false，有效零波动为 0/true。禁止模型通过环境重型模块导入常量。
- [ ] 定义新模型/replay 元数据的统一构造与校验接口：字段顺序、schema、归一化参数、exit_price_scale_floor 的实际值和来源，以及行情统计语义；后续票实际接入保存与加载。
- [ ] 保持现有四维计算、动作映射、独立输入、n-step 与去重保留语义；四维环境到模型/replay 回归通过，目标十一维纯契约校验有测试。

## 职责边界

不实现滚动统计、数据写出、退出事件或十一维运行切换。目标 schema 以本票正式契约为准，不从历史票中的示例名称推定已有实现。

共同遵循 [spec](../spec.md) 第 4–6 节与 [实施依赖总览](../implementation-plan.md)。Python/pytest 使用 `conda activate finetf`。不更改已确认范围，不启动正式重训。

## Comments

- 2026-11-01：完成共享 `trading_info.v2` 契约，集中定义 11 维有序字段、schema、有限值/宽度校验、退出记忆编码、18 步因果尺度计算、元数据 sidecar 和 `fc_trading` 输入宽度校验。运行环境仍保持原四维入口，十一维切换留给 08/03。
- 验证：在 `conda activate finetf` 环境执行 `python -m pytest tests/common/test_trading_observation.py -q`，结果 `5 passed`。

- 2026-11-01：按确认后的行情统计预计算、Q8 A 与 exit_price_scale_floor 命名重新切分职责；当前验收项全部未完成。

- 2026-11-01（回退后状态校正）：用户确认本功能所有代码改动已回退；重新打开本票，全部验收项恢复未完成。下方记录只描述回退前工作，不能证明当前源码已实现或验证通过。

### 回退前历史记录（不作为当前验收证据）

- 2026-11-01：完成 `common.trading_observation` 共享契约、float32/finite/width 校验、schema 元数据和 checkpoint encoder-width 校验；环境、模型与 replay 已接入。最终集成目标要求 11 维，运行时扩维在后续依赖票的同一工作区提前贯通，未保留静默四维兼容入口。
- 验证：`python -m pytest tests/common/test_trading_observation.py tests/model/test_low_level_trading_observation.py tests/rl/test_trading_info_replay_contract.py -q`（conda 环境 `finetf`）通过。
