# 交易 state 输入实施依赖总览

日期：2026-11-01
状态：已实施；01–08 及 06 的代码、定向验证和收尾文档已完成。正式历史数据重训未启动。

## 本次拆分

保留原 01–06 文件路径，避免破坏已有引用；新增 07、08。编号是稳定票号，执行顺序由依赖决定。

- 原 02 中的辅助数据跨流水线传递拆到 07；02 专注分区行情预计算与 scale-save 产物。
- 原 02 中的环境尺度参数/有效起点，及原 03 中的退出生命周期拆到 08。
- 03 聚焦模型、CPU replay、串行 learner 及产物契约接入。
- 01 提供共享观测与辅助数据契约；08 与 03 负责将全局运行宽度切换为十一维。
- 04、05 在 03 之后可分别实施，06 最终集成验收。

## 依赖与交付物

| 票号 | 任务 | 依赖 | 可验收交付物 |
|---|---|---|---|
| [01](issues/01-centralize-trading-observation-contract.md) | 集中观测与行情辅助数据契约 | 无 | 共享契约与四维行为回归 |
| [02](issues/02-configurable-causal-price-scale.md) | 在分区行情中预计算因果价格统计量 | 01 | 分区行情统计值/有效位及 scale-save 产物 |
| [07](issues/07-preserve-causal-price-auxiliary-data.md) | 贯通辅助列与切片坐标 | 02 | 辅助列贯通、切片与老师/compact 坐标一致 |
| [08](issues/08-environment-trading-memory-lifecycle.md) | 实现环境十一维观测与退出生命周期 | 01, 07 | 真实环境十一维观测及退出生命周期 |
| [03](issues/03-serial-eleven-dimensional-training.md) | 接通模型与串行训练闭环 | 01, 08 | 模型/CPU replay/串行真实更新及产物校验 |
| [04](issues/04-parallel-compact-replay-training.md) | 接通并行训练与紧凑经验传输 | 03 | 并行与 compact/CPU/GPU 闭环 |
| [05](issues/05-evaluation-and-expert-routing.md) | 接通独立评估与高层路由 | 03 | 独立评估与路由真实前向 |
| [06](issues/06-integrate-and-verify-training-state.md) | 完成数据到训练评估全链路验收 | 04, 05 | 数据到训练评估完整证据、ADR 和运行说明 |

执行顺序：`01 → 02 → 07 → 08 → 03 → (04、05) → 06`。

## 集成约定

08 与 03 是相邻的同一集成变更：环境输出宽度、模型输入与 CPU replay 一起完成切换，不把 08 单独宣称为可发布的训练版本，不引入长期四维兼容开关。08 用环境测试验证事件规则，03 用真实 learner 更新验证消费链路。04/05 尚未迁移时须记录受影响入口，06 完成后才宣称全入口可用。

ready-for-agent 表示设计具备实施条件；本轮依赖已按 `01 → 02 → 07 → 08 → 03 → (04、05) → 06` 完成。各票状态和证据记录在对应 issue 与 [verification.md](verification.md) 中；正式重训仍不属于本轮交付。
