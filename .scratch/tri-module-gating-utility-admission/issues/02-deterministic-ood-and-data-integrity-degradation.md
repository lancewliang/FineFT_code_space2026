# 02: 接入确定性 OOD 与数据完整性降级闭环

**What to build:** 在 tri-module 栈中实现只负责输入可信度的 Module 1。从确定性双轴 VAE 评分和数据完整性检查开始，完整覆盖降级进入、持续、防御处置、迟滞恢复和审计，使相同输入始终产生相同安全结论。

**Blocked by:** 01 — 扩展双门控栈边界与公共审计契约.

**Status:** ready-for-human

- [ ] 硬 OOD 评分使用确定性 VAE latent mean，不经过随机重参数采样。
- [ ] 每个 VAE 轴使用冻结的 train-ID 经验似然分位数，任一轴最佳槽位低于 0.005 即进入降级。
- [ ] 数据跳空、缺失字段、非单调事件次序和关键执行输入不可用拥有独立于 VAE OOD 的 reason code。
- [ ] 降级进入后阻断开仓、同向加仓和反手，并取消会继续扩大风险的未完成意图或动作持续状态。
- [ ] 行情可执行时进行 best-effort 减仓或退出；不可执行时不得生成理想化成交或虚假 Trade Record。
- [ ] 只有双轴均达到 0.01 且输入连续干净 `max(VAE rolling window, H)` 个有效决策步后才恢复。
- [ ] 任一恢复观察步重新变脏时，连续恢复计数按协议重置。
- [ ] 审计契约分别记录 OOD/data-invalid 的 enter、stay、recover 状态、阈值证据和具体原因。
- [ ] 两个 valid 外层 Fold 共用冻结的 VAE 权重、train scaler 和 train-ID likelihood reference。
- [ ] VAE 训练完成流程不自动运行或读取 test OOD 分析；test 分析只能在最终冻结流程中显式触发。
- [ ] 测试覆盖确定性重复运行、单轴失败、双轴恢复、数据错误、不可执行退出和优先级高于普通路由的防御行为。
