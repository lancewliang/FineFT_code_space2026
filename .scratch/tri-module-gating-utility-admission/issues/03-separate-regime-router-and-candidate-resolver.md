# 03: 拆分 Regime Router 与 Candidate Resolver

**What to build:** 让健康输入在 Module 2 中只完成二维 regime 槽位选择，再由独立 Candidate Resolver 将槽位映射为候选策略和候选动作。整个路径必须保留连续证据，但不再把模式清晰度或候选缺失变成隐式交易价值判断。

**Blocked by:** 02 — 接入确定性 OOD 与数据完整性降级闭环.

**Status:** ready-for-human

- [ ] Module 2 对健康输入输出唯一的 `(volatility_slot, slope_slot)` 和结构化路由证据。
- [ ] 路由证据包含可用的每轴似然、最大概率与首二差值，但这些值不在 Router 内触发交易硬拒绝。
- [ ] 模式间距很小但仍属分布内的输入继续路由到最佳二维槽位。
- [ ] Candidate Resolver 是独立可观察步骤，将有效槽位映射为候选策略及其实际候选动作。
- [ ] 没有合格模型的槽位显式返回 `NullCandidate`，不伪造 Flat 策略或 Utility Gate 拒绝。
- [ ] Router 不输出准入批准，Candidate Resolver 也不拥有经济效用准入权。
- [ ] VAE 和路由连续证据可以传给后续 Utility Model，但不产生未声明 veto。
- [ ] 路由槽位、连续证据、candidate identity 和 `NullCandidate` 原因进入公共审计契约。
- [ ] 测试覆盖明确模式、模糊但 in-distribution 模式、空槽位、候选可用性变化和 Module 1 已降级时跳过普通路由。
