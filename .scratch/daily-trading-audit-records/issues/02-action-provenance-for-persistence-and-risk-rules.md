# 02: 记录动作持续与规则风控来源

**What to build:** 让没有新低层策略推理的步骤仍能被准确审计。动作持续引用最初驱动该动作的 Prediction Record；warm-up、防御平仓、动作不可用、硬止损、冷静期和熔断以明确的 Decision Reason 进入 Trade Record，不伪造 Prediction Record。

**Blocked by:** 01: 建立最小审计级证据链.

**Status:** closed

- [x] 只有真实模型推理产生 Prediction Record，记录明确区分策略推理、路由推理和规则决策。
- [x] 动作持续步骤引用启动持续性的 Prediction ID，并保留自身的 Decision ID。
- [x] warm-up 和所有规则风控路径允许 Prediction ID 为空，同时保存明确原因。
- [x] 动作不可用导致重新推理或防御动作时，来源关系与实际控制路径一致。
- [x] 测试覆盖 warm-up、动作持续、防御平仓、硬止损、冷静期和熔断，不允许虚构 Action Value 或概率。
