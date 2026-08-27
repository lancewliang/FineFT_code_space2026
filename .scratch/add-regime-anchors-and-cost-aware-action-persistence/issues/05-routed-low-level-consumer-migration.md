# 05: 迁移路由后的 Low-level 动作消费者

**What to build:** 让 High-level 路由后调用的 Low-level Agent 也使用共享动作选择接缝，确保训练、独立测试和路由推理未来能一致执行成本感知规则。

**Blocked by:** 04: 建立共享 Low-level 贪心动作选择接缝。

**Status:** ready-for-agent

- [ ] 路由后的 Low-level 动作消费者全部经由共享选择接缝。
- [ ] 迟滞关闭时，路由推理动作与迁移前输出保持一致。
- [ ] High-level 的 Agent 路由、Label 语义和动作空间不改变。
- [ ] focused tests 使用相同 Q、当前动作和可用动作，验证直接 Low-level、独立测试和路由调用结果一致。
- [ ] 旧模型和缺少新增配置的产物仍走关闭模式。

