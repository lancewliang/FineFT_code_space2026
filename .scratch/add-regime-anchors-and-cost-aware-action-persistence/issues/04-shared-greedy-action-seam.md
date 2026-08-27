# 04: 建立共享 Low-level 贪心动作选择接缝

**What to build:** 将 Stage I 训练贪心分支和独立 Low-level 测试统一到一个确定性的动作选择接缝，为后续加入成本迟滞提供单一入口，同时在迟滞关闭时保持现有 argmax 动作。

**Blocked by:** 03: 按跨合约稳定性条件保留市场状态锚点。

**Status:** ready-for-agent

- [ ] 共享规则接收 Q 值、当前动作和可用动作，并在关闭模式下返回与现有 argmax 相同的动作。
- [ ] 可用动作掩码、并列动作的确定性顺序和当前动作编号语义保持不变。
- [ ] Stage I 训练贪心分支与独立 Low-level 测试均使用该接缝。
- [ ] epsilon 随机探索、固定风格 rollout 和 DP expert path 保持原行为。
- [ ] focused tests 验证关闭模式、掩码、并列 Q 值和训练/测试动作一致。
