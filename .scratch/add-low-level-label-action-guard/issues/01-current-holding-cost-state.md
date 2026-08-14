# 01 — 暴露当前持仓真实成本状态

**What to build:** 让 Base Futures Trading Environment 从真实订单簿开仓成交中维护当前持仓开仓价和当前持仓均价，使外部 validation 守卫可以在不访问环境私有结算状态的情况下获取可审计成本基准。

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] 多头与空头有效开仓价均使用实际成交额、实际开仓数量和已发生开仓税费，且订单簿成交价已反映滑点。
- [ ] 同向加仓保留当前持仓开仓价并按实际成交数量更新持仓均价；部分减仓不改变两个价格。
- [ ] 完全平仓后两个成本价均为 `0.0`；Reverse Position 成功时仅用新方向开仓腿重置，只平未开时保持零值。
- [ ] Initial-action 情景的非零初始仓位使用首行 mark price，不虚构开仓税费或滑点。
- [ ] 环境属性、reset info 和每个 step info 均暴露两个成本价，空仓口径统一为 `0.0`。
- [ ] 现有四维 Trading Process Feature、checkpoint 结构、Reverse Position best-effort 和交易结算结果六值兼容契约保持不变。
- [ ] 高层环境测试覆盖空仓、Initial-action、多空开仓、加仓、减仓、平仓、反手和部分成交，并与现有商品环境/反手/交易过程特征回归一起通过。
