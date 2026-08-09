# 08 — 生成分类诊断和六个聚合视图

**What to build:** 从完整 Window/Expanded 数据生成研究员可直接使用的分类诊断，以及 Kline、Strategy、Cross 三类 Scenario 和 triple 汇总，在保持 Initial-action 反事实语义的前提下支持跨 epoch 子 Agent 比较。

**Blocked by:** 07 — 贯通量价背离及全局分类优先级。

**Status:** ready-for-agent

- [ ] Classifier Diagnostics 使用固定英文长表 schema，支持 overall、Label、epoch、triple 和 Kline、Strategy、Cross 维度。
- [ ] Diagnostics 报告 window count、window ratio、total net PnL、p25/p50/p75、median range 和告警。
- [ ] 策略多选导致 Strategy/Cross window ratio 合计超过 1 时结果保持合法。
- [ ] 类别零命中、未分类率至少 30% 或 PnL 区分度弱只产生告警，不触发失败或自动调阈值。
- [ ] 三个 Scenario summary 使用固定 schema，并以 contract、df path 和 Initial-action 保留反事实情景身份。
- [ ] 三个 triple summary 使用固定 schema，只对实际命中目标形态的 Scenario 统计做算术平均。
- [ ] 未命中形态的 Scenario 不补零，不同 Initial-action 不直接相加，不跨策略形态计算账户总 PnL。
- [ ] observed count、expected count 和 Initial-action coverage ratio 与完整评估全集一致。
- [ ] 未分类哨兵保留在 Diagnostics，但不进入六个正式 summary。
- [ ] 测试证明展开、多选和情景聚合均不会放大账户盈亏。

