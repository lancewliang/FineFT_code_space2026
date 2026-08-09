# 05 — Initial-action Scenario/triple 聚合视图

**What to build:** 从唯一窗口事实生成 K 线、策略和 7×6 交叉的 Scenario 级与 triple 级汇总，在保留 Initial-action 反事实语义的同时防止多选形态放大账户 PnL。

**Blocked by:** 04 — 全候选窗口产物与完整性契约

**Status:** ready-for-agent

- [ ] 固定生成 K 线、策略和交叉三类视图的 Scenario 级与 triple 级 summary，共六个文件。
- [ ] Scenario 级按 Initial-action 输出 `total_net_pnl`, `window_count`, `pnl_p25`, `pnl_p50`, `pnl_p75`，net PnL 为默认绩效口径。
- [ ] triple 级每个形态组只对至少有一个窗口命中该形态的 Initial-action Scenario 做算术平均，输出 `mean_initial_action_*`。
- [ ] triple 级同时输出已命中情景数、期望情景数和 Initial-action 覆盖率。
- [ ] 未命中某形态的 Initial-action Scenario 不被伪造为零 PnL；期望 Initial-action Detail 行为轨迹本身缺失时立即失败。
- [ ] 账户总 PnL 只能在单个 Initial-action Scenario 中按窗口表的唯一 `window_id` 汇总，不从展开表跨策略形态求和。
- [ ] 未分类和策略未分类哨兵不进入任何正式 K 线、策略或交叉 summary，但仍保留在明细和展开表。
- [ ] smoke test 覆盖单窗多策略、数组展开、PnL 不放大、Initial-action 不相加、命中情景等权平均、覆盖率、不伪造零 PnL 和行为轨迹缺失失败。
