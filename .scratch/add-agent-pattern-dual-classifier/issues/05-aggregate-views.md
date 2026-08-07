# 05 — 聚合视图脚本:distinct 盈亏汇总

**What to build:** 端到端行为:用户给出 04 产出的完整 `agent_pattern_detail_table.csv`,脚本读它 → 按两种 distinct 维度聚合 → 输出 `agent_pattern_pnl_summary.csv`,让用户看到"每个 triple 在各形态下的总盈亏、窗口数、盈亏分位数"。

两种聚合视图(同一脚本产出,或分两个 CSV 输出):
1. 按 `(label, epoch, bin_index, K 线形态)` distinct 聚合:该 triple 在每个 K 线形态下的总盈亏(所有命中该 K 线形态的窗口盈亏求和)、窗口数、盈亏分位数(中位数 / p25 / p75)。因为 K 线形态单选,distinct 不放大。
2. 按 `(label, epoch, bin_index, 策略形态)` distinct 聚合:该 triple 在每个策略形态下的总盈亏、窗口数、盈亏分位数。关键正确性——策略形态多选导致同一窗口盈亏出现在多行,distinct 聚合时按形态独立计算总盈亏(每个形态独立 sum/mean,不跨形态求和),避免放大。

输出列:`label, epoch, bin_index, 形态(标签), 形态类型(K线/策略), 总盈亏, 窗口数, 盈亏中位数, p25, p75`。

验证:总盈亏不放大(distinct 语义正确——多选策略形态下,同一窗口盈亏在 ST1 行和 SM3 行各计入一次,但聚合按形态 distinct,ST1 总盈亏只含命中 ST1 的窗口,SM3 总盈亏只含命中 SM3 的窗口,不混入对方)。

这是薄 groupby 逻辑,一个 smoke test 验证 distinct 语义即可。

**Blocked by:** 04 — 多 triple orchestrator(顺序执行:需要完整明细表作为输入)

**Status:** ready-for-agent

- [ ] 聚合脚本实现:读 `agent_pattern_detail_table.csv`
- [ ] 按 `(label, epoch, bin_index, K 线形态)` distinct 聚合:总盈亏、窗口数、盈亏分位数
- [ ] 按 `(label, epoch, bin_index, 策略形态)` distinct 聚合:总盈亏、窗口数、盈亏分位数
- [ ] 输出 `agent_pattern_pnl_summary.csv`(或分两个 CSV)
- [ ] smoke test:合成明细表(含多选策略形态行)→ 验证 distinct 聚合不放大
- [ ] distinct 语义验证:ST1 行和 SM3 行的同一窗口盈亏各计入各形态总盈亏,不混入
