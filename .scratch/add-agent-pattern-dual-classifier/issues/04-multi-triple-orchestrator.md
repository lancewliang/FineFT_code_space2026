# 04 — 多 triple orchestrator:从完整 Detail CSV 生成全明细表

**What to build:** 把 01 的单 triple orchestrator 扩展到遍历完整 Detail CSV 的所有 `(label, epoch, bin_index)` triple × 所有 segment 文件,输出完整 `agent_pattern_detail_table.csv`。

端到端行为:用户给出一个 `trading_action_detail_epoch_*.csv` 路径,脚本读它 → 按 `(标签, 分箱索引, 数据文件)` 分组得到所有 trajectory → 对每个 trajectory:label ∈ {0,6} 直接标 KX1 产出一行 / label ∈ {1..5} 按 N=20 不重叠切窗,每窗口跑两个完整分类器(02/03 已实现) → 产出明细表行(K线形态单选 1 个,策略形态多选 N 个 → 该窗口产出 1×N 行) → 写出 CSV。

关键正确性:
- 盈亏归因:每行盈亏 = 窗口内 `(cumulative_realized_pnl[end]-[start]) + (unrealized_pnl[end]-[start])`。
- K 线形态单选 → 一个窗口的盈亏只归到其唯一 K 线形态行,不重复计入。
- 策略形态多选 → 一个窗口的盈亏重复计入各命中策略形态行(同一窗口盈亏出现在多行)。
- label_0/6 短路:不进窗口识别,直接标 KX1,盈亏仍按 trajectory 整体算(或按窗口?需在实现时确认——label_0/6 trajectory 极短 median=3,整体算即可)。
- 明细表行数符合预期(label_1~5 窗口数 × 命中策略形态数 + label_0/6 trajectory 数)。

输入数据:真实 Detail CSV 位于 `analysis_result/DiHFT/low_level/<dataset>/<experiment>/trading_action_detail_epoch_*.csv`(n≈366k 行/triple 级别)。输出 `agent_pattern_detail_table.csv` 同目录。

**Blocked by:** 03 — 完整 策略形态分类器(顺序执行:需要两个分类器都完整才能 scale out)

**Status:** ready-for-agent

- [ ] orchestrator 扩展:遍历完整 Detail CSV 的所有 `(label, epoch, bin_index)` triple × 所有 segment
- [ ] trajectory 分组:按 `(标签, 分箱索引, 数据文件)` 分组,`initial_action` 进分组键但不进分类输出
- [ ] label_0/6 短路:直接标 KX1,不切窗
- [ ] label_1~5 切 N=20 不重叠窗口,每窗口调两个完整分类器
- [ ] K 线形态单选 + 策略形态多选的行展开逻辑(1 个窗口 → 1×N 行)
- [ ] 盈亏归因正确(已实现+浮动 PnL 变化)
- [ ] 输出 `agent_pattern_detail_table.csv`,列含 `label, epoch, bin_index, K线形态, 策略形态, 盈亏` + 窗口起止 step
- [ ] smoke test:真实 Detail CSV 样本上跑通,行数符合预期、盈亏列非空
- [ ] 语义验证 smoke:K 线形态单选不重复(同窗口只 1 个 K 线形态)、策略形态多选重复(同窗口盈亏出现在多行)
