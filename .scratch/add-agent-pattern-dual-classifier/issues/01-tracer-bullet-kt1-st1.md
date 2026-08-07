# 01 — Tracer bullet: KT1+ST1 单 triple 端到端

**What to build:** 给定单个 `(label, epoch, bin_index)` agent triple 在单个 segment 上的 Detail CSV 行（已含 `mark_price` / `volume` / `position_after` / `cumulative_realized_pnl` / `unrealized_pnl` / `标签` / `分箱索引` / `初始动作` / `数据文件` 列），产出一行 Agent 形态明细表：`(label, epoch, bin_index, K 线形态 ∈ {KT1, 未分类}, 策略形态 ∈ {ST1, 空}, 盈亏)`。

这是 tracer bullet：穿透所有层，验证架构走通。具体包含：
- 新建 `classify_agent` 模块包骨架。
- N=20 步不重叠窗口切分（步长 = N）。
- label_0 / label_6 短路标 KX1（不在其上跑窗口识别）。
- K 线形态分类器纯函数：实现 KT1 突破即时型判别 + "未分类"兜底，命中即止（KT1 → 未分类）。
- 策略形态分类器纯函数：实现 ST1 突破即时型判别（`max(|Δpos|)/max_hold≥0.8` 且阶跃后维持 ≥10 步），命中输出 ST1、未命中输出空集合。
- 多选机制骨架：策略形态返回标签集合（此 ticket 只填 ST1，后续 ticket 补其他类）。
- 盈亏归因：每行盈亏 = 窗口内 `(cumulative_realized_pnl[end]-[start]) + (unrealized_pnl[end]-[start])`。
- 明细表 schema 落地：列含 `label, epoch, bin_index, K线形态, 策略形态, 盈亏`（可加窗口起止 step 便于追溯）。
- 明细表生成 orchestrator 骨架（此 ticket 只处理单 triple 单 segment，后续 ticket 扩展到全量）。

测试 seam = 纯函数 seam（已在 spec 中与用户确认），prior art 为 [test_pick_agent.py](../../../FineFT/tests/analysis/test_pick_agent.py) 的合成输入 + 期望输出风格。

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] `classify_agent` 模块包骨架建立（含 `__init__.py`）
- [ ] K 线形态分类器纯函数实现 KT1 + 未分类，命中顺序 `KT1 → 未分类`
- [ ] 策略形态分类器纯函数实现 ST1，返回标签集合（多选机制骨架）
- [ ] N=20 不重叠窗口切分函数（输入 trajectory 序列 → 窗口列表）
- [ ] label_0/6 短路标 KX1（不进窗口识别）
- [ ] 盈亏归因函数（窗口内已实现+浮动 PnL 变化）
- [ ] 单 triple orchestrator：读 Detail CSV 一个 triple 的行 → 切窗 → 调两个分类器 → 写一行明细表
- [ ] 明细表 schema 落地（CSV 列定义）
- [ ] `test_kline_pattern_classifier.py` 覆盖 KT1 合成输入 + 未分类（随机游走）+ 阈值边界
- [ ] `test_strategy_pattern_classifier.py` 覆盖 ST1 合成输入（完美阶跃）+ 未命中（非阶跃）+ 阈值边界
- [ ] orchestrator smoke test：单 triple 单 segment 端到端跑通，输出表行 schema 正确、盈亏列非空
- [ ] 阈值参数化（fixture 可注入，便于标定后只改 fixture）
