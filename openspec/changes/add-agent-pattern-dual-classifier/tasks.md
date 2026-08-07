# Tasks — add-agent-pattern-dual-classifier

> Spec 见 [proposal.md](./proposal.md)。架构决策见 [ADR-0006](../../../docs/adr/0006-agent-pattern-dual-classifier.md)。
>
> 状态：**ready-for-agent**（已与用户确认测试 seam，阈值待标定但实现路径明确）。

## 1. K 线形态分类器（纯函数）

- [ ] 新建分类器模块包（建议 `FineFT/analysis/classify_agent/`）
- [ ] 实现 K 线形态分类器纯函数：输入 N=20 步窗口的 `mark_price`/`volume` 序列，输出单一 K 线形态标签
- [ ] 实现 7 类判别（KX1 由 label 决定，不入纯函数；纯函数负责 KM1/KT2/KT1/KT3/KM2/KM3/未分类）
- [ ] 实现命中顺序 `KM1→KT2→KT1→KT3→KM2→KM3→未分类`（KT2 优先于 KT1）
- [ ] 阈值参数化（fixture 可注入，便于标定后只改 fixture）
- **验证**：`FineFT/tests/analysis/test_kline_pattern_classifier.py` 全部通过（合成输入覆盖 7 类 + 命中顺序 + 未分类 + 阈值边界）

## 2. 策略形态分类器（纯函数）

- [ ] 实现策略形态分类器纯函数：输入 N=20 步窗口的行为轨迹整体（`position_after`/`mark_price`/`volume`/`cumulative_realized_pnl`），输出策略形态标签集合（多选）
- [ ] 实现 6 类判别（ST1/ST3/ST2/SM1/SM2/SM3）
- [ ] 多选语义：各类独立判定，命中几个算几个
- [ ] 阈值参数化
- **验证**：`FineFT/tests/analysis/test_strategy_pattern_classifier.py` 全部通过（合成输入覆盖 6 类 + 多选组合如 ST1+SM3 + 未分类 + 阈值边界）

## 3. 明细表生成脚本（薄 orchestrator）

- [ ] 实现明细表生成：读 `trading_action_detail_epoch_*.csv` → 按 `(label, epoch, bin_index)` triple 遍历 → label_0/6 直接标 KX1 / label_1~5 切 N=20 不重叠窗口 → 调两个分类器 → 产出明细表行
- [ ] 盈亏归因：每行盈亏 = 窗口内 `(cumulative_realized_pnl[end]-[start]) + (unrealized_pnl[end]-[start])`
- [ ] K 线形态单选不重复计入；策略形态多选重复计入各命中行（一个窗口产出 1 个 K 线形态 × N 个策略形态 = N 行）
- [ ] 输出 `agent_pattern_detail_table.csv`，列含 `label, epoch, bin_index, K线形态, 策略形态, 盈亏`（可加窗口起止 step 便于追溯）
- **验证**：一个 smoke test 跑通端到端（小样本 Detail CSV → 表行数符合预期、盈亏列非空）

## 4. 聚合视图脚本

- [ ] 按 `(label, epoch, bin, K 线形态)` distinct 聚合策略形态盈亏（总盈亏、窗口数、盈亏分位数）
- [ ] 按 `(label, epoch, bin, 策略形态)` distinct 聚合 K 线形态盈亏
- [ ] 验证总盈亏不放大（distinct 检查：同一窗口的盈亏在多选策略形态下被计入多行，但聚合按形态 distinct 不跨形态求和）
- **验证**：一个 smoke test 验证 distinct 语义

## 5. 阈值标定（goal-driven 循环）

- [ ] 用提议阈值跑真实 Detail CSV（如 `trading_action_detail_epoch_58.csv`，n≈366k 行）
- [ ] 看未分类率（目标 < 30%）、7 类 K 线形态分布、6 类策略形态分布
- [ ] 看每个判别特征在真实窗口上的分布（直方图）
- [ ] 调阈值 → 重跑 → 验证分布改善
- [ ] 锁定阈值 → 更新 fixture → 回归测试覆盖
- **验证**：未分类率 < 30%、无某类从不命中、盈亏在各类间有区分度

## 6. 数值稳定性验证

- [ ] 验证 ST2 `corr(pos[t], price[t-k])` 在 N=20 窗口的稳定性
- [ ] 验证 ST3 指数拟合在 N=20 窗口的稳定性
- [ ] 验证 SM3 背离段活跃度计算的稳定性
- [ ] 不稳定的特征降级或换公式（实现期风险，记录在 ADR-0006 后果段）
- **验证**：无 NaN/inf、无方差过大导致恒命中或恒不命中
