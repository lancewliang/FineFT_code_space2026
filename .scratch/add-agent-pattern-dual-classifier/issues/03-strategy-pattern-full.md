# 03 — 完整 策略形态分类器：补齐 ST2/ST3/SM1/SM2/SM3 + 多选

**What to build:** 在 01 建立的策略形态分类器纯函数上，补齐其余 5 类判别（ST2 回调加仓 / ST3 金字塔递增 / SM1 硬边界抄底 / SM2 网格微调 / SM3 背离增强），实现单窗口多选——各类独立判定，一个窗口可命中多类（如 ST1+SM3：阶跃动作 + 背离过滤不互斥）。

关键技术点：
- 读行为轨迹整体（`position_after` + `mark_price` + `volume` + `cumulative_realized_pnl`），判别 agent 动作与行情的关系模式，而非纯动作形状（01 已建立此输入契约，本 ticket 补类）。
- 各类判别公式与提议阈值见 [proposal.md](../../../openspec/changes/add-agent-pattern-dual-classifier/proposal.md)（ST3: pos~a·exp(b·cum_pnl), R²≥0.6；ST2: corr(pos[t],price[t-k]) 峰值 k≥2；SM1: |pos| 在 |z_price|≥2.0 时均值 ≥ 其他 ×3.0；SM2: pos~-α·z_price, R²≥0.7；SM3: 背离段 |Δpos| 均值 ≤ 全局 ×0.5）。
- 多选无命中顺序：各类独立判定，命中几个算几个，返回标签集合。
- 阈值参数化（沿用 01 的 fixture 注入机制）。
- 数值稳定性风险（ST2 互相关、ST3 指数拟合、SM3 背离段）在 N=20 小窗口上未验证——实现时如发现方差过大无法用，需降级或换公式。这是 ADR-0006 记录的实现期风险。

测试覆盖每类合成输入 + 多选组合 + 未分类。纯函数层，不涉及 orchestrator。

**Blocked by:** 02 — 完整 K 线形态分类器（顺序执行：先完成 K 线形态再补策略形态）

**Status:** ready-for-agent

- [ ] ST3 金字塔递增判别实现（指数拟合 R²≥0.6，b 与 label 方向同号）
- [ ] ST2 回调加仓判别实现（corr(pos,price[t-k]) 峰值 k≥2 + 回踩时 Δpos 显著）
- [ ] SM1 硬边界抄底判别实现（|pos| 在 |z_price|≥2.0 时均值 ≥ 其他 ×3.0）
- [ ] SM2 网格微调判别实现（pos~-α·z_price, R²≥0.7）
- [ ] SM3 背离增强判别实现（背离段 |Δpos| 均值 ≤ 全局 ×0.5）
- [ ] 多选机制验证：一个窗口命中多个策略形态标签（如 ST1+SM3）
- [ ] `test_strategy_pattern_classifier.py` 补齐每类合成输入测试
- [ ] 多选组合测试：构造"阶跃+背离"序列断言命中 {ST1, SM3}
- [ ] 阈值边界测试
- [ ] 未命中测试：构造无显著模式的序列断言返回空集合
- [ ] 数值稳定性验证：ST2/ST3/SM3 在 N=20 窗口无 NaN/inf、无恒命中/恒不命中
