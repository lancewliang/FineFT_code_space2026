# 02 — 完整 K 线形态分类器：补齐 KM1/KT2/KT3/KM2/KM3 + 命中顺序

**What to build:** 在 01 建立的 K 线形态分类器纯函数上，补齐其余 5 类判别（KM1 V 反转 / KT2 回调 / KT3 加速 / KM2 箱体 / KM3 背离），实现完整的互斥单选命中顺序 `KX1→KM1→KT2→KT1→KT3→KM2→KM3→未分类`。其中 KX1 由 label 决定（label ∈ {0,6} → KX1），不在纯函数内判定；纯函数负责 KM1/KT2/KT1/KT3/KM2/KM3/未分类 的命中顺序。

关键技术点：
- KT2 优先于 KT1：构造"突破+回踩+再延续"序列时，命中 KT2 而非 KT1。KT2 的判别包含"突破触发 + 回踩到突破点 ±0.5% 内 + 再延续"。
- 各类判别公式与提议阈值见 [proposal.md](../../../openspec/changes/add-agent-pattern-dual-classifier/proposal.md) Implementation Decisions 段（KM1: Z≥2.0 后反向 +0.5%；KT3: 指数拟合 R²≥0.6；KM2: roll_std<0.3% + 往返 ≥2 次；KM3: vol_trend<0 + |cum_ret|≥0.5%）。
- 命中即止：按命中顺序判定，命中第一类即返回，不再判后续。
- 阈值参数化（沿用 01 的 fixture 注入机制）。

测试覆盖每类合成输入 + 命中顺序 + 阈值边界。这是纯函数层，不涉及 orchestrator。

**Blocked by:** 01 — Tracer bullet: KT1+ST1 单 triple 端到端（需要 01 建的分类器骨架与 KT1 作为参照）

**Status:** ready-for-agent

- [ ] KM1 V 反转判别实现（Z≥2.0 触及 + 反向后 |cum_ret_from_trough|≥0.5%）
- [ ] KT2 回调判别实现（突破 + 回踩 ±0.5% + 再延续）
- [ ] KT3 加速判别实现（指数拟合 R²≥0.6，斜率同号于趋势方向）
- [ ] KM2 箱体判别实现（roll_std<0.3% + 往返 ≥2 次）
- [ ] KM3 背离判别实现（vol_trend<0 + |cum_ret|≥0.5%）
- [ ] 命中顺序 `KM1→KT2→KT1→KT3→KM2→KM3→未分类` 落地（KT2 优先于 KT1）
- [ ] `test_kline_pattern_classifier.py` 补齐每类合成输入测试
- [ ] 命中顺序测试：构造"突破+回调"序列断言命中 KT2 而非 KT1
- [ ] 阈值边界测试：构造刚好在阈值上下的序列，断言分类翻转
- [ ] 未分类测试：构造随机游走序列断言 = 未分类
