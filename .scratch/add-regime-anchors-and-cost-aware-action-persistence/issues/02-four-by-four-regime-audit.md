# 02: 发布 train-only 16 格条件市场状态审计

**What to build:** 让 Feature Selection 在共享的 train-only 斜率/波动率阈值下，对 train 和 valid 的全部 16 个二维市场状态组合生成可复现的步骤分布和条件 IC/RankIC 报告。

**Blocked by:** 01: 生成因果市场状态锚点 State Feature。

**Status:** ready-for-agent

- [ ] 在全部训练合约的成熟步骤上拟合共享的 25%、50%、75% 斜率和波动率阈值。
- [ ] valid 只复用 train 阈值，不能重新定标或改写 train State Feature。
- [ ] 16 个斜率格×波动率格全部显式输出，零样本格也必须记录；格计数之和等于可用步骤数。
- [ ] 每格记录步骤数、占比、参与合约数、可评估特征数，以及逐合约和聚合 IC/RankIC。
- [ ] Manifest 记录公式、窗口、阈值、warm-up、定标范围和产物版本；二维格不影响 Dynamic Label 语义、路由或动作约束。
- [ ] focused tests 覆盖跨合约共享阈值、空格、计数守恒、valid 复用和确定性重跑。

