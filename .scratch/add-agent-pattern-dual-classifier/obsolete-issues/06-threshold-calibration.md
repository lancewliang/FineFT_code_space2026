# 06 — 真实数据阈值标定与最终验收

**What to build:** 使用重生成的真实 Detail 数据运行完整双分类分析，生成特征分布和类别/PnL 诊断，经人工确认后锁定阈值，并完成全契约验收。

**Blocked by:** 05 — Initial-action Scenario/triple 聚合视图

**Status:** ready-for-agent

- [ ] 重新生成范围内的商品期货数据集与 Market Dynamic Segment，确认原始 volume 传递正确且未进入 State Feature。
- [ ] 重跑已有评估 epoch 以生成新 Detail CSV 和动作空间 sidecar，不重新训练模型。
- [ ] 使用提议阈值运行全部候选形态分析，输出 7 类 K 线形态、6 类策略形态、未分类和 PnL 区分度诊断。
- [ ] 诊断报告对未分类哨兵单独输出窗口数、比率和 PnL 分布，不将哨兵纳入正式 summary。
- [ ] 生成每个仍参与判定的特征分布，包括突破幅度、前后半窗斜率比、回撤幅度和 volume 下降比例。
- [ ] 未分类率达到告警线、某正式类别零命中或 PnL 区分度弱时只生成诊断告警，不自动调参或作为硬失败条件。
- [ ] 基于真实分布与业务语义进行人工阈值确认，不以达到固定类别占比为目标。
- [ ] 确认后的阈值写入权威配置/fixture，全部合成边界回归测试通过。
- [ ] 最终验收覆盖输入/输出 schema、唯一键、必需字段非法空值、Agent Pattern Candidate Universe、Initial-action 完整性、窗口 PnL 守恒、文件指纹和同输入/配置确定性。
