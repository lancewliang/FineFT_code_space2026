# 06: 建立两个外层 Fold 的严格 Prequential 管道

**What to build:** 建立可复现的两次外层未见评价流程。每个 Fold 只从此前历史重新生成 valid 派生资产、Agent Selection、Utility Model、OOF 分数和校准器；完成评价后再用全 valid 训练最终原始模型，但不让其训练内预测进入最终校准器。

**Blocked by:** 05 — 训练并版本化四头 Utility Model Bundle.

**Status:** ready-for-human

- [ ] 初始历史固定为 2024-07-23 至 2024-10-31，只用于首次训练，不产生外层评价分数。
- [ ] 外层 Fold 1 固定评价 2024-11-01 至 2025-02-28，Fold 2 固定评价 2025-03-01 至 2025-06-27。
- [ ] 所有边界按官方 TradingDay 对齐，并执行至少 H=3 的 embargo；持仓 episode 不跨边界拆分。
- [ ] 每折只读取该折之前的历史前缀，并重新生成所有依赖 valid 的统计、切片和 Agent Selection 产物。
- [ ] 历史前缀内部执行必要的 inner prequential/cross-fit，生成严格 OOF 原始效用与概率分数。
- [ ] Isotonic Regression 只用 OOF 分数校准连续效用，Platt scaling 只用 OOF 分数校准扣费后盈利概率。
- [ ] λ 组合与其他预注册数值超参只通过 inner prequential 选择，并先满足风险与单调性约束再按保守净效用排名。
- [ ] Selector、模型、校准器、残差与阈值在每个外层块预测前冻结，该块只预测一次且不能回看修订。
- [ ] 每个符合资格的 post-warmup valid 样本最多进入一次外层未见评价；embargo、无效数据和不完整 episode 有明确排除记录。
- [ ] 两次外层评价完成后以全 valid 重新训练最终 Selector 与原始模型；该步骤不计作第三个外层 Fold。
- [ ] 最终校准器只使用累计的 leakage-free OOF/外层未见预测和结果，不接收最终全 valid 模型的 in-sample 分数。
- [ ] VAE 权重、train scaler 与 train-ID likelihood reference 在整个流程中保持冻结。
- [ ] 特征、模型搜索空间、λ 网格、分位方法、证据规则、显著性规则、OOD 与恢复阈值形成预注册协议；协议变化会使既有 Fold 产物校验失败。
- [ ] test 数据与 test 分析在此管道中不可见，也不能参与选择、校准或协议冻结。
- [ ] 泄漏测试使用未来块 sentinel 数据证明其不会影响当前 Fold 的 Selector、模型、校准器、残差或阈值。
