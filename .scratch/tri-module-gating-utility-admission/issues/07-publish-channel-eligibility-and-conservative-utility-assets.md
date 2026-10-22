# 07: 发布通道资格与保守效用准入资产

**What to build:** 将两个外层 Fold 的未见证据转换为每个 `(slot, direction)` 的上线资格和在线保守效用规则。稀疏、无显著性或不单调的通道必须单独禁用，不能借用全局表现或降低标准。

**Blocked by:** 06 — 建立两个外层 Fold 的严格 Prequential 管道.

**Status:** ready-for-human

- [ ] 每个 `(slot, direction)` 独立统计完整 candidate episode、五分位样本、合约和有效 TradingDay block 支持度。
- [ ] 最低支持度固定为 150 个完整 episode、每个五分位至少 30 个、至少 3 个合约和至少 20 个有效 TradingDay block。
- [ ] 未满足任一支持度要求的通道标记为 insufficient 并禁用，不合并分箱、不降阈值、不使用全局 pass 替代。
- [ ] 统计显著性使用 `contract × TradingDay` block 的单侧 95% bootstrap，并对多通道结果执行 FDR 控制。
- [ ] 两个外层 Fold 的未见预测按固定方法汇总为五个效用分位，Fold 级表格只作为诊断。
- [ ] Q1 至 Q5 的扣费后胜率和盈亏比必须逐档严格上升，且两者的 Spearman ρ 均至少为 0.80。
- [ ] bootstrap 总体趋势斜率的单侧置信下界必须大于零。
- [ ] 盈亏比使用正收益均值除以负收益均值绝对值；缺少足量赢家或输家、零分母或非有限结果均判为证据不足。
- [ ] 从严格 OOF 残差计算第五百分位保守修正，并与校准效用组合成在线 `U_lower` 所需资产。
- [ ] eligibility 产物记录支持度、bootstrap、FDR、单调性、残差分位、λ、校准器及失败原因。
- [ ] 只有支持度、显著性和单调性全部通过的通道标记 eligible。
- [ ] 测试覆盖合格、各类支持度不足、FDR 失败、点位不单调、Spearman 失败、趋势下界失败和未定义盈亏比。
