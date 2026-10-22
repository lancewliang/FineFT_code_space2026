# 04: 生成可审计的反事实候选 Episode 数据集

**What to build:** 建立离线 shadow-counterfactual 数据生成流程，为每个可评价时点的全部可用风险增加候选动作生成真实经济标签，并发布具有完整血缘的候选 episode 数据集。数据集必须比较批准动作与拒绝基线，同时与真实执行账本严格隔离。

**Blocked by:** 03 — 拆分 Regime Router 与 Candidate Resolver.

**Status:** ready-for-human

- [ ] 数据生成器覆盖空仓开仓、同向加仓和反手等全部可用风险增加候选动作，而不是只记录当前门控已批准的动作。
- [ ] 每个 episode 显式记录候选动作、拒绝基线、slot、direction、候选来源、起止 TradingDay 和 H。
- [ ] fu 10min 标签视界固定为 H=3，并与 action persistence 协议一致。
- [ ] 标签使用实际市场与账户量计算 gross PnL、commission、slippage、impact 和扣费净 PnL，统一以 bps 表示。
- [ ] 交易摩擦只扣除一次；启用 1.5 倍压力成本时在标签构造中一次性体现并留下成本模型版本。
- [ ] 标签同时提供未来净 PnL、5% 下行 CVaR 所需结果、H 内 MAE 和扣费后盈利标记。
- [ ] 反手标签正确比较“平旧并开新”与“拒绝后平旧保持 Flat”的经济差异。
- [ ] 不足 H 的尾部、跨 Fold/embargo 边界的持仓 episode、输入无效 episode 和不可可靠定价 episode 被剔除并统计原因。
- [ ] shadow counterfactual 成交不进入真实 Prediction、Trade 或 Account 账本，也不使用真实执行 ID 冒充成交。
- [ ] 数据集 manifest 记录特征版本、候选规则、成本规则、基线规则、H、数据范围、剔除计数和输入资产 hash。
- [ ] 重复运行在相同输入上产生稳定的 episode identity、排序、标签和 manifest。
- [ ] 测试以可手工核算的开仓、加仓、反手、成本压力、MAE、尾部和边界样例验证标签恒等式与账本隔离。
