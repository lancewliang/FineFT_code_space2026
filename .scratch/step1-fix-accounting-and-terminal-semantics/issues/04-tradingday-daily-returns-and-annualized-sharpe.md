# 04: 基于真实 TradingDay 的日度收益与 242 天夏普比率重构 (TradingDay Daily Return & 242-Day Annualized Sharpe)

**What to build:** 将绩效评估中固定步长截取 12 根 bar（2小时）的伪日度计算，重构为基于中国商品期货真实业务交易日（TradingDay，包含夜盘归属）的日度组合净值聚合。按年交易日 242 天标准计算年化夏普比率、索提诺比率与日收益波动率，同时保留步级高水位线最大回撤，真实反映日度与盘中风险。

**Blocked by:** 03: 统一 12,000 共享组合资金分母与单合约除零消除 (Unified 12k Shared Capital & Zero-Denominator Elimination)

**Status:** ready-for-agent

- [ ] 组合净值序列严格按业务 TradingDay 分组，夜盘与日盘正确归入对应交易日
- [ ] 提取每个交易日结算时点的最终组合净值，构建真实的日收益率时间序列
- [ ] 年化夏普比率、索提诺比率和日度波动率基于 242 个交易日年化系数计算
- [ ] 步级高水位线最大回撤（MDD）与日末净值最大回撤分别独立计算并清晰呈现
- [ ] 报告中彻底纠正历史遗留的 30min 文案，统一标明 10min 颗粒度与 TradingDay 统计口径
