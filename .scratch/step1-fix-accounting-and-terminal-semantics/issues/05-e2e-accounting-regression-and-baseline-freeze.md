# 05: 计量修复全链路回归验收与基线冻结 (E2E Accounting Regression & Baseline Freeze)

**What to build:** 在 Valid 数据集上运行端到端完整回归验证，一次性验收净值守恒、前缀不变性与搜索回放一致性。重新回放生成最新的基线指标与证据链，明确记录修复后的净利润、TR 与 MDD（严格声明不预先承诺获得关闭全部交割防御对照中的 1,286.40 增益），并将全套基线产物版本化冻结，为后续组合风控与策略优化建立可信基准。

**Blocked by:** 01: 终局边界语义解耦与 ADR-0060 正式确立 (Terminal Boundary Decoupling & ADR-0060 Adoption), 02: 真实成交执行快照与订单级审计对账修复 (Real Execution Snapshot & Trade Audit Conservation), 03: 统一 12,000 共享组合资金分母与单合约除零消除 (Unified 12k Shared Capital & Zero-Denominator Elimination), 04: 基于真实 TradingDay 的日度收益与 242 天夏普比率重构 (TradingDay Daily Return & 242-Day Annualized Sharpe)

**Status:** ready-for-agent

- [ ] 真实成交审计脚本（--check-audit）通过，所有调仓记录手续费与成交金额 100% 匹配
- [ ] 终局前缀不变性脚本通过，追加未来数据对历史决策的改变步数为 0
- [ ] 搜索回放与最终组合回放指标一致，无计算分母与屏障口径偏差
- [ ] 组合全时段净值守恒，逐步净盈亏之和严格等于期末总净利润，各回合利润汇总吻合
- [ ] 输出并冻结修复计量后的新版基线业绩指标与诊断报告，不包含任何虚增或未见数据泄漏
