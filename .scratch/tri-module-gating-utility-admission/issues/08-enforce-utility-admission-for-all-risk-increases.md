# 08: 让 Trading Utility Gate 管辖所有风险增加动作

**What to build:** 将已校准的模型包和 channel eligibility 接入 tri-module 在线路径，使所有开仓、同向加仓和反手都由统一的保守净效用规则裁决，并把候选、拒绝基线、最终动作和原因写入公共审计链。

**Blocked by:** 07 — 发布通道资格与保守效用准入资产.

**Status:** ready-for-human

- [ ] Utility Gate 只在 Module 1 健康且 Candidate Resolver 返回真实候选时评估风险增加动作。
- [ ] 在线效用由校准后的净效用加严格 OOF 残差第五百分位得到，只有对应通道 eligible 且 `U_lower > 0` 才批准增加风险。
- [ ] 在线路径不重复扣除已包含在训练标签中的 commission、slippage、impact 或压力成本。
- [ ] 校准盈利概率进入审计与单调性解释，但不存在 `P(win)` 在线硬门槛。
- [ ] 拒绝空仓开仓时通过动作空间符号映射保持 Flat，不硬编码 Action 1。
- [ ] 拒绝同向加仓时保持当前仓位，不新增暴露，也不因拒绝本身强制退出。
- [ ] 拒绝反手时关闭旧方向并保持 Flat，不继续旧风险，也不建立新方向。
- [ ] 单纯保持现有仓位或按既定计划减仓不被错误分类为新增风险。
- [ ] Module 1 降级、硬止损和断路器继续拥有高于普通 Utility Gate 的处置优先级。
- [ ] bundle、calibrator、eligibility、schema 或 hash 不匹配时 tri-module fail-closed，并输出可审计原因。
- [ ] 公共审计记录 gate stack、OOD 状态、route、candidate、denied baseline、final action、原始/校准效用、`U_lower`、盈利概率、eligibility 和拒绝阶段。
- [ ] Utility Gate 拒绝写为决策层事件，真实执行拒绝继续由 Trade Record 的执行语义表达。
- [ ] 测试覆盖多空开仓、分档加仓、双向反手、零边界效用、ineligible 通道、NullCandidate、资产损坏和高优先级风控抢占。
