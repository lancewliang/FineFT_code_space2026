# Tasks: add-regime-anchors-and-cost-aware-action-persistence

- [ ] 实现七个因果市场状态锚点 State Feature，并补公式、数值安全、尺度不变和前缀不变测试。
- [ ] 扩展 Feature Selection 的 train-only 4×4 定标、逐合约条件指标、聚合指标和 Manifest 审计字段。
- [ ] 实现只针对市场状态锚点的条件保留规则，并覆盖样本/合约/符号/置信界退化场景。
- [ ] 为环境增加无副作用逐动作预计换仓成本，并与克隆环境真实执行结果做配对测试。
- [ ] 实现共享成本感知动作选择规则，接入训练贪心分支、Low-level 测试和路由后的 Low-level 推理。
- [ ] 持久化动作迟滞配置与逐步诊断，验证旧配置默认关闭且行为兼容。
- [ ] 在固定 `fu/30min` 数据和训练预算上完成 A/B/C 消融，输出全部 16 格和目标四格的收益、风险、换仓及执行成本报告。
- [ ] 使用 `conda activate finetf` 运行 focused tests、OpenSpec strict validation 和必要的 smoke test。
