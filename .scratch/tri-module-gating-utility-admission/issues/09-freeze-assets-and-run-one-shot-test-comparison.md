# 09: 冻结生产资产并执行一次性 Test 对照

**What to build:** 在两次 valid 外层评价和最终全 valid 训练完成后，冻结全部生产资产并执行唯一一次 test。输出可复现的 legacy/tri-module 对照证据，但不让 test 结果修改任何模型、阈值、默认门控或代码保留策略。

**Blocked by:** 08 — 让 Trading Utility Gate 管辖所有风险增加动作.

**Status:** ready-for-agent

- [ ] 冻结清单包含最终 Selector、四头模型、Isotonic、Platt、残差分布、eligibility、λ、特征 schema、成本模型、OOD 规则和全部 hash。
- [ ] 冻结前校验所有资产来自相容的数据范围、Fold 协议、代码版本和 bundle identity；不相容时停止 test。
- [ ] test 范围固定为 2025-06-30 至 2026-02-05，并在冻结之后只执行一次正式评价。
- [ ] test 执行不能更新 Selector、模型、校准器、残差、eligibility、阈值、λ 或预注册协议。
- [ ] legacy 与 tri-module 使用可比较的数据范围、成本口径和公共审计 envelope，同时保留各自真实决策路径。
- [ ] legacy 不为填充 tri-module 专属字段而运行候选效用推理；不适用字段保持 null。
- [ ] 对照报告分别呈现交易表现、风险、覆盖率、OOD、route、utility rejection、channel support 和审计完整性，不宣称消灭亏损交易。
- [ ] test 优劣不会自动切换默认门控、删除 legacy、改写阈值或触发二次 test。
- [ ] test 产物记录一次性运行标识、冻结清单 hash、输入数据 identity 和禁止调参声明。
- [ ] 工作流拒绝已存在正式 test 结果后的覆盖式重跑，除非由新的显式研究协议产生全新身份且不冒充原结果。
- [ ] 测试使用合成 final-test 数据验证冻结、只读行为、相同审计 schema、无自动切换和无自动删码。
