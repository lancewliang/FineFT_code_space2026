# 10: 完成端到端协议与故障路径验收

**What to build:** 用一个紧凑但完整的多合约 tracer bullet 验证从双门控栈选择、两个外层 Fold、最终生产训练到一次性 test 和审计发布的整条工作流，并验证关键故障路径均以可解释、不可泄漏的方式失败。

**Blocked by:** 09 — 冻结生产资产并执行一次性 Test 对照.

**Status:** ready-for-human

- [x] 端到端夹具覆盖初始历史、Fold 1、扩展历史、Fold 2、全 valid 最终训练和一次冻结 test。
- [x] 证据证明每个外层块在预测前只读取历史前缀，且每个符合资格的 post-warmup valid 样本最多贡献一次未见评价。
- [x] 端到端产物能够从 gate stack 和 artifact manifest 追溯到 route、candidate、utility decision、真实 Trade 与 Account 结果。
- [x] legacy 在完全没有 Utility Model 资产时仍能独立完成相同输入范围的运行与公共审计发布。
- [x] tri-module 的缺失 head、损坏 hash、错误 feature schema、过期 eligibility 或 Fold 协议不一致均明确 fail-closed。
- [x] OOD 数据缺失路径不会生成虚假成交，Utility rejection 不会被归类为 exchange rejection，shadow fills 不会进入真实账本。
- [x] 开仓、加仓、反手、NullCandidate、ineligible channel、`U_lower <= 0`、硬止损和断路器均在端到端链路中至少出现一次。
- [x] 协议冻结后改变特征、模型搜索空间、λ 网格、分位规则、证据门槛或 OOD 规则会使旧外层产物失效并要求从 Fold 1 重跑。
- [x] 重复构建相同非 test 资产产生稳定的 identity、排序、manifest 与允许误差内结果。
- [x] 官方输出采用原子发布；任何校验失败不得留下看似完成的部分生产资产或部分 test 报告。
- [x] 运行文档清楚区分两个外层评价 Fold、inner OOF 训练次数和最终全 valid 生产拟合。
- [x] 验收报告明确 legacy 是否保留、默认门控未改变、test 未参与调参以及 legacy 删除仍需用户另行决定。
