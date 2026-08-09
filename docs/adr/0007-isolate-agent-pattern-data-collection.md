# Agent 形态分析采用旁路数据采集

Agent 形态分析由独立的 `test_agents_indexs.py` 在单个模型参数目录内重新执行全部 epoch、bin、Label、数据文件和 Initial-action 组合，并把逐步事实、窗口明细、统计、聚合和可重现 manifest 全部写入独立输出目录。该入口不修改、导入或消费既有单 Agent 测试、Agent 选择、Scale Save 或 Selection Manifest 链路，因为本能力只负责研究数据采集与分析数据生产；隔离会带来单文件内重复部分评估编排逻辑的成本，但避免改变既有产物契约和下游行为。

双轴窗口分类与 Initial-action 不可相加语义继续遵循 ADR-0006；本决策只替代 ADR-0006 中“消费既有 Detail CSV 并用 Selection Manifest 标记候选”的数据来源边界。
