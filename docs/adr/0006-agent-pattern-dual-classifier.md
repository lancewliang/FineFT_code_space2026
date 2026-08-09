# Agent 形态分类采用双轴窗口模型

为了同时回答“Agent 面对什么行情”和“Agent 采取什么策略”，形态分析采用两个分离的输出轴：单选的 K 线形态和多选的策略二阶形态。两轴共享同一形态识别窗口和行情事件检测语义，但不把行情形态与 Agent 策略合并成一个类别。这一边界使同一行情下的多种 Agent 行为可比较，也使同一 Agent 在不同行情中的策略变化可观测。

分析以 `(label, epoch, bin_index)` 作为 Agent triple，覆盖所有存在 Detail CSV 的候选；`selection_manifest.json` 只标记当前已选 Agent，不缩小候选全集。`initial_action` 保留为反事实情景维度，不进入 Agent triple，不同情景的盈亏不直接相加。窗口级产物每个窗口只保存一组盈亏，展开级产物仅用于形态组合分析，避免多选策略放大账户盈亏。

## 否决的替代

- 不采用“K 线形态即 Agent 策略”的单列分类，因为它无法表达同一行情中的不同 Agent 行为。
- 不将 `segment` 或 `initial_action` 加入 Agent triple：前者是数据切片，后者是反事实起步条件，都不是部署 Agent 的身份。
