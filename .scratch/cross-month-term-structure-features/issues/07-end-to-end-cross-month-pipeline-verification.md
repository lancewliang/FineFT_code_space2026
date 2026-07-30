# 07 — 跨月特征管线端到端验证

**要构建什么:** 增加一个商品期货端到端 fixture，验证跨月合约结构特征的完整流转路径：特征生成、daily merge、Feature Selection 强制保留，以及 Scale Save Rolling Robust Scaling。

**被阻塞于:** 05 — 让跨月特征通过 Feature Selection 强制保留。06 已跳过，因为现有 Scale Save 机制无需生产代码修改。完整链路还要求 04a 已经实际写出 `CROSS_MONTH_FEATURE` 文件。

**Status:** ready-for-agent

- [ ] 一个小型、确定性的 fixture 同时覆盖主力/次主力动态配对和到期月份序列配对。
- [ ] 管线输出的 future-state frame 包含跨月特征。
- [ ] Feature Selection 在 `state_features.npy` 中保留跨月特征。
- [ ] Scale Save 将跨月特征记录为 scaled feature，而不是 passthrough feature。
- [ ] 如果生成的跨月特征集中出现原始价格水平或原始价格差，端到端验证会失败。
