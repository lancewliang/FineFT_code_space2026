# 05: 训练并版本化四头 Utility Model Bundle

**What to build:** 提供独立的 Utility Model 训练入口和 shell 启动方式，从候选 episode 数据训练一个按市场/频率共享的四头模型包，并发布可显式加载和完整校验的 manifest。模型内部可替换，但对外输出及血缘稳定。

**Blocked by:** 04 — 生成可审计的反事实候选 Episode 数据集.

**Status:** ready-for-human

- [ ] 训练入口独立于 VAE 和低层策略训练，可由专用 shell 启动并接收显式输入、输出和协议配置。
- [ ] 模型包包含 Mean Net PnL、多个下分位点 Lower Tail、MAE 和 Win Probability 四类输出。
- [ ] Lower Tail 输出能够按冻结规则近似 5% CVaR，所有回归损失量与标签单位保持为 bps。
- [ ] 一个市场/频率使用共享模型包，slot、direction、candidate、状态、路由及允许的宏微观证据进入冻结特征契约。
- [ ] 样本权重或采样策略防止单一合约、slot 或 direction 垄断训练，并将方法写入 manifest。
- [ ] 首选模型族通过可替换 estimator 接口接入，运行时消费者不依赖其私有对象布局。
- [ ] 四个 head 共享同一 feature schema、candidate-label schema、Fold metadata 和 bundle identity。
- [ ] CVaR 与 MAE 系数均为非零全局参数，候选值只来自 `{0.25, 0.5, 1, 2}`。
- [ ] manifest 记录模型类型、参数、随机种子、数据范围、特征与标签 hash、head 资产、训练代码版本和 λ 候选网格。
- [ ] 加载器只接受显式 manifest，不解析隐式 latest；缺失 head、hash 不符或 schema 不兼容时拒绝加载。
- [ ] 相同种子和输入能够复现 bundle identity 与允许误差内的预测；不同模型实现仍遵守相同公共输出契约。
- [ ] 测试覆盖独立启动、四头输出、manifest 往返、资产损坏、schema 不匹配和 legacy 无此 bundle 时不受影响。
