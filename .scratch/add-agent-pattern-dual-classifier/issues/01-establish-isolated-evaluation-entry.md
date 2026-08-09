# 01 — 建立隔离的全新评估入口

**What to build:** 提供一个不依赖既有单 Agent 测试和选择链路的全新评估入口，使研究员能够针对单个模型版本、单个子 Agent 和单个数据文件执行全部 Initial-action，并在独立输出目录获得英文逐步 Detail 数据。这一纵向切片要证明模型、验证环境、动作空间和逐步执行账本可以在新入口中完整贯通，同时不改变任何既有 Python 或产物契约。

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] 新入口可通过 CLI 接收单个模型参数目录、验证数据根目录、独立输出目录、数据集/实验身份和共享动作空间配置。
- [ ] 最小 smoke 场景可加载一个 epoch 的模型，运行一个 bin、一个 Label 数据文件和全部 Initial-action。
- [ ] 每步结果使用英文机器列名，并包含 epoch、Label、bin index、contract、df path、Initial-action、timestep、volume、mark price、动作、执行前后仓位、奖励、手续费、滑点、已实现 PnL 和浮动 PnL。
- [ ] Initial-action 集合由显式动作空间配置生成，不从观测结果反推。
- [ ] contract 或原始 volume 缺失、空值或非有限时立即失败，不回退到处理后的 volume 特征。
- [ ] 所有输出只写入独立输出目录；目标文件已存在时在评估开始前失败。
- [ ] 新入口不导入或读取既有单 Agent 测试专用代码及其 Detail/Aggregate 产物，也不读取 Selection Manifest。
- [ ] 回归检查证明既有 Scale Save、单 Agent 测试、Agent 选择和 Selection Manifest 相关 Python 均未修改。

