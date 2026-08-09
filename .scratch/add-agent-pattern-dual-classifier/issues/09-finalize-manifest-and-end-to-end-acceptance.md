# 09 — 完成 Manifest 与全量验收

**What to build:** 为完整旁路分析运行生成可重现的 Analysis Manifest，并通过多 epoch 端到端验收证明固定文件清单、英文 schema、候选覆盖、PnL、确定性、指纹和失败边界全部成立，同时既有系统保持零变更。

**Blocked by:** 08 — 生成分类诊断和六个聚合视图。

**Status:** ready-for-agent

- [ ] Analysis Manifest 使用固定顶层键，记录数据集/实验身份、逻辑根、评估配置、动作空间、窗口配置、分类阈值、发现/分析/缺失模型的 epoch、候选全集和告警。
- [ ] 每个实际模型、验证数据和生成输出记录逻辑相对路径、字节数和 SHA-256。
- [ ] Manifest 不记录 Selection Manifest、is selected 或自身指纹，绝对输出目录不构成窗口或数据身份。
- [ ] 多 epoch CLI smoke test 覆盖多个 bin、Label、数据文件和全部 Initial-action。
- [ ] 验收逐步 Detail 按 epoch 分区，其余 Window、Expanded、Coverage、Diagnostics 和六个 summary 跨 epoch汇总。
- [ ] 所有固定 CSV 文件名、英文表头、列顺序和唯一键与 spec 完全一致。
- [ ] 验收完整候选与行为轨迹覆盖、窗口和尾部 PnL 守恒、聚合不放大及相同输入配置的确定性。
- [ ] 验收输出碰撞、缺失/非法 contract 或 volume、非法 timestep、动作空间不一致、模型加载失败和指纹异常均按 spec 失败。
- [ ] 使用项目指定 conda 环境运行目标测试与相关低层回归测试。
- [ ] 最终 diff 证明除唯一新入口及其测试/规格外，没有修改既有 Scale Save、单 Agent 测试、Agent 选择或下游实现。
