# 02 — 扩展到完整 Agent 形态候选全集

**What to build:** 将最小评估路径扩展成一次完整的数据采集运行：对单个模型参数目录中的全部可用 epoch，执行所有 bin、Label、数据文件和 Initial-action，并按 epoch 生成逐步 Detail，同时用 Coverage Report 证明预期组合均已执行。

**Blocked by:** 01 — 建立隔离的全新评估入口。

**Status:** ready-for-agent

- [ ] 只扫描模型参数目录的直接 `epoch_<N>` 子目录，并按 epoch 数值确定稳定执行顺序。
- [ ] 模型文件存在的 epoch 全部进入评估；缺少模型文件的 epoch 记录为 `missing_model` 并跳过；模型加载失败则整次运行失败。
- [ ] 每个可用 epoch 执行完整的 `bin_index × label × df_path × initial_action` 组合，不按 Agent 选择结果过滤。
- [ ] 所有 epoch 使用同一套动作空间，实际环境与共享 Position Level 不一致时失败。
- [ ] 每个已分析 epoch 恰好生成一个英文逐步 Detail 文件，且文件中的 epoch 字段与版本身份一致。
- [ ] 每条行为轨迹的 timestep 从 0 开始、唯一、连续且非负；重复、缺口或负值立即失败。
- [ ] Coverage Report 同时生成 epoch 和 trajectory 记录，并按固定英文 schema 报告 expected count、observed count、coverage ratio、status 和 message。
- [ ] 一个多 epoch、小数据 smoke test 证明 Detail 按 epoch 分区且完整候选全集无缺失。

