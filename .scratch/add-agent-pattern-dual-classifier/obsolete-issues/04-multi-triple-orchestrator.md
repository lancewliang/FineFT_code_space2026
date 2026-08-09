# 04 — 全候选窗口产物与完整性契约

**What to build:** 将完整双分类器接入薄 orchestrator，对多 epoch 的 Agent Pattern Candidate Universe 生成可追溯、确定、可审计的窗口表、展开表、Detail Coverage 报告和分析 manifest。

**Blocked by:** 03 — 补全策略二阶形态分类器

**Status:** ready-for-agent

- [ ] 提供独立的 model root、Selection Manifest 和 output directory 输入，Selection Manifest 仅标记已选 Agent，不过滤候选。
- [ ] Selection Manifest 在匹配前校验数据集/实验逻辑归属、7 个 Label 唯一完整性、epoch 路径一致性和 checkpoint 存在性，且不依赖机器绝对路径前缀。
- [ ] 输入表头规范化为唯一英文内部 schema；缺少原始 volume、合约或 sidecar 时提示重生成 Detail，不使用派生 volume 替代。
- [ ] 行为轨迹按逻辑身份分组并以 timestep 升序排序；timestep 必须从 0 开始、唯一、连续且非负，CSV 全局行序不影响结果。
- [ ] 每个已观测 `(epoch, label, bin_index, contract, df_path)` 都具有 sidecar 推导的全部期望 Initial-action Detail 行为轨迹；任一轨迹缺失时立即失败。
- [ ] 普通 Label 只产生长度 20、步长 20 的完整窗口；尾部丢弃步数和 gross/net PnL 按行为轨迹写入 Detail Coverage。
- [ ] 涨跌停 Label 每条轨迹恰好产生一个 KX1 事件窗口；即使策略未分类也保留该窗口。
- [ ] 窗口表包含已约定追溯字段、单选 K 线 JSON 数组、多选策略 JSON 数组和唯一 gross/net PnL；每个 `window_id` 恰好一行。
- [ ] 展开表以 `(window_id, kline_pattern, strategy_pattern)` 为唯一键，继承 Selection 和全部追溯字段。
- [ ] `window_id` 仅由已确认的逻辑窗口身份生成，相对数据路径使用规范 POSIX 形式，不纳入 Selection、形态、PnL、阈值或绝对目录。
- [ ] Detail Coverage 区分未选 epoch 缺失告警、已选 triple 缺失失败、epoch 身份冲突和重复 Detail 失败。
- [ ] 分析 manifest 记录阈值/窗口配置、缺少 Detail 的 checkpoint，并为所有输入与输出文件记录逻辑相对路径、字节数和 SHA-256。
- [ ] 端到端 smoke test 覆盖多 epoch、全候选保留、Selection 标记、乱序稳定性、timestep 异常、Initial-action 缺失、Selection 错配、epoch 冲突、PnL 守恒和文件指纹。
