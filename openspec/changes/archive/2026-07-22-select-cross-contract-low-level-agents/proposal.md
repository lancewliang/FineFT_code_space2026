# select-cross-contract-low-level-agents

## 背景与目标

当前 `FineFT/RL/DiHFT/low_level/test_agent_index.py` 和 `FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py` 仍沿用旧的低层结果组织方式，实际回测目录已经切换到按商品合约分层、按 `label` 切片的商品期货数据结构。旧脚本会把 `fu2409/label_0` 这类路径串当作 `label`，导致无法稳定筛选出“跨商品期货合约通用”的 low level agent。

本次变更目标是把测试输出和筛选输入改成新的、显式的合约 + label schema，同时保留现有 picker 的两阶段选择算法，最终为每个 `label_i` 选出一个可跨合约复用的低层子 agent。

## 用户场景

### 场景 1：跨合约回测输出可被直接消费

用户在商品期货数据集上运行低层测试后，期望每条结果都能明确区分：

- 合约 `contract`
- 市场动态 `label`
- 初始动作 `initial_action`
- 子网络索引 `bin_index`
- 对应的多条 validation slice 结果

这样后续 picker 不需要从路径中反推语义，也不会把合约目录名误当作 label。

### 场景 2：按 label 选出跨合约通用低层 agent

用户希望对每个 `label_i` 选出一个通用 low level agent，而不是为每个合约分别选一个 agent。筛选时仍保留当前代码逻辑：

- 第一步：在单个 epoch 内，对每个 `label + initial_action` 选择最佳 `bin_index`
- 第二步：跨 epoch / 参数目录，对每个 `label` 按 `label + bin_index + epoch_path` 聚合，不同 `initial_action` 的 `trans_reward_mean` 求平均后选最优

### 场景 3：保留模型重组产物并增加审计信息

筛选完成后，用户仍希望得到后续高层路由可直接使用的 `model.pth`，同时希望有一份机器可读的选择记录，说明每个 label 最终选中了哪个 epoch、哪个 bin_index、评分和样本数。

## 设计方向

采用“新 schema + 旧 picker 算法 + 样本等权”的方案。

`test_agent_index.py` 改为只识别商品期货 valid 目录下的三层结构：

```text
valid/<contract>/<label>/df_*.feather
```

每个 `label + initial_action + bin_index` 的结果保留为一条记录，但记录中的样本来自所有合约 slice。记录会显式包含并对齐：

```text
contract
df_path
reward_sum
df_length
turnover
```

其中 `reward_sum / df_length` 的每个 slice 仍按样本等权参与评分。

`FineFT_single_agent_with_different_position.py` 保留当前两阶段算法，不改筛选语义，只改输入数据结构：

- 第一阶段：在单个 epoch 内，按 `label + initial_action` 选最佳 `bin_index`
- 第二阶段：沿用当前逻辑，使用 `result_all`，按 `label + bin_index + epoch_path` 对不同 `initial_action` 的 `trans_reward_mean` 求平均后选最终结果

输出保持现有 `model.pth` 形态，并新增 `selection_manifest.json` 记录每个 label 的最终选择。

## 关键决策

- `test_agent_index.py` 只支持新 schema，不兼容旧的 `fu2409/label_0` 结果格式。
- 结果记录中的 `label` 必须是纯 `label_<整数>`。
- 测试结果按纯 `label + initial_action + bin_index` 聚合，但样本来源覆盖所有合约 slice。
- 第一阶段评分继续使用样本等权：`reward_sum / df_length`。
- 第二阶段保持当前代码逻辑：对不同 `initial_action` 的 `trans_reward_mean` 求平均后选最优。
- picker 不要求每个合约都包含每个 label，只使用实际存在的 slice。
- 最终仍重组输出 `result/DiHFT/potential_model/<dataset>/<experiment>/model.pth`。
- 额外输出 `selection_manifest.json` 作为可审计产物。

## 范围边界

**包含：**
- 更新 `FineFT/RL/DiHFT/low_level/test_agent_index.py` 的 valid 发现逻辑和结果 schema。
- 更新 `FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py` 以消费新 schema。
- 保留当前 picker 的两阶段选择算法。
- 新增 `selection_manifest.json`。
- 更新相关测试与商品期货启动脚本，使其指向新 schema。

**不包含（本次）：**
- 修改低层训练算法本身。
- 修改环境动作映射语义。
- 修改 VAE 路由或高层 agent 选择逻辑。
- 修改 `model.pth` 的结构。
- 兼容旧 `analysis_result.npy` 的旧 schema。

## 验收标准

- [ ] `test_agent_index.py` 只能从 `valid/<contract>/<label>/df_*.feather` 读取测试数据。
- [ ] 测试结果中的 `label` 为纯 `label_<整数>`，不再包含合约前缀。
- [ ] 结果记录显式包含并对齐 `contract`、`df_path`、`reward_sum`、`df_length`、`turnover`。
- [ ] 第一阶段评分仍按样本等权计算。
- [ ] 第二阶段仍按当前代码逻辑，对不同 `initial_action` 的 `trans_reward_mean` 求平均后选最优。
- [ ] `model.pth` 仍能按 label 顺序组装并被后续流程使用。
- [ ] 生成 `selection_manifest.json`，记录每个 label 的最终选择与评分。
- [ ] 旧 schema 输入在 picker 中明确失败，提示需要重新运行测试。

