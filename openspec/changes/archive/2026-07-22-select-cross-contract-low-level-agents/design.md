# Design: select-cross-contract-low-level-agents

## Context

商品期货验证数据已经按合约和 label 分层：

```text
dataset/<freq>/<symbol>/valid/<contract>/label_*/df_*.feather
```

当前低层测试脚本递归发现 feather 文件后把相对目录整体当作 `label`，例如 `fu2409/label_0`。这会污染 picker 的 label 语义，使 `FineFT_single_agent_with_different_position.py` 无法按纯 `label_i` 选择跨合约通用 agent。

## Decisions

1. 测试输出先修正 schema。`test_agent_index.py` 只接受 `valid/<contract>/<label>/df_*.feather`，并显式输出 `contract` 和纯 `label`。
2. 聚合粒度保持旧 picker 需要的形状：每条 `analysis_result` 仍对应 `label + initial_action + bin_index`，但数组字段覆盖所有合约 slice。
3. 样本等权不引入新权重系统。picker 继续对每条 slice 使用 `reward_sum / df_length`，再计算 `trans_reward_mean` 和 `trans_reward_std`。
4. 最终选择逻辑保持当前代码行为：使用 `result_all`，按 `label + bin_index + epoch_path` 分组，对不同 `initial_action` 的 `trans_reward_mean` 求平均并取最大值。
5. 只支持新 schema。旧的 `label="fu2409/label_0"` 结果必须失败并提示重新运行测试。
6. 输出继续包含 `model.pth`，并新增 `selection_manifest.json` 记录每个 label 的最终选择。

## Data Shape

`analysis_result.npy` 中每条记录形态：

```python
{
    "label": "label_0",
    "initial_action": 0,
    "bin_index": 2,
    "contract": ["fu2409", "fu2501"],
    "df_path": ["fu2409/label_0/df_0.feather", "fu2501/label_0/df_0.feather"],
    "reward_sum": [1.0, 2.0],
    "df_length": [100, 200],
    "turnover": [0.0, 1.0],
}
```

`contract`、`df_path`、`reward_sum`、`df_length`、`turnover` 必须按 list index 对齐。

## Failure Policy

- `valid` 下没有任何三层 label slice 时失败。
- 发现旧 schema 输入时 picker 失败。
- label 集合与 `--num_label` 不一致时 picker 失败。
- 结果中数组字段长度不一致、`df_length <= 0`、非有限 reward 或重复 `df_path` 时失败。
- 每个 label 未能选出唯一 `(epoch_path, bin_index)` 时不生成最终 `model.pth`。

