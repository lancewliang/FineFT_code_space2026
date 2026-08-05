# Design: add-label-semantic-agent-selection

## Context

FineFT 的三阶段链路中，Stage II 负责从不同 market dynamics label 下筛选低层 agent，Stage III 再用 VAE 判断当前状态更接近哪个 label 并路由到对应 agent。

因此 `label_i -> agent_i` 不是普通的回测排行榜问题，而是一个语义绑定问题。若 label 表示“大涨”，被绑定的 agent 应当具备做多盈利能力；若 label 表示“大跌”，被绑定的 agent 应当具备做空盈利能力；若 label 表示“震荡”，被绑定的 agent 应避免强趋势敞口。商品期货场景下 label 还可能表示涨跌停状态；涨停/跌停不是普通的涨跌强度，而是带有流动性和可成交性约束的市场状态。

现有 `fineft-low-level-agent-selection` spec 已规定：

- 第一阶段在单个 epoch 内按 `(label, initial_action)` 选择 `bin_index`。
- 第一阶段评分为 `trans_reward_mean - std_preference * trans_reward_std`。
- 最终阶段使用 `result_all`，按 `label + bin_index + epoch_path` 聚合并选择平均 `trans_reward_mean` 最大的候选。
- `model.pth` 的 qnet index `i` 对应 `label_i`。

本变更保留这些结构，只在候选进入排序前增加语义匹配门槛。

## Data Contracts

### Label semantic manifest

当前商品期货 label 编号约定为：

```text
label_0       = 跌停 / 接近跌停
label_1..n    = 非涨跌停的普通动态 label，例如下跌、震荡、上涨
label_{n+1}   = 涨停 / 接近涨停
```

其中 `n = dynamic_number`，`label_number = dynamic_number + 2`。例如 `dynamic_number = 5` 时，总 label 数为 7，普通动态 label 为 `label_1` 到 `label_5`，`label_0` 和 `label_6` 分别保留给跌停与涨停。

新增机器可读文件：

```text
<dataset_root>/label_semantics.json
```

或由 picker 参数显式指定：

```text
--label_semantics_path <path>
```

建议数据形态：

```json
{
  "dataset_name": "30min/fu/fu2305",
  "labeling_method": "slope",
  "dynamic_number": 5,
  "label_number": 7,
  "labels": [
    {
      "label": "label_0",
      "direction": "strong_down",
      "direction_sign": -1,
      "strength": 2,
      "description": "跌停",
      "limit_state": "limit_down",
      "limit_state_sign": -1
    },
    {
      "label": "label_1",
      "direction": "down",
      "direction_sign": -1,
      "strength": 1,
      "description": "下跌",
      "limit_state": "none",
      "limit_state_sign": 0
    },
    {
      "label": "label_2",
      "direction": "sideways",
      "direction_sign": 0,
      "strength": 0,
      "description": "震荡",
      "limit_state": "none",
      "limit_state_sign": 0
    },
    {
      "label": "label_3",
      "direction": "sideways",
      "direction_sign": 0,
      "strength": 0,
      "description": "震荡偏弱",
      "limit_state": "none",
      "limit_state_sign": 0
    },
    {
      "label": "label_4",
      "direction": "up",
      "direction_sign": 1,
      "strength": 1,
      "description": "上涨",
      "limit_state": "none",
      "limit_state_sign": 0
    },
    {
      "label": "label_5",
      "direction": "strong_up",
      "direction_sign": 1,
      "strength": 2,
      "description": "大涨",
      "limit_state": "none",
      "limit_state_sign": 0
    },
    {
      "label": "label_6",
      "direction": "strong_up",
      "direction_sign": 1,
      "strength": 3,
      "description": "涨停",
      "limit_state": "limit_up",
      "limit_state_sign": 1
    }
  ]
}
```

约束：

- `labels` 必须覆盖 `label_0` 到 `label_<num_label-1>`。
- `direction_sign` 只能是 `-1`、`0`、`1`。
- `direction` 必须与 `direction_sign` 一致。
- `limit_state` 必须是 `none`、`near_limit_up`、`limit_up`、`near_limit_down`、`limit_down` 之一。
- `limit_state_sign` 只能是 `-1`、`0`、`1`，其中涨停类为 `1`，跌停类为 `-1`，非涨跌停为 `0`。
- `label_0` 必须具有 `limit_state = "limit_down"` 或 `"near_limit_down"`。
- 最后一个 label，即 `label_<label_number-1>` / `label_{n+1}`，必须具有 `limit_state = "limit_up"` 或 `"near_limit_up"`。
- 中间 label `label_1..label_n` 默认必须具有 `limit_state = "none"`，除非后续显式扩展出更多涨跌停子状态。
- `slope` 和 `quantile` 可以按分段顺序生成默认 manifest；`DTW` 必须显式提供。
- 如果 label 同时表示方向和涨跌停，例如“大涨+涨停”，manifest 必须同时记录 `direction_sign = 1` 和 `limit_state = "limit_up"`。
- manifest 必须进入 `selection_manifest.json` 的审计记录。

### Agent directional behavior metrics

低层测试聚合结果需要在现有字段外增加候选行为指标。每个数组字段必须与 `contract`、`df_path`、`reward_sum`、`df_length` 按 list index 对齐。

建议字段：

```python
{
    "mean_position": [...],
    "mean_abs_position": [...],
    "long_step_ratio": [...],
    "short_step_ratio": [...],
    "flat_step_ratio": [...],
    "long_reward_sum": [...],
    "short_reward_sum": [...],
    "flat_reward_sum": [...],
    "net_position_exposure": [...],
    "limit_up_step_ratio": [...],
    "limit_down_step_ratio": [...],
    "limit_up_long_reward_sum": [...],
    "limit_down_short_reward_sum": [...],
    "limit_up_reverse_short_ratio": [...],
    "limit_down_reverse_long_ratio": [...],
}
```

计算建议：

- `position_after > 0` 视为多头步。
- `position_after < 0` 视为空头步。
- `position_after == 0` 视为空仓步。
- `long_reward_sum` 为多头步上的 `step_reward` 求和。
- `short_reward_sum` 为空头步上的 `step_reward` 求和。
- `net_position_exposure = mean_position / max_holding_number`，应裁剪或校验在 `[-1, 1]`。
- `limit_up_step_ratio` 来自验证 slice 中 `limit_up_single_sided_ratio > 0` 或当前价触及 `UpperLimitPrice` 的步数占比。
- `limit_down_step_ratio` 来自验证 slice 中 `limit_down_single_sided_ratio > 0` 或当前价触及 `LowerLimitPrice` 的步数占比。
- `limit_up_long_reward_sum` 为涨停状态且 `position_after > 0` 的 `step_reward` 求和。
- `limit_down_short_reward_sum` 为跌停状态且 `position_after < 0` 的 `step_reward` 求和。
- `limit_up_reverse_short_ratio` 为涨停状态下 `position_after < 0` 的步数占比。
- `limit_down_reverse_long_ratio` 为跌停状态下 `position_after > 0` 的步数占比。
- 涨跌停识别必须沿用商品期货预处理/特征工程中的 `LowerLimitPrice`、`UpperLimitPrice`、`limit_up_single_sided_ratio`、`limit_down_single_sided_ratio` 语义，不在 picker 中重新解释原始行情合法性。

## Selection Algorithm

### Stage A: transform result rows

保留现有样本等权逻辑：

```text
normalized_reward = reward_sum / df_length
trans_reward_mean = mean(normalized_reward)
trans_reward_std = std(normalized_reward)
```

新增行为聚合：

```text
candidate_long_ratio = mean(long_step_ratio)
candidate_short_ratio = mean(short_step_ratio)
candidate_flat_ratio = mean(flat_step_ratio)
candidate_mean_exposure = mean(net_position_exposure)
candidate_long_reward_mean = mean(long_reward_sum / df_length)
candidate_short_reward_mean = mean(short_reward_sum / df_length)
candidate_limit_up_ratio = mean(limit_up_step_ratio)
candidate_limit_down_ratio = mean(limit_down_step_ratio)
candidate_limit_up_long_reward_mean = mean(limit_up_long_reward_sum / df_length)
candidate_limit_down_short_reward_mean = mean(limit_down_short_reward_sum / df_length)
candidate_limit_up_reverse_short_ratio = mean(limit_up_reverse_short_ratio)
candidate_limit_down_reverse_long_ratio = mean(limit_down_reverse_long_ratio)
```

### Stage B: semantic gate

默认阈值建议：

```text
min_directional_exposure = 0.10
min_directional_step_ratio = 0.35
max_neutral_abs_exposure = 0.20
max_limit_reverse_ratio = 0.20
```

语义规则：

- `direction_sign = 1`：候选必须满足 `candidate_mean_exposure >= min_directional_exposure`，`candidate_long_ratio >= min_directional_step_ratio`，且 `candidate_long_reward_mean > 0`。
- `direction_sign = -1`：候选必须满足 `candidate_mean_exposure <= -min_directional_exposure`，`candidate_short_ratio >= min_directional_step_ratio`，且 `candidate_short_reward_mean > 0`。
- `direction_sign = 0`：候选必须满足 `abs(candidate_mean_exposure) <= max_neutral_abs_exposure`。
- `limit_state = "limit_up"` 或 `"near_limit_up"`：候选必须满足 `candidate_limit_up_long_reward_mean > 0`，且 `candidate_limit_up_reverse_short_ratio <= max_limit_reverse_ratio`。
- `limit_state = "limit_down"` 或 `"near_limit_down"`：候选必须满足 `candidate_limit_down_short_reward_mean > 0`，且 `candidate_limit_down_reverse_long_ratio <= max_limit_reverse_ratio`。

语义规则应作为硬过滤。若某个 label 没有候选通过，picker 必须失败，并输出候选的行为指标摘要用于诊断。

### Stage C: financial ranking

通过语义过滤后，排序保持现有财务逻辑：

- 第一阶段：在 `(label, initial_action)` 内选择 `trans_reward_mean - std_preference * trans_reward_std` 最大的候选。
- 最终阶段：使用 `result_all`，按 `label + bin_index + epoch_path` 聚合不同 `initial_action` 的 `trans_reward_mean`，选择均值最大候选。

如果需要在最终阶段加入语义得分，也必须保持可审计：

```text
final_score = financial_score + semantic_alignment_weight * semantic_alignment_score
```

但本变更建议先采用硬过滤，不引入新权重，避免调参扩大范围。

## Output Changes

`selection_manifest.json` 的每个 label entry 应扩展：

```json
{
  "label": "label_6",
  "description": "大涨",
  "direction": "strong_up",
  "direction_sign": 1,
  "limit_state": "limit_up",
  "limit_state_sign": 1,
  "epoch_path": "...",
  "model_path": ".../trained_model.pkl",
  "bin_index": 3,
  "score": 0.0123,
  "source_rows": 8,
  "semantic_filter": {
    "min_directional_exposure": 0.1,
    "min_directional_step_ratio": 0.35,
    "max_neutral_abs_exposure": 0.2,
    "max_limit_reverse_ratio": 0.2
  },
  "behavior_summary": {
    "candidate_mean_exposure": 0.42,
    "candidate_long_ratio": 0.71,
    "candidate_short_ratio": 0.04,
    "candidate_long_reward_mean": 0.008,
    "candidate_limit_up_ratio": 0.31,
    "candidate_limit_up_long_reward_mean": 0.004,
    "candidate_limit_up_reverse_short_ratio": 0.02
  },
  "selection_reason": "passed bullish and limit-up semantic gates, then ranked by trans_reward_mean"
}
```

## Failure Policy

- 缺少 `label_semantics.json` 时，除非 `labeling_method` 为 `slope` 或 `quantile` 且用户显式允许按“首尾涨跌停、中间普通动态”的当前约定自动生成，否则失败。
- `DTW` label 缺少显式语义时失败。
- label 覆盖与 `num_label` 不一致时失败。
- `label_0` 未声明为跌停类，或最后一个 label 未声明为涨停类时失败。
- 候选缺少方向行为指标时失败，并提示需要重新运行支持语义指标的 `test_agent_index.py`。
- 候选缺少涨跌停行为指标且 label 语义包含 `limit_state != "none"` 时失败，并提示需要重新运行支持涨跌停语义指标的 `test_agent_index.py`。
- 某个 label 没有候选通过语义过滤时失败，不生成 `model.pth`。
- 指标存在非有限值、数组长度不对齐或 `df_length <= 0` 时失败。

## Verification Strategy

- 使用小型构造数据测试 picker，不依赖真实 GPU 或完整商品期货数据。
- 覆盖大涨 label：高收益但偏空候选应被拒绝，偏多且多头正收益候选应被选中。
- 覆盖大跌 label：高收益但偏多候选应被拒绝，偏空且空头正收益候选应被选中。
- 覆盖震荡 label：强多/强空候选应被拒绝，低净敞口候选应被选中。
- 覆盖缺失语义、DTW 无语义、缺失行为指标、无匹配候选等失败路径。
- 保留现有跨合约 schema、样本等权、manifest 写入和 label 顺序组装测试。

## Risks

- 方向行为指标依赖验证集，可能引入新的过拟合风险；必须继续只使用 validation 做选择，test 保留最终评估。
- 对震荡 label 的定义较弱，阈值需要在文档和 manifest 中显式记录。
- 如果历史 agent 没有明确方向专门化，硬过滤可能导致无候选通过；这是有效诊断信号，不应静默忽略。
- 商品期货数据量较大，逐步明细 CSV 不应成为必要输入；建议在聚合结果中直接保存方向行为指标。
