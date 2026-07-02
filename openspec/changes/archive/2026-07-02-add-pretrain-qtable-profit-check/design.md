# add-pretrain-qtable-profit-check 设计

## 背景

原规格只要求在 `weight_advantage_pretrain.py` 中预生成 sample plan、缓存 qtable 并打印每个 sample 的盈利诊断。修订后新增三项技术要求：

- qtable 计算需要使用多进程并行。
- 同一个 `df_index` 重复出现时，qtable 可缓存，但必须按 sample 独立验证和独立导出日志。
- qtable 诊断明细需要导出为 CSV，便于人工查看行情、动作、滑点、费率和累计利润。

这些要求会让训练主脚本变复杂，因此将诊断逻辑拆到独立 Python 模块。

## 设计决策

### 独立模块

新增 `FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py`，负责：

- 生成 `sample_plan`。
- 根据 `sample_plan` 提取唯一 `df_index`。
- 多进程计算唯一 `df_index` 对应的 qtable。
- 按 sample 的 `initial_action` 回放 DP action path。
- 打印 sample 级诊断。
- 写出 sample 级 CSV。

`weight_advantage_pretrain.py` 只保留训练流程调用，不内联多进程 worker 或 CSV 行构造逻辑。

### 多进程边界

多进程只用于唯一 `df_index` 的 qtable 计算。每个 worker 读取自己的 `df_{df_index}.feather` 并调用现有 `create_optimal_q_table_from_df(...)`。主进程收集结果后形成 `q_table_cache[df_index]`。

DP path 的 env 回放和 CSV 写出在主进程按 sample 顺序执行，保证日志顺序稳定，并避免 env 对象跨进程序列化。

### CSV 输出

CSV 默认写入：

```text
self.model_path/qtable_diagnostics/
```

每个 sample 一个 CSV，文件名包含 `sample_index`、`df_index` 和 `initial_action`。

CSV 每行对应一个 DP path 时间步，包含：

- sample 与动作：`sample_index`、`df_index`、`initial_action`、`step_index`、`action`、`previous_action`
- 行情：`timestamp`、`open`、`high`、`low`、`close`、`volume`、`mark_price`
- 交易状态：`position`、`leverage`、`commission_rate`
- 诊断指标：`step_slippage`、`step_reward`、`cumulative_profit`、`profitable`

`step_slippage` 使用当前 `env.slippage_sum` 与上一步 `env.slippage_sum` 的差值；`cumulative_profit` 使用 env reward 累计值。

## 取舍

- qtable 按唯一 `df_index` 缓存，减少重复计算。
- CSV 按 sample 独立输出，重复 `df_index` 时会重复回放和输出，但日志最直观。
- 多进程只计算 qtable，不并行 env 回放，避免日志顺序和 env 状态复杂化。
