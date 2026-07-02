## ADDED Requirements

### Requirement: 预训练 qtable 预计算与盈利诊断
系统 SHALL 在 `FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py` 的 Stage I 训练采样循环前，预生成本次训练实际使用的 sample 计划，通过独立 Python 模块多进程预计算所需 qtable，并按 sample 打印和导出 DP 专家路径盈利诊断。

#### Scenario: qtable 诊断逻辑位于独立 Python 模块
- **WHEN** 变更实现完成
- **THEN** qtable 诊断相关的 sample plan 生成、qtable 预计算、DP action path 回放和 CSV 导出逻辑 SHALL 位于 `FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py`
- **AND** `weight_advantage_pretrain.py` SHALL 只调用该模块提供的函数或类
- **AND** `weight_advantage_pretrain.py` SHALL NOT 内联实现 qtable 多进程 worker 或 CSV 行构造逻辑

#### Scenario: 训练循环前生成 sample plan
- **WHEN** `DQN.train()` 启动并进入 `for sample in range(self.num_sample)` 前
- **THEN** 系统生成长度等于 `self.num_sample` 的 `sample_plan`
- **AND** `sample_plan` 的每个元素包含该 sample 实际使用的 `df_index` 和 `initial_action`
- **AND** `df_index` 的取值范围保持为现有 `range(self.total_df_index_length)`
- **AND** `initial_action` 的取值范围保持为现有 `range(self.position_choices)`
- **AND** 训练循环 SHALL 从 `sample_plan[sample]` 读取 `df_index` 和 `initial_action`，不再在循环内重新随机抽取这两个值

#### Scenario: 按唯一 df_index 预计算并缓存 qtable
- **WHEN** `sample_plan` 已生成
- **THEN** 系统对 `sample_plan` 中出现的每个唯一 `df_index` 读取一次 `df_{df_index}.feather`
- **AND** 系统使用多进程并行计算唯一 `df_index` 对应的 qtable
- **AND** 每个 worker 使用现有 `create_optimal_q_table_from_df(...)` 和训练脚本当前 qtable 参数计算 qtable
- **AND** 系统将 qtable 缓存在以 `df_index` 为键的缓存结构中
- **AND** pretrain 阶段 SHALL 从缓存 qtable 生成 `self.perfection_action_list`
- **AND** pretrain 阶段 SHALL NOT 对同一个 `df_index` 再次调用 `create_optimal_q_table_from_df(...)`

#### Scenario: 每个 sample 打印 DP 专家路径盈利诊断
- **WHEN** 系统完成某个 sample 的 qtable 预计算
- **THEN** 系统使用该 sample 的 `initial_action` 调用 `get_dp_action_from_qtable(q_table, initial_action)` 生成 DP action path
- **AND** 系统使用与训练循环一致的 df、初始仓位、初始杠杆、初始状态和 `initiate_demo_env(...)` 参数初始化回放环境
- **AND** 系统按 DP action path 调用 `env.step(action)` 直到 episode 结束或 action path 用完
- **AND** 系统累计 env 返回的 reward 为 `episode_reward_sum`
- **AND** 系统打印或记录 `sample_index`、`df_index`、`initial_action`、`episode_reward_sum` 和 `profitable`
- **AND** `profitable` SHALL 为 `episode_reward_sum > 0`
- **AND** 盈利判断 SHALL 使用 env 回放累计 reward，不直接累加 qtable 中的 Q 值

#### Scenario: 每个 sample 导出独立 CSV 明细
- **WHEN** 系统完成某个 sample 的 DP 专家路径回放
- **THEN** 系统 SHALL 为该 sample 写出一个独立 CSV 文件
- **AND** CSV 默认目录 SHALL 为 `self.model_path/qtable_diagnostics/`
- **AND** CSV 文件名 SHALL 包含 `sample_index`、`df_index` 和 `initial_action`
- **AND** CSV 每一行 SHALL 对应 DP action path 的一个时间步
- **AND** CSV SHALL 包含 `sample_index`、`df_index`、`initial_action`、`step_index`、`timestamp`、`open`、`high`、`low`、`close`、`volume`、`mark_price`、`action`、`previous_action`、`position`、`leverage`、`commission_rate`、`step_slippage`、`step_reward`、`cumulative_profit` 和 `profitable`
- **AND** `step_slippage` SHALL 使用当前 `env.slippage_sum` 与上一步 `env.slippage_sum` 的差值
- **AND** `cumulative_profit` SHALL 为截至当前行的 env reward 累计值

#### Scenario: 重复 df_index 仍按 sample 独立验证
- **WHEN** `sample_plan` 中多个 sample 使用同一个 `df_index`
- **THEN** 系统 SHALL 只缓存并复用该 `df_index` 的 qtable
- **AND** 系统 SHALL 为每个 sample 按自己的 `initial_action` 独立执行 DP 路径回放
- **AND** 系统 SHALL 为每个 sample 独立打印诊断日志
- **AND** 系统 SHALL 为每个 sample 独立写出 CSV 明细

#### Scenario: 亏损诊断不中断训练
- **WHEN** 某个 sample 的 DP 专家路径 `episode_reward_sum <= 0`
- **THEN** 系统打印或记录 `profitable=False`
- **AND** 系统继续执行后续训练流程
- **AND** 系统 SHALL NOT 因亏损诊断跳过该 sample
- **AND** 系统 SHALL NOT 因亏损诊断停止训练

#### Scenario: 数据或 qtable 计算错误 fail-fast
- **WHEN** 预计算阶段读取 `df_{df_index}.feather` 失败、qtable 计算失败、DP action path 生成失败或 env 回放失败
- **THEN** 系统抛出错误并停止训练
- **AND** 错误信息包含当前处理的 `sample_index` 或 `df_index`

#### Scenario: 轻量验证命令
- **WHEN** 变更实现完成
- **THEN** `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py` SHALL 成功
- **AND** 如果本地训练数据可用，小参数 smoke run SHALL 在训练循环前输出 qtable 盈利诊断日志
- **AND** 如果本地训练数据可用，小参数 smoke run SHALL 在 `qtable_diagnostics/` 下生成每个 sample 对应的 CSV 明细
