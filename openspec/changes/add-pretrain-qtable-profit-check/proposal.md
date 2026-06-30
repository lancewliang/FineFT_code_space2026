# add-pretrain-qtable-profit-check

## 背景与目标

`FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py` 当前在 `for sample in range(self.num_sample)` 循环内随机选择 `df_index` 和 `initial_action`，并且只在 pretrain 阶段现场计算当前 df 的专家 qtable。这样训练开始前无法提前看到本次训练实际会使用哪些 df，也无法提前判断专家 qtable 生成的 DP action path 是否在环境回放中累计盈利。

目标是在训练循环前预生成本次训练的 sample 计划，提前计算实际会用到的 qtable，并打印每个 sample 对应 qtable/初始动作组合的盈利检查情况。该能力用于诊断 Stage I 预训练专家路径质量，帮助发现亏损专家路径或数据/qtable 计算异常。

## 用户场景

- 训练低层智能体前，维护者希望先看到本次训练将使用的 `df_index`、`initial_action` 以及对应专家路径累计收益。
- 当某个 qtable 或初始动作组合不盈利时，维护者希望日志中明确显示 `profitable=False` 和累计收益，但训练仍继续执行。
- 当同一个 `df_index` 被多个 sample 使用时，维护者希望 qtable 只计算一次，同时每个 sample 都按自己的 `initial_action` 打印盈利诊断。

## 设计方向

采用 **预生成 sample 计划 + qtable 缓存 + 盈利诊断打印**。

在 `train()` 中、进入 `for sample in range(self.num_sample)` 前：

1. 生成 `sample_plan`，长度等于 `self.num_sample`，每项包含本轮实际使用的 `df_index` 和 `initial_action`。
2. 按 `sample_plan` 中出现的唯一 `df_index` 读取 `df_{df_index}.feather`，调用现有 `create_optimal_q_table_from_df(...)` 计算 qtable，并缓存到 `q_table_cache[df_index]`。
3. 对每个 sample：
   - 从 `q_table_cache[df_index]` 读取 qtable；
   - 调用 `get_dp_action_from_qtable(q_table, initial_action)` 生成 DP action path；
   - 用对应 df、`initial_action` 和现有 demo env 初始化逻辑回放该 action path；
   - 累计 env 返回的 `episode_reward_sum`，打印 `sample_index`、`df_index`、`initial_action`、`episode_reward_sum`、`profitable=True/False`。
4. 原训练循环改为从 `sample_plan[sample]` 读取 `df_index` 和 `initial_action`。pretrain 阶段直接复用 `q_table_cache[df_index]`，不再现场重复计算 qtable。

盈利判断口径使用 env 回放得到的 `episode_reward_sum`，而不是直接累加 qtable 中的 Q 值，避免把 `reward + future max` 形式的累计 Q 值重复计数。该口径更贴近实际训练中 `env.step()` 返回的 reward。

## 关键决策

- 使用方案 B：训练前预生成每轮 sample 的 `df_index + initial_action`，只验证本次训练实际会用到的组合。
- qtable 按唯一 `df_index` 缓存，避免同一个 df 在预计算和 pretrain 阶段重复计算。
- 每个 sample 都打印盈利诊断，即使多个 sample 共享同一个 qtable，也按各自 `initial_action` 独立检查。
- `episode_reward_sum <= 0` 只打印亏损告警并继续训练；df 读取失败、qtable 计算失败或 env 回放失败应抛异常停止。
- 不新增命令行参数，不修改 `create_optimal_q_table_from_df()` 和 `get_dp_action_from_qtable()` 的函数签名，保持现有训练脚本兼容。

## 范围边界

**包含：**

- 修改 `weight_advantage_pretrain.py` 的 `train()` 流程，在 sample 循环前预生成 sample 计划。
- 预计算并缓存本次训练实际用到的 qtable。
- 打印每个 sample 的 qtable 盈利检查结果。
- pretrain 阶段复用缓存 qtable 生成 `self.perfection_action_list`。
- 增加轻量语法检查或 smoke 验证说明。

**不包含（本次）：**

- 不改变 `create_optimal_q_table_from_df()` 的计算逻辑。
- 不改变 `get_dp_action_from_qtable()` 的动作选择逻辑。
- 不新增 CLI 参数或配置文件。
- 不在 qtable 亏损时跳过 sample 或停止训练。
- 不新增完整回测指标、模型选择逻辑或策略评价体系。
- 不修改 Stage II/Stage III、VAE routing、agent filtering 或其他 baseline 脚本。

## 验收标准

- [ ] `for sample in range(self.num_sample)` 前会生成长度等于 `self.num_sample` 的 sample plan。
- [ ] 训练循环使用预生成的 `sample_plan[sample]`，不再在循环内重新随机抽取 `df_index` 和 `initial_action`。
- [ ] 本次训练涉及的每个唯一 `df_index` 只计算一次 qtable，并缓存复用。
- [ ] 每个 sample 都会打印 `sample_index`、`df_index`、`initial_action`、`episode_reward_sum` 和 `profitable`。
- [ ] 盈利判断基于 env 回放 DP action path 的累计 reward，而不是直接累加 qtable Q 值。
- [ ] pretrain 阶段从缓存 qtable 生成 `self.perfection_action_list`。
- [ ] `episode_reward_sum <= 0` 不会中断训练，只输出亏损诊断。
- [ ] `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py` 通过。
- [ ] 如果本地训练数据可用，小参数 smoke run 能看到训练循环前的 qtable 诊断打印。

## Amendments

### 2026-06-30: 独立诊断模块、多进程 qtable 计算与 sample CSV 导出

原因：训练前 qtable 诊断需要更严格地按 sample 输出独立记录；同时 qtable 计算可能较慢，需要多进程并行；生成 qtable 后需要导出便于人工查阅的 CSV 明细；相关逻辑应从训练主脚本中拆到独立 Python 文件，降低 `weight_advantage_pretrain.py` 的复杂度。

修订摘要：

- 诊断策略改为“预生成 sample 计划 + 每个 sample 独立验证”。即使同一个 `df_index` 重复出现，也必须按 sample 打印独立记录并导出独立 CSV。
- qtable 仍按唯一 `df_index` 缓存，但唯一 `df_index` 的 qtable 预计算应使用多进程并行执行。
- 新增独立模块 `FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py`，承载 sample plan、qtable 多进程预计算、DP 路径回放和 CSV 导出逻辑。
- 每个 sample 的诊断 CSV 默认输出到 `self.model_path/qtable_diagnostics/`，文件名包含 sample 序号、`df_index` 和 `initial_action`。
- CSV 以 DP action path 的时间步为行，包含行情 `open`、`high`、`low`、`close`、`volume`、`mark_price`，以及滑点、费率、动作、单步 reward、累计利润等诊断列。
