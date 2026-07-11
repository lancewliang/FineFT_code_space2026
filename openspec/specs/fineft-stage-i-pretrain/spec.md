# fineft-stage-i-pretrain Specification

## Purpose
TBD - created by archiving change add-full-df-warmup-pretrain. Update Purpose after archive.
## Requirements
### Requirement: 全量训练分块预训练 warmup
系统 SHALL 在 `FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py` 的 Stage I sample 训练循环前，默认对每个训练分块 `df_index` 执行一次专家/规则 warmup，并在 warmup 阶段直接更新网络参数。

#### Scenario: 默认启用 full-df warmup
- **WHEN** `DQN.train()` 启动并完成 qtable diagnostics/cache 准备
- **THEN** 系统 SHALL 在进入 `for sample in range(self.num_sample)` 前执行 full-df warmup
- **AND** full-df warmup SHALL 默认启用
- **AND** 系统 SHALL 提供 CLI 关闭开关，用于跳过 full-df warmup 并恢复旧训练路径

#### Scenario: 每个 df_index 只 warmup 一次
- **WHEN** full-df warmup 启用
- **THEN** 系统 SHALL 遍历 `range(self.total_df_index_length)` 中的每个 `df_index`
- **AND** 每个 `df_index` SHALL 只执行一次 full-df warmup
- **AND** full-df warmup SHALL NOT 遍历所有 `df_index × initial_action` 组合

#### Scenario: full-df warmup 固定使用空仓初始动作
- **WHEN** 系统为 full-df warmup 构造某个 `df_index` 的初始状态
- **THEN** 系统 SHALL 使用空仓对应的 `initial_action`
- **AND** 空仓动作 SHALL 根据现有动作语义定位
- **AND** 空仓动作 SHALL NOT 通过硬编码动作编号确定
- **AND** 如果无法定位空仓动作，系统 SHALL 抛出错误并停止训练

#### Scenario: full-df warmup 覆盖所有训练 df 的 qtable 和 df cache
- **WHEN** full-df warmup 启用
- **THEN** 系统 SHALL 确保 `df_0` 到 `df_{self.total_df_index_length - 1}` 的 df 和 qtable 都存在于 cache 中
- **AND** 系统 SHALL 复用 qtable diagnostics 已经计算过的 df 和 qtable cache
- **AND** 系统 SHALL 只对 cache 中缺失的 `df_index` 读取 `df_{df_index}.feather` 并计算 qtable
- **AND** 读取 df 失败或 qtable 计算失败时，系统 SHALL 抛出错误并停止训练

#### Scenario: full-df warmup 直接更新网络参数
- **WHEN** 系统执行某个 `df_index` 的 full-df warmup
- **THEN** 系统 SHALL 使用该 df 的 qtable 和空仓初始动作生成 DP expert action path
- **AND** 系统 SHALL 使用与 sample-level pretrain 一致的 demo env 初始化逻辑创建环境
- **AND** 系统 SHALL 跑现有 4 种预训练策略：专家最优、最大多仓、最大空仓、空仓
- **AND** warmup transition SHALL 写入 `buffer_pretrain`
- **AND** 满足现有 pretrain update 条件时，系统 SHALL 调用 `update_pretrain()` 更新网络参数
- **AND** full-df warmup SHALL NOT 只预填 replay buffer 后跳过参数更新

#### Scenario: full-df warmup 日志和亏损处理
- **WHEN** 系统完成某个 `df_index` 的 full-df warmup
- **THEN** 系统 SHALL 记录该 `df_index` 的累计 reward、最终余额、收益率和更新次数
- **AND** 如果专家/规则路径累计收益不盈利，系统 SHALL 记录 warning
- **AND** 系统 SHALL NOT 因不盈利 warning 跳过该 `df_index`
- **AND** 系统 SHALL NOT 因不盈利 warning 停止训练
- **AND** full-df warmup 完成后，系统 SHALL 记录总体 warmup 摘要

#### Scenario: pretrain_epoch 默认改为零且保留显式兼容
- **WHEN** 用户未显式传入 `--pretrain_epoch`
- **THEN** `pretrain_epoch` SHALL 默认为 `0`
- **AND** 默认训练流程 SHALL 在 full-df warmup 后直接进入 diverse training
- **WHEN** 用户显式传入 `--pretrain_epoch` 且值大于 `0`
- **THEN** 系统 SHALL 在 full-df warmup 后继续保留现有 sample-level pretrain 行为

#### Scenario: 轻量验证命令
- **WHEN** 变更实现完成
- **THEN** `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py` SHALL 成功
- **AND** focused tests SHALL 覆盖默认启用、关闭开关、空仓动作定位、每个 df 只 warmup 一次和 `pretrain_epoch` 默认值

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

### Requirement: 串行 Stage I 训练 SHALL 使用实验名隔离输出
系统 SHALL 在 `FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py` 中支持实验名参数，用于隔离串行 Stage I 训练输出，同时保持现有输入数据目录语义不变。

#### Scenario: 默认实验名参数
- **WHEN** 用户解析 `weight_advantage_pretrain.py` CLI 参数且未传入 `--experiment_name`
- **THEN** `experiment_name` SHALL 默认为 `default`
- **AND** 用户 SHALL 能通过 `--experiment_name <name>` 显式指定实验名

#### Scenario: 输入数据目录保持 base_path dataset_name 语义
- **WHEN** `Weighted_Contexts_DQN.__init__` 初始化训练数据路径
- **THEN** 系统 SHALL 继续使用 `os.path.join(args.base_path, args.dataset_name, "train")` 作为 `self.train_data_path`
- **AND** 系统 SHALL 继续从 `os.path.join(args.base_path, args.dataset_name, "state_features.npy")` 读取特征列表
- **AND** 系统 SHALL 继续从 `os.path.join(args.base_path, args.dataset_name, "maintenance_margin_ratio_dict.npy")` 读取保证金比例字典
- **AND** 本变更 SHALL NOT 新增或改用独立输入数据目录参数

#### Scenario: 模型和 TensorBoard 输出包含实验名层级
- **WHEN** 用户使用 `--dataset_name fu --experiment_name window_5min_gamma097` 启动串行训练
- **THEN** `self.model_path` SHALL 等于 `os.path.join(args.result_path, "fu", "window_5min_gamma097", "weights_advantage_pretrain")`
- **AND** `self.log_path` SHALL 位于 `self.model_path` 下的 `log`
- **AND** epoch 模型、TensorBoard 文件和 `qtable_diagnostics` SHALL 继续写入 `self.model_path` 及其子目录

#### Scenario: 训练日志输出包含实验名层级
- **WHEN** `configure_logger` 为 `dataset_name=fu` 和 `experiment_name=window_5min_gamma097` 配置文件日志
- **THEN** 日志文件 SHALL 写入 `log_futures/fu/low_level/train/window_5min_gamma097/advantage.log`
- **AND** 不同 `experiment_name` 的训练日志 SHALL 写入不同目录

#### Scenario: 串行商品训练脚本传递实验名
- **WHEN** 用户运行串行商品训练 shell
- **THEN** shell SHALL 将实验名传递给 `weight_advantage_pretrain.py`
- **AND** shell SHALL 提供默认实验名 `default`
- **AND** shell 的 stdout 重定向路径 SHALL 与实验名日志目录一致

#### Scenario: 并行训练文件不受影响
- **WHEN** 本变更实现完成
- **THEN** 系统 SHALL NOT 修改 `FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py`
- **AND** 并行训练输出路径语义 SHALL 保持本变更前的行为

#### Scenario: 轻量验证命令
- **WHEN** 变更实现完成
- **THEN** `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py` SHALL 成功
- **AND** focused tests SHALL 覆盖默认实验名、显式实验名、输入路径不变、模型输出路径和日志路径包含实验名
- **AND** shell 语法检查 SHALL 覆盖被修改的串行训练 shell

