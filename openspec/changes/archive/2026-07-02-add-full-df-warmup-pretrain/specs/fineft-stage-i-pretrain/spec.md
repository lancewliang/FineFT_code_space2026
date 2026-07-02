## ADDED Requirements

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
