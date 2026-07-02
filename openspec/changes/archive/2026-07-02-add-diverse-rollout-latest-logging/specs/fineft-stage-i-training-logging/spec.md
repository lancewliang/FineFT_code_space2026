## ADDED Requirements

### Requirement: Stage I 多样化训练 SHALL 记录每个 df 和 rollout 的最新表现
系统 SHALL 在 `FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py` 的 `Weighted_Contexts_DQN.train()` 生命周期内维护一个 train 级最新明细缓存，用于记录多样化训练中每个 `df_index + rollout_index` 的最新累计奖励、最终余额和收益率。

#### Scenario: 多样化 rollout 完成后更新最新明细
- **WHEN** `Weighted_Contexts_DQN.train()` 进入多样化训练分支
- **AND** 某个 `df_index` 上的某个 `rollout_index` 回合完成
- **THEN** 系统 SHALL 将该 `df_index + rollout_index` 的 `reward_sum`、`final_balance` 和 `return_rate` 写入 train 级缓存
- **AND** `return_rate` SHALL 使用现有多样化回合日志口径 `final_balance / (required_money + 1e-12) - 1`
- **AND** 系统 SHALL NOT 改变现有 TensorBoard scalar 写入
- **AND** 系统 SHALL NOT 改变现有单回合多样化训练日志

#### Scenario: 重复 df 和 rollout 覆盖旧明细
- **WHEN** 同一个 `df_index + rollout_index` 在训练后续 sample 中再次完成多样化回合
- **THEN** 系统 SHALL 使用最新的 `reward_sum`、`final_balance` 和 `return_rate` 覆盖该 key 的旧值
- **AND** 系统 SHALL 只保留截至当前训练进度的最新表现

#### Scenario: 预训练和 full-df warmup 不写入最新明细
- **WHEN** 系统执行 sample-level pretrain 或 full-df warmup
- **THEN** 系统 SHALL NOT 将这些回合写入多样化训练最新明细缓存
- **AND** 系统 SHALL 保留现有 pretrain 和 full-df warmup 日志行为

### Requirement: Stage I epoch 完成日志 SHALL 打印多样化训练最新明细
系统 SHALL 在 `len(epoch_reward_sum_train_list) == epoch_number` 触发时，先打印多样化训练最新明细，再打印现有 epoch 均值日志。

#### Scenario: epoch 边界打印当前所有最新明细
- **WHEN** `len(epoch_reward_sum_train_list) == epoch_number`
- **AND** 多样化训练最新明细缓存不为空
- **THEN** 系统 SHALL 在现有 epoch 均值日志之前打印缓存中的所有明细
- **AND** 明细日志 SHALL 按 `df_index` 升序、`rollout_index` 升序输出
- **AND** 明细日志 SHALL 使用格式 `第 %d 轮 epoch 训练完成 | 多样化训练最新明细 | df_index=%d | rollout_index=%d | 累计奖励=%.4f | 最终余额=%.4f | 收益率=%.6f | %s`
- **AND** 最后的 `%s` SHALL 在 `return_rate > 0` 时为 `盈利`
- **AND** 最后的 `%s` SHALL 在 `return_rate <= 0` 时为 `亏损`

#### Scenario: epoch 边界缓存为空时保持旧日志
- **WHEN** `len(epoch_reward_sum_train_list) == epoch_number`
- **AND** 多样化训练最新明细缓存为空
- **THEN** 系统 SHALL NOT 抛出错误
- **AND** 系统 SHALL 跳过多样化训练最新明细日志
- **AND** 系统 SHALL 保留现有 epoch 均值日志和模型保存行为

#### Scenario: epoch 均值计算和模型保存行为保持不变
- **WHEN** 系统打印多样化训练最新明细
- **THEN** 系统 SHALL NOT 改变 `epoch_return_rate_train_list`、`epoch_final_balance_train_list` 和 `epoch_reward_sum_train_list` 的均值计算方式
- **AND** 系统 SHALL NOT 改变 `epoch_return_rate_train`、`epoch_final_balance_train` 和 `epoch_reward_sum_train` 的 TensorBoard 写入
- **AND** 系统 SHALL NOT 改变 epoch 模型保存路径和保存时机
