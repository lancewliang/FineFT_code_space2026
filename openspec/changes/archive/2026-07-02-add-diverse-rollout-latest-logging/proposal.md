# add-diverse-rollout-latest-logging

## Why

`FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py` 在多样化训练中会逐个 `rollout_index` 记录单次回合结果，并在 `len(epoch_reward_sum_train_list) == epoch_number` 时输出 epoch 均值日志。当前 epoch 日志只包含平均收益率、平均最终余额和平均累计奖励，无法直接看出每个子策略在不同 `df_index` 上截至当前训练进度的最新表现。

训练低层多样化策略时，维护者需要在 epoch 边界查看每个子策略在不同训练 df 上的最新累计奖励、最终余额、收益率和盈亏状态。当同一个 `df_index + rollout_index` 在后续 sample 再次出现时，维护者只关心最新一次结果，而不是历史全部结果。

## What Changes

- 在 `Weighted_Contexts_DQN.train()` 生命周期内维护一个 train 级最新明细缓存，保存每个 `df_index + rollout_index` 的最新 `{reward_sum, final_balance, return_rate}`。
- 只在多样化训练分支更新缓存；同一个 `df_index + rollout_index` 后续再次出现时覆盖旧值。
- 当 `len(epoch_reward_sum_train_list) == epoch_number` 触发时，在现有 epoch 均值日志之前，按 `df_index` 升序、`rollout_index` 升序打印缓存中的所有最新明细。
- 明细日志格式为：`第 %d 轮 epoch 训练完成 | 多样化训练最新明细 | df_index=%d | rollout_index=%d | 累计奖励=%.4f | 最终余额=%.4f | 收益率=%.6f | %s`。
- 最后的 `%s` 按收益率正负判断：`return_rate > 0` 打印 `盈利`，否则打印 `亏损`。收益率口径沿用现有多样化回合日志：`final_balance / (required_money + 1e-12) - 1`。
- 如果 epoch 边界触发时还没有任何多样化训练明细，则跳过明细日志，只保留现有 epoch 均值日志。

## Impact

- 影响范围：FineFT Stage I 低层多样化训练日志。
- 主要文件：`FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`。
- 测试文件：`FineFT/tests/rl/test_weight_advantage_pretrain_logging.py`。
- 不新增 CLI、不改变训练输入输出、不改变 TensorBoard scalar。
- 不记录 sample-level pretrain 或 full-df warmup 的最新明细。
- 不导出 CSV、JSON 或额外诊断文件。
- 不改变 sample_plan、df 采样逻辑、策略更新逻辑、reward 计算逻辑或 epoch 均值计算方式。
