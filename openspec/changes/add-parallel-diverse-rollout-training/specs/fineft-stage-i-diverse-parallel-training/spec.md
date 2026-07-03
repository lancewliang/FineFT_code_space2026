## ADDED Requirements

### Requirement: Stage I 主 sample 循环 SHALL 不再执行 sample-level pretrain
系统 SHALL 将预训练固定保留在主 sample 循环之前的 `full_df_warmup` 路径中；进入 `for sample in range(self.num_sample)` 后，每次 sample 迭代 SHALL 进入多样化训练路径，系统 SHALL 不再提供 `pretrain_epoch` 或禁用 warmup 的 CLI 参数。

#### Scenario: 前置 warmup 后所有 sample 都进入多样化训练
- **WHEN** `Weighted_Contexts_DQN.train()` 完成 qtable diagnostics/cache 准备
- **AND** `full_df_warmup` 已按现有逻辑执行
- **THEN** 系统 SHALL 进入 `for sample in range(self.num_sample)` 主循环
- **AND** 每个 sample 迭代 SHALL 执行并行多样化训练
- **AND** 系统 SHALL NOT 在主 sample 循环内计算或使用 `pretrain = sample < self.pretrain_epoch` 来选择 sample-level pretrain
- **AND** 系统 SHALL NOT 从主 sample 循环调用 sample-level pretrain rollout/training 分支
- **AND** CLI SHALL reject `--pretrain_epoch` and `--no_full_df_warmup`

#### Scenario: full_df_warmup 固定在主 sample 循环前执行
- **WHEN** `Weighted_Contexts_DQN.train()` 完成 qtable diagnostics/cache 准备
- **THEN** 系统 SHALL 在主 `for sample in range(self.num_sample)` 循环之前执行现有 `full_df_warmup` 行为
- **AND** 该前置 warmup SHALL NOT 被并行 diverse worker 替代
- **AND** `pretrain_epoch` SHALL NOT exist as a parser argument or trainer field

### Requirement: Stage I 多样化训练 SHALL 支持同步窗口式多进程 df 探索
系统 SHALL 在 `FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py` 的多样化训练阶段并行探索所有训练 df，其中每个 df 使用一个子进程 worker 独立维护 env 时序状态，主进程负责共享 replay buffer 和模型训练。

#### Scenario: 多样化训练启动所有 df worker
- **WHEN** `Weighted_Contexts_DQN.train()` 进入多样化训练阶段
- **AND** parallel diverse training 启用
- **THEN** 系统 SHALL 为 `range(self.total_df_index_length)` 中的每个 `df_index` 启动一个 worker
- **AND** 每个 worker SHALL 只探索自己的 `df_index`
- **AND** worker SHALL 独立创建和维护自己的 env
- **AND** 系统 SHALL NOT 并行化 full-df warmup

#### Scenario: worker 不直接训练或写 replay buffer
- **WHEN** worker 探索产生 transitions
- **THEN** worker SHALL 通过队列把 transitions 返回主进程
- **AND** worker SHALL NOT 直接写入 `buffer_diverse`
- **AND** worker SHALL NOT 调用 optimizer step
- **AND** worker SHALL NOT 更新主进程的 `eval_net` 或 `target_net`
- **AND** 主进程 SHALL 串行调用 `buffer_diverse.add(...)` 写入共享经验池

#### Scenario: 主进程和 worker 通过队列通信
- **WHEN** 主进程控制并行多样化探索
- **THEN** 主进程 SHALL 为每个 worker 维护一个 command queue
- **AND** worker SHALL 通过共享 result queue 返回结果
- **AND** worker SHALL 在窗口结束后批量返回 transitions、步数、done 状态和 rollout 指标
- **AND** 队列消息 SHALL NOT 包含 CUDA tensor
- **AND** 模型同步 SHALL 使用 CPU 版 `state_dict`

### Requirement: Stage I 多样化训练 SHALL 按同步窗口触发训练
系统 SHALL 在多样化训练 epoch 内按窗口暂停探索并训练，其中窗口目标 transition 数为 `rollout_steps * df_count`，`df_count` 为当前 epoch 参与探索的总 df 数量。

#### Scenario: 每个未完成 df 优先探索 rollout_steps 步
- **WHEN** 一个多样化训练窗口开始
- **THEN** 主进程 SHALL 指令每个未 done worker 最多探索 `self.rollout_steps` 步
- **AND** 已 done worker SHALL NOT 继续探索
- **AND** 每个 worker SHALL 在本 df done 或达到本次指令步数后返回窗口结果

#### Scenario: 未完成 df 补足窗口目标
- **WHEN** 当前窗口已收集的 transition 数小于 `self.rollout_steps * df_count`
- **AND** 至少一个 worker 尚未 done
- **THEN** 主进程 SHALL 继续指令未 done worker 探索额外 steps
- **AND** 系统 SHALL 继续收集 transitions 直到达到窗口目标或所有 worker done

#### Scenario: 窗口结束后暂停探索并训练
- **WHEN** 一个窗口达到 `self.rollout_steps * df_count`
- **OR** 所有 worker 都 done
- **THEN** 主进程 SHALL 暂停 worker 探索
- **AND** 主进程 SHALL 将本窗口 transitions 写入 `buffer_diverse`
- **AND** 如果 `buffer_diverse` 足够 sample 一个 batch，主进程 SHALL 调用 `update(...)` 训练 `self.update_times` 次
- **AND** 如果 `buffer_diverse` 不足以 sample 一个 batch，主进程 SHALL 记录日志并跳过该窗口训练

#### Scenario: epoch 末尾不足窗口仍训练
- **WHEN** 一个 epoch 结束
- **AND** 最后一个窗口产生了至少一个 transition
- **AND** 最后一个窗口未达到 `self.rollout_steps * df_count`
- **THEN** 主进程 SHALL 对该 partial window 执行一次窗口训练流程
- **AND** 如果 `buffer_diverse` 足够 sample 一个 batch，主进程 SHALL 训练 `self.update_times` 次

### Requirement: Stage I 多样化训练 SHALL 在 epoch 边界管理参数和模型保存
系统 SHALL 将多样化训练 epoch 定义为所有训练 df 完整探索一遍，并在 epoch 边界更新探索/训练参数和保存模型。

#### Scenario: epoch 开始时更新 epsilon ada lr
- **WHEN** 一个新的多样化训练 epoch 开始
- **THEN** 主进程 SHALL 更新 `self.epsilon`、`self.ada` 和 `self.lr` 一次
- **AND** 主进程 SHALL 将更新后的 `self.lr` 写入 optimizer param groups
- **AND** epoch 内的窗口训练 SHALL NOT 再改变 `self.epsilon`、`self.ada` 或 `self.lr`
- **AND** 同一个 epoch 内所有 worker SHALL 使用同一个 `self.epsilon`

#### Scenario: 窗口训练后同步最新模型
- **WHEN** 主进程完成一个窗口的训练流程
- **AND** 至少一个 worker 尚未 done
- **THEN** 主进程 SHALL 把最新 `eval_net.state_dict()` 同步给未 done worker
- **AND** worker SHALL 在继续探索前加载最新模型权重

#### Scenario: epoch 结束后保存模型
- **WHEN** 一个多样化训练 epoch 结束
- **THEN** 主进程 SHALL 保存一次模型
- **AND** 保存 SHALL 使用主进程训练后的模型状态
- **AND** worker SHALL NOT 保存模型

### Requirement: Stage I 并行多样化训练 SHALL 限制 n_step 并处理 worker 错误
系统 SHALL 在并行多样化训练中 fail fast 处理不支持的 n-step 设置，并在 worker 异常时停止所有 worker。

#### Scenario: n_step 不为 1 时拒绝启动
- **WHEN** parallel diverse training 启用
- **AND** `self.n_step != 1`
- **THEN** 系统 SHALL 在启动 worker 前抛出错误
- **AND** 错误信息 SHALL 说明并行多样化训练当前只支持 `n_step=1`
- **AND** 系统 SHALL NOT 启动任何 worker

#### Scenario: worker 异常回传并停止训练
- **WHEN** worker 探索、模型加载或日志写入过程中发生异常
- **THEN** worker SHALL 通过 result queue 返回 `df_index`、错误类型、错误信息和 traceback
- **AND** 主进程 SHALL 停止所有 worker
- **AND** 主进程 SHALL 抛出错误并停止训练

### Requirement: Stage I 多样化 worker SHALL 记录独立探索日志和 rollout 指标
系统 SHALL 为每个多样化训练 worker 生成独立日志文件，并在 worker 内记录探索过程产生的 rollout 指标。

#### Scenario: worker 创建独立日志文件
- **WHEN** worker 启动
- **THEN** worker SHALL 创建或使用 `log_futures/<dataset_name>/low_level/train/diverse_workers/df_<df_index>.log`
- **AND** worker 日志 SHALL 记录 `df_index`、df 长度、设备、epoch index、window index 和模型同步事件

#### Scenario: worker 记录 rollout 结果指标
- **WHEN** worker 完成一个 df 的 rollout
- **THEN** worker SHALL 计算并记录 `episode_reward_sum`
- **AND** worker SHALL 计算并记录 `final_balance`
- **AND** worker SHALL 使用 `final_balance / (required_money + 1e-12) - 1` 计算并记录 `return_rate`
- **AND** worker SHALL 在 worker 进程内执行 `record_diverse_rollout_latest_metric` 等价语义
- **AND** worker SHALL 将 rollout 指标返回主进程用于汇总日志

#### Scenario: 主进程记录训练和汇总日志
- **WHEN** 主进程处理并行多样化训练窗口
- **THEN** 主进程 SHALL 记录窗口 transition 数、`buffer_diverse` 大小、是否触发训练、训练 loss、模型同步和 epoch 模型保存
- **AND** 主进程 SHALL 汇总 worker 返回的 rollout 指标
- **AND** 主进程 SHALL NOT 代替 worker 生成 worker 独立探索日志
