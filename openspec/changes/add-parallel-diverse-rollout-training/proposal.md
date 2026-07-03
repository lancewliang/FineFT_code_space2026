# add-parallel-diverse-rollout-training

## 背景与目标

当前低层多样性训练阶段按 df 顺序探索并训练，不能同时利用多个 df 进行探索。目标是在 DQN off-policy 训练前提下，将多样性训练改为同步窗口式多进程探索：每个 df 一个子进程负责 rollout，主进程持有共享经验池并负责训练，从而提高探索吞吐，同时保持训练/探索边界清晰。

## 用户场景

- 训练多样性阶段时，所有训练 df 在同一个 epoch 内并行探索，各 df 保持自己的时序环境状态。
- 主进程在每个探索窗口结束后暂停探索，集中训练模型，再把最新模型同步给未结束的 worker。
- 每个 df worker 独立记录探索过程日志，包括累计奖励、最终余额和收益率。

## 设计方向

采用同步窗口式多进程架构，只修改 `weight_advantage_pretrain.py` 的主训练流程中多样性训练路径；`full_df_warmup` 作为主循环前置预训练保持现状，主 `sample` 循环内不再执行 sample-level pretrain 分支。

主进程负责创建并持有 `buffer_diverse`、主模型、target net、optimizer、训练计数和模型保存。每个 df 启动一个 worker，worker 持有自己的 env 和 GPU 推理模型副本，只执行 rollout，不训练 optimizer，不写 replay buffer。

一个 epoch 定义为所有 df 完整探索一遍。每个 epoch 开始时主进程统一更新一次 `epsilon`、`ada`、`lr`，epoch 内这些参数固定。epoch 内按训练窗口触发训练：每个 df 优先探索 `rollout_steps` 步，窗口目标 transition 数为 `rollout_steps * df_count`；已经 done 的 df 不再探索，未 done 的 df 继续补足窗口目标。窗口结束后所有 worker 暂停，主进程把 worker 返回的 transitions 串行加入 `buffer_diverse`，训练 `update_times` 次，然后把最新 `eval_net.state_dict()` 同步给未 done worker。epoch 末尾如果最后一个窗口不足目标 transition 数，也训练一次 `update_times`。每个 epoch 结束后保存一次模型。

worker 通过队列与主进程通信。主进程为每个 worker 维护一个 command queue，所有 worker 共享一个 result queue。worker 窗口结束后批量返回 transitions、步数、done 状态和 rollout 指标。队列中不传 CUDA tensor；worker 返回 numpy/dict/float/bool 结构，模型同步时主进程发送 CPU 版 `state_dict`，worker 收到后加载到自己的 GPU 模型。

本次不修改 replay buffer 接口，并要求并行多样性训练使用 `n_step=1`。如果启用并行多样性训练但 `n_step != 1`，训练启动时应直接报错。未来若需要 `n_step > 1`，再单独扩展 replay buffer 以按 df/env 隔离 n-step 序列。

日志分为 worker 日志和主进程日志。每个 worker 创建独立日志文件，例如 `log_futures/<dataset_name>/low_level/train/diverse_workers/df_<df_index>.log`，记录 worker 启动参数、epoch/window 进度、探索步数、`episode_reward_sum`、`final_balance`、`return_rate`、模型同步和异常 traceback。`record_diverse_rollout_latest_metric` 的语义在 worker 内执行，因为这些指标来自探索过程；主进程只汇总 worker 返回的指标，并记录训练 loss、buffer 状态、模型同步和模型保存。

## 关键决策

- 只改主训练流程的多样性训练路径；`full_df_warmup` 保持前置执行，主 `sample` 循环内的 sample-level pretrain 分支移除。
- 每个 df 一个 worker，严格按 df 数启动子进程；单 GPU 下 worker 也使用 GPU 推理。
- 探索和训练同步切换：训练时 worker 暂停，训练后再继续探索。
- `epsilon`、`ada`、`lr` 只在新 epoch 开始时更新一次，epoch 内固定。
- 窗口目标为 `rollout_steps * df_count`；done 的 df 跳过，未 done 的 df 负责补足窗口目标。
- epoch 末尾不足一个窗口时仍训练 `update_times` 次。
- 每个 epoch 结束后保存一次模型。
- worker 通过队列批量返回 transitions，主进程串行写共享 `buffer_diverse`。
- 本次要求 `n_step=1`，不修改 replay buffer 接口。
- worker 独立记录 rollout 指标和过程日志；主进程记录训练与汇总日志。

## 范围边界

**包含：**
- 多样性训练阶段的多进程 worker/coordinator 设计。
- 同步窗口训练触发逻辑。
- 主进程共享 replay buffer 写入逻辑。
- 训练后模型同步给未 done worker。
- epoch 级参数衰减和模型保存。
- worker 独立日志与 rollout 指标记录。
- `n_step=1` 启动校验。
- worker 异常回传与主进程统一停止。

**不包含（本次）：**
- sample-level pretrain 并行化；主 `sample` 循环内不再执行 sample-level pretrain。
- `full_df_warmup` 并行化。
- replay buffer 接口改造或 `n_step > 1` 支持。
- 多 GPU worker 调度。
- 动态限制 worker 数量；本次严格每个 df 一个 worker。

## 验收标准

- [ ] `full_df_warmup` 仍在主 `sample` 循环前按现状执行；主 `sample` 循环内不再通过 `pretrain = sample < self.pretrain_epoch` 执行 sample-level pretrain。
- [ ] 多样性训练启动时每个 df 创建一个 worker。
- [ ] 并行多样性训练在 `n_step != 1` 时拒绝启动并给出明确错误。
- [ ] 每个 epoch 开始时只更新一次 `epsilon`、`ada`、`lr`，epoch 内窗口训练不会改变这些参数。
- [ ] 每个窗口中每个未 done df 优先探索 `rollout_steps` 步，窗口目标为 `rollout_steps * df_count`。
- [ ] 已 done 的 df 不再探索，未 done 的 df 会继续补足当前窗口目标。
- [ ] 每个窗口结束后主进程暂停探索，将 worker transitions 加入 `buffer_diverse`，并在 buffer 足够时训练 `update_times` 次。
- [ ] epoch 末尾最后一个不足窗口如果产生了 transition，也会训练 `update_times` 次。
- [ ] 每次窗口训练后，主进程把最新模型同步给未 done worker。
- [ ] 每个 epoch 结束后保存一次模型。
- [ ] 每个 worker 生成独立日志文件，记录探索过程、`episode_reward_sum`、`final_balance`、`return_rate` 和异常 traceback。
- [ ] 主进程日志记录窗口 transition 数、buffer 大小、训练 loss、模型同步和 epoch 保存。
- [ ] worker 不直接写 replay buffer，不执行 optimizer step。

## Amendments

### 2026-07-03: 移除主 sample 循环内的 sample-level pretrain

- 原因：预训练已经前置到主训练流程之前，主 `sample` 循环内不需要再按 `sample < self.pretrain_epoch` 区分“预训练/多样化训练”。
- 需求变化：`full_df_warmup` 保持主循环前置执行；之后 `for sample in range(self.num_sample)` 的每次迭代都进入并行多样性训练，不再执行 sample-level pretrain 分支。
- 影响：实现和测试需要确认 `pretrain_epoch` 不再 gate 主 `sample` 循环内的训练路径。
