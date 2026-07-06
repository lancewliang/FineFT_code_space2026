# refactor-parallel-rollout-training

## 背景与目标

`FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py` 当前多样化训练 rollout 与主进程训练更新耦合在同一串行流程中，训练速度受环境探索吞吐限制。

本变更目标是将该脚本改造成 **df-worker 并行探索、主进程串行训练** 架构：多个 df worker 并行生成探索 transitions，主进程集中写 replay buffer 并串行更新模型，从而提高训练速度，同时避免多个进程同时修改 PyTorch 模型。

## 用户场景

- 训练低层 `Weighted_Contexts_DQN` 时，希望利用多个 df 分片并行探索，加快 rollout 数据生成。
- 希望主模型、optimizer、replay buffer 和 TensorBoard writer 仍只由主进程持有，降低并发训练风险。
- 希望保留探索批次与训练批次交替的训练节奏：每个 worker 探索固定步数后，主进程执行固定次数更新。

## 设计方向

采用 **df-worker 并行探索、主进程串行训练**。

主进程直接替换 `parallel_weight_advantage_pretrain.py` 的多样化训练 rollout 逻辑，不保留旧串行回退。`full_df_warmup` 保持现状，不纳入并行化。

一个常驻 worker 进程绑定一个有效 df。有效 df 范围沿用当前 `range(self.total_df_index_length)` 语义，不额外再排除 df。主进程按 `epoch -> context_index -> initial_action -> round` 顺序调度训练。

每个 round 开始时，主进程向 active df worker 发送最新 CPU `state_dict`、当前 `epsilon`、`context_index`、`initial_action` 和 `rollout_steps`。worker 在 GPU 上加载模型副本，并在自己的 df 环境中最多探索 `rollout_steps` 步后返回 transitions。主进程收齐结果后按固定顺序写入 replay buffer；如果 buffer 足够采样，则固定执行 `update_times` 次 `self.update()`。

`num_sample` 的 sample 语义改为 epoch 语义，建议替换为 `num_epoch`。`num_epoch=10` 表示完整探索所有有效 df 10 遍。每个 epoch 结束保存一次模型。

## 关键决策

- 只并行化多样化训练 rollout；`full_df_warmup` 保持现状。
- 直接替换现有训练逻辑，不新增串行回退开关。
- 一个 worker 进程绑定一个有效 df；worker 数等于有效 df 数。
- 有效 df 范围沿用 `range(self.total_df_index_length)`，避免重复减去最后一个 df。
- 主循环顺序为 `epoch -> context_index -> initial_action -> round`。
- `rollout_steps` 表示每个 active worker 每个 round 最多探索的固定步数。
- worker 内 rollout 跨 round 不重置；未 done 时下个 round 继续。
- rollout 完成标准遵循现有 env 的 `done` 逻辑。
- 每个 round 开始同步最新模型 `state_dict`；worker 使用 GPU 推理。
- worker 不访问 replay buffer、不更新模型、不写 TensorBoard。
- 每个 round 后主进程固定执行 `update_times` 次 update；`update_times` 不按 `round_steps` 折算。
- `epsilon / ada / lr` 只在 epoch 开始时按 `num_epoch` 计算衰减值；同一个 epoch 内所有 context 和 round 都使用该 epoch 的参数快照。
- `epsilon` 按 epoch 均匀线性衰减；`ada` 和 `lr` 前半程保持初始值，后半程按 epoch 线性衰减。
- TensorBoard 和日志 global step 使用全局 `round_counter`。
- GPU OOM 或 worker 启动失败不自动降级，让程序直接报错。

## 范围边界

**包含：**
- 重构 `parallel_weight_advantage_pretrain.py` 的多样化训练阶段为 df-worker 并行探索。
- 使用 `torch.multiprocessing`、`spawn`、常驻 worker 和 Queue 通信。
- 每个 worker 一个 input queue，所有 worker 共用一个 result queue。
- 支持消息类型：`reset_task`、`explore_round`、`round_result`、`worker_error`、`shutdown`。
- 主进程发送 CPU `state_dict`，worker 加载到 GPU 模型副本。
- worker 返回 numpy / Python 标量结构的 transitions 和 metrics，不通过 Queue 传 CUDA tensor。
- 新增或调整参数，使 epoch 语义清晰，例如 `--num_epoch`。
- 每个 epoch 结束保存模型。
- 增加调度语义单元测试。

**不包含（本次）：**
- 不并行化 `full_df_warmup`。
- 不修改 `FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`；串行训练版本必须保持不变。
- 不修改 `FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py`；qtable 诊断与 cache 模块必须保持不变。
- 不实现多进程并行训练更新。
- 不实现异步 actor-learner 架构。
- 不实现 GPU 显存自动估算、自动降级或自动限制并发。
- 不引入 Ray 等外部 actor 框架。
- 不做真实长训练集成测试。

## 主流程

```text
for epoch in range(num_epoch):
    compute epsilon / ada / lr from epoch_index and num_epoch

    for context_index in range(N):
        for initial_action in range(position_choices):
            reset active df workers for current context + initial_action

            while not all active df workers done:
                broadcast latest state_dict + epsilon + context_index + initial_action
                each active worker explores at most rollout_steps
                collect worker results

                round_steps = sum(worker_steps)
                step_counter_diverse += round_steps

                write transitions to replay buffer in df_index -> step_index order

                if replay buffer is ready:
                    run update_times serial updates in main process

    save model at epoch end
```

## 参数衰减语义

主进程在每个 epoch 开始时根据 `epoch_index` 和 `num_epoch` 计算本 epoch 的 `epsilon / ada / lr`。同一个 epoch 内所有 context、initial action 和 round 都使用这组参数；round 间只同步最新模型，不重新计算这些参数。

`epsilon` 按 epoch 均匀线性衰减：

```text
if num_epoch <= 1:
    epsilon = epsilon_init
else:
    epsilon = max(
        epsilon_min,
        epsilon_init - (epsilon_init - epsilon_min) * epoch_index / (num_epoch - 1),
    )
```

`ada` 前半程保持 `ada_init`，后半程按 epoch 线性衰减到 `ada_min`：

```text
ada_hold_epochs = num_epoch // 2
ada_decay_epochs = max(num_epoch - ada_hold_epochs - 1, 1)

if epoch_index < ada_hold_epochs:
    ada = ada_init
else:
    ada = max(
        ada_min,
        ada_init - (ada_init - ada_min) * (epoch_index - ada_hold_epochs) / ada_decay_epochs,
    )
```

`lr` 前半程保持 `lr_init`，后半程按 epoch 线性衰减到 `lr_min`：

```text
lr_hold_epochs = num_epoch // 2
lr_decay_epochs = max(num_epoch - lr_hold_epochs - 1, 1)

if epoch_index < lr_hold_epochs:
    lr = lr_init
else:
    lr = max(
        lr_min,
        lr_init - (lr_init - lr_min) * (epoch_index - lr_hold_epochs) / lr_decay_epochs,
    )
```

主进程在 epoch 开始时同步 optimizer param group 的学习率。worker 只读取本 epoch 的 `epsilon` 快照，不修改全局训练参数。

## 进程通信

使用 `torch.multiprocessing`，启动方式为 `spawn`。

主进程到 worker 的消息：

```text
reset_task:
  epoch_index
  context_index
  initial_action

explore_round:
  epoch_index
  context_index
  initial_action
  round_counter
  state_dict
  epsilon
  rollout_steps

shutdown
```

worker 到主进程的消息：

```text
round_result:
  df_index
  epoch_index
  context_index
  initial_action
  round_counter
  worker_steps
  transitions
  rollout_metrics
  done
  progress

worker_error:
  df_index
  epoch_index
  context_index
  initial_action
  round_counter
  traceback
```

任一 worker 返回 `worker_error` 时，主进程停止训练并尽量 shutdown 其他 worker。

## 日志与指标

rollout 级别记录：
- `epoch_index`
- `context_index`
- `initial_action`
- `df_index`
- `transition_count`
- `reward_sum`
- `final_balance`
- `return_rate`

round 级别记录：
- `round_counter`
- `epoch_index`
- `context_index`
- `initial_action`
- `round_steps`
- active worker 数
- replay buffer size
- 固定 update 次数
- 实际 update 次数

epoch 级别记录：
- 平均收益率
- 平均最终余额
- 平均累计奖励
- 模型保存路径

## 验收标准

- [ ] 多样化训练主循环顺序为 `epoch -> context_index -> initial_action -> round`。
- [ ] `full_df_warmup` 行为保持现状。
- [ ] 有效 df 范围沿用 `range(self.total_df_index_length)`，不会重复排除最后一个 df。
- [ ] worker 数等于有效 df 数，每个 worker 绑定一个 df。
- [ ] 每个 active worker 每个 round 最多探索 `rollout_steps` 步。
- [ ] worker rollout 跨 round 不重置，直到现有 env 返回 `done=True`。
- [ ] 每个 round 开始同步最新 CPU `state_dict`，worker 不通过 Queue 传 CUDA tensor。
- [ ] 主进程按 `df_index -> step_index` 固定顺序写 replay buffer。
- [ ] 每个 round 后固定执行 `update_times` 次 update，且不按 `round_steps` 折算。
- [ ] `epsilon / ada / lr` 只在 epoch 开始按 `num_epoch` 计划计算，且 epoch 内 context 和 round 不重新衰减。
- [ ] 每个 epoch 结束保存一次模型。
- [ ] TensorBoard 和日志使用全局 `round_counter`。
- [ ] 调度语义有单元测试覆盖，不依赖真实长训练。
