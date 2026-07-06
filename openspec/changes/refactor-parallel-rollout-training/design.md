# refactor-parallel-rollout-training 设计

## 背景

`parallel_weight_advantage_pretrain.py` 当前多样化训练 rollout 在主进程串行执行，环境探索吞吐成为训练速度瓶颈。用户希望保留“探索一批 transition 后主进程串行训练”的节奏，但将探索并行到多个训练 df 分片上。

本设计只重构多样化训练阶段。`full_df_warmup` 保持现状，避免同时改变预热训练和多样化训练两条路径。

本变更只实现并行训练版本。串行训练脚本 `FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py` 不允许修改，`FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py` 也不允许修改。

## 设计决策

### df-worker 并行探索，主进程串行训练

主进程仍唯一持有主模型、target model、optimizer、replay buffer 和 TensorBoard writer。每个有效 `df_index` 启动一个常驻 worker 进程，worker 只负责该 df 的环境探索和动作推理。

有效 df 范围沿用当前 `range(self.total_df_index_length)`。当前代码已经通过 `len(os.listdir(train_data_path)) - 1` 排除了最后一个 df，因此新逻辑不得再次减去最后一个 df。

### 训练循环顺序

新的多样化训练循环顺序为：

```text
for epoch in range(num_epoch):
    compute epsilon / ada / lr from epoch schedule
    for context_index in range(N):
        for initial_action in range(position_choices):
            reset df workers for this context + initial_action
            while not all workers done:
                run one rollout round
                main process writes replay buffer
                main process runs update_times serial updates
```

`num_sample` 的旧 sample 语义替换为 epoch 语义，建议使用 `num_epoch`。`num_epoch=10` 表示所有有效 df 会完整探索 10 遍。

### round 语义

一个 round 是一次“每个 active worker 探索最多 `rollout_steps` 步，然后主进程训练”的同步批次。`rollout_steps` 是每个 worker 的固定探索步数，不是全局总步数。

worker 内部 rollout 不在 round 间重置。某个 `context_index + initial_action` 的 rollout 如果本 round 未完成，下个 round 继续同一个 env 状态；直到现有 env 返回 `done=True` 后，该 df worker 在当前 `context_index + initial_action` 下完成。

### 参数衰减

`epsilon / ada / lr` 只在 epoch 开始时根据 `epoch_index` 和 `num_epoch` 计算。同一个 epoch 内所有 context、initial action 和 round 使用同一组参数快照；round 间只更新模型参数。

`epsilon` 按 epoch 从 `epsilon_init` 均匀线性衰减到 `epsilon_min`。`num_epoch <= 1` 时保持 `epsilon_init`：

```text
epsilon = max(
    epsilon_min,
    epsilon_init - (epsilon_init - epsilon_min) * epoch_index / max(num_epoch - 1, 1),
)
```

`ada` 前半程保持 `ada_init`，后半程按 epoch 线性衰减到 `ada_min`：

```text
hold_epochs = num_epoch // 2
decay_epochs = max(num_epoch - hold_epochs - 1, 1)

if epoch_index < hold_epochs:
    ada = ada_init
else:
    ada = max(
        ada_min,
        ada_init - (ada_init - ada_min) * (epoch_index - hold_epochs) / decay_epochs,
    )
```

`lr` 使用与 `ada` 相同的“前半程保持、后半程线性衰减”节奏，从 `lr_init` 衰减到 `lr_min`。主进程在 epoch 开始时更新 optimizer param group。worker 只读取本 epoch 的 `epsilon` 快照。

### 模型同步与通信

使用 `torch.multiprocessing`，启动方式为 `spawn`。每个 worker 有一个 input queue，所有 worker 共用一个 result queue。

每个 round 开始时，主进程将最新 CPU `state_dict` 发送给 active worker。worker 加载到自己的 GPU 模型副本后执行动作推理。Queue 中不得传 CUDA tensor；worker 返回 numpy 或 Python 标量结构，主进程负责写 replay buffer。

支持的消息类型：

- `reset_task`
- `explore_round`
- `round_result`
- `worker_error`
- `shutdown`

任一 worker 返回 `worker_error` 时，主进程停止训练并尽量关闭其他 worker。GPU OOM 不做自动降级。

## 取舍

- 选择 df-worker 粒度而不是 `df + context + initial_action` 粒度，是为了降低 GPU 模型副本数量和显存压力。
- 保留主进程串行训练，避免多个进程同时修改 PyTorch 模型。
- 每个 round 都同步最新模型，提高探索使用模型的新鲜度；代价是 `state_dict` 传输有通信开销。
- 不实现自动显存估算或降级，避免训练语义在资源不足时悄悄变化。
