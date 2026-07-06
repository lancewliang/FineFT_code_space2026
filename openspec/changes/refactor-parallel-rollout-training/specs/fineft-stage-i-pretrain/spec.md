## ADDED Requirements

### Requirement: Stage I 并行多样化探索 SHALL 使用 df-worker 架构
系统 SHALL 将 `FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py` 的多样化训练阶段重构为 df-worker 并行探索、主进程串行训练架构。

#### Scenario: full-df warmup 保持现状
- **WHEN** `Weighted_Contexts_DQN.train()` 执行训练流程
- **THEN** 系统 SHALL 保留现有 full-df warmup 执行路径
- **AND** 本变更 SHALL NOT 并行化 full-df warmup
- **AND** 本变更 SHALL NOT 改变 full-df warmup 的 update、日志和 cache 行为

#### Scenario: 串行训练版本和 qtable 诊断模块保持不变
- **WHEN** 本变更实现并行训练版本
- **THEN** 系统 SHALL NOT 修改 `FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`
- **AND** 系统 SHALL NOT 修改 `FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py`
- **AND** 串行训练版本 SHALL 保持当前行为不变
- **AND** qtable 诊断与 cache 模块 SHALL 保持当前行为不变

#### Scenario: 有效 df 范围沿用现有 total_df_index_length
- **WHEN** 系统创建多样化探索 worker
- **THEN** 系统 SHALL 使用 `range(self.total_df_index_length)` 作为有效 df 范围
- **AND** worker 数 SHALL 等于有效 df 数
- **AND** 系统 SHALL NOT 在 `self.total_df_index_length` 基础上再次减去最后一个 df
- **AND** 每个 worker SHALL 绑定一个唯一 `df_index`

#### Scenario: 主循环按 epoch context initial_action round 调度
- **WHEN** full-df warmup 完成并进入多样化训练阶段
- **THEN** 系统 SHALL 按 `epoch -> context_index -> initial_action -> round` 顺序调度探索和训练
- **AND** `epoch` SHALL 表示所有有效 df 完整探索一遍
- **AND** CLI 训练轮数 SHALL 使用 epoch 语义，例如 `--num_epoch`
- **AND** `num_epoch=10` SHALL 表示所有有效 df 完整探索 10 遍
- **AND** 旧 `num_sample` 的 sample 语义 SHALL NOT 继续作为并行多样化训练的主循环语义

#### Scenario: 每个 round 每个 active worker 固定最多探索 rollout_steps
- **WHEN** 系统处于某个 `context_index + initial_action`
- **AND** 仍有 active df worker 未完成当前 rollout
- **THEN** 主进程 SHALL 向每个 active worker 发送一个 `explore_round` 任务
- **AND** 每个 active worker SHALL 最多探索 `rollout_steps` 个 transition
- **AND** `rollout_steps` SHALL 表示每个 worker 每个 round 的固定探索步数
- **AND** 主进程 SHALL 等待本 round 的所有 active worker 返回后再写 replay buffer 和训练
- **AND** `round_steps` SHALL 等于本 round 所有 worker 实际返回的 transition 数之和

#### Scenario: worker rollout 跨 round 保持环境状态
- **WHEN** 某个 worker 在当前 round 探索满 `rollout_steps` 但 env 尚未 `done`
- **THEN** worker SHALL 保留当前 env 状态、累计指标和进度
- **AND** 下一个 round SHALL 从该状态继续探索
- **AND** worker SHALL NOT 因 round 边界重置 env
- **AND** worker SHALL 仅在现有 env 返回 `done=True` 后标记当前 `df_index + context_index + initial_action` 完成
- **AND** 已完成的 worker SHALL NOT 再收到当前 `context_index + initial_action` 的 `explore_round` 任务

#### Scenario: 每个 round 后主进程串行固定次数训练
- **WHEN** 主进程收齐某个 round 的 worker 结果
- **THEN** 主进程 SHALL 按 `df_index -> step_index` 固定顺序写入 `buffer_diverse`
- **AND** 主进程 SHALL 将 `step_counter_diverse` 增加 `round_steps`
- **AND** 如果 replay buffer 已满足采样条件，主进程 SHALL 串行执行固定 `update_times` 次 `self.update()`
- **AND** `update_times` SHALL NOT 按 `round_steps` 或 worker 数折算
- **AND** worker SHALL NOT 访问 replay buffer
- **AND** worker SHALL NOT 更新主模型或 optimizer

### Requirement: Stage I 并行探索 SHALL 由主进程统一同步模型和 epoch 衰减参数
系统 SHALL 在并行多样化训练中由主进程统一管理模型同步、按 epoch 计算的 `epsilon / ada / lr` 和 TensorBoard step。

#### Scenario: 每个 round 同步最新模型快照
- **WHEN** 主进程开始一个新的探索 round
- **THEN** 主进程 SHALL 向每个 active worker 发送最新 CPU `state_dict`
- **AND** worker SHALL 将该 `state_dict` 加载到自己的 GPU 模型副本
- **AND** worker SHALL 使用 GPU 执行动作推理
- **AND** worker SHALL NOT 通过 Queue 发送或接收 CUDA tensor
- **AND** GPU OOM 或 worker 启动失败时，系统 SHALL fail fast 且不自动降级为 CPU 或限制并发

#### Scenario: epoch 开始时计算 epsilon 线性衰减
- **WHEN** 主进程进入新的 `epoch_index`
- **THEN** 主进程 SHALL 根据 `epoch_index` 和 `num_epoch` 计算本 epoch 的 `epsilon`
- **AND** `epsilon` SHALL 从 `epsilon_init` 按 epoch 均匀线性衰减到 `epsilon_min`
- **AND** `epsilon` SHALL NOT 小于 `epsilon_min`
- **AND** 当 `num_epoch <= 1` 时，`epsilon` SHALL 保持 `epsilon_init`
- **AND** 同一个 epoch 内所有 `context_index`、`initial_action` 和 round SHALL 使用同一个 `epsilon`

#### Scenario: epoch 开始时计算 ada 后半程衰减
- **WHEN** 主进程进入新的 `epoch_index`
- **THEN** 主进程 SHALL 根据 `epoch_index` 和 `num_epoch` 计算本 epoch 的 `ada`
- **AND** `ada` SHALL 在前半程 epoch 保持 `ada_init`
- **AND** `ada` SHALL 在后半程 epoch 按 epoch 线性衰减到 `ada_min`
- **AND** `ada` SHALL NOT 小于 `ada_min`
- **AND** 同一个 epoch 内所有 `context_index`、`initial_action` 和 round SHALL 使用同一个 `ada`

#### Scenario: epoch 开始时计算 lr 后半程衰减
- **WHEN** 主进程进入新的 `epoch_index`
- **THEN** 主进程 SHALL 根据 `epoch_index` 和 `num_epoch` 计算本 epoch 的 `lr`
- **AND** `lr` SHALL 在前半程 epoch 保持 `lr_init`
- **AND** `lr` SHALL 在后半程 epoch 按 epoch 线性衰减到 `lr_min`
- **AND** `lr` SHALL NOT 小于 `lr_min`
- **AND** 主进程 SHALL 将本 epoch 的 `lr` 写入 optimizer param group
- **AND** 同一个 epoch 内所有 `context_index`、`initial_action` 和 round SHALL 使用同一个 `lr`

#### Scenario: 同一 epoch 内 context 和 round 间不重新衰减参数
- **WHEN** 系统处于同一个 `epoch_index`
- **AND** 已完成一个或多个 round 训练更新
- **THEN** 系统 SHALL 在后续 context 和 round 继续使用该 epoch 开始时确定的 `epsilon / ada / lr`
- **AND** 系统 SHALL NOT 在同一 epoch 的 context 间或 round 间重新计算 `epsilon / ada / lr`
- **AND** 系统 SHALL 允许 round 间同步最新模型参数

#### Scenario: 使用 torch multiprocessing 常驻 worker 和 Queue 通信
- **WHEN** 系统启动并行多样化探索
- **THEN** 系统 SHALL 使用 `torch.multiprocessing` 和 `spawn` 启动常驻 worker
- **AND** 每个 worker SHALL 拥有一个 input queue
- **AND** 所有 worker SHALL 共用一个 result queue
- **AND** 主进程 SHALL 支持向 worker 发送 `reset_task`、`explore_round` 和 `shutdown` 消息
- **AND** worker SHALL 返回 `round_result` 或 `worker_error` 消息
- **AND** worker 返回 `worker_error` 时，主进程 SHALL 停止训练并尽量 shutdown 其他 worker

#### Scenario: TensorBoard 和模型保存使用新 epoch round 语义
- **WHEN** 系统记录并行多样化训练日志或 TensorBoard scalar
- **THEN** 系统 SHALL 使用全局 `round_counter` 作为 round 级 global step
- **AND** rollout 级日志 SHALL 包含 `epoch_index`、`context_index`、`initial_action`、`df_index`、`transition_count`、`reward_sum`、`final_balance` 和 `return_rate`
- **AND** round 级日志 SHALL 包含 `round_counter`、`round_steps`、active worker 数、replay buffer size 和实际 update 次数
- **AND** 每个 epoch 结束时系统 SHALL 保存一次模型

#### Scenario: 轻量验证命令
- **WHEN** 变更实现完成
- **THEN** `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py` SHALL 成功
- **AND** focused tests SHALL 覆盖 worker 范围、主循环顺序、round 步数语义、固定 update_times、epoch 级参数衰减计划、结果排序和 worker error 处理
