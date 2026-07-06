## 1. Parallel rollout scheduling

- [x] 1.0 Complete parallel rollout scheduling. <!-- 已实现: 添加并验证并行 rollout 调度 helper 与 epoch 参数 wiring -->
- [x] 1.1 Add epoch-based CLI/state wiring for `parallel_weight_advantage_pretrain.py`, replacing the diverse-training use of sample semantics with `num_epoch`.
- [x] 1.2 Add focused scheduling helpers for effective df range, `epoch -> context_index -> initial_action -> round` ordering, round step accounting, and context-step decay accounting.
- [x] 1.3 Add unit tests for effective df range, loop ordering, per-worker `rollout_steps`, and fixed `update_times` behavior.

## 2. Worker process protocol

- [x] 2.0 Complete worker process protocol. <!-- 已实现: 添加并验证 worker 协议 helper、错误回传和 CPU state_dict helper -->
- [x] 2.1 Implement a `torch.multiprocessing` spawn-based df worker entrypoint with per-worker input queues and a shared result queue.
- [x] 2.2 Implement `reset_task`, `explore_round`, `round_result`, `worker_error`, and `shutdown` message handling.
- [x] 2.3 Ensure workers load CPU `state_dict` into GPU model replicas, never pass CUDA tensors through queues, and return replay-buffer-compatible transition payloads.
- [x] 2.4 Add unit tests for worker result ordering and worker error propagation without running a long training job.

## 3. Main-process serial training integration

- [x] 3.0 Complete main-process serial training integration. <!-- 已实现: 替换 diverse rollout 为 df-worker dispatch，并固定主进程串行 update_times 更新 -->
- [x] 3.1 Replace the diverse-training rollout loop with df-worker dispatch while leaving full-df warmup unchanged.
- [x] 3.2 Write returned transitions into `buffer_diverse` in deterministic `df_index -> step_index` order and update `step_counter_diverse` from actual `round_steps`.
- [x] 3.3 Run exactly `update_times` serial `self.update()` calls after each ready round, without scaling update count by `round_steps`.
- [x] 3.4 Synchronize latest model state at each round while keeping `epsilon / ada / lr` fixed inside the current epoch.

## 4. Logging, saving, and verification

- [x] 4.0 Complete logging, saving, and verification. <!-- 已实现: 添加 round/rollout 日志、epoch 保存 helper 并完成 focused pytest、py_compile 和 OpenSpec 校验 -->
- [x] 4.1 Add rollout, round, and epoch logs using global `round_counter` and save one model per epoch.
- [x] 4.2 Add focused tests for epoch-level `epsilon / ada / lr` schedules and global round counter usage.
- [x] 4.3 Run `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py`.
- [x] 4.4 Run focused pytest coverage for the new scheduling and worker protocol tests.
- [x] 4.5 Run `openspec validate refactor-parallel-rollout-training --strict`.
