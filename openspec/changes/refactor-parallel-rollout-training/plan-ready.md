# 实现计划：refactor-parallel-rollout-training

## 来源
- 提案：openspec/changes/refactor-parallel-rollout-training/proposal.md
- 设计：openspec/changes/refactor-parallel-rollout-training/design.md
- 规格：openspec/changes/refactor-parallel-rollout-training/specs/fineft-stage-i-pretrain/spec.md
- 任务：openspec/changes/refactor-parallel-rollout-training/tasks.md

## 实现步骤

## 全局约束
- 禁止修改：`FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`
- 禁止修改：`FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py`
- 本次只实现并行训练版本，串行训练版本保持不变。

### Task 1: Parallel rollout scheduling
- [x] **任务完成**（与 superpowers plan `Task 1`、`tasks.md` 对应条目同步勾选）
- 目标：为 `parallel_weight_advantage_pretrain.py` 建立 epoch 语义、调度顺序、round 统计和 epoch 级参数衰减 helper，不接入真实 worker。
- 改动文件：`FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py`；测试文件 `FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py`。
- 验证方式：运行 focused pytest，覆盖有效 df 范围、`epoch -> context -> initial_action -> round` 顺序、每 worker `rollout_steps` 语义、固定 `update_times` 和 epoch 级 `epsilon / ada / lr` schedule。

### Task 2: Worker process protocol
- [x] **任务完成**（与 superpowers plan `Task 2`、`tasks.md` 对应条目同步勾选）
- 目标：实现 `torch.multiprocessing` spawn 常驻 df worker、Queue 消息协议、CPU `state_dict` 到 GPU 模型副本加载、worker error 回传。
- 改动文件：`FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py`；测试文件 `FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py`。
- 验证方式：运行 focused pytest，使用 fake worker/env 验证消息处理、结果排序和 error propagation，不运行真实长训练。

### Task 3: Main-process serial training integration
- [x] **任务完成**（与 superpowers plan `Task 3`、`tasks.md` 对应条目同步勾选）
- 目标：用 df-worker dispatch 替换多样化训练 rollout loop，保留 full-df warmup；主进程按 round 写 replay buffer 并固定执行 `update_times` 次 update。
- 改动文件：`FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py`；测试文件 `FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py`。
- 验证方式：运行 focused pytest，验证 full-df warmup 调用点不变、round 结果按 `df_index -> step_index` 写 buffer、`update_times` 不按 `round_steps` 折算。

### Task 4: Logging, saving, and verification
- [x] **任务完成**（与 superpowers plan `Task 4`、`tasks.md` 对应条目同步勾选）
- 目标：补齐 rollout/round/epoch 日志、全局 `round_counter` TensorBoard step、每 epoch 保存模型，并执行最终验证。
- 改动文件：`FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py`；测试文件 `FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py`；规格文件 `openspec/changes/refactor-parallel-rollout-training/tasks.md` 仅在 build 完成时同步勾选。
- 验证方式：运行 `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py`、focused pytest、`openspec validate refactor-parallel-rollout-training --strict`。
