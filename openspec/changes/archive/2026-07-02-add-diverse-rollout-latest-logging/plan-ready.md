# 实现计划：add-diverse-rollout-latest-logging

## 来源
- 提案：openspec/changes/add-diverse-rollout-latest-logging/proposal.md
- 设计：无（OpenSpec 判定无需；本变更只新增 Stage I 训练日志和小型内部 helper）
- 规格：openspec/changes/add-diverse-rollout-latest-logging/specs/fineft-stage-i-training-logging/spec.md
- 任务：openspec/changes/add-diverse-rollout-latest-logging/tasks.md

## 实现步骤

### Task 1: Latest metrics helpers
- [x] **任务完成**（与 superpowers plan `Task 1`、`tasks.md` 对应条目同步勾选）
- 目标：在 `weight_advantage_pretrain.py` 中增加最小内部 helper，用于记录 `df_index + rollout_index` 最新多样化训练指标，并按固定顺序打印 epoch 明细日志；先用 focused tests 锁定覆盖、排序、盈亏标签和空缓存行为。
- 改动文件：`FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`、`FineFT/tests/rl/test_weight_advantage_pretrain_logging.py`
- 验证方式：运行 `conda activate finetf && pytest FineFT/tests/rl/test_weight_advantage_pretrain_logging.py::test_record_diverse_rollout_latest_metric_overwrites_existing_key FineFT/tests/rl/test_weight_advantage_pretrain_logging.py::test_log_diverse_rollout_latest_metrics_sorts_and_labels_profit_loss FineFT/tests/rl/test_weight_advantage_pretrain_logging.py::test_log_diverse_rollout_latest_metrics_skips_empty_cache -q`，预期新增 tests 先失败、实现 helper 后通过。

### Task 2: Training loop integration
- [x] **任务完成**（与 superpowers plan `Task 2`、`tasks.md` 对应条目同步勾选）
- 目标：在 `Weighted_Contexts_DQN.train()` 中初始化 train 级缓存，只在多样化训练 rollout 完成后更新缓存，并在 `len(epoch_reward_sum_train_list) == epoch_number` 触发时先打印最新明细再打印现有 epoch 均值日志。
- 改动文件：`FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`
- 验证方式：代码检查确认缓存初始化位于 sample loop 前，缓存更新只在 `else` 多样化分支内，明细打印位于现有 epoch 均值日志之前；运行 focused logging tests 确认 helper 行为稳定。

### Task 3: Verification
- [x] **任务完成**（与 superpowers plan `Task 3`、`tasks.md` 对应条目同步勾选）
- 目标：完成 focused tests、Python 语法检查和 OpenSpec strict 校验，确认变更没有影响训练输入输出、TensorBoard scalar、pretrain/full-df warmup 日志或 epoch 均值保存逻辑。
- 改动文件：`FineFT/tests/rl/test_weight_advantage_pretrain_logging.py`、`FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`、OpenSpec 文档 checkbox 状态（build 阶段完成后同步）
- 验证方式：运行 `conda activate finetf && pytest FineFT/tests/rl/test_weight_advantage_pretrain_logging.py -q`；运行 `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`；运行 `openspec validate add-diverse-rollout-latest-logging --strict`。
