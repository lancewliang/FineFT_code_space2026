# 实现计划：refactor-rl-diagnostics-dataclasses

## 来源
- 提案：openspec/changes/refactor-rl-diagnostics-dataclasses/proposal.md
- 设计：openspec/changes/refactor-rl-diagnostics-dataclasses/design.md
- 规格：openspec/changes/refactor-rl-diagnostics-dataclasses/specs/fineft-rl-diagnostics-dataclasses/spec.md
- 任务：openspec/changes/refactor-rl-diagnostics-dataclasses/tasks.md

## 实现步骤

### Task 1: Update focused tests for dataclass diagnostics contracts
- [x] **任务完成**（与 superpowers plan `Task 1`、`tasks.md` 对应条目同步勾选）
- 目标：先让测试要求 dataclass 返回值、属性访问和 `.to_dict()` 兼容结构，覆盖 loss NaN、qtable diagnostics 和 parallel rollout。
- 改动文件：`FineFT/tests/rl/test_weight_advantage_pretrain_logging.py`、`FineFT/tests/rl/test_pretrain_qtable_diagnostics.py`、`FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py`
- 验证方式：`conda activate finetf && pytest FineFT/tests/rl/test_weight_advantage_pretrain_logging.py FineFT/tests/rl/test_pretrain_qtable_diagnostics.py FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py -q` 应在实现前暴露旧 dict/tuple 接口失败。

### Task 2: Refactor loss NaN diagnostics dataclasses
- [x] **任务完成**（与 superpowers plan `Task 2`、`tasks.md` 对应条目同步勾选）
- 目标：在 `loss_nan_diagnostics.py` 定义 `NumericValueSummary`、`NonfiniteLocation` 和 `LossNanDiagnostics`，让诊断生成和日志输出使用属性访问。
- 改动文件：`FineFT/RL/DiHFT/low_level/loss_nan_diagnostics.py`、`FineFT/tests/rl/test_weight_advantage_pretrain_logging.py`
- 验证方式：`conda activate finetf && pytest FineFT/tests/rl/test_weight_advantage_pretrain_logging.py::test_build_loss_nan_diagnostics_identifies_nonfinite_training_data -q` 通过。

### Task 3: Refactor qtable diagnostics dataclasses
- [x] **任务完成**（与 superpowers plan `Task 3`、`tasks.md` 对应条目同步勾选）
- 目标：在 `pretrain_qtable_diagnostics.py` 定义 sample item、manifest、CSV row、sample diagnostic、worker result 和 prepare result dataclass，保持 JSON/CSV 格式兼容。
- 改动文件：`FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py`、`FineFT/tests/rl/test_pretrain_qtable_diagnostics.py`
- 验证方式：`conda activate finetf && pytest FineFT/tests/rl/test_pretrain_qtable_diagnostics.py -q` 通过。

### Task 4: Update training callers for qtable dataclass results
- [x] **任务完成**（与 superpowers plan `Task 4`、`tasks.md` 对应条目同步勾选）
- 目标：把串行和并行训练脚本中对 `prepare_pretrain_qtable_diagnostics()`、sample plan 和 sample action cache 的访问改为 dataclass 属性。
- 改动文件：`FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`、`FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py`、相关 RL tests
- 验证方式：`conda activate finetf && pytest FineFT/tests/rl/test_pretrain_qtable_diagnostics.py FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py -q` 通过；`python -m py_compile` 覆盖两个训练脚本。

### Task 5: Refactor parallel rollout dataclass contracts
- [x] **任务完成**（与 superpowers plan `Task 5`、`tasks.md` 对应条目同步勾选）
- 目标：把 `parallel_weight_advantage_pretrain.py` 的 rollout task、epoch params、worker queue payload、worker result/error、transition record、metrics 和 summary 改成 dataclass。
- 改动文件：`FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py`、`FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py`
- 验证方式：`conda activate finetf && pytest FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py -q` 通过。

### Task 6: Run focused verification
- [x] **任务完成**（与 superpowers plan `Task 6`、`tasks.md` 对应条目同步勾选）
- 目标：运行 focused tests、相关模块语法检查和 OpenSpec strict validation，确认实现满足规格且没有扩大范围。
- 改动文件：`openspec/changes/refactor-rl-diagnostics-dataclasses/tasks.md`、`openspec/changes/refactor-rl-diagnostics-dataclasses/plan-ready.md`、`docs/superpowers/plans/2026-07-22-refactor-rl-diagnostics-dataclasses.md`
- 验证方式：运行 focused pytest、`conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/loss_nan_diagnostics.py FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py`、`openspec validate refactor-rl-diagnostics-dataclasses --strict`。
