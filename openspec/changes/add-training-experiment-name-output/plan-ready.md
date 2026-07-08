# 实现计划：add-training-experiment-name-output

## 来源
- 提案：openspec/changes/add-training-experiment-name-output/proposal.md
- 设计：无（OpenSpec 判定无需）
- 规格：openspec/changes/add-training-experiment-name-output/specs/fineft-stage-i-pretrain/spec.md
- 任务：openspec/changes/add-training-experiment-name-output/tasks.md

## 实现步骤

### Task 1: Experiment-name output isolation
- [x] **任务完成**（与 superpowers plan `Task 1`、`tasks.md` 对应条目同步勾选）
- 目标：为串行 `weight_advantage_pretrain.py` 增加默认实验名输出层级，使模型、TensorBoard、qtable diagnostics 和训练日志按 `<dataset_name>/<experiment_name>` 分开保存，同时保持输入路径仍由 `base_path/dataset_name` 控制。
- 改动文件：`FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`、`FineFT/tests/rl/test_weight_advantage_pretrain_logging.py`、`FineFT/script/train/train_commodity_fu.sh`、`FineFT/script/train/train_commodity_al.sh`。
- 验证方式：运行 `conda activate finetf && pytest FineFT/tests/rl/test_weight_advantage_pretrain_logging.py -q`；运行 `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`；运行 `bash -n FineFT/script/train/train_commodity_fu.sh FineFT/script/train/train_commodity_al.sh`；运行 `openspec validate add-training-experiment-name-output --strict`。
