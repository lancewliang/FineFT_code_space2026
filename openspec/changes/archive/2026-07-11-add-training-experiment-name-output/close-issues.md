# Close Issues: add-training-experiment-name-output

## 验证日期

2026-07-11

## 阶段 1：实现验证

通过：

- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest FineFT/tests/rl/test_weight_advantage_pretrain_logging.py -q`
  - 结果：`19 passed in 3.04s`
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`
  - 结果：exit code 0
- `bash -n FineFT/script/train/train_commodity_fu_5.sh FineFT/script/train/train_commodity_fu_10.sh FineFT/script/train/train_commodity_al_5.sh FineFT/script/train/train_commodity_al_10.sh`
  - 结果：exit code 0
- `openspec validate add-training-experiment-name-output --strict`
  - 结果：`Change 'add-training-experiment-name-output' is valid`
- `git diff --check -- FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py FineFT/tests/rl/test_weight_advantage_pretrain_logging.py FineFT/script/train/train_commodity_fu_5.sh FineFT/script/train/train_commodity_fu_10.sh FineFT/script/train/train_commodity_al_5.sh FineFT/script/train/train_commodity_al_10.sh openspec/changes/add-training-experiment-name-output docs/superpowers/plans/2026-07-08-add-training-experiment-name-output.md`
  - 结果：exit code 0

目录级测试限制：

- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest FineFT/tests -q`
  - 结果：collection 阶段失败；`FineFT/tests/env/test_env.py` 在 import 时读取缺失的外部数据文件 `/data2/mlqin/HFT4Ind2/dataset/BNBUSDT/train/df_0.feather`。

判断：目录级测试失败发生在 collection 阶段，原因与本变更的实验名输出隔离无直接关系。本变更的 focused tests、编译、shell 语法和 OpenSpec strict validation 均通过。

## 阶段 3：规格一致性验证

### Completeness

- `openspec/changes/add-training-experiment-name-output/tasks.md` 所有任务 checkbox 均为 `[x]`。
- `openspec/changes/add-training-experiment-name-output/plan-ready.md` 的任务完成 checkbox 为 `[x]`。
- `docs/superpowers/plans/2026-07-08-add-training-experiment-name-output.md` 的 Step 和 Task complete checkbox 均为 `[x]`。
- 实现证据：
  - `FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py` 包含 `--experiment_name`、`build_serial_model_path(...)`、`build_training_data_paths(...)` 和 `build_train_log_path(...)`。
  - `FineFT/tests/rl/test_weight_advantage_pretrain_logging.py` 覆盖默认实验名、显式实验名、输出路径和输入路径不变语义。
  - `FineFT/script/train/train_commodity_fu_5.sh`、`FineFT/script/train/train_commodity_fu_10.sh`、`FineFT/script/train/train_commodity_al_5.sh` 与 `FineFT/script/train/train_commodity_al_10.sh` 传递 `--experiment_name "${EXPERIMENT_NAME}"`。

### Correctness

- 默认实验名：测试覆盖 `parser.parse_args([]).experiment_name == "default"`。
- 显式实验名：测试覆盖 `--experiment_name 5min_gamma097`。
- 输入路径不变：测试覆盖 `dataset_5min/fu/train`、`state_features.npy` 和 `maintenance_margin_ratio_dict.npy` 仍由 `base_path/dataset_name` 拼接。
- 模型输出路径：测试覆盖 `<result_path>/<dataset_name>/<experiment_name>/weights_advantage_pretrain`。
- 训练日志路径：测试覆盖 `log_futures/<dataset_name>/low_level/train/<experiment_name>/advantage.log`。
- 串行 shell：语法检查通过，且四个实际串行商品训练脚本均传递 `--experiment_name`。

### Coherence

- 本变更无 `design.md`，符合 OpenSpec 可选规则。
- 实现保持既有串行训练脚本结构，仅新增小型路径 helper 和 CLI 参数。
- 未新增输入数据目录参数，符合用户缩小后的需求。
- 未在 close 阶段修改代码。

## 代码审查

未执行额外子代理代码审查。当前 close 依据 focused tests、实现对照、OpenSpec strict validation 和 diff 检查完成验证。

## 阶段 4：归档验证

通过：

- `openspec archive add-training-experiment-name-output --yes`
  - 结果：同步更新 `openspec/specs/fineft-stage-i-pretrain/spec.md`，并归档为 `openspec/changes/archive/2026-07-11-add-training-experiment-name-output/`。
- `openspec validate --all --strict`
  - 结果：6 passed, 0 failed。

## 阶段 5：开发分支收尾

未执行 merge / PR / branch cleanup。`finishing-a-development-branch` 要求完整测试套件通过后才能进入分支收尾；当前 `pytest FineFT/tests -q` 在 collection 阶段因缺失外部数据文件失败。

## 结论

无 CRITICAL 问题。

WARNING：

- `FineFT/tests` 目录级测试无法在当前环境完整运行，原因是 collection 阶段存在外部数据文件依赖问题。该问题记录为环境/既有测试限制，不阻塞本变更 focused close。
