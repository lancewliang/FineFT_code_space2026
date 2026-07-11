## 1. Implementation

- [x] 1.0 Complete experiment-name output isolation for serial Stage I training. <!-- 已实现: 串行 Stage I 输出按 experiment_name 隔离，并完成验证 -->
- [x] 1.1 Add `--experiment_name` to `FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py` with default `default`. <!-- 已实现 -->
- [x] 1.2 Update serial training model/TensorBoard output path to include `<dataset_name>/<experiment_name>/weights_advantage_pretrain` while preserving input path logic. <!-- 已实现 -->
- [x] 1.3 Update serial training file logger to include `<experiment_name>` under `log_futures/<dataset_name>/low_level/train/`. <!-- 已实现 -->
- [x] 1.4 Update serial commodity training shell scripts to pass `--experiment_name` and redirect stdout to the experiment-specific log path. <!-- 已实现 -->
- [x] 1.5 Add focused tests for CLI default, explicit experiment name, output path construction, logger path construction, and unchanged input path semantics. <!-- 已实现 -->
- [x] 1.6 Run `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`. <!-- 已验证 -->
- [x] 1.7 Run focused pytest for `FineFT/tests/rl/test_weight_advantage_pretrain_logging.py`. <!-- 已验证 -->
- [x] 1.8 Run `bash -n` for modified serial training shell scripts. <!-- 已验证 -->
- [x] 1.9 Run `openspec validate add-training-experiment-name-output --strict`. <!-- 已验证 -->
