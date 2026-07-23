# 实现计划：refactor-multi-contract-scale-save-robust-scaler

## 来源
- 提案：openspec/changes/refactor-multi-contract-scale-save-robust-scaler/proposal.md
- 设计：openspec/changes/refactor-multi-contract-scale-save-robust-scaler/design.md
- 规格：openspec/changes/refactor-multi-contract-scale-save-robust-scaler/specs/
- 任务：openspec/changes/refactor-multi-contract-scale-save-robust-scaler/tasks.md

## 实现步骤

### Task 1: Add robust scale-save regression tests
- [x] **任务完成**
- 目标：先写出能抓住旧问题的回归测试，覆盖 train-only fit、统一 apply、clip、manifest/diagnostics，以及 fail-fast 行为。
- 改动文件：`data_preprocess/tests/test_feature_selection_polars.py`
- 验证方式：先运行单个新增测试确认失败，再在实现后用 `conda activate finetf && pytest data_preprocess/tests/test_feature_selection_polars.py -k "multi_contract_scale_save or robust_scaler" -q` 确认通过。

### Task 2: Implement train-only robust scaler in multi-contract scale save
- [x] **任务完成**
- 目标：把 `muti_contract_scale_save.py` 改成 train-only robust scaler 的唯一 commodity split-stage 入口，保留输出目录结构和 debug CSV。
- 改动文件：`data_preprocess/operator_futures/scale_describe_save/muti_contract_scale_save.py`
- 验证方式：运行更新后的 focused tests、检查 `SCALE_SAVE/fu/5min/scaler_manifest.json` 和 `SCALE_SAVE/fu/5min/scale_diagnostics.csv`，并确认 `wap_1`/`awap` 不再出现 per-file 10x 跳档。

### Task 3: Validate spec and Python artifacts
- [x] **任务完成**
- 目标：对变更做最终校验，确保 spec、tasks、plan-ready 和 Python 代码契约一致。
- 改动文件：无代码文件；验证 `openspec/changes/refactor-multi-contract-scale-save-robust-scaler/{proposal.md,design.md,specs/,tasks.md,plan-ready.md}` 与 `data_preprocess/operator_futures/scale_describe_save/muti_contract_scale_save.py`
- 验证方式：运行 `conda activate finetf && python -m py_compile data_preprocess/operator_futures/scale_describe_save/muti_contract_scale_save.py data_preprocess/tests/test_feature_selection_polars.py && openspec validate refactor-multi-contract-scale-save-robust-scaler --strict`。
