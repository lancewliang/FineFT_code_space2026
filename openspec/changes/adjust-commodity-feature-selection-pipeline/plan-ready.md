# 实现计划：adjust-commodity-feature-selection-pipeline

## 来源
- 提案：openspec/changes/adjust-commodity-feature-selection-pipeline/proposal.md
- 设计：openspec/changes/adjust-commodity-feature-selection-pipeline/design.md
- 规格：openspec/changes/adjust-commodity-feature-selection-pipeline/specs/
- 任务：openspec/changes/adjust-commodity-feature-selection-pipeline/tasks.md

## Amendments

### 2026-07-18: 同步指标计算与筛选实现口径
- 原因：IC、RankIC、CatBoost Importance、多窗口计算和 Composite Score 筛选逻辑在实现中被修正，需要回写到 OpenSpec，避免规格遗漏关键行为。
- 影响规格：`openspec/changes/adjust-commodity-feature-selection-pipeline/specs/commodity-futures-support/spec.md`
- 影响任务：`tasks.md` 追加 1.7 和 2.4。

## 实现步骤

### Task 1: Add full-process ordering tests
- [x] **任务完成**（与 superpowers plan `Task 1`、`tasks.md` 对应条目同步勾选）
- 目标：更新商品 full-process focused shell tests，明确 `merge_clean -> dataset_split -> feature_selection_train -> feature_selection_valid -> scale_save -> maintenance_margin_dict` 顺序，并拒绝旧的 `merge_clean -> scale_save -> dataset_split` 顺序。
- 改动文件：`data_preprocess/tests/test_commodity_main_contract_cli.py`
- 验证方式：`bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest data_preprocess/tests/test_commodity_main_contract_cli.py::test_commodity_full_process_writes_step_logs_and_preserves_child_log_paths data_preprocess/tests/test_commodity_main_contract_cli.py::test_commodity_full_process_shell_runs_scale_after_feature_selection_valid -q'`，实现前预期失败。
- 对应 OpenSpec 任务：``- [ ] 1.1 Add focused tests for the split-after-merge-clean full-process order: all contracts run through `merge_clean`, `dataset_split` runs once, `feature_selection_train` then `feature_selection_valid` run once, per-contract `scale_save` runs after valid feature selection, and old immediate post-`merge_clean` scale-save ordering is rejected.``

### Task 2: Add multi-contract feature selection tests
- [x] **任务完成**（与 superpowers plan `Task 2`、`tasks.md` 对应条目同步勾选）
- 目标：新增 split 后多合约 feature selection 的 focused tests，覆盖 train candidate、valid candidate 限定、per-contract 明细、聚合统计、filtered `df.feather`、manifest 和 fail-fast。
- 改动文件：`data_preprocess/tests/test_commodity_multi_contract_feature_selection.py`
- 验证方式：`bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest data_preprocess/tests/test_commodity_multi_contract_feature_selection.py -q'`，实现前预期 collection 或 import 失败。
- 对应 OpenSpec 任务：``- [ ] 1.2 Add focused tests for a new multi-contract feature selection module covering train candidate output, valid candidate-restricted output, per-contract metric artifacts, aggregate `Mean` / `Std` / `Median` outputs, filtered contract `df.feather` outputs, manifest contents, and fail-fast behavior for missing input, empty candidate features, empty final features, and missing selected feature columns.``

### Task 3: Implement multi-contract feature selection module
- [x] **任务完成**（与 superpowers plan `Task 3`、`tasks.md` 对应条目同步勾选）
- 目标：实现 `data_preprocess/operator_futures/feature_selection/muti_contract/`，包含 metrics、aggregation、filters、manifest、filtered output 和 `--stage train|valid` CLI。
- 改动文件：`data_preprocess/operator_futures/feature_selection/muti_contract/__init__.py`、`data_preprocess/operator_futures/feature_selection/muti_contract/metrics.py`、`data_preprocess/operator_futures/feature_selection/muti_contract/pipeline.py`、`data_preprocess/operator_futures/feature_selection/muti_contract/__main__.py`
- 验证方式：`bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest data_preprocess/tests/test_commodity_multi_contract_feature_selection.py -q'`，预期通过。
- 对应 OpenSpec 任务：``- [ ] 1.3 Implement `data_preprocess/operator_futures/feature_selection/muti_contract/` with metric helpers for `Permutation Importance`, `CatBoost Importance`, `IC`, `RankIC`, `Sharpe`, aggregation helpers, ordered filters (`Hard Filter`, `Stability Filter`, `Composite Score`, `Correlation Filter`), manifest writing, and a CLI that supports `--stage train` and `--stage valid`.``

### Task 4: Update commodity full-process orchestration
- [x] **任务完成**（与 superpowers plan `Task 4`、`tasks.md` 对应条目同步勾选）
- 目标：修改 `fu_full_process.sh`，让 `dataset_split` 读取 `ALL_FEATURE`，新增 train/valid feature selection 调度，valid feature selection 后再按合约执行 `scale_save`。
- 改动文件：`data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`
- 验证方式：`bash -n data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh` 和 Task 1 focused pytest，预期通过。
- 对应 OpenSpec 任务：``- [ ] 1.4 Update `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh` so dataset split reads `ALL_FEATURE`, feature selection runs after dataset split, scale-save runs after valid feature selection, and step logs include `feature_selection_train` and `feature_selection_valid`.``

### Task 5: Add scale-save filtered input routing
- [x] **任务完成**（与 superpowers plan `Task 5`、`tasks.md` 对应条目同步勾选）
- 目标：扩展 `scale_save.py` 的输入解析，让商品 full process 可以读取 filtered `FEATURE_SELECTION/{target_freq}/{symbol}/valid/{contract}/df.feather` 和最终 `state_features.npy`，同时保持旧 `IC_RESULT` 行为。
- 改动文件：`data_preprocess/operator_futures/scale_describe_save/scale_save.py`、`data_preprocess/tests/test_feature_selection_polars.py`
- 验证方式：`bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest data_preprocess/tests/test_feature_selection_polars.py::test_scale_save_cli_reads_feature_selection_filtered_input data_preprocess/tests/test_feature_selection_polars.py::test_scale_save_cli_writes_expected_files -q'`，预期通过。
- 对应 OpenSpec 任务：``- [ ] 1.5 Update `data_preprocess/operator_futures/scale_describe_save/scale_save.py` routing so commodity full process can read filtered `FEATURE_SELECTION/{target_freq}/{symbol}/valid/{contract}/df.feather` and matching final `state_features.npy`, while preserving existing `IC_RESULT` behavior for old callers.``

### Task 6: Update commodity pipeline documentation
- [x] **任务完成**（与 superpowers plan `Task 6`、`tasks.md` 对应条目同步勾选）
- 目标：更新商品预处理文档，说明新顺序和 `FEATURE_SELECTION/{target_freq}` 产物边界。
- 改动文件：`docs/datahandler/data_preparation_analysis.zh_cn.md`、`docs/上海商品交易所/commodity_futures_preprocess.md`
- 验证方式：`rg -n "FEATURE_SELECTION|feature_selection_train|feature_selection_valid|dataset_split.*feature_selection|SPLIT-TRAIN-VALID-TEST" docs/datahandler/data_preparation_analysis.zh_cn.md docs/上海商品交易所/commodity_futures_preprocess.md`，预期包含新流程描述。
- 对应 OpenSpec 任务：``- [ ] 1.6 Update focused documentation for the commodity preprocessing pipeline to describe `dataset_split -> feature_selection(train) -> feature_selection(valid) -> scale_save -> maintenance_margin_dict` and the `FEATURE_SELECTION/{target_freq}` artifact layout.``

### Task 10: Amend metric and filter semantics in OpenSpec
- [x] **任务完成**（与 superpowers plan `Task 10`、`tasks.md` 对应条目同步勾选）
- 目标：把已实现的特征指标和筛选细节回写到 OpenSpec，包括默认多窗口 `[1,6,12]`、IC/RankIC/CatBoost Importance 原始脚本兼容口径、Sharpe 和 Permutation Importance 公式、Composite Score 优先级、底部 10% 删除和 manifest 字段。
- 改动文件：`openspec/changes/adjust-commodity-feature-selection-pipeline/proposal.md`、`openspec/changes/adjust-commodity-feature-selection-pipeline/design.md`、`openspec/changes/adjust-commodity-feature-selection-pipeline/specs/commodity-futures-support/spec.md`、`openspec/changes/adjust-commodity-feature-selection-pipeline/tasks.md`、`openspec/changes/adjust-commodity-feature-selection-pipeline/plan-ready.md`、`docs/superpowers/plans/2026-07-18-adjust-commodity-feature-selection-pipeline.md`
- 验证方式：`openspec validate adjust-commodity-feature-selection-pipeline --strict`，预期通过。
- 对应 OpenSpec 任务：``- [x] 1.7 Amend OpenSpec artifacts to document the implemented feature metric semantics and filter semantics: default `windows_list=[1,6,12]`, original-compatible IC/RankIC/CatBoost Importance, Sharpe and Permutation Importance formulas, Composite Score priority order, bottom 10% composite drop, and manifest fields.``

### Task 7: Run OpenSpec validation
- [x] **任务完成**（与 superpowers plan `Task 7`、`tasks.md` 对应条目同步勾选）
- 目标：确认 OpenSpec change 仍符合 strict validation。
- 改动文件：无代码改动；只在 build 阶段完成后同步 checkbox。
- 验证方式：`openspec validate adjust-commodity-feature-selection-pipeline --strict`，预期通过。
- 对应 OpenSpec 任务：``- [ ] 2.1 Run strict OpenSpec validation for `adjust-commodity-feature-selection-pipeline`.``

### Task 8: Run focused pytest validation
- [x] **任务完成**（与 superpowers plan `Task 8`、`tasks.md` 对应条目同步勾选）
- 目标：运行本变更涉及的 shell、multi-contract feature selection、scale-save focused tests。
- 改动文件：无代码改动；只在 build 阶段完成后同步 checkbox。
- 验证方式：`bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest data_preprocess/tests/test_commodity_main_contract_cli.py data_preprocess/tests/test_commodity_multi_contract_feature_selection.py data_preprocess/tests/test_feature_selection_polars.py -q'`，预期通过或只暴露与本变更无关的既有失败。
- 对应 OpenSpec 任务：``- [ ] 2.2 Run focused pytest commands with `conda activate finetf` for commodity full-process shell tests, multi-contract feature selection tests, and scale-save routing tests.``

### Task 9: Run static syntax validation
- [x] **任务完成**（与 superpowers plan `Task 9`、`tasks.md` 对应条目同步勾选）
- 目标：运行 shell syntax 和 Python compile 检查，确认新模块与修改脚本语法正确。
- 改动文件：无代码改动；只在 build 阶段完成后同步 checkbox。
- 验证方式：`bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && bash -n data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh && python -m py_compile data_preprocess/operator_futures/feature_selection/muti_contract/*.py data_preprocess/operator_futures/scale_describe_save/scale_save.py'`，预期通过。
- 对应 OpenSpec 任务：``- [ ] 2.3 Run `bash -n` on changed shell scripts and `python -m py_compile` on changed Python modules with `conda activate finetf`.``

### Task 11: Re-run OpenSpec validation after amend
- [x] **任务完成**（与 superpowers plan `Task 11`、`tasks.md` 对应条目同步勾选）
- 目标：确认补充后的 OpenSpec 仍通过 strict validation。
- 改动文件：无代码改动；只在 amend 完成后同步 checkbox。
- 验证方式：`openspec validate adjust-commodity-feature-selection-pipeline --strict`，预期通过。
- 对应 OpenSpec 任务：``- [x] 2.4 Re-run strict OpenSpec validation after metric/filter semantics amend.``
