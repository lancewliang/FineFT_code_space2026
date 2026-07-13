# 实现计划：refactor-commodity-feature-selection-union

## 来源
- 提案：openspec/changes/refactor-commodity-feature-selection-union/proposal.md
- 设计：openspec/changes/refactor-commodity-feature-selection-union/design.md
- 规格：openspec/changes/refactor-commodity-feature-selection-union/specs/
- 任务：openspec/changes/refactor-commodity-feature-selection-union/tasks.md

## 实现步骤

### Task 1: Candidate-only artifact tests
- [x] **任务完成**（与 superpowers plan `Task 1`、`tasks.md` 对应条目同步勾选）
- 目标：用聚焦 CLI 测试固化商品 IC candidate-only 模式只写候选 artifact，不写最终 `df.feather` 和标准 `state_features.npy`。
- 改动文件：`data_preprocess/tests/test_feature_selection_polars.py`。
- 验证方式：`conda run -n finetf pytest data_preprocess/tests/test_feature_selection_polars.py::test_ic_correlation_candidate_only_writes_candidate_artifacts -q`，新增测试先失败后通过。

### Task 2: Candidate-only implementation
- [x] **任务完成**（与 superpowers plan `Task 2`、`tasks.md` 对应条目同步勾选）
- 目标：在 `ic_correlation.py` 增加 candidate-only CLI 参数和输出分支，同时保持默认 IC 输出兼容。
- 改动文件：`data_preprocess/operator_futures/feature_selection/ic_correlation.py`、`data_preprocess/tests/test_feature_selection_polars.py`。
- 验证方式：运行 Task 1 新增测试和既有 `test_ic_correlation_cli_writes_expected_files`，确认 candidate-only 与默认输出都正确。

### Task 3: Union finalize happy-path tests
- [x] **任务完成**（与 superpowers plan `Task 3`、`tasks.md` 对应条目同步勾选）
- 目标：测试 union finalize 从合约 candidate 生成品种级 `FEATURE_UNION`，并为每个合约写出统一特征集的标准 `IC_RESULT`。
- 改动文件：`data_preprocess/tests/test_commodity_feature_pipeline.py`。
- 验证方式：`conda run -n finetf pytest data_preprocess/tests/test_commodity_feature_pipeline.py::test_write_contract_feature_union_finalizes_ic_result_from_candidates -q`，新增测试先失败后通过。

### Task 4: Union finalize implementation
- [x] **任务完成**（与 superpowers plan `Task 4`、`tasks.md` 对应条目同步勾选）
- 目标：扩展 `contract_feature_union.py`，支持读取 IC candidate、构建 union、校验所有合约列，并写出每合约过滤后的标准 `IC_RESULT`。
- 改动文件：`data_preprocess/operator_futures/feature_selection/contract_feature_union.py`、`data_preprocess/tests/test_commodity_feature_pipeline.py`。
- 验证方式：运行 Task 3 新增测试和现有 `test_write_contract_feature_union_writes_symbol_level_manifest`，确认旧默认行为和新 finalize 行为兼容。

### Task 5: Union finalize fail-fast tests
- [x] **任务完成**（与 superpowers plan `Task 5`、`tasks.md` 对应条目同步勾选）
- 目标：覆盖缺 candidate、空 union、union 特征在某合约 `ALL_FEATURE` 缺列时的 fail-fast 行为。
- 改动文件：`data_preprocess/tests/test_commodity_feature_pipeline.py`、`data_preprocess/operator_futures/feature_selection/contract_feature_union.py`。
- 验证方式：运行新增 fail-fast 测试，确认错误信息包含合约和缺失路径或缺失特征。

### Task 6: Commodity full-process shell ordering
- [x] **任务完成**（与 superpowers plan `Task 6`、`tasks.md` 对应条目同步勾选）
- 目标：调整 `fu_full_process.sh`，合约循环内运行 `ic_candidate`，循环后运行一次 `ic_union_finalize`，再按合约运行 `scale_save`，并更新 shell 测试。
- 改动文件：`data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`、`data_preprocess/tests/test_commodity_main_contract_cli.py`.
- 验证方式：运行 commodity shell 聚焦测试，确认日志阶段和调用顺序符合新流程，且旧独立 `feature_union` 阶段不再出现。

### Task 7: Pipeline artifact and manifest regression
- [x] **任务完成**（与 superpowers plan `Task 7`、`tasks.md` 对应条目同步勾选）
- 目标：更新商品 feature pipeline 或验证入口测试，确认新最终 artifact 布局、manifest 字段和 `validate_features.sh` 的检查语义一致。
- 改动文件：`data_preprocess/tests/test_commodity_feature_pipeline.py`、`data_preprocess/tests/test_commodity_main_contract_cli.py`，必要时更新 `data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh`。
- 验证方式：运行相关聚焦测试，确认 `FEATURE_UNION/state_features.npy`、`feature_union_manifest.json` 和每合约标准 `IC_RESULT` 均可被检查。

### Task 8: OpenSpec validation
- [x] **任务完成**（与 superpowers plan `Task 8`、`tasks.md` 对应条目同步勾选）
- 目标：验证 OpenSpec 变更本身仍然有效。
- 改动文件：无实现文件；只读取 `openspec/changes/refactor-commodity-feature-selection-union/*`。
- 验证方式：`openspec validate refactor-commodity-feature-selection-union --strict`，期望通过。

### Task 9: Focused pytest regression
- [x] **任务完成**（与 superpowers plan `Task 9`、`tasks.md` 对应条目同步勾选）
- 目标：运行 feature selection、商品 feature pipeline 和商品主合约 CLI 的聚焦回归测试。
- 改动文件：无新增改动；验证实现。
- 验证方式：`conda run -n finetf pytest data_preprocess/tests/test_feature_selection_polars.py data_preprocess/tests/test_commodity_feature_pipeline.py data_preprocess/tests/test_commodity_main_contract_cli.py -q`，期望通过。

### Task 10: Static shell assertions
- [x] **任务完成**（与 superpowers plan `Task 10`、`tasks.md` 对应条目同步勾选）
- 目标：用静态断言或已有 shell 测试确认 `fu_full_process.sh` 不再保留旧的 post-loop `feature_union` 阶段，且 `scale_save` 只在 `ic_union_finalize` 后执行。
- 改动文件：`data_preprocess/tests/test_commodity_main_contract_cli.py`。
- 验证方式：运行新增或更新后的 shell 静态测试，期望通过。
