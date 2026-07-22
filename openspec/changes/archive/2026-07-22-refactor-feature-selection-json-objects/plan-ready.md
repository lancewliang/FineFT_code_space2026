# 实现计划：refactor-feature-selection-json-objects

## 来源
- 提案：openspec/changes/refactor-feature-selection-json-objects/proposal.md
- 设计：openspec/changes/refactor-feature-selection-json-objects/design.md
- 规格：openspec/changes/refactor-feature-selection-json-objects/specs/operator-futures-feature-selection-json-objects/spec.md
- 任务：openspec/changes/refactor-feature-selection-json-objects/tasks.md

## 实现步骤

### Task 1: Add focused tests for feature selection JSON dataclass objects
- [x] **任务完成**（与 superpowers plan `Task 1`、`tasks.md` 对应条目同步勾选）
- 目标：先用测试锁定 dataclass 返回值、属性访问和 `to_dict()` 与落盘 JSON 一致，覆盖 multi-contract feature selection、feature union、IC 和 Rank IC。
- 改动文件：`data_preprocess/tests/test_commodity_multi_contract_feature_selection.py`、`data_preprocess/tests/test_commodity_feature_pipeline.py`、`data_preprocess/tests/test_feature_selection_polars.py`
- 验证方式：`conda activate finetf && pytest data_preprocess/tests/test_commodity_multi_contract_feature_selection.py data_preprocess/tests/test_commodity_feature_pipeline.py data_preprocess/tests/test_feature_selection_polars.py -q` 应在实现前暴露旧返回值/缺失类型失败。

### Task 2: Add feature selection manifest dataclasses
- [x] **任务完成**（与 superpowers plan `Task 2`、`tasks.md` 对应条目同步勾选）
- 目标：新增 `manifests.py`，定义 `FeatureSelectionManifest`、`FeatureUnionManifest`、`FeatureScoreWindow` 和四个 result 对象，集中维护 JSON 契约。
- 改动文件：`data_preprocess/operator_futures/feature_selection/manifests.py`
- 验证方式：`conda activate finetf && python -m py_compile data_preprocess/operator_futures/feature_selection/manifests.py` 通过。

### Task 3: Refactor multi-contract feature selection manifest boundary
- [x] **任务完成**（与 superpowers plan `Task 3`、`tasks.md` 对应条目同步勾选）
- 目标：把 `muti_contract/pipeline.py` 的 train/valid manifest 拼装和返回值改为 dataclass，同时保持 `feature_selection_manifest.json` 结构兼容。
- 改动文件：`data_preprocess/operator_futures/feature_selection/muti_contract/pipeline.py`、`data_preprocess/tests/test_commodity_multi_contract_feature_selection.py`
- 验证方式：`conda activate finetf && pytest data_preprocess/tests/test_commodity_multi_contract_feature_selection.py -q` 通过。

### Task 4: Refactor contract feature union manifest boundary
- [x] **任务完成**（与 superpowers plan `Task 4`、`tasks.md` 对应条目同步勾选）
- 目标：把 `contract_feature_union.py` 的 manifest 拼装和返回值改为 dataclass，同时保持 `feature_union_manifest.json` 结构兼容。
- 改动文件：`data_preprocess/operator_futures/feature_selection/contract_feature_union.py`、`data_preprocess/tests/test_commodity_feature_pipeline.py`
- 验证方式：`conda activate finetf && pytest data_preprocess/tests/test_commodity_feature_pipeline.py -q` 通过。

### Task 5: Refactor IC and Rank IC JSON score boundaries
- [x] **任务完成**（与 superpowers plan `Task 5`、`tasks.md` 对应条目同步勾选）
- 目标：把 `ic_correlation.py` 和 `rank_ic_correlation.py` 的窗口分数 JSON 改为 `FeatureScoreWindow` 写出，并返回 result 对象；窗口 JSON 仍保持顶层 `{feature: score}`。
- 改动文件：`data_preprocess/operator_futures/feature_selection/ic_correlation.py`、`data_preprocess/operator_futures/feature_selection/rank_ic_correlation.py`、`data_preprocess/tests/test_feature_selection_polars.py`
- 验证方式：`conda activate finetf && pytest data_preprocess/tests/test_feature_selection_polars.py -q` 通过。

### Task 6: Run focused verification
- [x] **任务完成**（与 superpowers plan `Task 6`、`tasks.md` 对应条目同步勾选）
- 目标：运行 focused tests、相关模块语法检查和 OpenSpec strict validation，确认实现满足规格且没有扩大范围。
- 改动文件：`openspec/changes/refactor-feature-selection-json-objects/tasks.md`、`openspec/changes/refactor-feature-selection-json-objects/plan-ready.md`、`docs/superpowers/plans/2026-07-22-refactor-feature-selection-json-objects.md`
- 验证方式：运行 `conda activate finetf && pytest data_preprocess/tests/test_commodity_multi_contract_feature_selection.py data_preprocess/tests/test_commodity_feature_pipeline.py data_preprocess/tests/test_feature_selection_polars.py`、`conda activate finetf && python -m py_compile data_preprocess/operator_futures/feature_selection/manifests.py data_preprocess/operator_futures/feature_selection/muti_contract/pipeline.py data_preprocess/operator_futures/feature_selection/contract_feature_union.py data_preprocess/operator_futures/feature_selection/ic_correlation.py data_preprocess/operator_futures/feature_selection/rank_ic_correlation.py`、`openspec validate refactor-feature-selection-json-objects --strict`。
