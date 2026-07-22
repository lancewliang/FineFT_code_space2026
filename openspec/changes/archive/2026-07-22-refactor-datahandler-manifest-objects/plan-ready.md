# 实现计划：refactor-datahandler-manifest-objects

## 来源
- 提案：openspec/changes/refactor-datahandler-manifest-objects/proposal.md
- 设计：openspec/changes/refactor-datahandler-manifest-objects/design.md
- 规格：openspec/changes/refactor-datahandler-manifest-objects/specs/
- 任务：openspec/changes/refactor-datahandler-manifest-objects/tasks.md

## 实现步骤

### Task 1: Add focused datahandler manifest object tests
- [x] **任务完成**（与 superpowers plan `Task 1`、`tasks.md` 对应条目同步勾选）
- 目标：先用失败测试固定新的对象返回值、属性访问方式、`to_dict()` 与 JSON 文件兼容关系。
- 改动文件：`FineFT/tests/datahandler/test_commodity_contract_dataset.py`、`FineFT/tests/datahandler/test_slice_model.py`
- 验证方式：`conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py FineFT/tests/datahandler/test_slice_model.py`，预期在实现前因缺少 manifest dataclass 或返回类型不匹配而失败。

### Task 2: Add datahandler manifest dataclass models
- [x] **任务完成**（与 superpowers plan `Task 2`、`tasks.md` 对应条目同步勾选）
- 目标：新增 `FineFT/datahandler/manifests.py`，定义 split、dataset、slice manifest dataclass、`from_dict()`、更新方法、排序方法和 `to_dict()`。
- 改动文件：`FineFT/datahandler/manifests.py`
- 验证方式：`conda activate finetf && python -m py_compile FineFT/datahandler/manifests.py`，并运行 Task 1 中新增的对象模型测试。

### Task 3: Refactor commodity_contract_dataset.py to use manifest objects
- [x] **任务完成**（与 superpowers plan `Task 3`、`tasks.md` 对应条目同步勾选）
- 目标：让 `load_dataset_split_manifest()` 返回 `DatasetSplitManifest`，让 `build_dataset_manifest()`、`run_dataset_generation()` 返回 `DatasetManifest`，并让 stage 写出与 train slice 更新使用对象属性和方法。
- 改动文件：`FineFT/datahandler/commodity_contract_dataset.py`、`FineFT/tests/datahandler/test_commodity_contract_dataset.py`
- 验证方式：`conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py`，预期通过并保持现有错误场景。

### Task 4: Refactor slice_model.py to use SliceManifest
- [x] **任务完成**（与 superpowers plan `Task 4`、`tasks.md` 对应条目同步勾选）
- 目标：让 `slice_model.py` 的 manifest 读取、contract 更新、skip 更新、label 聚合、排序和 JSON 写出通过 `SliceManifest` 完成。
- 改动文件：`FineFT/datahandler/slice_model.py`、`FineFT/tests/datahandler/test_slice_model.py`
- 验证方式：`conda activate finetf && pytest FineFT/tests/datahandler/test_slice_model.py`，预期通过并保持 `slice_manifest.json` 结构兼容。

### Task 5: Run focused verification
- [x] **任务完成**（与 superpowers plan `Task 5`、`tasks.md` 对应条目同步勾选）
- 目标：运行 datahandler focused tests、Python 编译检查和 OpenSpec strict 校验，确认实现与规格一致。
- 改动文件：无代码改动；根据验证结果只在必要时修正前面任务引入的问题。
- 验证方式：`conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py FineFT/tests/datahandler/test_slice_model.py`、`conda activate finetf && python -m py_compile FineFT/datahandler/manifests.py FineFT/datahandler/commodity_contract_dataset.py FineFT/datahandler/slice_model.py`、`openspec validate refactor-datahandler-manifest-objects --strict` 全部通过。

## Amendments

### 2026-07-22: typed skipped contract records
- 原因：复查发现 dataset split/output 的 `skipped_contracts` 仍为 `list[dict]`，属于当前 manifest 对象化范围内的剩余结构化数据。
- 影响规格：`openspec/changes/refactor-datahandler-manifest-objects/specs/fineft-datahandler-manifest-objects/spec.md`
- 影响任务：`tasks.md` 条目 `- [ ] 1.6 Refactor dataset split/output skipped_contracts...`

## 追加实现步骤

### Task 6: Refactor dataset skipped contracts to dataclass records
- [x] **任务完成**（与 superpowers plan `Task 6`、`tasks.md` 对应条目同步勾选）
- 目标：新增 `DatasetSkippedContract`，让 `DatasetSplitSet.skipped_contracts` 和 `DatasetSetManifest.skipped_contracts` 使用对象列表表达，同时保持 `to_dict()` JSON payload 兼容。
- 改动文件：`FineFT/datahandler/manifests.py`、`FineFT/datahandler/commodity_contract_dataset.py`、`FineFT/tests/datahandler/test_commodity_contract_dataset.py`
- 验证方式：`conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py`、`conda activate finetf && python -m py_compile FineFT/datahandler/manifests.py FineFT/datahandler/commodity_contract_dataset.py`、`openspec validate refactor-datahandler-manifest-objects --strict` 全部通过。
