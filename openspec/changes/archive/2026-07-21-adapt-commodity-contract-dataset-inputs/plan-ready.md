# 实现计划：adapt-commodity-contract-dataset-inputs

## 来源
- 提案：openspec/changes/adapt-commodity-contract-dataset-inputs/proposal.md
- 设计：无（OpenSpec 判定无需）
- 规格：openspec/changes/adapt-commodity-contract-dataset-inputs/specs/
- 任务：openspec/changes/adapt-commodity-contract-dataset-inputs/tasks.md

## 实现步骤

### Task 1: Update commodity contract dataset tests for new input contract
- [x] **任务完成**（与 superpowers plan `Task 1`、`tasks.md` 对应条目同步勾选）
- 目标：用测试定义新的 `dataset_split_manifest.json` 输入、阶段化 `SCALE_SAVE` 路径、`--state_features_path`、`{contract}.feather` 输出和 fail-fast 行为。
- 改动文件：`FineFT/tests/datahandler/test_commodity_contract_dataset.py`
- 验证方式：`conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py -q`，预期先失败，失败点指向旧 API 或旧路径。

### Task 2: Refactor commodity_contract_dataset main path
- [x] **任务完成**（与 superpowers plan `Task 2`、`tasks.md` 对应条目同步勾选）
- 目标：让 `FineFT/datahandler/commodity_contract_dataset.py` 读取 `dataset_split_manifest.json`，从阶段化 `SCALE_SAVE` 复制数据，复制 `--state_features_path`，并移除主路径中的 split boundary/date filtering。
- 改动文件：`FineFT/datahandler/commodity_contract_dataset.py`
- 验证方式：`conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py -q`，预期新输入契约相关测试通过或仅剩 train slice/script 测试失败。

### Task 3: Keep train slices working from contract-named files
- [x] **任务完成**（与 superpowers plan `Task 3`、`tasks.md` 对应条目同步勾选）
- 目标：确认 `train/{contract}.feather` 作为 train slice 来源，slice 连续编号、短 slice 和 manifest row count 行为保持正确。
- 改动文件：`FineFT/datahandler/commodity_contract_dataset.py`、`FineFT/tests/datahandler/test_commodity_contract_dataset.py`
- 验证方式：`conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py::test_write_train_slices_uses_contiguous_indices_and_single_contract_files -q`。

### Task 4: Update commodity data handler scripts
- [x] **任务完成**（与 superpowers plan `Task 4`、`tasks.md` 对应条目同步勾选）
- 目标：更新燃料油和铝 FineFT data handler，使其传入 `--dataset_split_manifest_path` 与 `--state_features_path`，并扫描 `valid/*.feather` 调用 `slice_model.py`。
- 改动文件：`FineFT/script/data/commodity_data_handler_fu.sh`、`FineFT/script/data/commodity_data_handler_al.sh`
- 验证方式：`conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py::test_commodity_data_handler_scripts_use_contract_dataset_tool -q`。

### Task 5: Remove old path assertions from FineFT commodity tests
- [x] **任务完成**（与 superpowers plan `Task 5`、`tasks.md` 对应条目同步勾选）
- 目标：清理或替换仍断言 `df_<contract>.feather`、`--summary_path`、`--feature_union_path`、valid `df_*.feather` 的 FineFT 商品数据集测试。
- 改动文件：`FineFT/tests/datahandler/test_commodity_contract_dataset.py`
- 验证方式：`conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py -q`。

### Task 6: Run commodity contract dataset test suite
- [x] **任务完成**（与 superpowers plan `Task 6`、`tasks.md` 对应条目同步勾选）
- 目标：验证 FineFT 商品多合约 dataset 工具、脚本契约和 train slice 行为整体通过。
- 改动文件：无代码改动；执行验证命令。
- 验证方式：`conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py -q`。

### Task 7: Validate OpenSpec change
- [x] **任务完成**（与 superpowers plan `Task 7`、`tasks.md` 对应条目同步勾选）
- 目标：确认规格仍通过 OpenSpec 严格校验。
- 改动文件：无代码改动；执行验证命令。
- 验证方式：`openspec validate adapt-commodity-contract-dataset-inputs --strict`。
