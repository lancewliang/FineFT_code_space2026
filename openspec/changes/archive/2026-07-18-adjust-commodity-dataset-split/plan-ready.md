# 实现计划：adjust-commodity-dataset-split

## 来源
- 提案：openspec/changes/adjust-commodity-dataset-split/proposal.md
- 设计：openspec/changes/adjust-commodity-dataset-split/design.md
- 规格：openspec/changes/adjust-commodity-dataset-split/specs/
- 任务：openspec/changes/adjust-commodity-dataset-split/tasks.md

## 实现步骤

### Task 1: Add dataset split tests
- [x] **任务完成**（与 superpowers plan `Task 1`、`tasks.md` 对应条目同步勾选）
- 目标：新增 focused tests，覆盖第 9 阶段 dataset split 的边界计算、合约交集、跳过集合、所有列保留、合约级输出、顶层 merged 输出、manifest 行数和 fail-fast。
- 改动文件：`data_preprocess/tests/test_commodity_dataset_split.py`
- 验证方式：`conda activate finetf && pytest data_preprocess/tests/test_commodity_dataset_split.py -q`，预期在实现前因 module 不存在或函数未实现失败。
- 对应 OpenSpec 任务：``- [ ] 1.1 Add focused tests for `operator_futures.dataset_split.dataset_split` covering boundary calculation, contract/date intersections, skipped sets, all-column preservation, merged outputs, manifest row counts, and fail-fast cases.``

### Task 2: Implement dataset split operator
- [x] **任务完成**（与 superpowers plan `Task 2`、`tasks.md` 对应条目同步勾选）
- 目标：新增 `operator_futures.dataset_split.dataset_split`，实现 CLI、5:3:2 时间边界、合约级 stage feather、顶层纵向合并 feather 和 `dataset_split_manifest.json`。
- 改动文件：`data_preprocess/operator_futures/dataset_split/__init__.py`、`data_preprocess/operator_futures/dataset_split/dataset_split.py`
- 验证方式：`conda activate finetf && pytest data_preprocess/tests/test_commodity_dataset_split.py -q`，预期通过。
- 对应 OpenSpec 任务：``- [ ] 1.2 Implement `data_preprocess/operator_futures/dataset_split/dataset_split.py` with CLI arguments, contract-level stage writing, top-level vertical concatenation, and `dataset_split_manifest.json`.``

### Task 3: Add ninth-stage shell wrapper
- [x] **任务完成**（与 superpowers plan `Task 3`、`tasks.md` 对应条目同步勾选）
- 目标：新增 `future_upgraded/9_dataset_split/dataset_split.sh`，负责激活 `finetf` 并调用 Python module。
- 改动文件：`data_preprocess/script_preprocess/future_upgraded/9_dataset_split/dataset_split.sh`
- 验证方式：`bash -n data_preprocess/script_preprocess/future_upgraded/9_dataset_split/dataset_split.sh`。
- 对应 OpenSpec 任务：``- [ ] 1.3 Add `data_preprocess/script_preprocess/future_upgraded/9_dataset_split/dataset_split.sh` that activates `finetf` and calls `operator_futures.dataset_split.dataset_split`.``

### Task 4: Update full process orchestration tests
- [x] **任务完成**（与 superpowers plan `Task 4`、`tasks.md` 对应条目同步勾选）
- 目标：修改现有 shell tests，要求旧 IC candidate/union 步骤消失，`scale_save` 位于合约循环内 `merge_clean` 后，`dataset_split` 在所有合约完成后只执行一次。
- 改动文件：`data_preprocess/tests/test_commodity_main_contract_cli.py`
- 验证方式：`conda activate finetf && pytest data_preprocess/tests/test_commodity_main_contract_cli.py::test_commodity_full_process_shell_exposes_expected_functions data_preprocess/tests/test_commodity_main_contract_cli.py::test_commodity_full_process_writes_step_logs_and_preserves_child_log_paths data_preprocess/tests/test_commodity_main_contract_cli.py::test_commodity_full_process_shell_runs_scale_after_merge_clean_and_dataset_split_after_loop -q`，预期实现前失败。
- 对应 OpenSpec 任务：``- [ ] 1.4 Update `fu_full_process.sh` tests to reject old `ic_candidate` / `ic_union_finalize` functions and steps, require `scale_save` inside the contract loop after `merge_clean`, and require one post-loop `dataset_split`.``

### Task 5: Update fu full process orchestration
- [x] **任务完成**（与 superpowers plan `Task 5`、`tasks.md` 对应条目同步勾选）
- 目标：修改 `fu_full_process.sh`，删除旧函数和步骤，合约循环内 `merge_clean -> scale_save`，循环后 `dataset_split -> maintenance_margin_dict`。
- 改动文件：`data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`
- 验证方式：`bash -n data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh` 和 Task 4 的 focused pytest，预期通过。
- 对应 OpenSpec 任务：``- [ ] 1.5 Update `fu_full_process.sh` to remove old IC candidate/union functions and steps, run `scale_save` after each contract `merge_clean`, and run `dataset_split` once before `maintenance_margin_dict`.``

### Task 6: Update focused documentation
- [x] **任务完成**（与 superpowers plan `Task 6`、`tasks.md` 对应条目同步勾选）
- 目标：更新商品数据准备文档，说明第 9 阶段读取 `SCALE_SAVE`、输出合约级目录并额外生成顶层 merged feather；移除“不生成 train.feather/valid.feather/test.feather”的旧描述。
- 改动文件：`docs/datahandler/data_preparation_analysis.zh_cn.md`
- 验证方式：`rg -n "9_dataset_split|dataset_split|train\\.feather|valid\\.feather|test\\.feather" docs/datahandler/data_preparation_analysis.zh_cn.md`，预期包含新阶段和 merged 输出说明。
- 对应 OpenSpec 任务：``- [ ] 1.6 Update focused documentation to reflect the ninth dataset split stage and merged `train.feather`、`valid.feather`、`test.feather` outputs.``

### Task 7: Run validation
- [x] **任务完成**（与 superpowers plan `Task 7`、`tasks.md` 对应条目同步勾选）
- 目标：运行 OpenSpec、pytest 和 shell syntax validation，确认本变更可交付。
- 改动文件：无代码改动；更新执行记录时只勾选计划 checkbox。
- 验证方式：`openspec validate adjust-commodity-dataset-split --strict`；`conda activate finetf && pytest data_preprocess/tests/test_commodity_dataset_split.py data_preprocess/tests/test_commodity_main_contract_cli.py -q`；`bash -n data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh data_preprocess/script_preprocess/future_upgraded/9_dataset_split/dataset_split.sh`。
- 对应 OpenSpec 任务：``- [ ] 2.1 Run strict OpenSpec validation, focused pytest commands with `conda activate finetf`, and `bash -n` on changed shell scripts.``
