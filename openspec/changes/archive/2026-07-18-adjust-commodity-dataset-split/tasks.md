# Tasks

## 1. Implementation

- [x] 1.1 Add focused tests for `operator_futures.dataset_split.dataset_split` covering boundary calculation, contract/date intersections, skipped sets, all-column preservation, merged outputs, manifest row counts, and fail-fast cases. <!-- 已实现: 添加 dataset split focused tests 并确认 RED 失败 -->
- [x] 1.2 Implement `data_preprocess/operator_futures/dataset_split/dataset_split.py` with CLI arguments, contract-level stage writing, top-level vertical concatenation, and `dataset_split_manifest.json`. <!-- 已实现: 新增 dataset_split operator 并通过 focused tests -->
- [x] 1.3 Add `data_preprocess/script_preprocess/future_upgraded/9_dataset_split/dataset_split.sh` that activates `finetf` and calls `operator_futures.dataset_split.dataset_split`. <!-- 已实现: 新增第 9 阶段 shell wrapper 并通过 bash -n -->
- [x] 1.4 Update `fu_full_process.sh` tests to reject old `ic_candidate` / `ic_union_finalize` functions and steps, require `scale_save` inside the contract loop after `merge_clean`, and require one post-loop `dataset_split`. <!-- 已实现: 更新 full process 调度测试并确认 RED 失败 -->
- [x] 1.5 Update `fu_full_process.sh` to remove old IC candidate/union functions and steps, run `scale_save` after each contract `merge_clean`, and run `dataset_split` once before `maintenance_margin_dict`. <!-- 已实现: 调整 full process 调度并通过 focused shell tests -->
- [x] 1.6 Update focused documentation to reflect the ninth dataset split stage and merged `train.feather`、`valid.feather`、`test.feather` outputs. <!-- 已实现: 更新商品数据准备文档 -->

## 2. Validation

- [x] 2.1 Run strict OpenSpec validation, focused pytest commands with `conda activate finetf`, and `bash -n` on changed shell scripts. <!-- 已实现: 所有验证命令通过 -->
