# Tasks

## 1. Implementation

- [x] 1.1 Add focused tests for the split-after-merge-clean full-process order: all contracts run through `merge_clean`, `dataset_split` runs once, `feature_selection_train` then `feature_selection_valid` run once, per-contract `scale_save` runs after valid feature selection, and old immediate post-`merge_clean` scale-save ordering is rejected. <!-- 已实现: 更新 full-process 顺序测试并确认 RED/GREEN -->
- [x] 1.2 Add focused tests for a new multi-contract feature selection module covering train candidate output, valid candidate-restricted output, per-contract metric artifacts, aggregate `Mean` / `Std` / `Median` outputs, filtered contract `df.feather` outputs, manifest contents, and fail-fast behavior for missing input, empty candidate features, empty final features, and missing selected feature columns. <!-- 已实现: 新增 split 后 feature selection focused tests 并确认 RED/GREEN -->
- [x] 1.3 Implement `data_preprocess/operator_futures/feature_selection/muti_contract/` with metric helpers for `Permutation Importance`, `CatBoost Importance`, `IC`, `RankIC`, `Sharpe`, aggregation helpers, ordered filters (`Hard Filter`, `Stability Filter`, `Composite Score`, `Correlation Filter`), manifest writing, and a CLI that supports `--stage train` and `--stage valid`. <!-- 已实现: 新增 muti_contract 模块并通过 focused tests -->
- [x] 1.4 Update `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh` so dataset split reads `ALL_FEATURE`, feature selection runs after dataset split, scale-save runs after valid feature selection, and step logs include `feature_selection_train` and `feature_selection_valid`. <!-- 已实现: 重排商品 full-process 并通过 shell tests -->
- [x] 1.5 Update `data_preprocess/operator_futures/scale_describe_save/scale_save.py` routing so commodity full process can read filtered `FEATURE_SELECTION/{target_freq}/{symbol}/valid/{contract}/df.feather` and matching final `state_features.npy`, while preserving existing `IC_RESULT` behavior for old callers. <!-- 已实现: 增加 feature_selection_stage 路由并保留旧 IC_RESULT 行为 -->
- [x] 1.6 Update focused documentation for the commodity preprocessing pipeline to describe `dataset_split -> feature_selection(train) -> feature_selection(valid) -> scale_save -> maintenance_margin_dict` and the `FEATURE_SELECTION/{target_freq}` artifact layout. <!-- 已实现: 更新商品预处理文档与流程说明 -->
- [x] 1.7 Amend OpenSpec artifacts to document the implemented feature metric semantics and filter semantics: default `windows_list=[1,6,12]`, original-compatible IC/RankIC/CatBoost Importance, Sharpe and Permutation Importance formulas, Composite Score priority order, bottom 10% composite drop, and manifest fields. <!-- 已实现: 将实现口径回写到 proposal/design/spec/plan -->

## 2. Validation

- [x] 2.1 Run strict OpenSpec validation for `adjust-commodity-feature-selection-pipeline`. <!-- 已实现: openspec validate --strict 通过 -->
- [x] 2.2 Run focused pytest commands with `conda activate finetf` for commodity full-process shell tests, multi-contract feature selection tests, and scale-save routing tests. <!-- 已实现: 46 个 focused tests 通过 -->
- [x] 2.3 Run `bash -n` on changed shell scripts and `python -m py_compile` on changed Python modules with `conda activate finetf`. <!-- 已实现: shell syntax 与 py_compile 通过 -->
- [x] 2.4 Re-run strict OpenSpec validation after metric/filter semantics amend. <!-- 已实现: amend 后 openspec validate --strict 通过 -->
