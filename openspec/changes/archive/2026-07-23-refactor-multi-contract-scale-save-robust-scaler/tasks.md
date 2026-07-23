# Tasks

## 1. Implementation

- [x] 1.1 Add focused regression tests for train-only robust scaling in `data_preprocess/tests/test_feature_selection_polars.py`, covering train-fit once, consistent apply across train/valid/test, clip behavior, manifest/diagnostics files, and fail-fast behavior for missing train inputs, invalid clip bounds, and missing selected feature columns. <!-- 已实现: 新增 robust scaler RED 回归测试并确认旧实现失败 -->
- [x] 1.2 Implement the train-only robust scaler path in `data_preprocess/operator_futures/scale_describe_save/muti_contract_scale_save.py`, including train split fit, reusable manifest generation, apply-to-all-splits output writing, same-basename CSV debug output, and original `SCALE_SAVE/{symbol}/{target_freq}/{stage}/{contract}.feather` layout. <!-- 已实现: train-only robust scaler + manifest/diagnostics + same-basename CSV 输出 -->
- [x] 1.3 Run focused validation for the new scaler contract, including `pytest` for the updated scale-save tests, `python -m py_compile` for changed Python files, and `openspec validate --strict` for this change. <!-- 已实现: 语法检查、focused pytest、strict validate、产物落盘检查已完成 -->
