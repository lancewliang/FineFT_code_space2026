# Tasks

## 1. Implementation

- [x] 1.1 Add focused tests for commodity IC candidate-only output artifacts and absence of final `df.feather` / `state_features.npy`. <!-- 已实现: 添加 candidate-only CLI 回归测试并确认 RED 失败 -->
- [x] 1.2 Implement commodity candidate-only mode in `ic_correlation.py` while preserving default IC output compatibility. <!-- 已实现: ic_correlation 支持 candidate-only 输出并保持默认 df/state_features 输出兼容 -->
- [x] 1.3 Add focused tests for union finalize loading candidate features, writing品种级 `FEATURE_UNION`, and writing per-contract filtered `IC_RESULT`. <!-- 已实现: 添加 union finalize happy-path 测试并确认 RED 失败 -->
- [x] 1.4 Extend `contract_feature_union.py` to read IC candidates, build union, validate all contract columns, and write per-contract filtered `IC_RESULT` outputs. <!-- 已实现: contract_feature_union 支持 candidate finalize 并写出统一 union 的合约 IC_RESULT -->
- [x] 1.5 Add fail-fast tests for missing candidate files, empty union, and union features missing from a contract `ALL_FEATURE`. <!-- 已实现: 覆盖缺 candidate、空 union 和合约缺 union 特征的 fail-fast 测试 -->
- [x] 1.6 Update `fu_full_process.sh` tests and shell flow so candidate runs inside the contract loop, union finalize runs once after all candidates, and `scale_save` runs per contract after finalize. <!-- 已实现: 商品 full process 改为 ic_candidate -> ic_union_finalize -> per-contract scale_save -->
- [x] 1.7 Update validation entrypoint or commodity feature pipeline tests to cover the new final artifact layout and manifest content. <!-- 已实现: manifest 断言覆盖 candidate/all_feature/ic_result/finalize 输出字段并确认验证入口检查 FEATURE_UNION -->

## 2. Validation

- [x] 2.1 Run `openspec validate refactor-commodity-feature-selection-union --strict`. <!-- 已实现: strict 校验通过 -->
- [x] 2.2 Run `conda run -n finetf pytest data_preprocess/tests/test_feature_selection_polars.py data_preprocess/tests/test_commodity_feature_pipeline.py data_preprocess/tests/test_commodity_main_contract_cli.py -q`. <!-- 已实现: 聚焦回归 50 passed -->
- [x] 2.3 Run any focused static shell assertions that verify `fu_full_process.sh` no longer has the old separate post-loop `feature_union` stage and runs `scale_save` only after `ic_union_finalize`. <!-- 已实现: 静态 shell 顺序测试通过 -->
