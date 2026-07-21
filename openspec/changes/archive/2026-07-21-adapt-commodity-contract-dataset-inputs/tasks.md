## 1. Implementation

- [x] 1.1 Update `FineFT/tests/datahandler/test_commodity_contract_dataset.py` for the new manifest-driven input contract, stage file naming, state feature path, train slices, and failure cases. <!-- 已实现: 新增 manifest-driven 输入契约、阶段化 SCALE_SAVE、state_features_path、contract 命名输出和 fail-fast RED 测试 -->
- [x] 1.2 Refactor `FineFT/datahandler/commodity_contract_dataset.py` to read `dataset_split_manifest.json`, build a FineFT manifest from stage/contract metadata, copy staged SCALE_SAVE files, copy `--state_features_path`, and remove internal split-boundary filtering from the main path. <!-- 已实现: 工具主路径改为读取 dataset_split_manifest 并复制阶段化 SCALE_SAVE 与 state_features_path -->
- [x] 1.3 Keep train slice generation working from `train/{contract}.feather`, with continuous slice indices and manifest row counts. <!-- 已实现: train slice 从 contract 命名阶段文件生成并保留连续编号与行数记录 -->
- [x] 1.4 Update `FineFT/script/data/commodity_data_handler_fu.sh` and `FineFT/script/data/commodity_data_handler_al.sh` to pass `--dataset_split_manifest_path` and `--state_features_path`, and to scan `valid/*.feather` for `slice_model.py`. <!-- 已实现: 商品 data handler 脚本传新 dataset manifest/state features 参数并扫描 valid/*.feather -->
- [x] 1.5 Update any FineFT commodity dataset tests that assert old `df_<contract>.feather`, `--summary_path`, `--feature_union_path`, or valid `df_*.feather` contracts. <!-- 已实现: 旧路径正向断言已替换，仅保留 train slice df_* 合法断言和旧参数负断言 -->

## 2. Validation

- [x] 2.1 Run `conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py -q`. <!-- 已实现: FineFT commodity contract dataset 测试 14 passed -->
- [x] 2.2 Run `openspec validate adapt-commodity-contract-dataset-inputs --strict`. <!-- 已实现: OpenSpec strict validation 通过 -->
