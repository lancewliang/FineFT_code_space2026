# Tasks

## 1. Implementation

- [x] 1.1 Add focused tests for commodity split boundary calculation from multi-contract summary trading days.
- [x] 1.2 Add manifest-builder tests for 5:3:2 non-overlapping set assignment, contract/date intersections, input paths, stage output paths, and slice plans.
- [x] 1.3 Implement `FineFT/datahandler/commodity_contract_dataset.py` boundary calculation and manifest generation.
- [x] 1.4 Add stage dataset writer tests and implementation for `train/df_<contract>.feather`, `valid/df_<contract>.feather`, `test/df_<contract>.feather`, state feature copy, and absence of legacy `train.feather`/`valid.feather`/`test.feather`.
- [x] 1.5 Add train slice writer tests and implementation for continuous `train/slice/df_*.feather` numbering, no-cross-contract slices, and `early_stop` clipping inside train.
- [x] 1.6 Add valid dynamic slicing tests and implementation so valid slices are produced per contract under `valid/label_*/` without cross-contract concatenation.
- [x] 1.7 Update `vae_data_creation.py` tests and implementation so commodity VAE generation reads `valid/label_*/*.feather` and `test/df_<contract>.feather` without requiring `test.feather`.
- [x] 1.8 Update `commodity_data_handler_fu.sh` and `commodity_data_handler_al.sh` to call the new multi-contract dataset tool and stop calling legacy single-file preprocess/slice commands.
- [x] 1.9 Update or add datahandler documentation covering the commodity multi-contract dataset manifest workflow.
- [x] 1.10 Remove any `slice_model.py` import/call and valid label output responsibility from `commodity_contract_dataset.py`. <!-- 已实现: commodity_contract_dataset 不再调用 slice_model 或生成 valid label -->
- [x] 1.11 Update `commodity_data_handler_fu.sh` and `commodity_data_handler_al.sh` to call `slice_model.py` independently for each `valid/df_<contract>.feather` after dataset generation. <!-- 已实现: 商品 shell 逐个 valid/df_<contract>.feather 调用 slice_model.py -->
- [x] 1.12 Update tests and docs for shell-orchestrated valid slicing. <!-- 已实现: 更新测试和文档并完成校验 -->
- [x] 1.13 Add manifest row counts for each contract `output_path` and each split's `contracts_total_count`. <!-- 已实现: manifest 记录每个 output_path 行数和集合总行数 -->
- [x] 1.14 Keep short train slices below `chunk_length` and record slice `output_row_count` in manifest. <!-- 已实现: 保留短 train slice 并记录 slice 行数 -->
- [x] 1.15 Write valid processed and label slices under contract-scoped directories and update VAE reading. <!-- 已实现: valid processed 和 label 切片按合约隔离，VAE 递归读取 -->
- [x] 1.16 Make slope dynamic labeling robust when valid segment counts are small. <!-- 已实现: 小 segment 数量时 slope 阈值使用安全 min/max -->
- [x] 1.17 Generate `valid/slice_manifest.json` with contract and label row counts. <!-- 已实现: slice_model 记录合约/label 视角文件与行数 -->
- [x] 1.18 Write multi-contract VAE test arrays as `VAE_data/test/test_<contract>.npy` instead of one merged `test.npy`. <!-- 已实现: 商品 test npy 按合约输出 -->
- [x] 1.19 Write multi-contract VAE valid arrays as `VAE_data/<contract>/label_*.npy` instead of cross-contract `label_*.npy`. <!-- 已实现: 商品 valid npy 按合约输出 -->

## 2. Validation

- [x] 2.1 Run `openspec validate add-commodity-contract-dataset-manifest --strict`.
- [x] 2.2 Run focused datahandler tests with the `finetf` conda environment.
- [x] 2.3 Run shell/static checks that confirm commodity data handler scripts call the new tool and do not call legacy single-file commands.
