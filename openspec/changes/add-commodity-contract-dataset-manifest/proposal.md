# add-commodity-contract-dataset-manifest

## Why

商品期货特征已经按真实合约单位落盘，路径为 `SCALE_SAVE/{symbol}/{contract}/{target_freq}/{start_date}-{end_date}/df.feather`，并通过 `main_contract_summary.json` 描述每个合约的有效交易日窗口。FineFT 现有 datahandler 仍按单个 `df.feather` 行序切出 `train.feather`、`valid.feather` 和 `test.feather`，随后对单个 `valid.feather` 做动态切片。

这个旧流程会把多个真实合约当成连续行情处理，训练、验证、测试边界也无法先按合约和日期范围确定，存在跨合约切片和未来信息泄漏风险。

## What Changes

- 新增商品多合约 FineFT 数据集生成工具，读取 `main_contract_summary.json`、合约级 `SCALE_SAVE` 输出和品种级 `FEATURE_UNION/state_features.npy`。
- 新增边界计算能力：基于 summary 中所有合约有效 `trading_days` 的去重有序交易日轴，按 `train:valid:test = 5:3:2` 计算全局时序边界 `start/a/b/c`。
- 新增 `dataset_manifest.json`，记录全局边界、每个集合的日期范围、每个合约在集合中的交易日、输入路径、阶段输出路径和切片计划。
- 商品 FineFT 阶段数据集不再生成 `train.feather`、`valid.feather` 和 `test.feather`；改为生成 `train/df_<contract>.feather`、`valid/df_<contract>.feather` 和 `test/df_<contract>.feather`。
- 真正用于低层训练的文件生成到 `train/slice/df_*.feather`，连续编号，单个文件不跨合约且不跨 train 日期边界。
- valid 动态切片逐合约运行，processed 文件输出到 `valid/processed/valid_processed_<contract>.feather`，label 片段输出到 `valid/<contract>/label_*/df_*.feather`，任何 label 片段不跨合约。
- 直接升级 `FineFT/script/data/commodity_data_handler_fu.sh` 和 `FineFT/script/data/commodity_data_handler_al.sh`，让现有商品数据处理入口调用新工具。
- 保留旧 `preprocess_data.py` 给旧单文件数据路径使用。

## Amendments

### 2026-07-13: Shell-orchestrated valid slicing
- 原因：商品多合约数据集生成工具不应直接调用 `slice_model.py` 或嵌入动态切片逻辑；valid 动态切片应作为 shell 编排中的独立阶段，和数据集生成解耦。
- 摘要：`commodity_contract_dataset.py` 只负责边界计算、manifest、阶段数据和 train slice。`commodity_data_handler_fu.sh` 与 `commodity_data_handler_al.sh` 在新工具完成后，根据 manifest/valid 合约文件逐合约调用 `slice_model.py`，输出 `valid/<contract>/label_*/df_*.feather`。
- 行为影响：代码边界更清晰；valid 动态切片仍不跨合约，但由 shell 调度实现。

### 2026-07-13: Manifest row counts
- 原因：用户需要仅通过 `dataset_manifest.json` 判断每个阶段输出文件的行数，以及 train/valid/test 每个训练阶段的总行数。
- 摘要：每个含 `output_path` 的合约记录增加 `output_row_count`；每个集合增加 `contracts_total_count`，汇总该集合所有合约阶段输出文件行数。
- 行为影响：manifest 在阶段文件写出后记录实际 feather 行数，方便训练前审计数据规模。

### 2026-07-13: Train short slices and slice row counts
- 原因：少于 `chunk_length` 的 train 合约数据和尾部剩余数据仍需要参与训练，不能因为不足一个完整窗口而丢弃。
- 摘要：train slice 计划允许短 slice；每个 `train/slice/df_*.feather` 记录实际 `output_row_count`。
- 行为影响：`train/slice` 文件数量可能多于完整窗口数量，manifest 可直接审计每个训练 slice 的行数。

### 2026-07-13: Contract-scoped valid dynamic slices
- 原因：多个合约逐个调用 `slice_model.py` 时，旧的 `valid_processed.feather` 和 `valid/label_*/df_*.feather` 会在合约之间冲突。
- 摘要：valid processed 文件改为 `valid/processed/valid_processed_<contract>.feather`，动态切片输出改为 `valid/<contract>/label_*/df_*.feather`。
- 行为影响：不同合约的 valid 动态切片互不覆盖；VAE 数据生成需要递归读取合约级 label 目录并按 label 聚合。

### 2026-07-13: Robust slope labels for small valid segment counts
- 原因：部分合约 valid 动态切片经过合并后 segment 数量很少，旧 slope 阈值计算会越界并中断 shell。
- 摘要：slope 动态标签在 segment 数量较少时使用全量 min/max 生成阈值，避免 `IndexError`。
- 行为影响：小样本合约仍可生成 valid 动态标签；完全没有 segment 时给出明确错误。

### 2026-07-13: Valid slice manifest
- 原因：valid 动态切片后需要审计每个合约、每个 label 下生成了哪些文件，以及每个文件和 label 总行数。
- 摘要：`slice_model.py` 每处理一个合约后更新 `valid/slice_manifest.json`，记录合约视角和 label 视角的文件列表、文件数和行数。
- 行为影响：空 label 不写入 manifest；调用方可以不读取 feather 文件就知道 valid 动态切片规模。

### 2026-07-13: Contract-scoped VAE test arrays
- 原因：商品多合约 test 数据不能合并成单个 `test.npy`，否则会丢失合约边界。
- 摘要：当 test 数据来自 `test/df_<contract>.feather` 目录时，`vae_data_creation.py` 逐合约写出 `VAE_data/test/test_<contract>.npy`。
- 行为影响：旧单文件 `test.feather` 仍保持 `VAE_data/test.npy`；商品多合约 test 输出按合约隔离。

### 2026-07-13: Contract-scoped VAE valid arrays
- 原因：商品多合约 valid 动态片段不能跨合约聚合成单个 `VAE_data/label_x.npy`，否则会丢失合约边界。
- 摘要：当 valid 输入为 `valid/<contract>/label_*/df_*.feather` 时，`vae_data_creation.py` 写出 `VAE_data/<contract>/label_*.npy`。
- 行为影响：旧 `valid/label_*/*.feather` 仍保持 `VAE_data/label_*.npy`；商品多合约 valid 输出按合约隔离。

## Impact

- Affected specs: `commodity-futures-support`
- Affected code:
  - `FineFT/datahandler/commodity_contract_dataset.py`
  - `FineFT/datahandler/vae_data_creation.py`
  - `FineFT/script/data/commodity_data_handler_fu.sh`
  - `FineFT/script/data/commodity_data_handler_al.sh`
  - `FineFT/tests/datahandler/`
- Behavioral impact:
  - 商品 FineFT 数据集按合约和日期范围先确定集合归属，再切片。
  - 商品 data handler 不再依赖 `dataset/<symbol>/valid.feather` 的单文件动态切片入口。
  - 训练、验证、测试集合严格时序且日期不重叠。
