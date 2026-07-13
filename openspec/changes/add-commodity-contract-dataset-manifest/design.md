# Design: add-commodity-contract-dataset-manifest

## Context

商品预处理已经生成按合约分离的 scale-save 输出和品种级 state feature union。FineFT 数据准备层需要消费这些多合约产物，而不是把多个合约拼成一个连续 `df.feather` 后沿用旧切分逻辑。

## Goals

- 先用 summary 的合约交易日窗口计算全局 train/valid/test 日期边界。
- 生成可复现的 `dataset_manifest.json`，让数据集生成和切片计划可审计。
- 保持 train/valid/test 严格时序、不重叠，避免未来信息泄漏。
- 让 train slice 和 valid dynamic slice 都不跨合约。
- 让 `commodity_contract_dataset.py` 与 `slice_model.py` 解耦，valid 动态切片由 shell 独立调度。
- 保持 `commodity_data_handler_fu.sh` 和 `commodity_data_handler_al.sh` 作为用户入口。

## Non-Goals

- 不修改商品期货特征公式、feature selection 或 scale-save 行为。
- 不修改旧 crypto/single-file `preprocess_data.py` 行为。
- 不改低层训练算法的采样逻辑，除非只需要让其读取新的 `train/slice` 目录。

## Architecture

新增 `FineFT/datahandler/commodity_contract_dataset.py` 作为商品多合约数据集生成入口，内部保持四个职责边界：

1. `SplitBoundaryCalculator`
   - 从 summary 中收集所有合约有效 `trading_days[].date`。
   - 对日期去重、升序排序，按 `5:3:2` 计算 `start/a/b/c`。
   - 使用左闭右开集合范围：`train=[start,a)`、`valid=[a,b)`、`test=[b,c)`。

2. `DatasetManifestBuilder`
   - 对每个合约的交易日分别与全局集合范围求交。
   - 记录输入 `SCALE_SAVE` 路径、阶段输出路径、train slice 连续编号计划和 valid 动态切片计划。
   - 阶段文件写出后记录每个 `output_path` 的实际行数，并汇总集合级总行数。
   - 写出 `dataset_manifest.json`。

3. `ContractDatasetWriter`
   - 只按 manifest 过滤每个合约的 `df.feather` 并落盘。
   - 写出 `train/df_<contract>.feather`、`valid/df_<contract>.feather`、`test/df_<contract>.feather`。
   - 不生成 `train.feather`、`valid.feather`、`test.feather`。

4. `ContractSliceWriter`
   - 生成 `train/slice/df_*.feather`，编号连续，单个 slice 不跨合约、不跨 train 日期边界。
   - 合约 train 行数少于 `chunk_length` 或尾部剩余行数少于 `chunk_length` 时仍写出短 slice。
   - 写出 slice 后在 manifest 的 `slice_outputs` 中记录该 slice 文件实际行数。
   - 不调用 `slice_model.py`，不生成 valid 动态标签片段。

5. Shell valid slicing orchestration
   - `commodity_data_handler_fu.sh` 和 `commodity_data_handler_al.sh` 在数据集生成工具完成后，读取 `dataset_manifest.json` 或扫描 `valid/df_<contract>.feather`。
   - 对每个 valid 合约文件独立调用 `FineFT/datahandler/slice_model.py --data_path dataset/{target_freq}/{symbol}/valid/df_<contract>.feather --timestamp timestamp`。
   - 将每个合约的 processed 中间文件写到 `valid/processed/valid_processed_<contract>.feather`。
   - 将每个合约的动态片段写到 `valid/<contract>/label_*/df_*.feather`，且每个片段仍只包含单一合约。

## Data Contract

Input:

```text
PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/{symbol}/main_contract_summary.json
PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/{symbol}/{contract}/{target_freq}/{start_date}-{end_date}/df.feather
PREPROCESS_DATASET/commodity-futures/FEATURE_UNION/{symbol}/{target_freq}/{start_date}-{end_date}/state_features.npy
```

Output:

```text
dataset/{target_freq}/{symbol}/dataset_manifest.json
dataset/{target_freq}/{symbol}/state_features.npy
dataset/{target_freq}/{symbol}/train/df_<contract>.feather
dataset/{target_freq}/{symbol}/valid/df_<contract>.feather
dataset/{target_freq}/{symbol}/test/df_<contract>.feather
dataset/{target_freq}/{symbol}/train/slice/df_0.feather
dataset/{target_freq}/{symbol}/valid/processed/valid_processed_<contract>.feather
dataset/{target_freq}/{symbol}/valid/<contract>/label_<n>/df_*.feather
```

`dataset_manifest.json` 中每个含 `output_path` 的合约记录包含 `output_row_count`，每个集合包含 `contracts_total_count`。`train.slice_outputs[]` 中每个 slice 记录也包含 `output_row_count`。这些行数来自实际写出的 feather 文件，不用重新读取所有文件也能审计单文件、阶段总规模和训练 slice 规模。

## Error Handling

- Missing summary, malformed summary, missing contracts or missing valid trading days fail fast.
- Invalid split boundaries fail fast.
- Missing `SCALE_SAVE` contract `df.feather` for a planned non-empty set fails fast.
- Missing `FEATURE_UNION/state_features.npy` fails fast.
- Missing required data columns fail fast.
- If a valid contract has too little data for dynamic slicing, skip that contract's valid dynamic output and record the reason in manifest; do not concatenate another contract to satisfy length.
- `early_stop` is clipped inside the same contract's train range. If the clipped slice is too short to use, skip it and record the reason in manifest.

## Testing

- Unit-test split boundary calculation on multi-contract summary data.
- Unit-test manifest construction and no-overlap set assignment.
- Unit-test stage data generation and absence of legacy `train.feather`/`valid.feather`/`test.feather`.
- Unit-test train slice numbering and no-cross-contract/no-cross-set behavior.
- Unit-test that `commodity_contract_dataset.py` does not import or call `slice_model.py`.
- Smoke-test commodity data handler scripts invoke the new tool and then call `slice_model.py` per `valid/df_<contract>.feather`, never against `dataset/<symbol>/valid.feather`.
