# Design: adjust-commodity-dataset-split

## Context

商品期货 full process 当前仍包含旧的 `ic_candidate` 和 `ic_union_finalize` 调度，并且 `scale_save` 位于后置阶段。新的流程要求每个合约在 `merge_clean` 后立即执行 `scale_save`，全部合约完成后新增第 9 阶段数据集切分。

现有 `FineFT/datahandler/commodity_contract_dataset.py` 已实现可复用的时间切分算法：对 summary 中所有合约有效交易日取并集，按 `train:valid:test = 5:3:2` 计算全局边界，再按每个合约自身交易日与边界求交。本次只复用该算法，不复用其 `state_features.npy`、train slice、valid 动态切片或 VAE 相关职责。

## Goals

- 删除商品 full process 中旧的 `ic_candidate` 和 `ic_union_finalize` 调度。
- 将 `scale_save` 放回每个合约的 `merge_clean` 后。
- 新增 `future_upgraded/9_dataset_split` 阶段，全部合约 `scale_save` 完成后只运行一次。
- 新增 `operator_futures.dataset_split.dataset_split`，从合约级 `SCALE_SAVE` 输出生成阶段数据集。
- 保留合约级 train/valid/test 目录，并额外写出纵向合并后的 `train.feather`、`valid.feather`、`test.feather`。
- 切分与合并保留输入 feather 的所有列，不做特征筛选。

## Non-Goals

- 不改造 `FineFT/datahandler/commodity_contract_dataset.py` 的训练 slice、valid 动态切片或 VAE 数据流程。
- 不新增 feature union 替代方案。
- 不重新设计 `scale_save.py` 的特征选择和缩放逻辑。
- 不删除合约级阶段输出目录。

## Architecture

### Shell orchestration

`data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh` 继续负责全流程调度：

1. `stitch_main_contract`
2. `downscale_continuous_by_trading_day`
3. 对 summary 中每个合约循环运行：
   - `cross_section`
   - `merge`
   - `concat`
   - `time_feature`
   - `merge_clean`
   - `scale_save`
4. 所有合约完成后运行一次 `dataset_split`
5. `maintenance_margin_dict`

`data_preprocess/script_preprocess/future_upgraded/9_dataset_split/dataset_split.sh` 作为第 9 阶段入口，负责激活 `finetf` conda 环境并调用 Python module。

### Python operator

`data_preprocess/operator_futures/dataset_split/dataset_split.py` 承载核心逻辑：

1. 读取 `main_contract_summary.json`。
2. 收集所有合约 `trading_days[].date` 的去重升序并集。
3. 按 `train:valid:test = 5:3:2` 计算 `start/a/b/c`，形成左闭右开范围。
4. 对每个合约分别计算其交易日与 train/valid/test 范围的交集。
5. 读取 `SCALE_SAVE/{symbol}/{contract}/{target_freq}/{start_date}-{end_date}/df.feather`。
6. 按交易日过滤并按时间排序，写出合约级阶段文件。
7. 对每个阶段纵向合并所有合约输出，写出顶层阶段文件。
8. 写出 `dataset_split_manifest.json`，记录边界、输入输出路径、行数和 skipped 信息。

## Data Contract

Input:

```text
PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/{symbol}/main_contract_summary.json
PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/{symbol}/{contract}/{target_freq}/{start_date}-{end_date}/df.feather
```

Output:

```text
dataset/{target_freq}/{symbol}/train/{contract}.feather
dataset/{target_freq}/{symbol}/valid/{contract}.feather
dataset/{target_freq}/{symbol}/test/{contract}.feather
dataset/{target_freq}/{symbol}/train.feather
dataset/{target_freq}/{symbol}/valid.feather
dataset/{target_freq}/{symbol}/test.feather
dataset/{target_freq}/{symbol}/dataset_split_manifest.json
```

All output feather files preserve all input columns. The top-level stage files are row-wise concatenations of the corresponding contract-level files.

## Error Handling

- Missing summary, malformed summary, empty contracts, or insufficient effective trading days fail fast.
- Invalid split boundaries fail fast.
- Missing `SCALE_SAVE` contract `df.feather` for a planned non-empty set fails fast.
- If summary says a contract has trading days in a set but filtering produces no rows, the split fails fast.
- If any top-level `train.feather`, `valid.feather`, or `test.feather` cannot be generated because the set has no contract outputs, the split fails fast.
- A contract with no trading days in a set is skipped for that set and recorded in the manifest.

## Testing

- Unit-test split boundary calculation on multi-contract summary data.
- Unit-test contract/date set assignment and skipped contracts.
- CLI-test `dataset_split.py` with synthetic `SCALE_SAVE` feather files, verifying contract-level files, merged files, manifest row counts, and all-column preservation.
- Shell/static-test `fu_full_process.sh` to ensure deleted steps are absent, `scale_save` is inside the contract loop after `merge_clean`, and `dataset_split` runs once after all contracts.
- Validate with `conda activate finetf && pytest data_preprocess/tests/test_commodity_dataset_split.py data_preprocess/tests/test_commodity_main_contract_cli.py -q`.
