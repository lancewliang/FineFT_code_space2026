# adjust-commodity-dataset-split

## 背景与目标

商品期货全流程 `fu_full_process.sh` 的调度需要调整：旧的 `ic_candidate` 子功能和 `ic_union_finalize` 方案步骤不再需要；`scale_save` 应恢复为每个合约在 `merge_clean` 之后立即执行；全部合约完成 `scale_save` 后，需要新增第 9 阶段数据集切分步骤。

新增切分步骤复用 `commodity_contract_dataset.py` 中基于所有合约交易日并集计算 train/valid/test 时间边界的算法，但本次只做合约数据切分和纵向合并，不需要知道特征列集合，也不依赖 `state_features.npy`。

## 用户场景

- 运行商品期货 full process 时，每个合约独立完成特征合并清洗和 scale save。
- 全部合约 scale save 完成后，按统一时间边界切分 train/valid/test。
- 下游既可以读取保留的合约级阶段文件，也可以读取纵向合并后的阶段数据集。

## 设计方向

采用新增第 9 阶段 dataset split 的方案。

`fu_full_process.sh` 继续只负责调度：

1. 删除 `run_commodity_ic_candidate`、`run_commodity_ic_union_finalize` 及其 logged step。
2. 将 `run_commodity_scale_save` 放回合约循环内，紧跟每个合约的 `run_commodity_merge_and_clean` 后执行。
3. 所有合约循环结束后，只执行一次 `dataset_split` logged step。

Python 核心逻辑放在：

```text
data_preprocess/operator_futures/dataset_split/dataset_split.py
```

第 9 阶段 shell 入口放在：

```text
data_preprocess/script_preprocess/future_upgraded/9_dataset_split/
```

数据集切分读取：

```text
PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/{symbol}/main_contract_summary.json
PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/{symbol}/{contract}/{target_freq}/{start_date}-{end_date}/df.feather
```

输出：

```text
dataset/{target_freq}/{symbol}/train/{contract}.feather
dataset/{target_freq}/{symbol}/valid/{contract}.feather
dataset/{target_freq}/{symbol}/test/{contract}.feather
dataset/{target_freq}/{symbol}/train.feather
dataset/{target_freq}/{symbol}/valid.feather
dataset/{target_freq}/{symbol}/test.feather
dataset/{target_freq}/{symbol}/dataset_split_manifest.json
```

合约级 `train/`、`valid/`、`test/` 目录必须保留。顶层 `train.feather`、`valid.feather`、`test.feather` 只做纵向合并，所有列原样保留，不做特征筛选。

## 关键决策

- 删除 `ic_candidate` 和 `ic_union_finalize`，不再运行 `--candidate_only` 或 feature union finalize。
- `scale_save` 是每个合约的处理步骤，位置在该合约 `merge_clean` 之后。
- 新增 `9_dataset_split` 阶段，位置在所有合约的 `scale_save` 都完成之后。
- `dataset_split.py` 复用交易日并集的时间切分算法，默认比例为 train:valid:test = 5:3:2。
- `dataset_split.py` 不依赖特征列表，不读取或生成 `state_features.npy`。
- Python 脚本通过 `conda activate finetf` 后运行。

## 范围边界

**包含：**

- 调整 `fu_full_process.sh` 调度顺序。
- 新增 `data_preprocess/operator_futures/dataset_split/dataset_split.py`。
- 新增 `data_preprocess/script_preprocess/future_upgraded/9_dataset_split/` 阶段入口。
- 生成合约级 train/valid/test feather 文件。
- 生成纵向合并后的 train.feather、valid.feather、test.feather。
- 生成 `dataset_split_manifest.json`，记录边界、输入输出路径、行数和 skipped 信息。
- 更新相关测试，覆盖调度顺序、旧步骤删除、切分输出和所有列保留。

**不包含（本次）：**

- 不改造 FineFT 现有 `commodity_contract_dataset.py` 的训练 slice、valid 动态切片或 VAE 数据逻辑。
- 不新增 feature union 替代方案。
- 不重新设计 `scale_save.py` 的特征选择和缩放逻辑。
- 不删除合约级阶段输出目录。

## 错误处理

`dataset_split.py` 应 fail-fast：

- `summary_path` 不存在时报错。
- summary 无合约时报错。
- 交易日不足以切出非空 train/valid/test 时报错。
- 应处理合约缺少 `SCALE_SAVE/{symbol}/{contract}/{target_freq}/{start_date}-{end_date}/df.feather` 时报错。
- summary 显示某合约在某阶段有交易日，但过滤输入后为空时报错。
- 任一集合无法生成顶层 `train.feather`、`valid.feather` 或 `test.feather` 时报错。

某合约没有落入某个阶段时间范围时不报错，记录到 manifest 的 skipped 信息中。

## 验收标准

- [ ] `fu_full_process.sh` 不再定义或调度 `run_commodity_ic_candidate`。
- [ ] `fu_full_process.sh` 不再定义或调度 `run_commodity_ic_union_finalize`。
- [ ] `fu_full_process.sh` 在每个合约的 `merge_clean` 后调度 `scale_save`。
- [ ] `fu_full_process.sh` 在所有合约处理完成后只调度一次 `dataset_split`。
- [ ] 新增 `data_preprocess/operator_futures/dataset_split/dataset_split.py`。
- [ ] 新增 `data_preprocess/script_preprocess/future_upgraded/9_dataset_split/` 阶段入口。
- [ ] `dataset_split.py` 基于 summary 所有合约交易日并集按 5:3:2 计算 train/valid/test 边界。
- [ ] `dataset_split.py` 从 `SCALE_SAVE/{symbol}/{contract}/{target_freq}/{start_date}-{end_date}/df.feather` 读取输入。
- [ ] `dataset_split.py` 输出并保留 `train/{contract}.feather`、`valid/{contract}.feather`、`test/{contract}.feather`。
- [ ] `dataset_split.py` 输出纵向合并的 `train.feather`、`valid.feather`、`test.feather`。
- [ ] 切分和合并输出保留输入 feather 的所有列。
- [ ] 相关测试通过，使用 `conda activate finetf && pytest data_preprocess/tests/test_commodity_dataset_split.py data_preprocess/tests/test_commodity_main_contract_cli.py -q` 验证。
