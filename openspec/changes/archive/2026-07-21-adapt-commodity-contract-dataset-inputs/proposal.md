# adapt-commodity-contract-dataset-inputs

## 背景与目标

商品期货数据流水线的输入结构已经变化：上游 `dataset_split` 先生成阶段切分 manifest，后续 `muti_contract_scale_save.py` 再按阶段写出最终缩放后的合约文件。`FineFT/datahandler/commodity_contract_dataset.py` 现在不应再读取 `main_contract_summary.json` 并自行计算 train/valid/test 边界，也不应再按日期过滤合约数据。

本次目标是把 `commodity_contract_dataset.py` 收窄为 FineFT 商品数据集装配工具：读取 `dataset_split_manifest.json` 获取阶段和合约元数据，从阶段化 `SCALE_SAVE` 读取真实 feather 文件，复制到 FineFT dataset 目录，复制训练阶段选出的 state feature 清单，继续生成 train slices，并保留 valid label 由 shell 调度 `slice_model.py` 生成的职责边界。

## 用户场景

### 场景 1：从阶段化 SCALE_SAVE 装配 FineFT dataset

用户已经运行商品期货预处理、切分、特征选择和 scale save。系统读取：

```text
PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/{target_freq}/{symbol}/dataset_split_manifest.json
PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/{symbol}/{target_freq}/{train|valid|test}/{contract}.feather
```

然后写出 FineFT 可用的阶段合约文件：

```text
dataset/{target_freq}/{symbol}/{train|valid|test}/{contract}.feather
```

### 场景 2：复制训练阶段 state feature 清单

系统使用训练阶段特征选择结果作为 FineFT dataset 的 state feature 清单来源：

```text
PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/{target_freq}/{symbol}/train/state_features.npy
```

并复制到：

```text
dataset/{target_freq}/{symbol}/state_features.npy
```

### 场景 3：继续生成 train slices 和 valid labels

系统继续基于 FineFT dataset 中的 train 合约文件生成：

```text
dataset/{target_freq}/{symbol}/train/slice/df_*.feather
```

valid label 仍由 `commodity_data_handler_fu.sh` 和 `commodity_data_handler_al.sh` 在 `commodity_contract_dataset.py` 完成后循环调用 `slice_model.py` 生成。脚本扫描 valid 合约文件时应匹配：

```text
dataset/{target_freq}/{symbol}/valid/*.feather
```

## 设计方向

采用直接切换到新契约的方案，清理旧切分职责。

`commodity_contract_dataset.py` 的主输入改为：

```text
--dataset_split_manifest_path
--input_root
--state_features_path
--output_root
--symbol
--target_freq
--chunk_length
--early_stop
```

其中 `--input_root` 指向：

```text
PREPROCESS_DATASET/commodity-futures/SCALE_SAVE
```

真实数据路径固定解析为：

```text
{input_root}/{symbol}/{target_freq}/{stage}/{contract}.feather
```

输出路径固定解析为：

```text
{output_root}/{symbol}/{stage}/{contract}.feather
```

如果 `output_root` 由脚本传入 `dataset/{target_freq}`，最终落盘路径为：

```text
dataset/{target_freq}/{symbol}/{stage}/{contract}.feather
```

`dataset_split_manifest.json` 只作为阶段、合约、交易日和行数等审计元数据来源，不再驱动日期过滤。`commodity_contract_dataset.py` 不再计算 split boundaries，也不再按 trading days 切割阶段。

## 关键决策

- `commodity_contract_dataset.py` 不再读取 `main_contract_summary.json`。
- `dataset_split_manifest.json` 代替 `main_contract_summary.json`，作为阶段和合约列表来源。
- 真实数据文件只从 `SCALE_SAVE/{symbol}/{target_freq}/{stage}/{contract}.feather` 读取。
- FineFT dataset 阶段文件名改为 `{contract}.feather`，不再使用 `df_{contract}.feather`。
- `--feature_union_path` 改名为 `--state_features_path`。
- `--state_features_path` 来源为 `FEATURE_SELECTION/{target_freq}/{symbol}/train/state_features.npy`。
- 继续生成 train slices：`train/slice/df_*.feather`。
- valid label 继续由 shell 调用 `slice_model.py` 生成，不合并进 `commodity_contract_dataset.py`。
- `summary_path`、`start_date`、`end_date`、`train_ratio`、`valid_ratio`、`test_ratio` 不再作为该工具主路径参数。

## 范围边界

**包含：**
- 调整 `commodity_contract_dataset.py` 的输入契约，改读 `dataset_split_manifest.json`。
- 调整 `commodity_contract_dataset.py` 的真实数据输入路径，改读阶段化 `SCALE_SAVE`。
- 调整 FineFT dataset 阶段文件输出名为 `{contract}.feather`。
- 将 CLI 参数 `--feature_union_path` 改名为 `--state_features_path`。
- 复制 `state_features.npy` 到 FineFT dataset 根目录。
- 保留并适配 train slice 生成。
- 更新商品 FineFT 数据脚本，使 valid label 生成扫描 `valid/*.feather`。
- 更新相关测试以覆盖新输入结构、输出文件名、train slices、valid label 调度和 fail-fast 行为。

**不包含（本次）：**
- 修改 `dataset_split` 的阶段切分规则。
- 修改 `muti_contract_scale_save.py` 的缩放算法。
- 修改 `slice_model.py` 的 label 生成算法。
- 将 valid label 生成逻辑并入 `commodity_contract_dataset.py`。
- 保留旧 `main_contract_summary.json` 切分路径作为主流程兼容模式。

## 验收标准

- [ ] `commodity_contract_dataset.py` 支持 `--dataset_split_manifest_path`，并使用它代替 `--summary_path`。
- [ ] `commodity_contract_dataset.py` 支持 `--state_features_path`，并不再要求 `--feature_union_path`。
- [ ] `commodity_contract_dataset.py` 不再计算 train/valid/test split boundaries。
- [ ] `commodity_contract_dataset.py` 不再按 trading days 过滤 feather 数据。
- [ ] 系统从 `SCALE_SAVE/{symbol}/{target_freq}/{stage}/{contract}.feather` 读取真实数据。
- [ ] 系统写出 `dataset/{target_freq}/{symbol}/{stage}/{contract}.feather`。
- [ ] 系统复制 `FEATURE_SELECTION/{target_freq}/{symbol}/train/state_features.npy` 到 `dataset/{target_freq}/{symbol}/state_features.npy`。
- [ ] 系统继续写出 `dataset/{target_freq}/{symbol}/train/slice/df_*.feather`。
- [ ] `dataset_manifest.json` 记录阶段、合约、输入路径、输出路径、输出行数、阶段总行数和 train slice 输出。
- [ ] 商品 FineFT 数据脚本扫描 `dataset/{target_freq}/{symbol}/valid/*.feather` 并继续调用 `slice_model.py` 生成 valid label 数据。
- [ ] 缺少 `dataset_split_manifest.json`、缺少 `state_features.npy`、manifest symbol/freq 不匹配、缺少任一声明的 SCALE_SAVE 合约文件时，流程 fail-fast。
