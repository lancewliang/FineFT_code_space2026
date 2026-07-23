# fu 脚本主流程

下面整理仓库里的 `fu` 主流程 shell 脚本，包含所有 `*_fu.sh`，以及用户点名的 `train_commodity_fu_10.sh`。  
如果只想按依赖关系理解，可以直接看这个顺序：

`main_fu.sh` -> `commodity_data_handler_fu.sh` -> `train_commodity_fu_10.sh` -> `test_util_fu.sh` -> `low_level_fu.sh` -> `VAE_util_fu.sh`

注意：`train_commodity_fu_10.sh` 默认 `EXPERIMENT_NAME=default`，而 `test_util_fu.sh` 和 `low_level_fu.sh` 默认 `EXPERIMENT_NAME=10min_nstep6_costw5`。串联运行时需要显式设置同一个 `EXPERIMENT_NAME`。

## 1. 数据预处理入口

### `data_preprocess/script_preprocess/future_upgraded/commodity/main_fu.sh`

- 作用：燃料油 `fu` 的商品期货预处理总入口。
- 依赖：`fu_full_process.sh`
- 默认参数：
  - `ROOTPATH=$(pwd)`
  - `START_DATE=2023-01-01`
  - `END_DATE=2026-03-01`
  - `TARGET_FREQ=10min`
  - `SYMBOL=fu`
  - `COMMODITY_NAME=燃料油`
  - `MAX_PROCESSES=4`
- 输出：
  - 预处理数据到 `PREPROCESS_DATASET/commodity-futures/...`
  - 日志到 `log_futures/ticker_result/commodity/`
- 位置：整个 `fu` 流程的起点。

## 2. FineFT 数据准备

### `FineFT/script/data/commodity_data_handler_fu.sh`

- 作用：把商品期货预处理结果整理成 FineFT 训练需要的数据集结构，并生成 valid 动态切片和 VAE 数据。
- 依赖：
  - `PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/<target_freq>/fu/dataset_split_manifest.json`
  - `PREPROCESS_DATASET/commodity-futures/SCALE_SAVE`
  - `PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/<target_freq>/fu/train/state_features.npy`
- 默认参数：
  - `ROOTPATH=$(pwd)`
  - `SYMBOL=fu`
  - `TARGET_FREQ=10min`
  - `CHUNK_LENGTH=8000`
  - `EARLY_STOP=2`
- 主要步骤：
  - 激活 `finetf` conda 环境。
  - 调用 `FineFT/datahandler/commodity_contract_dataset.py` 生成 `dataset/<target_freq>/fu`。
  - 对 `dataset/<target_freq>/fu/valid/*.feather` 逐个调用 `slice_model.py`。
  - 调用 `vae_data_creation.py` 生成 VAE 训练/测试数据。
- 输出：
  - `dataset/10min/fu/...`，实际频率由 `TARGET_FREQ` 决定。
- 位置：承接商品预处理结果，是进入 FineFT 训练、回测和 VAE 的数据准备入口。

## 3. 低层 agent 训练

### `FineFT/script/train/train_commodity_fu_10.sh`

- 作用：训练 `fu` 的 10min 低层 agent。
- 默认参数：
  - `ROOTPATH=$(pwd)`
  - `EXPERIMENT_NAME=default`
- 固定训练参数：
  - `--base_path dataset/10min`
  - `--dataset_name fu`
  - `--max_holding_number 1`
  - `--position_choices 3`
  - `--transcation_cost 0.0004`
  - `--n_step 12`
  - `--gamma 0.99`
  - `--order_book_depth 5`
  - `--early_stop 2`
- 输出：
  - `log_futures/fu/low_level/train/<experiment_name>/advantage-10min.log`
- 位置：承接 `commodity_data_handler_fu.sh` 生成的 `dataset/10min/fu` 数据，是低层回测和 agent 筛选的前置训练步骤。

## 4. 低层回测/测试

### `FineFT/script/test/DiHFT/low_level/test_util_fu.sh`

- 作用：批量跑低层 agent 的测试回测，生成每个 epoch 的结果日志。
- 默认参数：
  - `DATASET_NAME=fu`
  - `BASE_PATH=dataset/10min`
  - `EPOCH_START=1`
  - `EPOCH_END=60`
  - `EXPERIMENT_NAME=10min_nstep6_costw5`
  - `MAX_HOLDING_NUMBER=1`
- 输出：
  - `log/DiHFT/fu/low_level/test/<experiment_name>/epoch_*.log`
- 位置：给后面的 agent 筛选提供测试结果。

## 5. 低层 agent 筛选

### `FineFT/script/analysis/pick_agent/low_level_fu.sh`

- 作用：根据低层测试结果筛选 agent。
- 默认参数：
  - `DATASET_NAME=fu`
  - `EXPERIMENT_NAME=10min_nstep6_costw5`
  - `BASE_PATH=dataset/10min`
  - `POSITION_CHOICES=3`
  - `NUM_LABEL=5`
- 输出：
  - `log/analysis/pick_agent/DiHFT/fu/<experiment_name>.log`
- 位置：承接低层测试结果，做筛选分析。

## 6. VAE 训练

### `FineFT/script/train/DiHFT/low_level/VAE_util_fu.sh`

- 作用：按 `label_0 ~ label_4` 批量训练 VAE。
- 默认参数：
  - `DATASET_NAME=fu`
  - `DATA_BASE_PATH=dataset/10min`
  - `LABEL_COUNT=5`
  - `EXPERIMENT_NAME=default`
  - `MAX_PARALLEL_JOBS=2`
- 输出：
  - `log/DiHFT/fu/VAE/<experiment_name>/train_label_*.log`
- 位置：按 README 的 Stage II 流程放在筛选之后。

## 一句话总结

- `main_fu.sh` 负责把原始 `fu` 数据预处理成可训练数据。
- `commodity_data_handler_fu.sh` 负责把预处理结果转换成 FineFT 数据集、valid label 切片和 VAE 数据。
- `train_commodity_fu_10.sh` 负责训练 10min 低层 agent。
- `test_util_fu.sh` 负责低层回测。
- `low_level_fu.sh` 负责低层筛选。
- `VAE_util_fu.sh` 负责 VAE 训练。
