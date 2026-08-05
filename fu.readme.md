# fu 脚本主流程

下面整理仓库里以 `30min` 频率为例的 `fu` (燃料油) 主流程 shell 脚本及其完整执行顺序、**各个步骤的前置依赖输入物**、关键产出物目录与文件（包含 Feather/NPY 数据集、`result/` 模型权重与回测诊断数据、`log/` 日志以及**用于分析与评估的 JSON 与 CSV 文件**）。

推荐串联运行顺序：

`main_30min_fu.sh` -> `commodity_data_handler_30min_fu.sh` -> `train_commodity_fu_30.sh` -> `test_util_fu_30.sh` -> `low_level_fu_30.sh` -> `VAE_util_fu_30.sh` -> `vae_optuna_fu_30.sh` -> `high_level_heurstic_fu_30_half.sh`

> **提示**：30min 专属脚本（`train_commodity_fu_30.sh`、`test_util_fu_30.sh`、`low_level_fu_30.sh`、`VAE_util_fu_30.sh`、`vae_optuna_fu_30.sh`）内部默认的 `EXPERIMENT_NAME` 均为 `30min`，保证了实验路径的一致性。

## 1. 数据预处理入口

### `data_preprocess/script_preprocess/future_upgraded/commodity/main_30min_fu.sh`

- **作用**：燃料油 `fu` 30min 频率下的商品期货预处理总入口。
- **依赖输入物**：
  - **脚本与算子库**：`data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh` 及 `operator_futures` 算子库
  - **原始行情数据**：`data/原始下载/` 目录下的 Level-2 Orderbook 与 K线 Tick CSV 文件
  - **规则配置**：`data_preprocess/operator_futures/commodity/config.py` 合约规范与交易时间段配置
- **默认参数**：
  - `ROOTPATH=$(pwd)`
  - `START_DATE=2023-01-01`
  - `END_DATE=2026-03-01`
  - `TARGET_FREQ=30min`
  - `SYMBOL=fu`
  - `COMMODITY_NAME=燃料油`
  - `MAX_PROCESSES=4`
- **关键产出物目录与文件**：
  - **连续与下采样基础数据目录**：
    - `PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/fu/`（主力合约连续原始数据及 `main_contract_summary.json`）
    - `PREPROCESS_DATASET/commodity-futures/BASE_FEATURE/fu/<contract>/30min/`（基础成交算子 Feather 数据）
    - `PREPROCESS_DATASET/commodity-futures/DOWNSCALE_ORDERBOOK_25/fu/<contract>/30min/`（25 档盘口 L5 下采样 Feather 数据）
    - `PREPROCESS_DATASET/commodity-futures/DOWNSCALE_DERTIC/fu/<contract>/30min/`（下采样派生 Tick 特征）
  - **截面算子特征目录 (`CROSS_SECTION/`)**：
    - `PREPROCESS_DATASET/commodity-futures/CROSS_SECTION/KLINE_FEATURE/fu/<contract>/30min/`（K 线算子截面特征）
    - `PREPROCESS_DATASET/commodity-futures/CROSS_SECTION/QUOTES_FEATURE/fu/<contract>/30min/`（Quotes 算子截面特征）
    - `PREPROCESS_DATASET/commodity-futures/CROSS_SECTION/SNAPSHOT_FEATURE/fu/<contract>/30min/`（Snapshot 算子截面特征）
  - **混合频率与状态特征计算目录**：
    - `PREPROCESS_DATASET/commodity-futures/DAILY_BASE_FEATURE/30min/fu/<contract>/`（日频基础特征）
    - `PREPROCESS_DATASET/commodity-futures/WEEKLY_BASE_FEATURE/30min/fu/<contract>/`（周频基础特征）
    - `PREPROCESS_DATASET/commodity-futures/CROSS_MONTH_FEATURE/30min/fu/<contract>/`（跨月/期限结构算子特征）
    - `PREPROCESS_DATASET/commodity-futures/DAILY_MIXED_FREQUENCY_FEATURE/30min/fu/<contract>/`（日频混合频率特征）
    - `PREPROCESS_DATASET/commodity-futures/WEEKLY_MIXED_FREQUENCY_FEATURE/30min/fu/<contract>/`（周频混合频率特征）
    - `PREPROCESS_DATASET/commodity-futures/MIXED_FREQUENCY_FEATURE/30min/fu/<contract>/`（综合混合频率特征）
  - **特征合并、时间扩展与清洗目录**：
    - `PREPROCESS_DATASET/commodity-futures/MERGE_FEATURE/30min/fu/<contract>/`（基础与状态特征合并数据）
    - `PREPROCESS_DATASET/commodity-futures/CONCAT_FEATURE/30min/fu/<contract>/`（时间拼接特征）
    - `PREPROCESS_DATASET/commodity-futures/TIME_FEATURE/30min/fu/<contract>/`（时间扩展算子特征数据）
    - `PREPROCESS_DATASET/commodity-futures/MERGE_CLEAN_FEATURE/30min/fu/<contract>/`（清洗与缺失值处理后全量特征数据）
  - **切分、特征选择与标准化目录**：
    - `PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/30min/fu/`（含 `dataset_split_manifest.json` 及 `train/`, `valid/`, `test/` 合约划分子目录）
    - `PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/30min/fu/`（含 `feature_selection_manifest.json`, `ic_window_*.json`, `rank_ic_window_*.json`, `correlation.csv`, `cat_boost_feature_importance_*.csv`, `aggregate_metrics.csv` 以及 `train/state_features.npy`）
    - `PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/fu/30min/`（含 `scaler_manifest.json`, `scale_diagnostics.csv` 及 `train/`, `valid/`, `test/` 标准化 `.feather` 数据）
  - **日志与保证金字典**：
    - `dataset/30min/fu/maintenance_margin_ratio_dict.npy`（维持保证金比例字典）
    - `log_futures/ticker_result/commodity/fu_30min_2023-01-01_2026-03-01.log`（主流程日志）
    - `log_futures/ticker_result/commodity/steps/`（包含 `cross_section`, `daily_base_feature`, `cross_month_feature`, `mixed_frequency_feature`, `merge`, `time_feature`, `scale_save` 等各个子步骤的分步日志）
- **位置**：整个 30min `fu` 流程的起点。

## 2. FineFT 数据准备

### `FineFT/script/data/commodity_data_handler_30min_fu.sh`

- **作用**：把 30min 商品期货预处理结果整理成 FineFT 训练所需的数据集结构，并生成 valid 动态切片和 VAE 数据。
- **依赖输入物**（由步骤 1 预处理产出）：
  - **数据集划分清单**：`PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/30min/fu/dataset_split_manifest.json`
  - **标准化特征数据**：`PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/fu/30min/`
  - **特征选择向量**：`PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/30min/fu/train/state_features.npy`
  - **数据处理 Python 脚本**：`FineFT/datahandler/commodity_contract_dataset.py`、`slice_model.py`、`vae_data_creation.py`
- **默认参数**：
  - `ROOTPATH=$(pwd)`
  - `SYMBOL=fu`
  - `TARGET_FREQ=30min`
  - `CHUNK_LENGTH=8000`
  - `EARLY_STOP=2`
- **主要步骤**：
  - 激活 `finetf` conda 环境。
  - 调用 `FineFT/datahandler/commodity_contract_dataset.py` 生成 `dataset/30min/fu`。
  - 对 `dataset/30min/fu/valid/*.feather` 逐个调用 `slice_model.py`。
  - 调用 `vae_data_creation.py` 生成 VAE 训练/测试数据。
- **关键产出物**：
  - **JSON 元数据清单**：
    - `dataset/30min/fu/dataset_manifest.json`（FineFT 数据集配置与入口元数据 JSON）
    - `dataset/30min/fu/valid/slice_manifest.json`（验证集切片划分清单 JSON）
  - **数据集与矩阵文件**：
    - `dataset/30min/fu/state_features.npy`、`maintenance_margin_ratio_dict.npy`（状态特征与维持保证金字典）
    - `dataset/30min/fu/train/*.feather`、`dataset/30min/fu/train/slice/df_*.feather`（训练合约数据及分块切片）
    - `dataset/30min/fu/valid/*.feather`、`valid/processed/valid_processed_*.feather`、`valid/<contract>/label_*/`（验证集切片与标签划分数据）
    - `dataset/30min/fu/test/*.feather`（测试集数据）
    - `dataset/30min/fu/VAE_data/<contract>/label_*.npy`、`dataset/30min/fu/VAE_data/test/test_*.npy`（VAE 训练/测试特征向量）
- **位置**：承接 30min 预处理结果，是进入 FineFT 训练、回测和 VAE 的数据准备入口。

## 3. 低层 agent 训练

### `FineFT/script/train/train_commodity_fu_30.sh`

- **作用**：训练 `fu` 的 30min 低层 agent。
- **依赖输入物**（由步骤 2 数据准备产出）：
  - **训练集数据与切片**：`dataset/30min/fu/train/*.feather` 或 `dataset/30min/fu/train/slice/df_*.feather`
  - **全局状态特征**：`dataset/30min/fu/state_features.npy`
  - **保证金字典**：`dataset/30min/fu/maintenance_margin_ratio_dict.npy`
  - **训练 Python 脚本**：`FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`
- **默认参数**：
  - `ROOTPATH=$(pwd)`
  - `EXPERIMENT_NAME=30min`
- **固定训练参数**：
  - `--base_path dataset/30min`
  - `--dataset_name fu`
  - `--experiment_name 30min`
  - `--initial_wallet_balance 10000`
  - `--batch_size 1024`
  - `--update_times 30`
  - `--max_holding_number 1`
  - `--position_choices 3`
  - `--transcation_cost 0.0004`
  - `--n_step 12`
  - `--gamma 0.99`
  - `--order_book_depth 5`
  - `--early_stop 2`
  - `--pretrain_epoch 100`
  - `--lr_init 0.0005`
  - `--epsilon_min 0.05`
  - `--allow_reverse_position`
- **关键产出物目录与文件**：
  - **`result/` 模型权重与诊断主产出目录**：
    - 主路径：`result/DiHFT/low_level/fu/30min/weights_advantage_pretrain/`
    - **各 Epoch 模型权重**：`epoch_1/` ~ `epoch_100/` 子目录，每个目录下包含 `trained_model.pkl`（各 Epoch 低层 Agent 强化学习神经网络模型文件，供后续回测 `test_util_fu_30.sh` 评估加载）
    - **TensorBoard 监控日志**：`result/DiHFT/low_level/fu/30min/weights_advantage_pretrain/log/`（包含 TensorBoard 训练日志 `events.out.tfevents.*`，用于可视化 Loss、Q 值标量及训练衰减曲线）
    - **Q 表预热与诊断数据**：`result/DiHFT/low_level/fu/30min/weights_advantage_pretrain/qtable_diagnostics/`（包含预热诊断元数据清单 `manifest.json` 以及动作-状态分布明细 CSV 文件 `df_*_initial_action_*.csv`）
  - **`log/` 运行文本日志**：
    - 控制台文本日志：`log/DiHFT/fu/low_level/train/30min/30min/advantage-30min.log`
- **位置**：承接 `commodity_data_handler_30min_fu.sh` 生成的 `dataset/30min/fu` 数据，是低层回测和 agent 筛选的前置训练步骤。

## 4. 低层回测/测试

### `FineFT/script/test/DiHFT/low_level/test_util_fu_30.sh`

- **作用**：批量跑 30min 低层 agent 的测试回测（epoch 11~100），生成每个 epoch 的结果日志与回测评估明细文件。
- **依赖输入物**（由步骤 2 & 步骤 3 产出）：
  - **低层 Agent 训练模型权重**（由步骤 3 产出）：`result/DiHFT/low_level/fu/30min/weights_advantage_pretrain/epoch_{epoch}/trained_model.pkl`（Epoch 11 ~ 100）
  - **验证集切片与数据**（由步骤 2 产出）：`dataset/30min/fu/valid/*.feather` 及 `dataset/30min/fu/valid/processed/valid_processed_*.feather`
  - **全局特征与保证金配置**（由步骤 2 产出）：`dataset/30min/fu/state_features.npy` 与 `maintenance_margin_ratio_dict.npy`
  - **测试 Python 脚本**：`FineFT/RL/DiHFT/low_level/test_agent_index.py`
- **默认参数**：
  - `DATASET_NAME=fu`
  - `BASE_PATH=dataset/30min`
  - `EPOCH_START=11`
  - `EPOCH_END=100`
  - `EXPERIMENT_NAME=30min`
  - `MAX_HOLDING_NUMBER=1`
- **关键产出物目录与文件**：
  - **`result/` 回测评估分析与交易动作明细产出**：
    - 主路径：`result/DiHFT/low_level/fu/30min/weights_advantage_pretrain/epoch_{epoch}/`
    - **评估汇总 CSV**：`analysis_result.csv`（保存该 Epoch 低层 Agent 在不同合约和 Label 上的收益、换手率、持仓步数等指标分析表，支持双语对照）
    - **评估矩阵 NPY**：`analysis_result.npy`（多维 NumPy 评估数据矩阵）
    - **逐笔交易动作明细 CSV**：（若开启 `--save_trading_detail_csv`）`trading_action_detail_epoch_{epoch}.csv`（记录逐 Step 动作、买卖平仓方向、持仓量及实时收益明细表）
  - **`log/` 回测测试日志**：
    - 控制台运行日志：`log/DiHFT/fu/low_level/test/30min/epoch_11.log` ~ `epoch_100.log`（各个 Epoch 在后台并行测试的日志文件）
- **位置**：给后面的 agent 筛选提供测试与评估数据。

## 5. 低层 agent 筛选与分析

### `FineFT/script/analysis/pick_agent/low_level_fu_30.sh`

- **作用**：根据 30min 低层测试结果筛选 agent。
- **依赖输入物**（由步骤 4 回测产出）：
  - **回测测试评估数据**（由步骤 4 产出）：`result/DiHFT/low_level/fu/30min/weights_advantage_pretrain/epoch_{epoch}/analysis_result.csv`（及 `analysis_result.npy`）
  - **验证集数据与标签配置**（由步骤 2 产出）：`dataset/30min/fu/valid/` 切片数据
  - **分析筛选 Python 脚本**：`FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py` 及 `conclude_metric.py`
- **默认参数**：
  - `DATASET_NAME=fu`
  - `EXPERIMENT_NAME=30min`
  - `BASE_PATH=dataset/30min`
  - `POSITION_CHOICES=3`
  - `NUM_LABEL=5`
  - `--epoch_num 100`
  - `--initial_position 0`
- **关键产出物**：
  - **分析与绩效评估 CSV / JSON**：
    - `log/analysis/pick_agent/DiHFT/fu/30min.log`（策略选拔执行日志）
    - `log/analysis/calculate_metric/fu.csv` / `fu.json`（各 Epoch 与不同持仓选择下的年化收益、夏普比率、最大回撤、胜率等财务与行为指标对比分析表）
    - `analysis_result/DiHFT/low_level/fu/30min/result.csv`（筛选后的 Agent 汇总结果）
    - `analysis_result/DiHFT/low_level/fu/30min/result_all.csv`（所有候选 Agent 的完整结果）
    - `analysis_result/DiHFT/low_level/fu/30min/selection_manifest.json`（最终 Label、模型路径、bin 及筛选信息）
- **位置**：承接低层测试结果，做筛选与财务指标分析。

## 6. VAE 训练与分布分析

### `FineFT/script/train/DiHFT/low_level/VAE_util_fu_30.sh`

- **作用**：按 `label_0 ~ label_4` 批量训练 30min VAE 模型，分析测试合约在各 Label 下的 Out-of-Distribution (OOD) 概率密度分布，并生成跨 Label 的路由对比摘要。
- **依赖输入物**（由步骤 2 & 步骤 5 产出）：
  - **VAE 向量数据集**（由步骤 2 产出）：`dataset/30min/fu/VAE_data/<contract>/label_*.npy` 及 `dataset/30min/fu/VAE_data/test/test_*.npy`
  - **Label 标注分布**（由步骤 2 & 步骤 5 确定）：`dataset/30min/fu/valid/` 下各个 label 的样本切片
  - **VAE 训练 Python 脚本**：`FineFT/RL/DiHFT/VAE/main.py`
- **默认参数**：
  - `DATASET_NAME=fu`
  - `DATA_BASE_PATH=dataset/30min`
  - `LABEL_COUNT=5`
  - `EXPERIMENT_NAME=30min`
  - `MAX_PARALLEL_JOBS=2`
- **关键产出物**：
  - **`result/` VAE 编解码器模型权重与分布评估分析文件**：
    - **分 Label 模型与诊断数据 (`result/DiHFT/vae_results/fu/30min/label_0/` ~ `label_4/`)**：
      - `model_latest.pth` / `model_epoch_*.pth`（各 Label 环境专属 VAE 编解码器 PyTorch 模型权重）
      - `id_logpx.npy`（In-Distribution 训练集样本基准 logpx 似然概率向量）
      - `ood_logpx_<contract>.npy` / `ood_logpx_<contract>.csv`（单合约 Out-of-Distribution 概率密度与对数似然估算）
      - `ood_logpx_all.npy` / `ood_logpx_all.csv`（全合约 OOD 似然分布向量与评估指标明细汇总）
      - `summary.json`（当前 Label VAE 的分布重构指标、分位数 `q01`~`q99` 及 OOD 接受度百分比评估摘要 JSON）
    - **跨 Label 路由对比摘要文件**：
      - `result/DiHFT/vae_results/fu/30min/routing_summary.json`（跨全套 VAE Label 模型的多环境 Winner 胜出分布、Margin 边际差异与样本对齐分析 JSON）
  - **`log/` 训练与分析日志**：
    - `log/DiHFT/fu/VAE/30min/train_label_0.log` ~ `train_label_4.log`（各 Label 环境并行训练及测试合约 OOD 分析日志）
- **位置**：按 README 的 Stage II 流程放在筛选之后。

## 7. 高层 VAE 路由 Optuna 优化与分析

### `FineFT/script/test/DiHFT/high_level/vae_optuna_fu_30.sh`

- **作用**：基于已训练好的 30min 低层 agent 和 VAE 模型，使用 Optuna 进行高层策略路由超参数搜索（如 `window_length`, `gamma`, `rule_base_threshold`），并输出策略回测诊断与多维时序数据。
- **依赖输入物**（前置各步骤联合依赖）：
  - **筛选出的最优低层 Agent 模型**（由步骤 3 & 步骤 5 产出）：`result/DiHFT/low_level/fu/30min/weights_advantage_pretrain/epoch_{best_epoch}/trained_model.pkl`
  - **已训练好的 VAE 路由模型与分布数据**（由步骤 6 产出）：`result/DiHFT/vae_results/fu/30min/label_*/` 下的 VAE 模型点与 `routing_summary.json`
  - **验证集切片与特征**（由步骤 2 产出）：`dataset/30min/fu/valid/` 及其 `processed/valid_processed_*.feather`
  - **全局特征与参数字典**（由步骤 2 产出）：`dataset/30min/fu/state_features.npy` 与 `maintenance_margin_ratio_dict.npy`
  - **Optuna 路由 Python 脚本**：`FineFT/RL/DiHFT/high_level/vae_routing_optuna.py`
- **默认参数**：
  - `DATASET_NAME=fu`
  - `BASE_PATH=dataset/30min`
  - `EXPERIMENT_NAME=30min`
  - `MAX_HOLDING_NUMBER=1`
- **关键产出物**：
  - **`result/` Optuna 寻优明细与路由策略回测诊断文件**：
    - **Optuna 寻优结果目录 (`result/DiHFT/high_level/fu/30min/vae_risk_aware_routing_optuna/`)**：
      - `optuna_results.csv`（Optuna 所有 Trial 参数组合、收益率、夏普比率、最大回撤等参数空间与回测明细表）
    - **路由策略回测诊断目录 (`result/DiHFT/high_level/fu/30min/vae_risk_aware_routing/gamma_{gamma}_window_{window}_threshold_{threshold}/`)**：
      - `contract_results.csv`（分合约的回测绩效明细，包含 `rows`, `reward_sum`, `require_money`, `return_rate`）
      - `trading_info.npy`（综合组合收益率、胜率、等权平均收益率等高层策略路由绩效指标字典）
      - **分合约回测细节向量 (`contracts/<contract>/`)**：
        - `reward_history.npy`（每步收益历史）、`total_asset_history.npy`（总资产变化历史）、`wallet_balance_history.npy`（钱包余额历史）、`initial_margin_history.npy`（初始保证金历史）、`unrealized_pnl_history.npy`（未实现盈亏历史）、`maintain_marigine_history.npy`（维持保证金历史）、`new_position_required_money_history.npy`（新开仓所需资金历史）
        - `micro_action_history.npy`（低层 Agent 微观动作历史）与 `macro_action.npy`（高层 VAE 路由宏观选择 Label 历史）
  - **`log/` 寻优控制台日志**：
    - `log/DiHFT/fu/high_level/optuna/30min/optuna.log`（Optuna 搜索过程、GPU 调度及各个 Trial 运行状态日志）
- **位置**：完成高层 VAE 路由与超参寻优。

## 8. 高层启发式路由策略筛选与分析

### `FineFT/script/analysis/pick_agent/high_level_heurstic_fu_30_half.sh`

- **作用**：基于步骤 7 高层 VAE 路由 Optuna 寻优得到的多组路由策略参数诊断数据，全量计算组合收益、最大回撤、夏普比率、卡玛比率等指标，筛选并保存最佳高层 Agent 路由配置与诊断结果，同时绘制单合约及全合约组合验证集收益对比曲线（PNG/PDF）。
- **依赖输入物**（前置各步骤联合依赖）：
  - **高层路由策略回测诊断数据**（由步骤 7 产出）：`result/DiHFT/high_level/fu/30min_multi/vae_risk_aware_routing/` 下各超参数目录（包含 `reward_history.npy`, `initial_margin_history.npy`, `maintain_marigine_history.npy`, `new_position_required_money_history.npy`, `unrealized_pnl_history.npy`, `wallet_balance_history.npy` 等）
  - **验证集 Feather 数据与行情戳**（由步骤 2 产出）：`dataset/30min/fu/valid/` 及其 `processed/valid_processed_*.feather`（用于计算基准 Buy & Hold 收益与画图 timestamp）
  - **启发式筛选与可视化 Python 脚本**：`FineFT/analysis/pick_agent/DiHFT_high_level_heurstic.py`
- **默认参数**：
  - `ROOTPATH=$(pwd)`
  - `BASE_PATH=dataset/30min`
  - `DATASET_NAME=fu`
  - `EXPERIMENT_NAME=30min_multi`
  - `SAVE_PATH=analysis_result/DiHFT/high_level_heurstic`
- **关键产出物**：
  - **`analysis_result/` 高层路由评估与对比图表目录 (`analysis_result/DiHFT/high_level_heurstic/fu/30min_multi/`)**：
    - `result.csv`（包含所有高层路由超参组合的平均收益率 `tr`、组合收益率 `portfolio_tr`、日波动率 `daily_vol`、最大回撤 `mdd`、夏普比率 `annual_sr`、卡玛比率 `daily_cr`、索提诺比率 `daily_SoR` 等多维指标汇总表）
    - `best_result.csv`（分别按最高收益率、夏普比率、卡玛比率及最低回撤等指标筛选出的最佳超参组合记录表）
    - `best_result_<contract>.png` / `.pdf`（各个单合约在验证集上 DiHFT 路由策略与 Buy & Hold 基准收益对比图）
    - `best_result_all_contracts.pdf` / `best_result.pdf`（全合约组合多面板验证集收益对比图）
  - **`result/` 最终 Agent 配置与诊断导出目录 (`result/DiHFT/final_result/fu/30min_multi/`)**：
    - `high_level_agent_para.txt`（记录最佳高层路由策略超参数目录名称）
    - `contracts/<contract>/` 及其根目录下导出的最终路由诊断数据向量（`.npy` 和 `.csv`）
  - **`log/` 筛选分析日志**：
    - `log/analysis/pick_agent/DiHFT/fu/high_level_heurstic/30min_multi.log`（筛选与画图控制台日志）
- **位置**：完成高层路由策略筛选与分析评估。

## 总结：全流程依赖输入与产出对照表

| 阶段 / 步骤 | 核心执行脚本 | 核心依赖输入物 | 关键产出物目录与核心文件 |
| :--- | :--- | :--- | :--- |
| **1. 数据预处理** | `main_30min_fu.sh` | `data/原始下载/` (Tick/KLine CSV)<br>`config.py` (合约规则) | `PREPROCESS_DATASET/commodity-futures/`<br>(`SPLIT-TRAIN-VALID-TEST`, `FEATURE_SELECTION`, `SCALE_SAVE`) |
| **2. FineFT 数据准备** | `commodity_data_handler_30min_fu.sh` | 步骤 1 产出的 `SCALE_SAVE`, `dataset_split_manifest.json`, `state_features.npy` | `dataset/30min/fu/`<br>(`train/`, `valid/`, `test/`, `VAE_data/`, `dataset_manifest.json`) |
| **3. 低层 Agent 训练** | `train_commodity_fu_30.sh` | 步骤 2 产出的 `dataset/30min/fu/train/`, `state_features.npy`, `margin_dict.npy` | `result/DiHFT/low_level/fu/30min/weights_advantage_pretrain/`<br>(`epoch_1/`~`epoch_100/trained_model.pkl`, `log/`, `qtable_diagnostics/`) |
| **4. 低层 Agent 测试** | `test_util_fu_30.sh` | 步骤 3 产出的 `trained_model.pkl` + 步骤 2 产出的 `valid/` 数据 | `result/DiHFT/low_level/fu/30min/weights_advantage_pretrain/epoch_{epoch}/`<br>(`analysis_result.csv`, `trading_action_detail_*.csv`, `analysis_result.npy`) |
| **5. 低层 Agent 筛选** | `low_level_fu_30.sh` | 步骤 4 产出的 `analysis_result.csv` + 步骤 2 产出的 `valid/` 数据 | `analysis_result/DiHFT/low_level/fu/30min/`<br>(`result.csv`, `result_all.csv`, `selection_manifest.json`) |
| **6. VAE 训练与评估** | `VAE_util_fu_30.sh` | 步骤 2 产出的 `VAE_data/<contract>/label_*.npy` + 步骤 5 的 Label 划分 | `result/DiHFT/vae_results/fu/30min/`<br>(`label_*/model_latest.pth`, `summary.json`, `ood_logpx_*.csv`, `routing_summary.json`) |
| **7. 高层 Optuna 寻优** | `vae_optuna_fu_30.sh` | 步骤 3/5 筛选的 Agent 模型 + 步骤 6 的 VAE 模型 + 步骤 2 的 `valid/` 数据 | `result/DiHFT/high_level/fu/30min/`<br>(`vae_risk_aware_routing_optuna/optuna_results.csv`, `vae_risk_aware_routing/.../contract_results.csv`, `trading_info.npy`, `macro_action.npy`) |
| **8. 高层路由筛选与可视化** | `high_level_heurstic_fu_30_half.sh` | 步骤 7 产出的 `vae_risk_aware_routing/` 诊断数据 + 步骤 2 的 `valid/*.feather` | `analysis_result/DiHFT/high_level_heurstic/fu/30min_multi/`<br>(`result.csv`, `best_result.csv`, `best_result_*.png/pdf`)<br>`result/DiHFT/final_result/fu/30min_multi/`<br>(`high_level_agent_para.txt`, 最终路由诊断向量 `.npy`/`.csv`) |
