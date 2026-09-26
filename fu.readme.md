# fu 脚本主流程

下面整理仓库里以 `10min` 并行版本为例的 `fu` (燃料油) 主流程 shell 脚本及其完整执行顺序、**各个步骤的前置依赖输入物**、关键产出物目录与文件（包含 Feather/NPY 数据集、`result/` 模型权重与回测诊断数据、`log/` 日志以及**用于分析与评估的 JSON 与 CSV 文件**）。

推荐串联运行顺序：

`main_10min_fu.sh` -> `commodity_data_handler_10min_fu.sh` -> `train_commodity_fu_10_parallel.sh` -> `test_util_fu_10.sh` -> `low_level_fu_10.sh` -> `VAE_util_fu_10.sh` -> `vae_optuna_fu_10.sh` -> `high_level_heurstic_fu_10.sh` -> `final_result_fu_10_p.sh`

> **提示**：10min 并行专属脚本（`train_commodity_fu_10_parallel.sh`、`test_util_fu_10.sh`、`low_level_fu_10.sh`、`VAE_util_fu_10.sh`、`vae_optuna_fu_10.sh`、`high_level_heurstic_fu_10.sh`、`final_result_fu_10_p.sh`）内部默认的 `EXPERIMENT_NAME` 均为 `10min_parallel`，保证了实验路径的一致性。

## 1. 数据预处理入口

### `data_preprocess/script_preprocess/future_upgraded/commodity/main_10min_fu.sh`

- **作用**：燃料油 `fu` 10min 频率下的商品期货预处理总入口。
- **依赖输入物**：
  - **脚本与算子库**：`data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh` 及 `operator_futures` 算子库
  - **原始行情数据**：`data/原始下载/` 目录下的 Level-2 Orderbook 与 K线 Tick CSV 文件
  - **规则配置**：`data_preprocess/operator_futures/commodity/config.py` 合约规范与交易时间段配置
- **默认参数**：
  - `ROOTPATH=$(pwd)`
  - `START_DATE=2023-01-01`
  - `END_DATE=2026-03-01`
  - `TARGET_FREQ=10min`
  - `SYMBOL=fu`
  - `COMMODITY_NAME=燃料油`
  - `MAX_PROCESSES=4`
  - `REGIME_BINS=3`
- **关键产出物目录与文件**：
  - **连续与下采样基础数据目录**：
    - `PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/fu/`（主力合约连续原始数据及 `main_contract_summary.json`）
    - `PREPROCESS_DATASET/commodity-futures/BASE_FEATURE/fu/<contract>/10min/`（基础成交算子 Feather 数据）
    - `PREPROCESS_DATASET/commodity-futures/DOWNSCALE_ORDERBOOK_25/fu/<contract>/10min/`（25 档盘口 L5 下采样 Feather 数据）
    - `PREPROCESS_DATASET/commodity-futures/DOWNSCALE_DERTIC/fu/<contract>/10min/`（下采样派生 Tick 特征）
  - **截面算子特征目录 (`CROSS_SECTION/`)**：
    - `PREPROCESS_DATASET/commodity-futures/CROSS_SECTION/KLINE_FEATURE/fu/<contract>/10min/`（K 线算子截面特征）
    - `PREPROCESS_DATASET/commodity-futures/CROSS_SECTION/QUOTES_FEATURE/fu/<contract>/10min/`（Quotes 算子截面特征）
    - `PREPROCESS_DATASET/commodity-futures/CROSS_SECTION/SNAPSHOT_FEATURE/fu/<contract>/10min/`（Snapshot 算子截面特征）
  - **混合频率与状态特征计算目录**：
    - `PREPROCESS_DATASET/commodity-futures/DAILY_BASE_FEATURE/10min/fu/<contract>/`（日频基础特征）
    - `PREPROCESS_DATASET/commodity-futures/WEEKLY_BASE_FEATURE/10min/fu/<contract>/`（周频基础特征）
    - `PREPROCESS_DATASET/commodity-futures/CROSS_MONTH_FEATURE/10min/fu/<contract>/`（跨月/期限结构算子特征）
    - `PREPROCESS_DATASET/commodity-futures/DAILY_MIXED_FREQUENCY_FEATURE/10min/fu/<contract>/`（日频混合频率特征）
    - `PREPROCESS_DATASET/commodity-futures/WEEKLY_MIXED_FREQUENCY_FEATURE/10min/fu/<contract>/`（周频混合频率特征）
    - `PREPROCESS_DATASET/commodity-futures/MIXED_FREQUENCY_FEATURE/10min/fu/<contract>/`（综合混合频率特征）
  - **特征合并、时间扩展与清洗目录**：
    - `PREPROCESS_DATASET/commodity-futures/MERGE_FEATURE/10min/fu/<contract>/`（基础与状态特征合并数据）
    - `PREPROCESS_DATASET/commodity-futures/CONCAT_FEATURE/10min/fu/<contract>/`（时间拼接特征）
    - `PREPROCESS_DATASET/commodity-futures/TIME_FEATURE/10min/fu/<contract>/`（时间扩展算子特征数据）
    - `PREPROCESS_DATASET/commodity-futures/MERGE_CLEAN_FEATURE/10min/fu/<contract>/`（清洗与缺失值处理后全量特征数据）
  - **切分、特征选择与标准化目录**：
    - `PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/10min/fu/`（含 `dataset_split_manifest.json` 及 `train/`, `valid/`, `test/` 合约划分子目录）
    - `PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/10min/fu/`（含 `feature_selection_manifest.json`, `ic_window_*.json`, `rank_ic_window_*.json`, `correlation.csv`, `cat_boost_feature_importance_*.csv`, `aggregate_metrics.csv` 以及 `train/state_features.npy`）
    - `PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/fu/10min/`（含 `scaler_manifest.json`, `scale_diagnostics.csv` 及 `train/`, `valid/`, `test/` 标准化 `.feather` 数据）
  - **日志与保证金字典**：
    - `dataset/10min/fu/maintenance_margin_ratio_dict.npy`（维持保证金比例字典）
    - `log_futures/ticker_result/commodity/fu_10min_2023-01-01_2026-03-01.log`（主流程日志）
    - `log_futures/ticker_result/commodity/steps/`（包含 `cross_section`, `daily_base_feature`, `cross_month_feature`, `mixed_frequency_feature`, `merge`, `time_feature`, `scale_save` 等各个子步骤的分步日志，如 `fu_10min_2023-01-01_2026-03-01_*.log`）
- **位置**：整个 10min `fu` 流程的起点。

## 2. FineFT 数据准备

### `FineFT/script/data/commodity_data_handler_10min_fu.sh`

- **作用**：把 10min 商品期货预处理结果整理成 FineFT 训练所需的数据集结构，校准验证集和训练集的动态标签切片，并生成 VAE 数据。
- **依赖输入物**（由步骤 1 预处理产出）：
  - **数据集划分清单**：`PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/10min/fu/dataset_split_manifest.json`
  - **标准化特征数据**：`PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/fu/10min/`
  - **特征选择向量**：`PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/10min/fu/train/state_features.npy`
  - **数据处理 Python 脚本**：`FineFT/datahandler/commodity_contract_dataset.py`、`valid_cross_contract_label_calibration.py`、`vae_data_creation.py`
- **默认参数**：
  - `ROOTPATH=$(pwd)`
  - `SYMBOL=fu`
  - `TARGET_FREQ=10min`
  - `CHUNK_LENGTH=5000`
  - `EARLY_STOP=2`
- **主要步骤**：
  - 激活 `finetf` conda 环境。
  - 调用 `FineFT/datahandler/commodity_contract_dataset.py` 生成 `dataset/10min/fu`。
  - 调用 `FineFT/datahandler/valid_cross_contract_label_calibration.py` 分别对 `valid` 与 `train` 数据集按 `slope` 和 `volatility` 进行跨合约标签校准与切片划分。
  - 调用 `FineFT/datahandler/vae_data_creation.py` 导出 VAE 训练数据（`slope` 与 `volatility`）。
- **关键产出物**：
  - **JSON 元数据清单**：
    - `dataset/10min/fu/dataset_manifest.json`（FineFT 数据集配置与入口元数据 JSON）
    - `dataset/10min/fu/valid/slice_manifest.json`（验证集切片划分清单 JSON）
  - **数据集与矩阵文件**：
    - `dataset/10min/fu/state_features.npy`、`maintenance_margin_ratio_dict.npy`（状态特征与维持保证金字典）
    - `dataset/10min/fu/train/*.feather`、`dataset/10min/fu/train/slice/df_*.feather`（训练合约数据及分块切片）
    - `dataset/10min/fu/valid/*.feather`、`valid/processed/valid_processed_*.feather`、`valid/<contract>/label_*/`（验证集切片与标签划分数据）
    - `dataset/10min/fu/test/*.feather`（测试集数据）
    - `dataset/10min/fu/VAE_data/<labeling_method>/<contract>/label_*.npy`（VAE 训练/测试特征向量）
- **位置**：承接 10min 预处理结果，是进入 FineFT 训练、回测和 VAE 的数据准备入口。

## 3. 低层 Agent 并行训练

### `FineFT/script/train/train_commodity_fu_10_parallel.sh`

- **作用**：训练 `fu` 的 10min 低层 agent，采用多 Worker 探索与权重优势预训练的并行训练机制。
- **依赖输入物**（由步骤 2 数据准备产出）：
  - **训练集数据与切片**：`dataset/10min/fu/train/*.feather` 或 `dataset/10min/fu/train/slice/df_*.feather`
  - **全局状态特征**：`dataset/10min/fu/state_features.npy`
  - **保证金字典**：`dataset/10min/fu/maintenance_margin_ratio_dict.npy`
  - **训练 Python 脚本**：`FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py`
- **默认参数**：
  - `ROOTPATH=$(pwd)`
  - `EXPERIMENT_NAME=10min_parallel`
- **固定训练参数**：
  - `--base_path dataset/10min`
  - `--dataset_name fu`
  - `--experiment_name 10min_parallel`
  - `--initial_wallet_balance 6000`
  - `--batch_size 102400`
  - `--update_times 600`
  - `--diverse_num_workers 96`（96 进程多 Worker 并行探索与采样）
  - `--max_holding_number 1`
  - `--short_estimated_rate 0`
  - `--long_estimated_rate 0`
  - `--position_choices 3`
  - `--transcation_cost 0.007`
  - `--n_step 18`
  - `--gamma 0.992`
  - `--order_book_depth 5`
  - `--early_stop 2`
  - `--N 13`（集成网络基模型数）
  - `--buffer_size 1000000`
  - `--pretrain_epoch 5`
  - `--curriculum_block_epochs 6`
  - `--num_epoch 75`
  - `--lr_init 0.0005`
  - `--lr_min 0.0001`
  - `--ada_init 96.0`
  - `--epsilon_min 0.05`
  - `--ada_min 0.1`
  - `--neighbor_size 2`
  - `--load_pretrain_model False`
- **关键产出物目录与文件**：
  - **`result/` 模型权重与诊断主产出目录**：
    - 主路径：`result/DiHFT/low_level/fu/10min_parallel/weights_advantage_pretrain/`
    - **各 Epoch 模型权重**：`epoch_1/` ~ `epoch_75/` 子目录，每个目录下包含 `trained_model.pkl`（各 Epoch 低层 Agent 强化学习神经网络模型文件，供后续回测 `test_util_fu_10.sh` 评估加载）
    - **TensorBoard 监控日志**：`result/DiHFT/low_level/fu/10min_parallel/weights_advantage_pretrain/log/`（包含 TensorBoard 训练日志 `events.out.tfevents.*`，用于可视化 Loss、Q 值标量及训练衰减曲线）
    - **评估日志**：`result/DiHFT/low_level/fu/10min_parallel/weights_advantage_pretrain/pretrain_evaluation.log`
  - **`log/` 运行文本日志**：
    - 控制台文本日志：`log/DiHFT/fu/low_level/train/10min/10min_parallel/advantage-10min-parallel.log`
- **位置**：承接 `commodity_data_handler_10min_fu.sh` 生成的 `dataset/10min/fu` 数据，是低层回测和 agent 筛选的前置训练步骤。

## 4. 低层回测/测试 (并行测试)

### `FineFT/script/test/DiHFT/low_level/test_util_fu_10.sh`

- **作用**：批量并行跑 10min 低层 agent 的测试回测（默认 epoch 1~75，同时覆盖 slope 和 volatility 两种环境切分），并发执行多达 96 个测试进程，生成每个 epoch 的结果日志与回测评估明细文件。
- **依赖输入物**（由步骤 2 & 步骤 3 产出）：
  - **低层 Agent 训练模型权重**（由步骤 3 产出）：`result/DiHFT/low_level/fu/10min_parallel/weights_advantage_pretrain/epoch_{epoch}/trained_model.pkl`（Epoch 1 ~ 75）
  - **验证集切片与数据**（由步骤 2 产出）：`dataset/10min/fu/valid/*.feather` 及 `dataset/10min/fu/valid/processed/valid_processed_*.feather`
  - **全局特征与保证金配置**（由步骤 2 产出）：`dataset/10min/fu/state_features.npy` 与 `maintenance_margin_ratio_dict.npy`
  - **测试 Python 脚本**：`FineFT/RL/DiHFT/low_level/test_agent_index.py`
- **默认参数**：
  - `DATASET_NAME=fu`
  - `BASE_PATH=dataset/10min`
  - `EPOCH_START=1`
  - `EPOCH_END=75`
  - `EXPERIMENT_NAME=10min_parallel`
  - `MAX_HOLDING_NUMBER=1`
  - `DEVICE=cpu`
  - `ENSEMBLE_NUMBER=13`
  - `MAX_PARALLEL=96`（最大 96 进程并发回测）
  - `LABEL_TYPES=("slope" "volatility")`
  - `--save_trading_detail_csv`
- **关键产出物目录与文件**：
  - **`result/` 回测评估分析与交易动作明细产出**：
    - 主路径：`result/DiHFT/low_level/fu/10min_parallel/weights_advantage_pretrain/epoch_{epoch}/{slope,volatility}/`
    - **评估汇总 CSV**：`analysis_result.csv`（保存该 Epoch 低层 Agent 在该 Label 切分下不同合约和 Label 上的收益、换手率、持仓步数等指标分析表，支持双语对照）
    - **评估矩阵 NPY**：`analysis_result.npy`（多维 NumPy 评估数据矩阵）
    - **逐笔交易动作明细 CSV**：`trading_action_detail_epoch_{epoch}.csv`（记录逐 Step 动作、买卖平仓方向、持仓量及实时收益明细表）
  - **`log/` 回测测试日志**：
    - 控制台运行日志：`log/DiHFT/fu/low_level/test/10min_parallel/{slope,volatility}/epoch_{epoch}.log`（各个 Epoch 在后台并行测试的日志文件）
- **位置**：给后面的二维 agent 筛选提供测试与评估数据。

## 5. 低层 Agent 二维联合筛选与分析

### `FineFT/script/analysis/pick_agent/low_level_fu_10.sh`

- **作用**：基于 10min 并行回测结果，针对 `slope`（斜率趋势）与 `volatility`（波动率）两个市场环境维度执行二维联合筛选（Two-Dimensional Agent Selection），通过 LCB 下置信界、各合约正收益比例、最差初始持仓收益门槛等鲁棒性准则选拔出各动态环境的最优 Agent Slot 与集成模型权重。
- **依赖输入物**（由步骤 4 回测产出）：
  - **回测测试评估数据**（由步骤 4 产出）：`result/DiHFT/low_level/fu/10min_parallel/weights_advantage_pretrain/epoch_{epoch}/{slope,volatility}/analysis_result.csv`
  - **验证集数据与切片配置**（由步骤 2 产出）：`dataset/10min/fu/valid/` 切片数据
  - **分析筛选 Python 脚本**：`FineFT/analysis/pick_agent/FineFT_two_dimensional_agent_selector.py`
- **默认参数**：
  - `DATASET_NAME=fu`
  - `EXPERIMENT_NAME=10min_parallel`
  - `BASE_PATH=dataset/10min`
  - `POSITION_CHOICES=3`
  - `NUM_LABELS=3`
  - `LCB_Z=0.20`
  - `MIN_MARGINAL_CONTRACTS=1`
  - `MIN_JOINT_CONTRACTS=1`
  - `MIN_POSITIVE_CONTRACT_RATIO=0.60`
  - `MIN_MEAN_RETURN=0.0`
  - `MIN_LCB=0.0`
  - `MIN_WORST_INITIAL_POSITION_RETURN=-1.0`
  - `MIN_WORST_INITIAL_POSITION_RETURN_V2=-3.0`
  - `MIN_WORST_INITIAL_POSITION_RETURN_V3=-5.0`
  - `MISSING_JOINT_POLICY=empty_model`
  - `CONTRACT_WEIGHTING=step_weighted`
  - `MIN_SLICE_STEPS=30`
- **关键产出物**：
  - **分析与绩效评估 CSV / JSON (`analysis_result/DiHFT/low_level/fu/10min_parallel/two_dimensional_selection/`)**：
    - `two_dimensional_selection_manifest.json`（核心选拔清单，记录选中的 Slot、对应 Epoch、环境 Label、权重路径及聚合指标）
    - `marginal_metrics.csv`（单维度边际收益与评估指标表）
    - `joint_metrics.csv`（两维度联合分布下的评估指标表）
    - `candidate_rankings.csv`（所有候选 Epoch / Agent 的综合评分与排名清单）
    - `selected_slots.csv`（最终锁定的各 Slot 最优策略清单）
  - **`log/` 选拔运行日志**：
    - `log/analysis/pick_agent/DiHFT/fu/10min_parallel.log`（二维筛选控制台日志）
- **位置**：衔接低层并行测试与高层 Optuna 寻优，为高层 VAE 路由提供低层 Agent 组合。

## 6. VAE 训练与评估 (多进程并行)

### `FineFT/script/train/DiHFT/low_level/VAE_util_fu_10.sh`

- **作用**：基于 10min 提取出的 VAE 动态特征，多进程并发对 `slope` 和 `volatility` 各动态标签（0 ~ 2，共 3 类标签）训练变分自编码器（VAE），拟合各市场环境的数据分布并计算样本重构似然值，为高层风险感知路由提供无监督环境分类与 OOD 检测能力。
- **依赖输入物**（由步骤 2 数据准备与步骤 5 标签划分产出）：
  - **VAE 训练与测试特征向量**（由步骤 2 产出）：`dataset/10min/fu/VAE_data/{slope,volatility}/<contract>/label_*.npy` 及 `dataset/10min/fu/VAE_data/test/test_*.npy`
  - **VAE 训练 Python 脚本**：`FineFT/RL/DiHFT/VAE/main.py`
- **默认参数**：
  - `ROOTPATH=$(pwd)`
  - `DATASET_NAME=fu`
  - `DATA_BASE_PATH=dataset/10min`
  - `LABEL_COUNT=3`
  - `EXPERIMENT_NAME=10min_parallel`
  - `MAX_PARALLEL_JOBS=2`（双任务并行并发调度）
  - `LABELING_METHODS=("slope" "volatility")`
  - `LOG_BASE_DIR=log/DiHFT`
- **关键产出物**：
  - **`result/` VAE 训练模型与 OOD 评估分析主目录 (`result/DiHFT/vae_results/fu/10min_parallel/{slope,volatility}/`)**：
    - **各 Label 模型权重**：`label_0/` ~ `label_2/` 子目录，每个目录下包含各 Step 检查点以及 `model_latest.pth`（最终保存的 VAE 编码器-解码器 PyTorch 权重模型）
    - **拟合汇总 JSON**：`label_{label_index}/summary.json`（记录 ELBO Loss、重构项对数似然度、KL 散度等收敛指标）
    - **测试合约 OOD 对数似然评估明细 CSV**：`label_{label_index}/ood_logpx_<contract>.csv` 与 `ood_logpx_all.csv`（测试合约各时间步的对数似然度量）
    - **矩阵向量 NPY**：`id_logpx.npy`（训练分布内部 log p(x)）、`ood_logpx_<contract>.npy` 与 `ood_logpx_all.npy`（测试集各时间步似然值矩阵）
    - **跨环境路由汇总 JSON**：`routing_summary.json`（跨全套 VAE Label 模型的多环境 Winner 胜出分布、Margin 边际差异与样本对齐分析 JSON）
  - **`log/` 训练与分析日志**：
    - `log/DiHFT/fu/VAE/10min_parallel/{slope,volatility}/train_label_{label_index}.log`（各 Label 环境并行训练及测试合约 OOD 分析日志）
- **位置**：按 README 的 Stage II 流程放在筛选之后。

## 7. 高层 VAE 路由 Optuna 优化与分析 (并行寻优)

### `FineFT/script/test/DiHFT/high_level/vae_optuna_fu_10.sh`

- **作用**：基于已训练好的 10min 低层 agent 和 VAE 模型，使用 Optuna 进行高层策略路由超参数搜索（如 `window_length`, `gamma`, `rule_base_threshold`），利用多 Worker CPU 并行（默认 45 进程）高效探索策略参数空间，开启非主力合约防御机制，并输出策略回测诊断与多维时序数据。
- **依赖输入物**（前置各步骤联合依赖）：
  - **筛选出的二维低层 Agent 清单与模型**（由步骤 3 & 步骤 5 产出）：`analysis_result/DiHFT/low_level/fu/10min_parallel/two_dimensional_selection/two_dimensional_selection_manifest.json` 及对应的 `trained_model.pkl`
  - **已训练好的 VAE 路由模型与分布数据**（由步骤 6 产出）：`result/DiHFT/vae_results/fu/10min_parallel/{slope,volatility}/label_*/` 下的 VAE 模型点与 `routing_summary.json`
  - **验证集切片与特征**（由步骤 2 产出）：`dataset/10min/fu/valid/` 及其 `processed/valid_processed_*.feather`
  - **全局特征与参数字典**（由步骤 2 产出）：`dataset/10min/fu/state_features.npy` 与 `maintenance_margin_ratio_dict.npy`
  - **Optuna 路由 Python 脚本**：`FineFT/RL/DiHFT/high_level/vae_routing_optuna.py`
- **默认参数**：
  - `DATASET_NAME=fu`
  - `BASE_PATH=dataset/10min`
  - `EXPERIMENT_NAME=10min_parallel`
  - `MAX_HOLDING_NUMBER=1`
  - `ENABLE_NON_MAIN_DEFENSE=1`
  - `N_WORKERS=45`（45 并发进程并行搜索）
  - `--initial_wallet_balance 6000`
  - `--position_choices 3`
  - `--order_book_depth 5`
  - `--transcation_cost 0.0005`
  - `--short_estimated_rate 0`
  - `--long_estimated_rate 0`
  - `--enable_non_main_contract_defense`
- **关键产出物**：
  - **`result/` Optuna 寻优明细与路由策略回测诊断文件**：
    - **Optuna 寻优结果目录 (`result/DiHFT/high_level/fu/10min_parallel/vae_risk_aware_routing_optuna/`)**：
      - `optuna_results.csv`（Optuna 所有 Trial 参数组合、收益率、夏普比率、最大回撤等参数空间与回测明细表）
    - **路由策略回测诊断目录 (`result/DiHFT/high_level/fu/10min_parallel/vae_risk_aware_routing/.../`)**：
      - `contract_results.csv`（分合约的回测绩效明细，包含 `rows`, `reward_sum`, `require_money`, `return_rate`）
      - `trading_info.npy`（综合组合收益率、胜率、等权平均收益率等高层策略路由绩效指标字典）
      - **分合约回测细节向量 (`contracts/<contract>/`)**：
        - `reward_history.npy`（每步收益历史）、`total_asset_history.npy`（总资产变化历史）、`wallet_balance_history.npy`（钱包余额历史）、`initial_margin_history.npy`（初始保证金历史）、`unrealized_pnl_history.npy`（未实现盈亏历史）、`maintain_marigine_history.npy`（维持保证金历史）、`new_position_required_money_history.npy`（新开仓所需资金历史）
        - `micro_action_history.npy`（低层 Agent 微观动作历史）与 `macro_action.npy`（高层 VAE 路由宏观选择 Label 历史）
  - **`log/` 寻优控制台日志**：
    - `log/DiHFT/fu/high_level/optuna/10min_parallel/optuna.log`（Optuna 搜索过程、多 Worker 调度及各个 Trial 运行状态日志）
- **位置**：完成高层 VAE 路由与超参寻优。

## 8. 高层启发式路由策略筛选与分析

### `FineFT/script/analysis/pick_agent/high_level_heurstic_fu_10.sh`

- **作用**：基于步骤 7 高层 VAE 路由 Optuna 寻优得到的多组路由策略参数诊断数据，全量计算组合收益、最大回撤、夏普比率、卡玛比率等指标，筛选并保存最佳高层 Agent 路由配置与诊断结果，同时绘制单合约及全合约组合验证集收益对比曲线（PNG/PDF）。
- **依赖输入物**（前置各步骤联合依赖）：
  - **高层路由策略回测诊断数据**（由步骤 7 产出）：`result/DiHFT/high_level/fu/10min_parallel/vae_risk_aware_routing/` 下各超参数目录（包含 `reward_history.npy`, `initial_margin_history.npy`, `maintain_marigine_history.npy`, `new_position_required_money_history.npy`, `unrealized_pnl_history.npy`, `wallet_balance_history.npy` 等）
  - **Optuna 寻优结果表**（由步骤 7 产出）：`result/DiHFT/high_level/fu/10min_parallel/vae_risk_aware_routing_optuna/optuna_results.csv`
  - **验证集 Feather 数据与行情戳**（由步骤 2 产出）：`dataset/10min/fu/valid/` 及其 `processed/valid_processed_*.feather`（用于计算基准 Buy & Hold 收益与画图 timestamp）
  - **启发式筛选与可视化 Python 脚本**：`FineFT/analysis/pick_agent/DiHFT_high_level_heurstic.py`
- **默认参数**：
  - `ROOTPATH=$(pwd)`
  - `BASE_PATH=dataset/10min`
  - `DATASET_NAME=fu`
  - `EXPERIMENT_NAME=10min_parallel`
  - `SAVE_PATH=analysis_result/DiHFT/high_level_heurstic`
  - `RESULT_PATH=result/DiHFT/high_level`
  - `SELECTION_METRIC=tr`
  - `EARLY_STOP=0`
  - `FOREGROUND=0`
- **关键产出物**：
  - **`analysis_result/` 高层路由评估与对比图表目录 (`analysis_result/DiHFT/high_level_heurstic/fu/10min_parallel/`)**：
    - `result.csv`（包含所有高层路由超参组合的平均收益率 `tr`、组合收益率 `portfolio_tr`、日波动率 `daily_vol`、最大回撤 `mdd`、夏普比率 `annual_sr`、卡玛比率 `daily_cr`、索提诺比率 `daily_SoR` 等多维指标汇总表）
    - `best_result.csv`（分别按最高收益率、夏普比率、卡玛比率及最低回撤等指标筛选出的最佳超参组合记录表）
    - `best_result_<contract>.png` / `.pdf`（各个单合约在验证集上 DiHFT 路由策略与 Buy & Hold 基准收益对比图）
    - `best_result_all_contracts.pdf` / `best_result.pdf`（全合约组合多面板验证集收益对比图）
  - **`result/` 最终 Agent 配置与诊断导出目录 (`result/DiHFT/final_result/fu/10min_parallel/`)**：
    - `high_level_agent_para.txt`（记录最佳高层路由策略超参数目录名称）
    - `contracts/<contract>/` 及其根目录下导出的最终路由诊断数据向量（`.npy` 和 `.csv`）
  - **`log/` 筛选分析日志**：
    - `log/analysis/pick_agent/DiHFT/fu/high_level_heurstic/10min_parallel.log`（筛选与画图控制台日志）
- **位置**：完成高层路由策略筛选与分析评估。

## 9. 高层路由测试集最终回测与评估 (Final Result)

### `FineFT/script/test/DiHFT/high_level/final_result_fu_10_p.sh`

- **作用**：加载步骤 8 筛选出的高层最优路由超参数与步骤 5 筛选出的低层 Agent 组合，在测试集（test stage）全量合约上运行最终的风险感知 VAE 宏观路由与微观交易回测评估，记录逐合约与整体投资组合的最终收益表现与动作历史，并开启非主力合约防御机制。
- **依赖输入物**（前置各步骤联合依赖）：
  - **高层最优超参数配置**（由步骤 8 产出）：`result/DiHFT/final_result/${DATASET_NAME}/${EXPERIMENT_NAME}/high_level_agent_para.txt`
  - **Optuna 寻优结果表**（由步骤 7 产出）：`result/DiHFT/high_level/${DATASET_NAME}/${EXPERIMENT_NAME}/vae_risk_aware_routing_optuna/optuna_results.csv`
  - **低层 Agent 筛选清单**（由步骤 5 产出）：`analysis_result/DiHFT/low_level/${DATASET_NAME}/${EXPERIMENT_NAME}/two_dimensional_selection/two_dimensional_selection_manifest.json`
  - **已训练好的 VAE 模型**（由步骤 6 产出）：`result/DiHFT/vae_results/${DATASET_NAME}/${EXPERIMENT_NAME}/` 下各 Label 的模型权重
  - **测试集 Feather 行情与特征**（由步骤 2 产出）：`dataset/10min/${DATASET_NAME}/test/*.feather`、`state_features.npy` 与 `maintenance_margin_ratio_dict.npy`
  - **最终评估 Python 脚本**：`FineFT/RL/DiHFT/high_level/vae_routing_final_result_macro_action.py`
- **默认参数**：
  - `ROOTPATH=$(pwd)`
  - `DATASET_NAME=fu`
  - `BASE_PATH=dataset/10min`
  - `EXPERIMENT_NAME=10min_parallel`
  - `MAX_HOLDING_NUMBER=1`
  - `POSITION_CHOICES=3`
  - `ORDER_BOOK_DEPTH=5`
  - `TRANSACTION_COST=0.0005`
  - `PARA_FILE=result/DiHFT/final_result/${DATASET_NAME}/${EXPERIMENT_NAME}/high_level_agent_para.txt`
  - `OPTUNA_CSV=result/DiHFT/high_level/${DATASET_NAME}/${EXPERIMENT_NAME}/vae_risk_aware_routing_optuna/optuna_results.csv`
  - `SELECTION_MANIFEST=analysis_result/DiHFT/low_level/${DATASET_NAME}/${EXPERIMENT_NAME}/two_dimensional_selection/two_dimensional_selection_manifest.json`
- **固定执行参数**：
  - `--eval_stage test`
  - `--initial_wallet_balance 5000`
  - `--short_estimated_rate 0`
  - `--long_estimated_rate 0`
  - `--enable_non_main_contract_defense`
- **关键产出物**：
  - **`result/` 最终测试集回测评估与轨迹产出目录 (`result/DiHFT/final_result/${DATASET_NAME}/${EXPERIMENT_NAME}/`)**：
    - `contract_results.csv`（测试集各合约绩效明细汇总，包含 `contract`, `source_file`, `rows`, `reward_sum`, `require_money`, `return_rate`）
    - `trading_info.npy`（测试集综合投资组合绩效字典，包含 `return_rate`, `portfolio_return_rate`, `win_rate`, `equal_weighted_mean_return`, `total_reward_sum`, `contract_count` 等）
    - **分合约测试细节与动作向量 (`contracts/<contract>/`)**：
      - `reward_history.npy`（每步测试收益历史）、`total_asset_history.npy`（总资产变化历史）、`wallet_balance_history.npy`（钱包余额历史）、`initial_margin_history.npy`（初始保证金历史）、`unrealized_pnl_history.npy`（未实现盈亏历史）、`maintain_marigine_history.npy`（维持保证金历史）、`new_position_required_money_history.npy`（新开仓所需资金历史）
      - `micro_action_history.npy`（低层 Agent 微观动作历史）、`macro_action.npy` / `macro_action_history.npy`（高层 VAE 路由宏观选择 Label 历史）以及分合约 `trading_info.npy`
  - **`log/` 最终回测日志**：
    - `log/DiHFT/${DATASET_NAME}/high_level/final_result/${EXPERIMENT_NAME}/final_result.log`（记录测试集各合约评估进度、逐合约结算与最终组合收益率的执行日志）
- **位置**：承接高层启发式筛选的最优超参和低层模型，是整个 DiHFT 算法体系在测试集（OOS）上的终检评估出口。

## 总结：全流程依赖输入与产出对照表

| 阶段 / 步骤 | 核心执行脚本 | 核心依赖输入物 | 关键产出物目录与核心文件 |
| :--- | :--- | :--- | :--- |
| **1. 数据预处理** | `main_10min_fu.sh` | `data/原始下载/` (Tick/KLine CSV)<br>`config.py` (合约规则) | `PREPROCESS_DATASET/commodity-futures/`<br>(`SPLIT-TRAIN-VALID-TEST`, `FEATURE_SELECTION`, `SCALE_SAVE`) |
| **2. FineFT 数据准备** | `commodity_data_handler_10min_fu.sh` | 步骤 1 产出的 `SCALE_SAVE`, `dataset_split_manifest.json`, `state_features.npy` | `dataset/10min/fu/`<br>(`train/`, `valid/`, `test/`, `VAE_data/`, `dataset_manifest.json`) |
| **3. 低层 Agent 并行训练** | `train_commodity_fu_10_parallel.sh` | 步骤 2 产出的 `dataset/10min/fu/train/`, `state_features.npy`, `margin_dict.npy` | `result/DiHFT/low_level/fu/10min_parallel/weights_advantage_pretrain/`<br>(`epoch_1/`~`epoch_75/trained_model.pkl`, `log/`, `pretrain_evaluation.log`) |
| **4. 低层 Agent 并行测试** | `test_util_fu_10.sh` | 步骤 3 产出的 `trained_model.pkl` + 步骤 2 产出的 `valid/` 数据 | `result/DiHFT/low_level/fu/10min_parallel/weights_advantage_pretrain/epoch_{epoch}/{slope,volatility}/`<br>(`analysis_result.csv`, `trading_action_detail_*.csv`, `analysis_result.npy`) |
| **5. 低层 Agent 二维筛选** | `low_level_fu_10.sh` | 步骤 4 产出的 `analysis_result.csv` + 步骤 2 产出的 `valid/` 数据 | `analysis_result/DiHFT/low_level/fu/10min_parallel/two_dimensional_selection/`<br>(`two_dimensional_selection_manifest.json`, `marginal_metrics.csv`, `joint_metrics.csv`, `candidate_rankings.csv`, `selected_slots.csv`) |
| **6. VAE 并行训练与评估** | `VAE_util_fu_10.sh` | 步骤 2 产出的 `VAE_data/{slope,volatility}/<contract>/label_*.npy` + 步骤 5 的 Label 划分 | `result/DiHFT/vae_results/fu/10min_parallel/{slope,volatility}/`<br>(`label_*/model_latest.pth`, `summary.json`, `ood_logpx_*.csv`, `routing_summary.json`) |
| **7. 高层 Optuna 并行寻优** | `vae_optuna_fu_10.sh` | 步骤 3/5 筛选的 Agent 模型 + 步骤 6 的 VAE 模型 + 步骤 2 的 `valid/` 数据 | `result/DiHFT/high_level/fu/10min_parallel/`<br>(`vae_risk_aware_routing_optuna/optuna_results.csv`, `vae_risk_aware_routing/.../contract_results.csv`, `trading_info.npy`, `macro_action.npy`) |
| **8. 高层路由筛选与可视化** | `high_level_heurstic_fu_10.sh` | 步骤 7 产出的 `vae_risk_aware_routing/` 诊断数据 + 步骤 2 的 `valid/*.feather` | `analysis_result/DiHFT/high_level_heurstic/fu/10min_parallel/`<br>(`result.csv`, `best_result.csv`, `best_result_*.png/pdf`)<br>`result/DiHFT/final_result/fu/10min_parallel/`<br>(`high_level_agent_para.txt`, 最终路由诊断向量 `.npy`/`.csv`) |
| **9. 最终测试与评估** | `final_result_fu_10_p.sh` | 步骤 8 产出的 `high_level_agent_para.txt` + 步骤 7 产出的 `optuna_results.csv` + 步骤 5 的 `two_dimensional_selection_manifest.json` + 步骤 2 的 `test/*.feather` | `result/DiHFT/final_result/${DATASET_NAME}/${EXPERIMENT_NAME}/`<br>(`contract_results.csv`, `trading_info.npy`, `contracts/<contract>/` 轨迹向量)<br>`log/DiHFT/${DATASET_NAME}/high_level/final_result/${EXPERIMENT_NAME}/final_result.log` |
