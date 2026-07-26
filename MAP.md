> Before creating a new component, module, UI, or API surface: check this file and the
> codebase for an existing equivalent. If one exists, use it. If you still intend to
> create something new, confirm with the human what it does that the existing one does not.
>
> LANGUAGE.md owns what things *mean*. This file owns where things *live* and how they
> *connect*. A term can appear in both.
>
> Keep entries sparse — document at the seam level only. Implementation details belong
> in the code.

# System Map

## Components

**Commodity Preprocessing** (`data_preprocess/operator_futures/commodity/`): 商品期货数据接入、主力合约拼接、五档下采样和特征生成的预处理管线；暴露 `stitch_main_contract.py`（主力合约日文件生成，并在 Main Contract Summary 记录每个合约最后交易日和完整交易日数量）和 `downscale_continuous_by_trading_day.py` 作为 CLI 入口；`downscale.py` 提供 BASE_FEATURE 基础列（含窗口末尾 `open_interest`）、OFI（`downscale_quote_ofi_features`）、深度不平衡（`_depth_imbalance_expr`）、队列压力（`_queue_change_expr`）和微观结构（`downscale_quote_microstructure_features`）特征计算器。

**Base_Time_feature Generator** (`PREPROCESS_DATASET/commodity-futures/Base_Time_feature/`): 与 `BASE_FEATURE` 平级的商品期货时间编码产物层，按 symbol/contract/target_freq/date 输出非绝对时间和合约生命周期特征，并在 daily merge 阶段并入 `FUTURE_FEATURE`。

**Cross-section Feature Generator** (`data_preprocess/operator_futures/cross_section/`): 从下采样后的 base feature 和 orderbook 生成 KLINE、QUOTE 和 SNAPSHOT 截面特征；支持 `--contract` 参数按合约读写日文件。

**Rolling Window Feature Generator** (`data_preprocess/operator_futures/time_operator/`): 从合并后的截面特征生成滚动窗口特征；已迁移到 Polars，支持 depth-aware 特征生成；负责需要历史窗口的风险状态特征和流动性状态特征，消费 `BASE_FEATURE` 提供的 `open/high/low/close/volume/tradeval/open_interest`。

**Feature Selection** (`data_preprocess/operator_futures/feature_selection/`): 多合约特征评估与筛选流水线，包含 IC、RankIC、CatBoost、Permutation Importance 和 Sharpe 指标；暴露 `pipeline.py` 作为多合约入口，输出 `FeatureSelectionManifest` dataclass 表达的 `feature_selection_manifest.json` 和 `state_features.npy`；`contract_feature_union.py` 输出 `FeatureUnionManifest` dataclass 表达的品种级统一特征列表。

**Scale Save** (`data_preprocess/operator_futures/scale_describe_save/`): 使用 train-only robust scaler 标准化 state feature 并裁剪输出；暴露 `muti_contract_scale_save.py` 处理多合约 split-stage 文件，从训练集拟合单一 scaler（`fit_scope="train_all_contracts"`），输出 `ScaleManifest` dataclass 表达的 `scaler_manifest.json` 和 `scale_diagnostics.csv`；前后执行 NaN 校验。

**Dataset Split** (`data_preprocess/operator_futures/dataset_split/`): 按时间边界将数据分为 train/valid/test 集合；输出 `dataset_split_manifest.json` 记录合约级归属和行数。

**Feature Validation** (`data_preprocess/operator_futures/feature_validation/`): 特征正确性验证框架，包含 pandas reference 实现、expected columns 和 comparison/validators 工具。

**Futures Trading Environment** (`FineFT/env/`): 期货交易环境实现，暴露 mark-price 估值、orderbook 执行、手续费、滑点、funding、保证金、强平约束、Reverse Position（best-effort 先平后开）和 `trading_info` 交易过程特征；`futures_util.py` 提供 `change_of_wallet` 和 `calculate_avaiable_action` 的交易动作路由与可用性计算；`commodity_env.py` 提供商品期货专用环境。

**Stage I Low-level Training** (`FineFT/RL/DiHFT/low_level/`): 价值基低层 agent 集成训练，消费 state、previous_action、time、avaliable_action 和 `trading_info` 五路 Q 网络输入，包含 full-df warmup、qtable 预计算、diverse training 和 parallel rollout；暴露 `weight_advantage_pretrain.py` 和 `parallel_weight_advantage_pretrain.py`。

**Stage II VAE Training** (`FineFT/RL/DiHFT/VAE/`): VAE 训练与分析，支持跨合约 label 训练数据物化、分合约测试分析和 routing summary 生成；暴露 `main.py` 作为 CLI 入口，`merge_vae_train.py` 负责跨合约训练数据合并和 `LabelTrainingManifest` 生成。

**Stage III High-level Routing** (`FineFT/RL/DiHFT/high_level/`): 使用 VAE 重构损失进行风险感知路由，包含 heuristic routing 和 Optuna 调参。

**Data Handler** (`FineFT/datahandler/`): FineFT 数据集装配、train slice 生成和 valid 动态切片；暴露 `commodity_contract_dataset.py`（从 `dataset_split_manifest.json` 读取拆分元数据）和 `slice_model.py`；使用 `DatasetSplitManifest` 和 `DatasetManifest` dataclass 表达内部状态。

**Low-level Agent Selection** (`FineFT/analysis/pick_agent/`): 低层 agent 两阶段选择算法，从跨合约 label 结果中选择最优 bin 和 epoch，输出 potential model 和 `selection_manifest.json`；重构后支持按合约和标签聚合结果。

**Diagnostics** (`FineFT/RL/DiHFT/low_level/`): Loss NaN 诊断、qtable 诊断和 parallel rollout metrics，使用 `LossNanDiagnostics`、`PretrainQTableDiagnosticsResult` 等 dataclass 对象表达内部状态。

**Baselines** (`FineFT/RL/base/` and `FineFT/RL/EarnHFT/`): 对比基线算法，包括 DQN、PPO、DRA、CRP/QR-DQN、rule-based、EDQN/SUNRISE 和 EarnHFT 层级 RL。

**Commodity Config** (`data_preprocess/operator_futures/commodity/config.py`): 商品期货品种配置，暴露 orderbook_depth、fee_rate、contract_unit 和 trading session 参数。

## User Interfaces

**Commodity Full Process** (`CLI: data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`): 燃料油端到端预处理脚本，从主力合约拼接到 scale save 的完整流程。

**Commodity Main Process** (`CLI: data_preprocess/script_preprocess/future_upgraded/commodity/main.sh`): 商品期货主流程脚本，支持日期范围和步骤日志。

**Commodity Data Handler** (`CLI: FineFT/script/data/commodity_data_handler_fu.sh`): 燃料油 FineFT 数据集装配脚本，调用多合约数据集工具和 valid 动态切片。

**Stage I Training** (`CLI: FineFT/script/train/DiHFT/low_level/advantage.sh`): 串行低层训练脚本，支持 experiment_name 参数。

**VAE Training** (`CLI: FineFT/script/train/DiHFT/low_level/VAE_util_fu.sh`): 燃料油 VAE 训练脚本，支持多 label 并行训练。

**Feature Validation** (`CLI: data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh`): 商品期货输出验证脚本。

## Related Systems

**Tardis Data** (`download_operator`): 原始加密货币期货数据下载和解压，为 operator_futures 预处理提供输入；商品期货不依赖此系统。

**OpenSpec** (`openspec/`): 规格驱动的设计与变更管理工具，维护 specs 和 changes 目录；本系统的 spec 文件是其输入。
