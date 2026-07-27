# System Map

This document preserves the project-level component map for humans. `CONTEXT.md` owns domain vocabulary; this file owns where major capabilities live and how they connect.

## Components

**Commodity Preprocessing** (`data_preprocess/operator_futures/commodity/`): 商品期货数据接入、主力合约拼接、五档下采样和特征生成的预处理管线。

**Base_Time_feature Generator** (`PREPROCESS_DATASET/commodity-futures/Base_Time_feature/`): 与 `BASE_FEATURE` 平级的商品期货时间编码产物层，按 symbol/contract/target_freq/date 输出非绝对时间和合约生命周期特征，并在 daily merge 阶段并入 `FUTURE_FEATURE`。

**Cross-section Feature Generator** (`data_preprocess/operator_futures/cross_section/`): 从下采样后的 base feature 和 orderbook 生成 KLINE、QUOTE 和 SNAPSHOT 截面特征。

**Rolling Window Feature Generator** (`data_preprocess/operator_futures/time_operator/`): 从合并后的截面特征生成滚动窗口特征，负责需要历史窗口的风险状态特征和流动性状态特征。

**Feature Selection** (`data_preprocess/operator_futures/feature_selection/`): 多合约特征评估与筛选流水线，输出 `feature_selection_manifest.json`、`state_features.npy` 和品种级统一特征列表。

**Scale Save** (`data_preprocess/operator_futures/scale_describe_save/`): 使用 train-only robust scaler 标准化 State Feature 并裁剪输出，前后执行 NaN 校验。

**Dataset Split** (`data_preprocess/operator_futures/dataset_split/`): 按时间边界将数据分为 train/valid/test 集合，输出 `dataset_split_manifest.json`。

**Feature Validation** (`data_preprocess/operator_futures/feature_validation/`): 特征正确性验证框架，包含 pandas reference 实现、expected columns 和 comparison/validators 工具。

**Futures Trading Environment** (`FineFT/env/`): 期货交易环境实现，维护交易执行、风控约束、Reverse Position 和 Trading Process Feature 契约。

**Stage I Low-level Training** (`FineFT/RL/DiHFT/low_level/`): 价值基低层 agent 集成训练，消费 state、previous_action、time、avaliable_action 和 `trading_info` 五路 Q 网络输入。

**Stage II VAE Training** (`FineFT/RL/DiHFT/VAE/`): VAE 训练与分析，支持跨合约 label 训练数据物化、分合约测试分析和 routing summary 生成。

**Stage III High-level Routing** (`FineFT/RL/DiHFT/high_level/`): 使用 VAE 重构损失进行风险感知路由，包含 heuristic routing 和 Optuna 调参。

**Data Handler** (`FineFT/datahandler/`): FineFT 数据集装配、train slice 生成和 valid 动态切片。

**Low-level Agent Selection** (`FineFT/analysis/pick_agent/`): 低层 agent 两阶段选择算法，从跨合约 label 结果中选择最优 bin 和 epoch，输出 potential model 和 `selection_manifest.json`。

**Diagnostics** (`FineFT/RL/DiHFT/low_level/`): Loss NaN 诊断、qtable 诊断和 parallel rollout metrics。

**Baselines** (`FineFT/RL/base/`, `FineFT/RL/EarnHFT/`): 对比基线算法，包括 DQN、PPO、DRA、CRP/QR-DQN、rule-based、EDQN/SUNRISE 和 EarnHFT 层级 RL。

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

## Relationships

- **Commodity Preprocessing** 生成主力合约日文件，供 **Cross-section Feature Generator** 和后续特征流水线消费。
- **Cross-section Feature Generator** 与 **Base_Time_feature Generator** 的产物进入 **Feature Selection**，形成 **State Feature**。
- **Feature Selection** 输出的 **State Feature** 经 **Scale Save** 标准化后，由 **Data Handler** 装配为训练数据集。
- **Dataset Split** 的 manifest 决定 **Data Handler** 的 train/valid/test 视角。
- **Futures Trading Environment** 定义执行语义和 `trading_info` 契约，**Stage I Low-level Training** 必须与该契约同步。
- **Stage I Low-level Training** 产出低层 agents，**Low-level Agent Selection** 在验证市场动态下筛选 **Potential Model**。
- **Stage II VAE Training** 为筛选后的 agents 建立能力边界，**Stage III High-level Routing** 使用 VAE 重构损失进行风险感知路由。
