# FineFT

本仓库提供完整的数据预处理流程，以及用于构建回测环境、训练 FineFT 算法并实现相关文献中基线方法的对应处理后数据集。此外，仓库还包含结果分析与可视化内容，用于说明我们方法背后的动机。

## 环境安装

使用 `conda create -n FineFT python==3.10.14` 创建对应的下载环境。

使用 `conda activate FineFT` 激活对应的下载环境。

使用 `pip install -r requirements.txt` 安装所有依赖项。

## 交易环境

交易环境的设计位于 [`env/env_class`](env/env_class/base_env.py)，附录 C 中描述的大部分交易流程实现于 [`utils`](env/env_class/futures_util.py)。不同环境会提供不同的历史记录。

要启用一个交易环境，你需要 [`df.feather`](dataset/BNBUSDT/df.feather)、[`state_features.npy`](dataset/BNBUSDT/state_features.npy) 和 [`maintenance_margin_ratio_dict.npy`](dataset/BNBUSDT/maintenance_margin_ratio_dict.npy)。前两者由前面的数据预处理流程提供，后者由[交易所](https://www.binance.com/en/futures/trading-rules/perpetual/leverage-margin)提供，用于计算维持保证金。

环境的基本元素如下所示。

![Env](fig/Environment.png)


## FineFT 算法

这里展示 FineFT 算法的训练、验证和测试流程。FineFT 的整体流程如下所示。

![pipline](fig/pipline.png)

### 额外数据预处理

创建训练、验证和测试数据集：[`python datahandler/preprocess_data.py`](datahandler/preprocess_data.py)。

将验证数据集切分为多种动态：[`python datahandler/slice_model.py`](datahandler/slice_model.py)。

创建 VAE 数据集：[`python datahandler/vae_data_creation.py`](datahandler/vae_data_creation.py)。

### 阶段 I：带选择性更新的高效集成

训练低层智能体：[`python RL/DiHFT/low_level/weight_advantage_pretrain.py`](RL/DiHFT/low_level/weight_advantage_pretrain.py)。

### 阶段 II：集成过滤与边界识别

回测集成模型：[`bash script/test/DiHFT/low_level/main.sh`](script/test/DiHFT/low_level/main.sh)。

过滤集成模型：[`python analysis/pick_agent/FineFT_single_agent_with_different_position.py`](analysis/pick_agent/FineFT_single_agent_with_different_position.py)。

训练 VAE：[`python RL/DiHFT/VAE/main.py`](RL/DiHFT/VAE/main.py)。


### 阶段 III：风险感知启发式路由

在验证数据集上调参：[`python RL/DiHFT/high_level/vae_routing_optuna.py`](RL/DiHFT/high_level/vae_routing_optuna.py)。

选择合适的参数：[`python analysis/pick_agent/DiHFT_high_level_heurstic.py`](analysis/pick_agent/DiHFT_high_level_heurstic.py)

在测试数据集上回测：[`python RL/DiHFT/high_level/vae_routing_final_result_macro_action.py`](RL/DiHFT/high_level/vae_routing_final_result_macro_action.py)。


### 脚本

我们在[这里](script)提供 shell 脚本，便于执行这些实验，其中也包含其他基线方法的额外实验。


## 可视化与分析工具

用于动机部分的三张 t-SNE 图：[`python analysis/motivation_finding/create_tsne_result.py`](analysis/motivation_finding/create_tsne_result.py)。


训练、验证、测试数据可视化：[`python analysis/plot/valid_test_comparison.py`](analysis/plot/valid_test_comparison.py)。
