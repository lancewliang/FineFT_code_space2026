# FineFT 代码空间

这里提供 FineFT 及其对比基线算法所使用的数据预处理代码空间和算法源代码。

# [数据](data_preprocess/README.zh_CN.md)
这里展示了在 FineFT 中使用高保真交易环境所需的数据格式。

## 环境安装

使用 `conda create -n data_preprocess python==3.10.14` 创建对应的下载环境。

使用 `conda activate data_preprocess` 激活对应的下载环境。

使用 `pip install -r requirements.txt` 安装所有依赖。

## 基础数据

这里介绍如何通过 tardis 下载数据以及这些数据的格式。同时也提供了额外信息源的建议，方便用户构建自己的技术指标（我们不提供额外信息源的数据预处理）。此外，还提供了用于检查下载内容完整性的代码。

### 概览
FineFT 代码提供的高保真环境中使用了 2 类特征：
- 奖励特征：这些特征用于计算环境中的奖励，包括 [`book_snapshot_25`](https://docs.tardis.dev/downloadable-csv-files#book_snapshot_25) 和 [`derivative_ticker`](https://docs.tardis.dev/downloadable-csv-files#derivative_ticker)。
- 市场状态特征：这些特征用于构建 FineFT 论文中描述的市场状态表示，包括 [`quotes`](https://docs.tardis.dev/downloadable-csv-files#quotes) 和 [`trades`](https://docs.tardis.dev/downloadable-csv-files#trades)。但这并不意味着你必须严格沿用这里的技术指标构建方式。你可以接入任何对 RL 智能体交易有帮助的信息源。

这里我们提供了基于 [tardis](https://tardis.dev/) 的 [`下载数据脚本`](script_download/download/all.sh)。你可能需要购买 API key 才能完整使用我们的代码。

总之，要完整使用后续的数据预处理代码，需要 4 类文件：[`book_snapshot_25`](https://docs.tardis.dev/downloadable-csv-files#book_snapshot_25)、[`derivative_ticker`](https://docs.tardis.dev/downloadable-csv-files#derivative_ticker)、[`quotes`](https://docs.tardis.dev/downloadable-csv-files#quotes)（可选）和 [`trades`](https://docs.tardis.dev/downloadable-csv-files#trades)（可选）。

### Book_Snapshot_25
[`Book_Snapshot_25`](https://docs.tardis.dev/downloadable-csv-files#book_snapshot_25) 是包含 25 档深度的限价订单簿快照。该数据用于在环境中计算市价单的成交价格。Tardis 在[这里](https://docs.tardis.dev/downloadable-csv-files#book_snapshot_25)提供了数据样例。另外，如果你想使用自己的数据，我们也提供了对应的列名和 dataframe 示例。

| Dataframe 快照 | 列名 |
|------------|------------|
| ![Dataframe 快照](./data_preprocess//plot/original_data/book_25_shot.png) | ![列名](./data_preprocess/plot/original_data/book_25_columns.png) |

### Derivative_Ticker
[`Derivative_Ticker`](https://docs.tardis.dev/downloadable-csv-files#derivative_ticker) 是资金费率和标记价格的快照。该数据用于计算强平条件以及保证金余额中的价值变化。Tardis 在[这里](https://docs.tardis.dev/downloadable-csv-files#derivative_ticker)提供了数据样例。另外，如果你想使用自己的数据，我们也提供了对应的列名和 dataframe 示例。

| Dataframe 快照 | 列名 |
|------------|------------|
| ![Dataframe 快照](./data_preprocess/plot/original_data/der_ticker_shot.png) | ![列名](./data_preprocess/plot/original_data/der_ticker_columns.png) |

请注意，要使用交易环境，我们可能只需要 `local timestamp`、`funding timestamp`、`funding rate` 和 `mark price`。

### Quotes
[`Quotes`](https://docs.tardis.dev/downloadable-csv-files#quotes) 记录限价订单簿中最优卖价和最优买价的事件。该数据用于构建有助于预测未来标记价格走势的技术指标。Tardis 在[这里](https://docs.tardis.dev/downloadable-csv-files#derivative_ticker)提供了数据样例。另外，如果你想使用自己的数据，我们也提供了对应的列名和 dataframe 示例。

| Dataframe 快照 | 列名 |
|------------|------------|
| ![Dataframe 快照](./data_preprocess/plot/original_data/quotes_shot.png) | ![列名](./data_preprocess/plot/original_data/quotes_columns.png) |

### Trades
[`Trades`](https://docs.tardis.dev/downloadable-csv-files#trades) 记录已成交订单。该数据用于构建有助于预测未来标记价格走势的技术指标。Tardis 在[这里](https://docs.tardis.dev/downloadable-csv-files#trades)提供了数据样例。另外，如果你想使用自己的数据，我们也提供了对应的列名和 dataframe 示例。

| Dataframe 快照 | 列名 |
|------------|------------|
| ![Dataframe 快照](./data_preprocess/plot/original_data/trades_shot.png) | ![列名](./data_preprocess/plot/original_data/trades_columns.png) |

### 额外信息源建议
[`liquidations`](https://docs.tardis.dev/downloadable-csv-files#liquidations) 可以被视为一种特殊类型的 trade，其订单是由于保证金余额不足而触发的。由于它与 trades 共享相同的数据结构，trades 中有用的算子也应当适用于 liquidations。

[`incremental_book_l2`](https://docs.tardis.dev/downloadable-csv-files#incremental_book_l2) 记录限价订单簿中发生的每一个事件。它是所有数据结构中数据量最大的一类。`Incremental_book_L2` 与 `trades` 共同记录了市场中的每一个事件。这也是大多数公司用于构建特征的数据格式。

### 完整性检查

你可以使用 [`python over_view/checkout_download.py`](data_preprocess/over_view/checkout_download.py) 检查所有必需文件是否已正确下载。

接下来，我们介绍如何将原始数据处理成可以被 FineFT 环境直接使用的格式。

## 期货数据预处理

这里展示我们如何处理数据、检查数据完整性，并最终得到交易环境所需的数据。我们首先概览各部分及其功能，然后分别介绍每一部分的细节。最后，我们提供一行脚本，方便你配置目标交易对、频率、希望同时运行的核心数以提升效率，以及你希望如何选择用于构建市场状态的技术指标。

### 概览
这里展示如何处理基础数据，包括：
- 将奖励数据（`book_snapshot_25` 和 `derivative_ticker`）降采样到可配置频率。
- 为技术指标数据（`trades` 和 `quotes`）创建基础特征（OHLCV），并按可配置频率去重。
- 为预处理后的数据（`down_scale_book_25`、`down_scale_derivative_ticker`、`OHLCV`、`quotes_wo_duplications`）创建瞬时特征，以捕捉未来标记价格走势。
- 基于时间戳合并奖励数据、基础特征和瞬时特征。
- 拼接可配置开始日期和结束日期范围内的所有合并数据。
- 基于合并数据创建时间滚动特征。
- 将时间滚动特征合并到已合并的数据中。
- 根据你从 3 个选项中选择的方法计算特征重要性。
- 构建交易环境中使用的最终数据，以及对应市场技术指标的名称。

### 奖励数据降采样

这里我们提供了可直接从原始数据降采样频率的代码，分别用于 [`book_snapshot_25`](data_preprocess/operator_futures/orderbook_25/down_scale_single_shot_base_other.py) 和 [`derivative_ticker`](data_preprocess/operator_futures/derivative_ticker/down_scale_single_shot.py)。该代码针对某个特定交易对的单个日期运行。不过，`book_snapshot_25` 极大，运行时可能导致程序内存超限。为了节省计算成本，我们也提供了从更细时间跨度而非直接从原始数据降采样的代码，分别用于 [`book_snapshot_25`](data_preprocess/operator_futures/orderbook_25/down_scale_single_shot_base_other.py) 和 [`derivative_ticker`](data_preprocess/operator_futures/derivative_ticker/down_scale_single_shot_base_other.py)。

我们在代码中加入了 `memory_profiler`，帮助记录所需 RAM。你可以据此估算希望同时运行多少个进程；该数量可以在最终脚本中轻松控制。

### 创建基础特征并去重

这里我们提供了用于将原始 trades 聚合为 OHLCV 以便后续构建技术指标、并对 quotes 去重以避免数据歧义的[代码](data_preprocess/operator_futures/features_related/base_feature.py)。该代码针对某个特定交易对的单个日期运行。

### 创建瞬时特征

这里我们提供了[代码](data_preprocess/operator_futures/features_related/create_feature.py)，用于基于单个时间戳的信息（不涉及时间滚动）创建基础技术指标，输入包括降采样后的 25 档订单簿快照、降采样后的 derivative ticker、OHLCV 和 quotes。该代码针对某个特定交易对的单个日期运行。

### 合并特征
这里我们提供了用于合并降采样奖励特征和目前已创建技术指标的[代码](data_preprocess/operator_futures/merge_concat/merge.py)。该代码针对某个特定交易对的单个日期运行。

### 拼接特征
这里我们提供了用于按日期拼接包含合并信息的 dataframe 的[代码](data_preprocess/operator_futures/merge_concat/concat.py)。输入应为 2 个具体日期，分别表示为某个特定交易对构建 dataframe 的开始日期和结束日期。我们使用多线程加速 IO，便于构建长时间跨度的数据集。

### 创建时间滚动特征
这里我们提供了用于创建时间滚动特征的[代码](data_preprocess/operator_futures/time_operator/create_feature.py)。你可以在这里指定窗口选项。我们参考 [`qlib`](https://github.com/microsoft/qlib/) 中的 [Alpha 158](https://github.com/microsoft/qlib/blob/98f569eed2252cc7fad0c120cad44f6181c3acf6/qlib/contrib/data/handler.py#L142)，为类似 OHLCV 的特征和单价格特征构建滚动窗口特征。

### 合并并清理
这里我们提供了将时间滚动特征合并到前一个 merge dataframe，并移除包含 NAN 列的[代码](data_preprocess/operator_futures/merge_all/merge_clean.py)。

### 特征选择
这里我们提供了用于选择有助于预测未来标记价格走势的技术指标的代码。我们提供 4 种方法选择技术指标：它们都会基于多种长度窗口下的标记价格走势计算特征重要性（这也是 4 种方法彼此不同之处），并计算重要特征之间的相关性以移除重复特征。

更具体地说，我们首先计算若干 target：未来 (1,6,12) 列的标记价格变化，这些也可以配置。市场状态覆盖不同时间尺度上的未来标记价格预测能力很重要，这样 RL 智能体才能学会规划，而不是只关注 1-step reward。

计算特征重要性的 4 种方式如下：
- `IC`：直接计算每个特征与 target 之间的 [`Pearson correlation`](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)，并使用相关系数的绝对值作为特征重要性。

- `Rank IC`：按时间戳在整个数据集上分别对每个 target 和每个技术特征进行排序，然后计算每个特征排名与 target 排名之间的 Pearson 相关性，并使用相关系数的绝对值作为特征重要性。该方法比 IC 更稳健，因为 `IC` 的结果很容易受到数据集中极端值的影响。

- `Linear Lasso`：使用线性回归基于技术指标预测 target，并使用参数作为特征重要性。该方法与 IC 类似，并且保持了所选特征的独立性。

- `Catboost`：使用 catboost 基于技术指标对 target 进行回归，并直接使用 catboost 提供的特征重要性。由于 catboost 涉及非线性算子，因此其结果适合用于构建市场状态表示（FineFT 中的市场状态表示也是这样构建的）。

### 缩放并保存
这里我们使用 log 算子识别每个已选特征的尺度，并利用 log 结果对技术特征进行归一化，使原始尺度不会影响训练结果。

### 完整性检查
由于从奖励数据降采样到合并特征的过程都在处理单日数据，很可能出现 RAM 超限并导致部分进程丢失。因此，我们提供了 [`完整性检查代码`](data_preprocess/over_view/checkout_read_create_feature.py)，以确保在 concat 之前没有数据缺失。

### 最终脚本
你可以直接使用类似 [`bash script_preprocess/future_upgraded/total_process/BNBUSDT/5min/20210401-20240101.sh`](data_preprocess/script_preprocess/future_upgraded/total_process/BNBUSDT/5min/20210401-20240101.sh) 的脚本，在其中配置目标频率 `target_freq=5min`、开始日期 `start_date=2022-01-01`、结束日期 `end_date=2024-01-01`、交易对 `symbol=BNBUSDT`、用于降采样 trades、quotes 和 derivative ticker 的进程数 `max_processes_1=100`（它们占用的 RAM 较小）、用于降采样 book snapshot 25 的进程数 `max_processes_2=20`（它占用的 RAM 较大），以及用于存储结果的根路径 `root_path="."`。请注意，我们没有将完整性检查集成到该脚本中。

## 本流程的最终结果与建议
预处理完成后，你应该会得到类似下图的结果。你可以将 `df.feather` 和 `state_features.npy` 作为 FineFT 代码的输入来构建环境。

| 整体结果 | FineFT 的输入 |
|------------|------------|
| ![Dataframe 快照](./data_preprocess/plot/processed_data/process_example.png) | ![列名](./data_preprocess/plot/processed_data/input.png) |

单个交易对 2 年长度的数据下载需要 800G。如果你将数据存储在 SATA 上，建议每 24 小时下载半年数据，否则密集 IO 操作可能导致 SATA 硬盘断开连接。

# [FineFT](FineFT/README.md)

本仓库提供完整的数据预处理流水线，以及用于构建回测环境、训练 FineFT 算法和实现相关文献中基线方法的对应已处理数据集。此外，它还包含结果分析和可视化内容，用于说明我们方法背后的动机。

## 环境安装

使用 `conda create -n FineFT python==3.10.14` 创建对应的下载环境。

使用 `conda activate FineFT` 激活对应的下载环境。

使用 `pip install -r requirements.txt` 安装所有依赖。

## 交易环境
交易环境设计位于 [`env/env_class`](FineFT/env/env_class/base_env.py)，附录 C 中描述的大部分交易流程都在 [`utils`](FineFT/env/env_class/futures_util.py) 中实现。不同环境会提供不同的历史记录。

要启动一个交易环境，你需要 [`df.feather`](dataset/BNBUSDT/df.feather)、[`state_features.npy`](FineFT/dataset/BNBUSDT/state_features.npy) 和 [`maintenance_margin_ratio_dict.npy`](FineFT/dataset/BNBUSDT/maintenance_margin_ratio_dict.npy)。其中前两者由前面的数据预处理流程提供，后者由[交易所](https://www.binance.com/en/futures/trading-rules/perpetual/leverage-margin)提供，用于计算维持保证金。

环境的基础元素如下所示。

![Env](FineFT/fig/Environment.png)

## FineFT 算法
这里展示 FineFT 算法的训练、验证和测试流程。FineFT 的整体流水线如下所示。

![pipline](FineFT/fig/pipline.png)

### 额外数据预处理
创建训练集、验证集和测试集：[`python datahandler/preprocess_data.py`](FineFT/datahandler/preprocess_data.py)。

将验证集划分为多种 dynamics：[`python datahandler/slice_model.py`](FineFT/datahandler/slice_model.py)。

创建 VAE 数据集：[`python datahandler/vae_data_creation.py`](FineFT/datahandler/vae_data_creation.py)。

### Stage I：Selective Update 下的高效集成
训练低层智能体：[`python RL/DiHFT/low_level/weight_advantage_pretrain.py`](FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py)。

### Stage II：集成筛选与边界识别
回测集成模型：[`bash script/test/DiHFT/low_level/main.sh`](FineFT/script/test/DiHFT/low_level/main.sh)。

筛选集成模型：[`python analysis/pick_agent/FineFT_single_agent_with_different_position.py`](FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py)。

训练 VAE：[`python RL/DiHFT/VAE/main.py`](FineFT/RL/DiHFT/VAE/main.py)。

### Stage III：风险感知启发式路由
在验证集上调参：[`python RL/DiHFT/high_level/vae_routing_optuna.py`](FineFT/RL/DiHFT/high_level/vae_routing_optuna.py)。

选择合适参数：[`python analysis/pick_agent/DiHFT_high_level_heurstic.py`](FineFT/analysis/pick_agent/DiHFT_high_level_heurstic.py)。

在测试集上回测：[`python RL/DiHFT/high_level/vae_routing_final_result_macro_action.py`](FineFT/RL/DiHFT/high_level/vae_routing_final_result_macro_action.py)。

### 脚本

我们在[这里](FineFT/script)提供了 shell 脚本，便于执行这些实验，其中也包含其他基线方法的额外实验。

## 可视化与分析工具
用于动机部分的三张 t-SNE 图：[`python analysis/motivation_finding/create_tsne_result.py`](FineFT/analysis/motivation_finding/create_tsne_result.py)。

训练集、验证集、测试集数据可视化：[`python analysis/plot/valid_test_comparison.py`](FineFT/analysis/plot/valid_test_comparison.py)。

## 许可证

FineFT: Efficient and Risk-Aware Ensemble Reinforcement Learning for Futures Trading  
© 2025 Molei Qin

本作品采用 Creative Commons Attribution 4.0 International License (CC BY 4.0) 授权。  
https://creativecommons.org/licenses/by/4.0/

仓库 DOI：10.5281/zenodo.18139892
