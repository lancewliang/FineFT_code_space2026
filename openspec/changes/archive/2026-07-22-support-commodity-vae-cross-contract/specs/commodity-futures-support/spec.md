## ADDED Requirements

### Requirement: 商品 VAE SHALL materialize cross-contract label training datasets
系统 SHALL 为商品期货 VAE 训练按 label 合并多合约样本，并将合并结果物化为训练输入文件。

#### Scenario: 合并存在的多合约 label 样本
- **WHEN** 用户为 `fu` 运行 VAE 训练并选择 `label_0`
- **THEN** 系统 SHALL 扫描 `dataset/10min/fu/VAE_data/<contract>/label_0.npy`
- **AND** 系统 SHALL 合并所有存在的二维 label 数组
- **AND** 系统 SHALL 写出 `dataset/10min/fu/VAE_data/train/label_0.npy`
- **AND** 合并后数组的列数 SHALL 等于每个源数组的 feature 维度
- **AND** 合并后数组的行数 SHALL 等于所有 included source arrays 的行数总和

#### Scenario: 写出训练集合并 manifest
- **WHEN** 系统写出 `VAE_data/train/label_0.npy`
- **THEN** 系统 SHALL 同步写出 `VAE_data/train/label_0_manifest.json`
- **AND** manifest SHALL 包含 `dataset_name`、`label`、`merged_path`、`total_samples` 和 `feature_dim`
- **AND** manifest SHALL 包含 `included_contracts`，其中每项记录 `contract`、`source_file` 和 `sample_count`
- **AND** manifest SHALL 包含 `missing_contracts`
- **AND** manifest SHALL 与合并训练集同一次生成并允许覆盖旧 manifest

#### Scenario: 缺失部分合约 label 时跳过并记录
- **WHEN** `VAE_data/fu2505/label_0.npy` 存在
- **AND** `VAE_data/fu2510/label_0.npy` 不存在
- **THEN** 系统 SHALL 使用 `fu2505` 的样本参与合并
- **AND** 系统 SHALL NOT 因 `fu2510` 缺少该 label 而停止训练
- **AND** 系统 SHALL 在 manifest 的 `missing_contracts` 中记录 `fu2510`

#### Scenario: 没有任何可用 label 样本时 fail-fast
- **WHEN** 用户为 `fu` 运行 VAE 训练并选择 `label_4`
- **AND** 没有任何 `VAE_data/<contract>/label_4.npy` 存在
- **THEN** 系统 SHALL 报错并停止训练
- **AND** 错误信息 SHALL 包含 `label_4` 和被扫描的 `VAE_data` 路径
- **AND** 系统 SHALL NOT 写出空的 `VAE_data/train/label_4.npy`

#### Scenario: 合并输入维度校验
- **WHEN** 系统合并 `label_0` 的合约数组
- **THEN** 每个源数组 SHALL 是二维数组
- **AND** 每个源数组 SHALL 至少包含一行样本
- **AND** 所有源数组的列数 SHALL 相同
- **AND** 任一数组不满足要求时系统 SHALL fail-fast
- **AND** 错误信息 SHALL 包含相关 `contract` 和 `source_file`

### Requirement: 商品 VAE SHALL train from the materialized cross-contract dataset
系统 SHALL 使用物化后的跨合约训练集训练每个 label 的通用 VAE 模型。

#### Scenario: VAE 训练读取物化训练集
- **WHEN** 用户为 `fu` 和 `label_0` 启动训练
- **THEN** VAE 训练 SHALL 读取 `dataset/10min/fu/VAE_data/train/label_0.npy`
- **AND** VAE 训练 SHALL NOT 读取 `dataset/10min/fu/VAE_data/label_0.npy`
- **AND** VAE 训练 SHALL NOT 要求旧的扁平 `VAE_data/label_0.npy` 存在

#### Scenario: 训练产物沿用 label 目录
- **WHEN** `label_0` 训练运行到保存 checkpoint 的 epoch
- **THEN** 系统 SHALL 在 `result/DiHFT/vae_results/fu/label_0/` 下保存 checkpoint
- **AND** `model_latest.pth` SHALL 表示最近一次保存的 checkpoint
- **AND** `model_latest.pth` MAY 被后续训练保存覆盖
- **AND** 系统 SHALL NOT 使用测试合约分析结果决定是否覆盖 `model_latest.pth`

#### Scenario: CLI 使用明确训练 flag
- **WHEN** 用户通过 `FineFT/RL/DiHFT/VAE/main.py` 启动商品 VAE 训练
- **THEN** CLI SHALL 支持明确的训练 flag，例如 `--train`
- **AND** 训练 flag SHALL 触发训练集合并、VAE 训练和训练后的分合约分析
- **AND** 系统 SHALL NOT 要求用户传入 `--if_train True` 才能训练

#### Scenario: 只重跑分析
- **WHEN** 用户希望使用已保存的 `model_latest.pth` 重跑测试分析
- **THEN** CLI SHALL 支持明确的 analyze-only 行为，例如 `--analyze-only`
- **AND** analyze-only SHALL 加载 `result/DiHFT/vae_results/fu/label_0/model_latest.pth`
- **AND** analyze-only SHALL NOT 重新合并训练集或重新训练模型

### Requirement: 商品 VAE SHALL analyze test contracts separately and produce aggregate outputs
系统 SHALL 对商品 VAE 测试合约逐合约分析，并输出分合约结果和总体汇总。

#### Scenario: 逐合约读取测试数组
- **WHEN** `dataset/10min/fu/VAE_data/test/test_fu2508.npy` 和 `test_fu2509.npy` 存在
- **THEN** 分析阶段 SHALL 分别读取每个 `test_<contract>.npy`
- **AND** 分析阶段 SHALL NOT 要求 `dataset/10min/fu/VAE_data/test.npy` 存在
- **AND** 分析阶段 SHALL NOT 在磁盘上写出合并后的 test 输入文件

#### Scenario: 输出分合约 logpx npy 和 csv
- **WHEN** 系统完成合约 `fu2508` 的 VAE 分析
- **THEN** 系统 SHALL 写出 `result/DiHFT/vae_results/fu/label_0/ood_logpx_fu2508.npy`
- **AND** 系统 SHALL 写出 `result/DiHFT/vae_results/fu/label_0/ood_logpx_fu2508.csv`
- **AND** CSV SHALL 包含列 `contract`、`source_file`、`row_index` 和 `logpx`
- **AND** CSV 的 `contract` 列 SHALL 等于 `fu2508`
- **AND** CSV 的 `source_file` 列 SHALL 等于该合约测试数组路径
- **AND** CSV 的 `row_index` SHALL 与 `ood_logpx_fu2508.npy` 中的 logpx 顺序一一对应

#### Scenario: 输出总体 logpx npy 和 csv
- **WHEN** 系统完成所有测试合约分析
- **THEN** 系统 SHALL 写出 `result/DiHFT/vae_results/fu/label_0/ood_logpx_all.npy`
- **AND** 系统 SHALL 写出 `result/DiHFT/vae_results/fu/label_0/ood_logpx_all.csv`
- **AND** `ood_logpx_all.npy` SHALL 按合约名稳定排序后的测试结果顺序拼接所有合约 logpx
- **AND** `ood_logpx_all.csv` SHALL 包含列 `contract`、`source_file`、`row_index` 和 `logpx`
- **AND** `ood_logpx_all.csv` SHALL 保留每一行对应的原始测试合约和源文件路径

#### Scenario: 输出 summary 统计
- **WHEN** 系统完成所有测试合约分析
- **THEN** 系统 SHALL 写出 `result/DiHFT/vae_results/fu/label_0/summary.json`
- **AND** summary SHALL 包含 `dataset_name` 和 `label`
- **AND** summary SHALL 为每个测试合约记录 `source_file`、`samples`、`logpx_mean`、`logpx_std`、`logpx_min` 和 `logpx_max`
- **AND** summary SHALL 记录总体 `samples`、`logpx_mean`、`logpx_std`、`logpx_min` 和 `logpx_max`
- **AND** summary SHALL NOT 输出 AUROC、AUPRC 或 FPR80，除非另有明确的 ID/OOD reference 定义

#### Scenario: 输出增强 summary 门控分析指标
- **WHEN** 系统完成 `label_0` 的训练后分析或 analyze-only 分析
- **THEN** `summary.json` SHALL 包含 `train_baseline`
- **AND** `train_baseline` SHALL 记录训练集 `source_file`、`input_samples`、`analyzed_samples`、`sample_mismatch`、`logpx_mean`、`logpx_std`、`logpx_min` 和 `logpx_max`
- **AND** `train_baseline` SHALL 包含 `quantiles`，其中至少包含 `q01`、`q05`、`q25`、`q50`、`q75`、`q95` 和 `q99`
- **AND** 每个测试合约 summary SHALL 记录 `input_samples`、`analyzed_samples` 和 `sample_mismatch`
- **AND** 每个测试合约 summary SHALL 包含相同 quantile keys 的 `quantiles`
- **AND** 每个测试合约 summary SHALL 包含 `acceptance`，其中至少包含 `ge_train_q01_pct`、`ge_train_q05_pct` 和 `ge_train_q50_pct`
- **AND** 总体 test summary SHALL 包含相同 quantile keys 的 `quantiles` 和相同 threshold keys 的 `acceptance`
- **AND** summary SHALL NOT 输出 accuracy、AUROC、AUPRC 或 FPR80，除非另有明确的真实 label 或 ID/OOD reference 定义

#### Scenario: 样本数完整性检查
- **WHEN** 分析阶段读取 `test_fu2508.npy`
- **THEN** 系统 SHALL 在 summary 中记录该文件的 `input_samples`
- **AND** 系统 SHALL 在 summary 中记录实际写入 logpx 的 `analyzed_samples`
- **AND** 当 `input_samples` 不等于 `analyzed_samples` 时 `sample_mismatch` SHALL 为 true
- **AND** 当 `input_samples` 等于 `analyzed_samples` 时 `sample_mismatch` SHALL 为 false
- **AND** `samples` SHALL 保持为 `analyzed_samples` 的兼容别名

#### Scenario: 输出跨 label routing summary
- **WHEN** `result/DiHFT/vae_results/fu/label_0` 到 `label_4` 的分合约 `ood_logpx_<contract>.npy` 均存在
- **THEN** 系统 SHALL 能生成 `result/DiHFT/vae_results/fu/routing_summary.json`
- **AND** routing summary SHALL 包含 `dataset_name`、`labels` 和 `score_type`
- **AND** routing summary SHALL 为每个测试合约记录 `samples`、`winner_counts` 和 `winner_pct`
- **AND** routing summary SHALL 记录总体 `samples`、`winner_counts` 和 `winner_pct`
- **AND** winner SHALL 表示同一 row 在所有 label logpx 中分数最高的 label
- **AND** routing summary SHALL 记录 `top1_top2_margin_mean`、`top1_top2_margin_q25` 和 `low_margin_pct`
- **AND** 若某合约在不同 label 下的 logpx 数量不一致，系统 SHALL 只比较共同长度并在 routing summary 中记录 sample mismatch 信息

#### Scenario: 测试输入维度校验
- **WHEN** 分析阶段读取 `test_fu2508.npy`
- **THEN** 测试数组 SHALL 是二维数组
- **AND** 测试数组的列数 SHALL 等于训练集 `feature_dim`
- **AND** 任一测试数组不满足要求时系统 SHALL fail-fast
- **AND** 错误信息 SHALL 包含相关 `contract` 和 `source_file`

### Requirement: 商品 VAE fu shell entry SHALL run the multi-contract workflow
系统 SHALL 提供适配当前商品多合约数据结构的 `fu` VAE shell 入口。

#### Scenario: fu VAE shell passes current data base path
- **WHEN** 用户运行 `FineFT/script/train/DiHFT/low_level/VAE_util_fu.sh`
- **THEN** 脚本 SHALL 激活 `finetf` conda 环境
- **AND** 脚本 SHALL 设置 `PYTHONPATH` 以包含 `FineFT`
- **AND** 脚本 SHALL 调用 `FineFT/RL/DiHFT/VAE/main.py`
- **AND** 每次调用 SHALL 传递 `--dataset_name fu`
- **AND** 每次调用 SHALL 传递 `--data_base_path dataset/10min`
- **AND** 每次调用 SHALL 使用明确训练 flag，例如 `--train`

#### Scenario: fu VAE shell launches all labels
- **WHEN** 用户运行 `VAE_util_fu.sh`
- **THEN** 脚本 SHALL 为 `label_0` 到 `label_4` 启动 VAE 训练
- **AND** 每个 label 的日志 SHALL 写入 `log/DiHFT/fu/VAE/`
- **AND** 每个 label 调用 SHALL 传递对应的 `--label_index`

#### Scenario: fu VAE shell limits concurrent training jobs
- **WHEN** 用户运行 `VAE_util_fu.sh`
- **THEN** 脚本 SHALL 默认最多同时运行 2 个 VAE 训练进程
- **AND** 脚本 SHALL 支持通过 `MAX_PARALLEL_JOBS` 环境变量覆盖并发上限
- **AND** `MAX_PARALLEL_JOBS` 不是正整数时脚本 SHALL fail-fast
- **AND** 任一 label 训练进程失败时脚本 SHALL 最终返回非 0
