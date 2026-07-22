# support-commodity-vae-cross-contract

## 背景与目标

当前商品期货 `fu` 的 VAE 数据按合约拆分：

```text
dataset/10min/fu/VAE_data/<contract>/label_*.npy
dataset/10min/fu/VAE_data/test/test_*.npy
```

现有 VAE 入口仍按旧的扁平路径读取 `VAE_data/label_*.npy` 和 `VAE_data/test.npy`，不能直接支持商品市场多合约训练与分合约测试。

本变更目标是让 `VAE_util_fu.sh`、`merge_vae_train.py`、`main.py` 和 `process.py` 支持当前商品多合约数据结构：按 label 合并所有可用合约样本训练跨合约通用 VAE，同时按测试合约分别输出分析结果，并提供总体汇总。

## 用户场景

1. 研究者使用 `dataset/10min/fu/VAE_data/<contract>/label_k.npy` 训练 `label_k` 的跨合约通用 VAE。
2. 研究者需要查看每个测试合约的 VAE logpx 表现，判断模型在不同合约上的泛化情况。
3. 研究者需要同时查看所有测试合约的总体 logpx 分布和统计汇总。

## 设计方向

采用“训练集物化 + 分合约测试分析”的方案。

`VAE_util_fu.sh` 作为商品 VAE 启动入口，只面向 `fu` 当前数据结构，显式使用 `dataset/10min` 作为 `data_base_path`，并对 `label_0..label_4` 启动训练。由于当前只有 1 个 GPU，脚本默认最多同时运行 2 个 VAE 训练进程。

`merge_vae_train.py` 负责商品多合约训练集发现和物化：

1. 对当前 `label_k` 扫描 `VAE_data/<contract>/label_k.npy`。
2. 合并所有存在的该 label 样本，落地为 `VAE_data/train/label_k.npy`。
3. 同步写入 `VAE_data/train/label_k_manifest.json`，记录合约、源文件、样本数、总样本数和缺失合约。

`main.py` 负责商品多合约 VAE orchestration：

1. 训练模式调用 `merge_vae_train.materialize_label_training_data(...)` 生成当前 label 的物化训练集。
2. VAE 训练读取 `VAE_data/train/label_k.npy`，训练产物保存到 `result/DiHFT/vae_results/fu/label_k/`。
3. 分析阶段逐个发现并读取 `VAE_data/test/test_<contract>.npy`，输出分合约结果和总体汇总。

`process.py` 负责支持逐合约分析输出：

```text
ood_logpx_<contract>.npy
ood_logpx_<contract>.csv
ood_logpx_all.npy
ood_logpx_all.csv
summary.json
```

CSV 列固定为：

```text
contract, source_file, row_index, logpx
```

`summary.json` 默认保存每个合约和总体的样本数、logpx 均值、标准差、最小值、最大值。不在没有明确 ID/OOD reference 定义时伪造 AUROC/AUPRC/FPR80。

## 关键决策

- 不兼容旧版扁平 VAE 输入结构，本次只支持商品多合约结构。
- 跨合约通用训练语义为：按 label 合并所有可用合约样本，而不是按合约分别训练。
- 合并后的训练集必须落地到 `VAE_data/train/label_k.npy`。
- 测试输入不落地合并为单个 test 文件，分析时按 `test_<contract>.npy` 逐合约读取。
- 分析结果既要有分合约 `.npy/.csv`，也要有总体 `ood_logpx_all.npy/.csv` 和 `summary.json`。
- 部分合约缺少某个 `label_k.npy` 时跳过该合约，并记录到 manifest；如果没有任何合约提供该 label，则报错。
- `model_latest.pth`、合并训练集和分析输出允许覆盖；合并训练集覆盖时必须同步覆盖 manifest。
- 命令行接口应避免继续使用含糊的 `--if_train True`，改为明确 flag，例如 `--train`，并可提供 `--analyze-only` 用于只重跑分析。
- `VAE_util_fu.sh` 默认 `MAX_PARALLEL_JOBS=2`，保证单 GPU 场景不会一次启动全部 label 训练进程；该值可通过环境变量覆盖，但必须是正整数。

## 范围边界

**包含：**

- 改造 `VAE_util_fu.sh`，使其适配 `fu` 商品多合约 VAE 训练入口。
- 新增/改造 `merge_vae_train.py`，支持发现并合并 `VAE_data/<contract>/label_k.npy`。
- 改造 `main.py`，集成训练集物化 helper 并编排训练与分合约分析流程。
- 改造 `process.py` 或相关 VAE 分析逻辑，支持分合约测试输出与总体汇总。
- 生成 `VAE_data/train/label_k.npy` 和 `label_k_manifest.json`。
- 生成分合约与总体的 `.npy`、`.csv`、`summary.json` 分析结果。
- 添加聚焦的单元测试和轻量 smoke test。

**不包含（本次）：**

- 不修改商品期货预处理和 `vae_data_creation.py` 的数据结构。
- 不改造低层 agent 训练、筛选或高层 routing 逻辑。
- 不引入 `model_best.pth` 或 validation split；`model_latest.pth` 仍表示最近一次保存的 checkpoint。
- 不为旧版 `VAE_data/label_k.npy` / `test.npy` 扁平结构提供兼容。
- 不计算 AUROC/AUPRC/FPR80，除非后续另行定义明确的 ID/OOD reference。

## 验收标准

- [ ] `VAE_util_fu.sh` 能以 `dataset/10min/fu` 的当前多合约 VAE 数据结构启动 `label_0..label_4` 训练。
- [ ] `VAE_util_fu.sh` 默认最多同时运行 2 个 label 训练进程，且 `MAX_PARALLEL_JOBS` 非正整数时 fail-fast。
- [ ] 训练 `label_k` 前，系统合并所有存在的 `VAE_data/<contract>/label_k.npy`，并写出 `VAE_data/train/label_k.npy`。
- [ ] 系统写出 `VAE_data/train/label_k_manifest.json`，记录 included contracts、missing contracts、source files、sample counts 和 total samples。
- [ ] 某些合约缺少 `label_k.npy` 时系统跳过并记录；所有合约都缺少该 label 时系统报错。
- [ ] test 阶段逐合约读取 `VAE_data/test/test_<contract>.npy`，不要求存在 `VAE_data/test.npy`。
- [ ] 每个测试合约生成 `ood_logpx_<contract>.npy` 和 `ood_logpx_<contract>.csv`。
- [ ] 系统生成 `ood_logpx_all.npy`、`ood_logpx_all.csv` 和 `summary.json`。
- [ ] CSV 文件包含 `contract, source_file, row_index, logpx` 列。
- [ ] `summary.json` 包含每个测试合约和总体的样本数、logpx 均值、标准差、最小值、最大值。
- [ ] 单元测试覆盖训练集合并、manifest、缺失 label 处理、分合约 CSV 输出和总体汇总。

## Amendments

### 2026-07-22: 增强 VAE summary 门控分析指标

训练完成后发现当前 `summary.json` 只包含 raw `logpx` 的均值、标准差、最小值和最大值，能确认输出存在，但不方便判断门控模型在不同合约上的可用性。由于每个 label 是针对不同行情状态训练的门控模型，单个 label 不应匹配总体 test 数据；分析应补充每个 label 自己的训练分布基准、test 分位数、相对训练阈值的接受率，以及跨 label 的 winner/margin 汇总。

新增验收标准：

- [ ] 每个 `label_k/summary.json` 包含该 label 训练集 logpx 基准统计。
- [ ] 每个 `label_k/summary.json` 的 test 合约与总体统计包含 logpx 分位数。
- [ ] 每个 `label_k/summary.json` 包含相对训练集分位阈值的接受率。
- [ ] 每个 `label_k/summary.json` 暴露输入样本数、分析样本数和样本数是否匹配。
- [ ] 所有 label 分析完成后，系统可生成跨 label `routing_summary.json`，统计逐合约和总体的 winner label 占比、top1/top2 margin 和低置信度比例。
- [ ] 不在没有真实 label 的情况下输出 accuracy、ROC-AUC、AUPRC 或 FPR80。
