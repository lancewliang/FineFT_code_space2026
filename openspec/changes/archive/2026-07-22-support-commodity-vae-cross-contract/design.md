# Design: support-commodity-vae-cross-contract

## Context

商品期货 `fu` 的 VAE 数据已经由数据生成阶段按合约拆分：

```text
dataset/10min/fu/VAE_data/<contract>/label_*.npy
dataset/10min/fu/VAE_data/test/test_<contract>.npy
```

当前 VAE 入口仍读取旧的扁平文件：

```text
VAE_data/label_*.npy
VAE_data/test.npy
```

这会阻止商品期货多合约数据直接训练跨合约通用 VAE，也无法观察每个测试合约的单独表现。

## Decisions

1. 只支持商品多合约 VAE 数据结构。本变更不保留旧扁平输入兼容。
2. 训练语义为按 label 跨合约合并。训练 `label_k` 时扫描所有 `VAE_data/<contract>/label_k.npy`，合并存在的样本。
3. `merge_vae_train.py` 独立负责训练数据发现、二维数组校验、跨合约合并、`VAE_data/train/label_k.npy` 物化和 `label_k_manifest.json` 写入。
4. `main.py` 只编排训练/分析流程：训练模式调用 `merge_vae_train.materialize_label_training_data(...)`，分析模式读取已物化训练集的 feature 维度；训练或分析完成后调用 `summary.py` 的 post-analysis helper 尝试生成跨 label routing summary。
5. VAE 训练只读取物化后的 `VAE_data/train/label_k.npy`，使训练入口保持单训练数组语义。
6. 测试输入不物化为单个 test 文件。分析阶段逐个读取 `VAE_data/test/test_<contract>.npy`。
7. 分析输出同时提供分合约结果和总体汇总：每个合约输出 `.npy/.csv`，总体输出 `ood_logpx_all.npy/.csv` 和 `summary.json`。所有 summary 相关写出逻辑集中在 `summary.py`，`process.py` 只负责运行 VAE 分析并把 contract results 交给 summary writer。
8. CLI 使用明确训练/分析 flag，避免继续依赖 `--if_train True` 这种容易误解析的 bool 参数。
9. `model_latest.pth`、合并训练集和分析输出允许覆盖；合并训练集每次覆盖时必须同步覆盖 manifest。
10. 不引入 `model_best.pth` 或 validation split。本次不使用 test 合约指标选择 checkpoint，避免测试集泄漏进模型选择。
11. 每个 label 的 `summary.json` 需要包含该 label 自己的训练集 logpx 基准。由于不同 label 的 VAE 独立训练，raw `logpx` 跨 label 直接比较不完全公平；summary 同时提供分位数和相对训练分位阈值的接受率，方便判断该 label 对某个测试合约的匹配程度。训练集 logpx 基准由 `main.py` 在分析阶段额外 forward 一遍物化训练集后传入 `summary.py`。
12. 跨 label 门控统计独立输出为 `result/DiHFT/vae_results/<dataset_name>/routing_summary.json`。`main.py` 不提供单独的 `--routing-summary` workflow flag；而是在 `piplinerunner.analyze_contracts()` 之后调用 `summary.maybe_write_routing_summary_after_analysis(args)`，该 helper 只有在所有 label 的所有测试合约 `ood_logpx_<contract>.npy` 都存在时才写出 routing summary。routing summary 负责 winner label、top1/top2 margin 和低置信度比例，不塞进单个 label 的 `summary.json`。

## Module Responsibilities

- `merge_vae_train.py`: 商品 VAE 训练数据发现、二维数组校验、跨合约合并、`VAE_data/train/label_k.npy` 物化和 manifest 写入。
- `main.py`: CLI 参数、训练/分析 workflow 编排、测试合约发现、训练集 baseline logpx 采集，以及分析后触发 routing summary 写出检查。
- `process.py`: DataLoader 准备、逐合约调用 VAE analyze、构造 contract result 列表；不直接写 `summary.json` 或 `routing_summary.json`。
- `summary.py`: per-label logpx `.npy/.csv`、`summary.json`、跨 label `routing_summary.json` 的统计和文件写出。

## Data Shape

训练 manifest 形态：

```json
{
  "dataset_name": "fu",
  "label": "label_0",
  "merged_path": "dataset/10min/fu/VAE_data/train/label_0.npy",
  "total_samples": 12345,
  "feature_dim": 46,
  "included_contracts": [
    {
      "contract": "fu2505",
      "source_file": "dataset/10min/fu/VAE_data/fu2505/label_0.npy",
      "sample_count": 1519
    }
  ],
  "missing_contracts": ["fu2510"]
}
```

分析 CSV 形态：

```text
contract,source_file,row_index,logpx
fu2508,dataset/10min/fu/VAE_data/test/test_fu2508.npy,0,-1.23
```

`summary.json` 形态：

```json
{
  "dataset_name": "fu",
  "label": "label_0",
  "train_baseline": {
    "source_file": "dataset/10min/fu/VAE_data/train/label_0.npy",
    "input_samples": 12428,
    "analyzed_samples": 12428,
    "sample_mismatch": false,
    "logpx_mean": -73.11,
    "logpx_std": 29.98,
    "logpx_min": -619.69,
    "logpx_max": -33.19,
    "quantiles": {
      "q01": -191.73,
      "q05": -122.66,
      "q25": -80.11,
      "q50": -66.65,
      "q75": -56.65,
      "q95": -46.56,
      "q99": -41.15
    }
  },
  "test": {
    "contracts": {
      "fu2508": {
        "source_file": "dataset/10min/fu/VAE_data/test/test_fu2508.npy",
        "input_samples": 1000,
        "analyzed_samples": 1000,
        "sample_mismatch": false,
        "samples": 1000,
        "logpx_mean": -1.23,
        "logpx_std": 0.45,
        "logpx_min": -3.0,
        "logpx_max": -0.1,
        "quantiles": {
          "q01": -2.9,
          "q05": -2.5,
          "q25": -1.7,
          "q50": -1.2,
          "q75": -0.8,
          "q95": -0.3,
          "q99": -0.2
        },
        "acceptance": {
          "ge_train_q01_pct": 98.0,
          "ge_train_q05_pct": 95.0,
          "ge_train_q50_pct": 50.0
        }
      }
    },
    "all": {
      "input_samples": 1000,
      "analyzed_samples": 1000,
      "sample_mismatch": false,
      "samples": 1000,
      "logpx_mean": -1.23,
      "logpx_std": 0.45,
      "logpx_min": -3.0,
      "logpx_max": -0.1,
      "quantiles": {
        "q01": -2.9,
        "q05": -2.5,
        "q25": -1.7,
        "q50": -1.2,
        "q75": -0.8,
        "q95": -0.3,
        "q99": -0.2
      },
      "acceptance": {
        "ge_train_q01_pct": 98.0,
        "ge_train_q05_pct": 95.0,
        "ge_train_q50_pct": 50.0
      }
    }
  }
}
```

`routing_summary.json` 形态：

```json
{
  "dataset_name": "fu",
  "labels": ["label_0", "label_1", "label_2", "label_3", "label_4"],
  "score_type": "raw_logpx",
  "low_margin_threshold": 1.0,
  "contracts": {
    "fu2508": {
      "samples": 1000,
      "winner_counts": {"label_0": 100, "label_1": 200, "label_2": 0, "label_3": 0, "label_4": 700},
      "winner_pct": {"label_0": 10.0, "label_1": 20.0, "label_2": 0.0, "label_3": 0.0, "label_4": 70.0},
      "top1_top2_margin_mean": 12.3,
      "top1_top2_margin_q25": 3.1,
      "low_margin_pct": 8.5,
      "input_samples_by_label": {"label_0": 1000, "label_1": 1000, "label_2": 1000, "label_3": 1000, "label_4": 1000},
      "sample_mismatch": false
    }
  },
  "all": {
    "samples": 1000,
    "winner_counts": {"label_0": 100, "label_1": 200, "label_2": 0, "label_3": 0, "label_4": 700},
    "winner_pct": {"label_0": 10.0, "label_1": 20.0, "label_2": 0.0, "label_3": 0.0, "label_4": 70.0},
    "top1_top2_margin_mean": 12.3,
    "top1_top2_margin_q25": 3.1,
    "low_margin_pct": 8.5
  }
}
```

## Failure Policy

- `VAE_data` 不存在时失败。
- `VAE_data/test` 不存在或没有 `test_*.npy` 时分析失败。
- 没有任何合约提供当前 `label_k.npy` 时训练失败。
- 部分合约缺少当前 `label_k.npy` 时跳过并记录到 manifest。
- 训练数组或测试数组不是二维时失败。
- 同一 label 的不同合约训练数组 feature 维度不一致时失败。
- 测试数组 feature 维度与训练 feature 维度不一致时失败。
- 合并后样本数为 0 时失败。
- 分析输出覆盖旧结果，但每次输出必须反映当前模型和当前测试文件。
- post-analysis routing summary 发现任一 label/contract 输出缺失时不写出 `routing_summary.json`，等待后续 label 分析进程补齐后再由最后完成的进程生成。
