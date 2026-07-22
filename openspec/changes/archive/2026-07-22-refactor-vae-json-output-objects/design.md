# Design: refactor-vae-json-output-objects

## Context

当前商品 VAE 跨合约流程已经输出三类 JSON：

```text
dataset/10min/fu/VAE_data/train/label_k_manifest.json
result/DiHFT/vae_results/fu/label_k/summary.json
result/DiHFT/vae_results/fu/routing_summary.json
```

这些 JSON 的业务含义已经稳定，但生成代码在 `merge_vae_train.py`、`main.py` 和
`summary.py` 中直接拼装和传递 dict。字段结构缺少显式对象边界，调用方依赖字符串 key
访问中间数据。后续调整 summary 或 routing summary 时，容易产生字段遗漏、拼写错误或
不小心改变 JSON 契约。

## Decisions

1. 新增 `FineFT/RL/DiHFT/VAE/manifests.py`，集中定义 VAE JSON 输出对象。
2. 使用标准库 `dataclass`，不引入 pydantic 或其他运行时 schema 依赖。
3. VAE 内部 manifest/summary 数据通过对象传递，不通过 dict 传递。
4. `to_dict()` 只用于 JSON 写入边界和测试中的 JSON 兼容断言。
5. JSON 文件字段名、层级和数值类型保持完全兼容。
6. 对象层不新增业务校验；已有路径发现、数组维度、样本数和文件写入错误行为保持原状。
7. 本次不重构非 VAE manifest，例如 feature selection manifest。
8. VAE 内部 source discovery 和 loader preparation 返回对象，不返回 `list[dict]`。
9. CSV row dict 和 JSON contract-keyed mapping 保留为写出边界结构，不强行对象化。

## Module Responsibilities

- `manifests.py`: 训练 manifest、label summary、routing summary 的 dataclass 定义和
  `to_dict()` 序列化；同时定义 VAE 内部 source 和 loader 数据对象。
- `merge_vae_train.py`: 继续负责 VAE 训练数据发现、数组校验、跨合约合并、训练数组写出和
  训练 manifest JSON 写出；返回 `LabelTrainingManifest`，并用 `LabelArraySource` 表达
  discovered label source。
- `main.py`: 继续负责 VAE workflow 编排；通过对象属性读取训练 manifest，不再通过 dict key
  读取；`discover_test_sources()` 返回 `TestContractSource` 列表。
- `process.py`: 继续负责 VAE DataLoader 准备和合约分析；`prepare_contract_dataset_loader_list()`
  返回 `ContractDatasetLoader` 列表，`analyze_contract_tests()` 使用对象属性访问 loader 元数据。
- `summary.py`: 继续负责 logpx 统计、per-contract CSV/Numpy 输出、`summary.json` 和
  `routing_summary.json` 写出；返回 `LabelSummary`、`RoutingSummary` 或 `None`。
- `FineFT/tests/rl/test_commodity_vae_cross_contract.py`: 覆盖对象返回值、属性访问和 JSON
  兼容性。

## Data Model

训练 manifest 对象：

```python
LabelTrainingManifest(
    dataset_name="fu",
    label="label_0",
    merged_path="dataset/10min/fu/VAE_data/train/label_0.npy",
    total_samples=12345,
    feature_dim=46,
    included_contracts=[
        LabelContractSource(
            contract="fu2505",
            source_file="dataset/10min/fu/VAE_data/fu2505/label_0.npy",
            sample_count=1519,
        )
    ],
    missing_contracts=["fu2510"],
)
```

`summary.py` 的对象模型保留当前 JSON 结构：

- `SampleIntegrity`: `input_samples`、`analyzed_samples`、`sample_mismatch`
- `LogpxStats`: `samples`、`logpx_mean`、`logpx_std`、`logpx_min`、`logpx_max`、`quantiles`
- `ContractLogpxSummary`: `source_file`、完整性、logpx 统计和可选 acceptance
- `LabelSummary`: `dataset_name`、`label`、`test.contracts`、`test.all` 和可选
  `train_baseline`
- `WinnerSummary`: routing 的 winner counts、winner pct、margin 和 low-margin 统计
- `ContractRoutingSummary`: `WinnerSummary` 加上 `input_samples_by_label` 和
  `sample_mismatch`
- `RoutingSummary`: `dataset_name`、`labels`、`score_type`、`low_margin_threshold`、
  `contracts` 和 `all`

追加内部 source/loader 对象：

- `LabelArraySource`: `contract`、`source_file`
- `TestContractSource`: `contract`、`source_file`
- `ContractDatasetLoader`: `contract`、`source_file`、`loader`

## Compatibility

本次是内部接口重构。以下 JSON 输出必须保持现有兼容格式：

- `label_k_manifest.json`
- `summary.json`
- `routing_summary.json`

现有 VAE 训练、分析、routing summary 的计算逻辑不变。测试应同时验证返回对象属性和
`returned.to_dict()` 与 `json.loads(...)` 的一致性。

`_logpx_rows()` 返回的 row dict 属于 `pandas.DataFrame(...).to_csv()` 写出边界，本次保持
不变。`LabelSummary.test.contracts` 和 `RoutingSummary.contracts` 使用 dict 是为了保持
外部 JSON 以 contract 为 key 的兼容结构，本次也保持不变。

## Failure Policy

- `VAE_data` 不存在时仍失败。
- 没有任何合约提供当前 `label_k.npy` 时仍失败。
- 训练或测试数组不是二维、为空或 feature 维度不一致时仍失败。
- JSON 写入失败时仍暴露底层文件写入异常。
- `maybe_write_routing_summary_after_analysis()` 发现任一 label/contract 输出缺失时仍返回
  `None`，不写出 `routing_summary.json`。
