## ADDED Requirements

### Requirement: VAE JSON outputs SHALL use dataclass objects internally
系统 SHALL 使用 dataclass 对象表达 VAE 训练 manifest、per-label summary 和 routing summary
数据，并在 VAE 内部函数之间传递对象而不是 dict。

#### Scenario: 训练 manifest 以对象返回和传递
- **WHEN** `materialize_label_training_data()` 完成 `dataset/10min/fu/VAE_data/train/label_0.npy`
  物化
- **THEN** 函数 SHALL 返回 `LabelTrainingManifest` 对象
- **AND** 返回对象 SHALL 暴露 `dataset_name`、`label`、`merged_path`、`total_samples`、
  `feature_dim`、`included_contracts` 和 `missing_contracts` 属性
- **AND** `included_contracts` 中的每个条目 SHALL 是包含 `contract`、`source_file` 和
  `sample_count` 属性的对象
- **AND** VAE 调用方 SHALL 使用对象属性访问训练 manifest，不得使用
  `train_manifest["merged_path"]` 这类 dict key 访问

#### Scenario: 非训练模式训练数据引用以对象传递
- **WHEN** `main.py` 在 analyze-only 或非训练模式下读取已物化的
  `VAE_data/train/label_0.npy`
- **THEN** 系统 SHALL 构造对象来表达训练数据路径、样本数和 feature 维度
- **AND** 后续 VAE workflow SHALL 使用对象属性读取 `merged_path`、`total_samples` 和
  `feature_dim`
- **AND** 系统 SHALL NOT 构造临时 dict 作为训练 manifest 在方法之间传递

#### Scenario: per-label summary 以对象返回和传递
- **WHEN** `write_contract_logpx_outputs()` 写出
  `result/DiHFT/vae_results/fu/label_0/summary.json`
- **THEN** 函数 SHALL 返回 `LabelSummary` 对象
- **AND** 返回对象 SHALL 暴露 `dataset_name`、`label`、`test` 和可选 `train_baseline`
  属性
- **AND** test contracts、test all、logpx stats、sample integrity 和 acceptance SHALL 在
  VAE 内部以对象表达
- **AND** summary 数据 SHALL NOT 在 VAE 内部以裸 dict 进行方法传递

#### Scenario: routing summary 以对象返回和传递
- **WHEN** `write_routing_summary()` 写出
  `result/DiHFT/vae_results/fu/routing_summary.json`
- **THEN** 函数 SHALL 返回 `RoutingSummary` 对象
- **AND** 返回对象 SHALL 暴露 `dataset_name`、`labels`、`score_type`、
  `low_margin_threshold`、`contracts` 和 `all` 属性
- **AND** winner summary、contract routing summary 和 aggregate routing summary SHALL 在
  VAE 内部以对象表达
- **AND** `maybe_write_routing_summary_after_analysis()` SHALL 返回 `RoutingSummary` 或
  `None`

### Requirement: VAE JSON object serialization SHALL preserve existing output contracts
系统 SHALL 只在 JSON 写入边界将 VAE JSON 输出对象序列化为 dict，并保持现有 JSON 文件结构
完全兼容。

#### Scenario: 训练 manifest JSON 结构保持兼容
- **WHEN** 系统写出 `dataset/10min/fu/VAE_data/train/label_0_manifest.json`
- **THEN** JSON SHALL 包含当前兼容字段 `dataset_name`、`label`、`merged_path`、
  `total_samples`、`feature_dim`、`included_contracts` 和 `missing_contracts`
- **AND** `included_contracts` 中每项 SHALL 包含 `contract`、`source_file` 和
  `sample_count`
- **AND** 写出的 JSON payload SHALL 等于 `LabelTrainingManifest.to_dict()` 的结果
- **AND** JSON 字段名、字段层级和数值类型 SHALL NOT 因对象化重构而改变

#### Scenario: summary JSON 结构保持兼容
- **WHEN** 系统写出 `result/DiHFT/vae_results/fu/label_0/summary.json`
- **THEN** JSON SHALL 保持当前兼容结构，包含 `dataset_name`、`label`、`test` 和可选
  `train_baseline`
- **AND** `test.contracts`、`test.all`、`quantiles`、`acceptance` 和
  sample integrity 字段 SHALL 保持当前字段名、层级和数值类型
- **AND** 写出的 JSON payload SHALL 等于 `LabelSummary.to_dict()` 的结果
- **AND** summary SHALL NOT 新增 accuracy、AUROC、AUPRC 或 FPR80 字段

#### Scenario: routing summary JSON 结构保持兼容
- **WHEN** 系统写出 `result/DiHFT/vae_results/fu/routing_summary.json`
- **THEN** JSON SHALL 保持当前兼容结构，包含 `dataset_name`、`labels`、`score_type`、
  `low_margin_threshold`、`contracts` 和 `all`
- **AND** 每个合约 routing summary SHALL 保持 `samples`、`winner_counts`、
  `winner_pct`、`top1_top2_margin_mean`、`top1_top2_margin_q25`、`low_margin_pct`、
  `input_samples_by_label` 和 `sample_mismatch`
- **AND** 写出的 JSON payload SHALL 等于 `RoutingSummary.to_dict()` 的结果
- **AND** JSON 字段名、字段层级和数值类型 SHALL NOT 因对象化重构而改变

#### Scenario: 对象层不改变现有错误处理
- **WHEN** VAE 训练数据发现、数组加载、数组维度校验、summary 写出或 routing summary
  补齐检查遇到现有错误条件
- **THEN** 系统 SHALL 保持现有失败行为和异常类型
- **AND** dataclass 对象层 SHALL NOT 新增独立业务校验或吞掉底层异常
- **AND** `maybe_write_routing_summary_after_analysis()` 在 label/contract 输出未全部存在时
  SHALL 继续返回 `None` 且不写出 `routing_summary.json`

### Requirement: VAE JSON object refactor SHALL be covered by focused tests
系统 SHALL 通过聚焦测试同时验证内部对象接口和外部 JSON 兼容性。

#### Scenario: focused tests assert object return types and attributes
- **WHEN** 执行 VAE JSON 输出相关测试
- **THEN** 测试 SHALL 断言 `materialize_label_training_data()` 返回
  `LabelTrainingManifest`
- **AND** 测试 SHALL 断言 `write_contract_logpx_outputs()` 返回 `LabelSummary`
- **AND** 测试 SHALL 断言 `write_routing_summary()` 返回 `RoutingSummary`
- **AND** 测试 SHALL 断言 `maybe_write_routing_summary_after_analysis()` 返回
  `RoutingSummary` 或 `None`
- **AND** 测试中针对返回值的业务断言 SHALL 使用对象属性访问

#### Scenario: focused tests assert JSON payload compatibility
- **WHEN** VAE focused tests 读取 `label_0_manifest.json`、`summary.json` 或
  `routing_summary.json`
- **THEN** 测试 SHALL 断言 `json.loads(...)` 的结果等于返回对象 `to_dict()` 的结果
- **AND** 测试 SHALL 保留关键字段、层级、样本数、统计值和 routing winner 字段的兼容性断言
- **AND** focused verification SHALL 使用
  `conda activate finetf && pytest FineFT/tests/rl/test_commodity_vae_cross_contract.py`

### Requirement: VAE source discovery SHALL return dataclass objects internally
系统 SHALL 使用 dataclass 对象表达 VAE label source、test contract source 和 contract
dataset loader 元数据，并避免在 VAE 内部方法之间传递 `list[dict]`。

#### Scenario: label source discovery returns objects
- **WHEN** `discover_label_sources()` 扫描 `dataset/10min/fu/VAE_data/<contract>/label_0.npy`
- **THEN** included sources SHALL 是 `LabelArraySource` 对象列表
- **AND** 每个 source SHALL 暴露 `contract` 和 `source_file` 属性
- **AND** `materialize_label_training_data()` SHALL 使用 source 属性访问合约和源文件路径
- **AND** 系统 SHALL NOT 通过 `source["contract"]` 或 `source["source_file"]` 访问 label source

#### Scenario: test source discovery returns objects
- **WHEN** `discover_test_sources()` 扫描 `dataset/10min/fu/VAE_data/test/test_fu2508.npy`
- **THEN** 函数 SHALL 返回 `TestContractSource` 对象列表
- **AND** 每个 source SHALL 暴露 `contract` 和 `source_file` 属性
- **AND** 调用方 SHALL NOT 通过 dict key 访问 test contract source

#### Scenario: contract loader preparation returns objects
- **WHEN** `prepare_contract_dataset_loader_list()` 为 discovered test sources 准备 DataLoader
- **THEN** 函数 SHALL 返回 `ContractDatasetLoader` 对象列表
- **AND** 每个 loader 对象 SHALL 暴露 `contract`、`source_file` 和 `loader` 属性
- **AND** `analyze_contract_tests()` SHALL 使用对象属性访问 loader 元数据
- **AND** 系统 SHALL NOT 在 VAE 内部方法之间传递包含 `contract`、`source_file` 和 `loader`
  的 dict

#### Scenario: serialization boundary dicts remain allowed
- **WHEN** 系统构造 per-contract CSV rows 或 JSON contract-keyed mappings
- **THEN** 系统 MAY 在写出边界使用 dict payload 保持 pandas CSV 和 JSON 兼容结构
- **AND** 这些边界 dict SHALL NOT 被用作 VAE 内部 source、loader、manifest 或 summary
  方法传递类型
