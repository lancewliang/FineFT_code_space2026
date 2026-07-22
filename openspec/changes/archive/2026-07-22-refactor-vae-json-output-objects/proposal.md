# refactor-vae-json-output-objects

## 背景与目标

VAE 相关 JSON 输出目前在生成流程中直接使用 dict 拼装和传递，包括
`label_k_manifest.json`、`summary.json` 和 `routing_summary.json`。这让字段结构隐含在
多个函数体里，调用方也通过字符串 key 访问中间数据，后续维护时容易出现字段遗漏、
拼写错误或 JSON 契约漂移。

本次重构目标是将 VAE JSON 输出数据改为使用明确的 dataclass 对象表达，并让 VAE
内部函数之间传递对象而不是 dict。JSON 文件结构必须保持完全兼容，重构不改变现有
业务行为、失败时机或输出契约。

## 用户场景

- 开发者阅读 VAE 训练数据合并流程时，可以通过对象属性理解训练 manifest 的字段含义。
- 开发者维护 per-label summary 或 routing summary 时，可以在对象模型中看到结构边界，
  而不是在多个裸 dict 中推断字段层级。
- 测试仍能通过 JSON 文件内容确认对外输出兼容，同时通过返回对象确认内部接口已对象化。

## 设计方向

采用 VAE 专用 dataclass 对象模型方案。新增 `FineFT/RL/DiHFT/VAE/manifests.py`，
集中定义训练 manifest、logpx summary、routing summary 相关对象。对象只负责表达数据
和序列化，不负责读写 `.npy`、运行模型、发现路径或计算统计。

`merge_vae_train.py` 继续负责训练数据发现、二维数组校验、跨合约合并和文件落盘。
`materialize_label_training_data()` 改为返回 `LabelTrainingManifest`，调用方使用
`train_manifest.merged_path`、`train_manifest.feature_dim` 等属性访问，不再通过
`train_manifest["merged_path"]` 访问。

`summary.py` 继续负责统计计算和 JSON 写出。`write_contract_logpx_outputs()` 返回
`LabelSummary` 对象，`write_routing_summary()` 返回 `RoutingSummary` 对象，
`maybe_write_routing_summary_after_analysis()` 返回 `RoutingSummary | None`。
`to_dict()` 只在 JSON 写入边界和测试兼容断言中使用。

错误处理保持现状：对象层不新增业务校验，不改变现有 `FileNotFoundError`、`ValueError`
或文件写入异常的抛出行为。

## 关键决策

- 只覆盖 VAE 相关 JSON 输出：`label_k_manifest.json`、`summary.json`、
  `routing_summary.json`。
- 使用标准库 `dataclass`，不引入 pydantic 或其他运行时 schema 依赖。
- VAE 内部函数之间传递 manifest/summary 对象，不再用 dict 进行方法传递。
- 对外 JSON 文件结构保持完全兼容，包括字段名、层级和数值类型。
- `to_dict()` 只用于 JSON 写入边界和测试中与 `json.loads(...)` 的兼容性对比。
- 不在对象层新增反序列化、schema 校验或额外错误处理。

## 范围边界

**包含：**
- 新增 VAE JSON 输出对象模型模块。
- 将 `materialize_label_training_data()` 的返回值改为训练 manifest 对象。
- 将 `main.py` 中 VAE 训练 manifest 访问改为对象属性访问。
- 将 `write_contract_logpx_outputs()` 的返回值改为 label summary 对象。
- 将 `write_routing_summary()` 的返回值改为 routing summary 对象。
- 将 `maybe_write_routing_summary_after_analysis()` 调整为返回 routing summary 对象或
  `None`。
- 更新 VAE 相关 focused tests，覆盖对象返回值和 JSON 兼容性。

**不包含（本次）：**
- 不重构非 VAE 模块的 manifest，例如 feature selection manifest。
- 不改变 `label_k_manifest.json`、`summary.json`、`routing_summary.json` 的外部结构。
- 不新增 JSON 反序列化 API。
- 不引入第三方数据校验库。
- 不改变 VAE 训练、分析或 routing summary 的业务计算逻辑。

## 验收标准

- [ ] `materialize_label_training_data()` 返回 `LabelTrainingManifest`，调用方不再通过
  dict key 访问训练 manifest。
- [ ] `write_contract_logpx_outputs()` 返回 `LabelSummary`，内部 summary 数据传递使用对象。
- [ ] `write_routing_summary()` 返回 `RoutingSummary`，routing summary 数据传递使用对象。
- [ ] `maybe_write_routing_summary_after_analysis()` 返回 `RoutingSummary | None`。
- [ ] `label_k_manifest.json`、`summary.json`、`routing_summary.json` 的 JSON 字段结构与
  现有输出兼容。
- [ ] 测试断言返回对象的属性访问，并断言 `returned.to_dict()` 与写出的 JSON 内容一致。
- [ ] 使用 `conda activate finetf && pytest FineFT/tests/rl/test_commodity_vae_cross_contract.py`
  通过 focused test 验证。

## Amendments

### 2026-07-22: 扩大 VAE 内部 list[dict] 对象化范围

实现后复查发现，VAE JSON 输出对象化已经覆盖 manifest、summary 和 routing summary，
但当前 VAE 设计范围内仍有两条内部方法链路在传递 `list[dict]`：

- `merge_vae_train.discover_label_sources()` 返回 included label source dict 列表。
- `main.discover_test_sources()` 到 `process.prepare_contract_dataset_loader_list()` 再到
  `analyze_contract_tests()` 之间传递 test contract source / loader dict 列表。

这些不是 JSON/CSV 写出边界，而是 VAE 内部方法之间的数据传递，符合本变更“使用对象而不是
dict 书写代码”的目标。因此本变更追加范围：将 label array source、test contract source
和 contract dataset loader 也改为 dataclass 对象。CSV row payload 和 JSON contract-keyed
mapping 暂不纳入本次扩大范围，因为它们属于序列化边界或外部 JSON 结构。
