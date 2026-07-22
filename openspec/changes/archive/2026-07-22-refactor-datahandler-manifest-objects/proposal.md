# refactor-datahandler-manifest-objects

## 背景与目标

`FineFT/datahandler` 中生成和读写 JSON manifest 的代码目前直接使用嵌套 dict
拼装和传递数据，主要包括 `dataset_manifest.json`、`slice_manifest.json`，以及输入的
`dataset_split_manifest.json`。字段结构散落在 `commodity_contract_dataset.py` 和
`slice_model.py` 的函数体里，调用方通过字符串 key 访问多层数据，后续维护时容易出现
字段遗漏、拼写错误或 JSON 契约漂移。

本次重构目标是将 datahandler manifest 数据改为使用明确的 dataclass 对象表达，让
代码在进程内通过对象属性和对象方法读写 manifest 数据，同时保持现有 JSON 文件结构、
目录结构、业务处理流程和主要错误行为兼容。

## 用户场景

- 开发者阅读商品合约 dataset 生成流程时，可以通过 `DatasetManifest` 和相关对象理解
  stage、contract、slice 输出字段，而不是在嵌套 dict 中推断结构。
- 开发者维护 valid 切片流程时，可以通过 `SliceManifest` 的更新与聚合方法理解
  `contracts`、`labels`、`skipped_contracts` 的关系。
- 测试仍能通过 JSON 文件内容确认外部输出兼容，同时通过返回对象确认内部接口已对象化。

## 设计方向

采用 datahandler 专用 manifest dataclass 模块方案。新增
`FineFT/datahandler/manifests.py`，集中定义输入 split manifest、输出 dataset manifest
和 slice manifest 相关对象。对象只负责表达 JSON 数据结构、必要的边界转换、排序规则、
聚合更新和 `to_dict()` 序列化，不负责 pandas/numpy 文件 I/O，也不改变数据处理业务逻辑。

`commodity_contract_dataset.py` 继续负责 split manifest 读取、stage 文件复制、
state features 复制、train slice plan 重建、train slice 写出和
`dataset_manifest.json` 落盘。`load_dataset_split_manifest()` 返回
`DatasetSplitManifest`；`build_dataset_manifest()` 和 `run_dataset_generation()` 返回
`DatasetManifest`。内部访问改为对象属性访问，不再依赖裸 dict key 传递 manifest 数据。

`slice_model.py` 继续负责单个 valid contract 的动态标签切片流程。`slice_manifest.json`
的读取、合并、跳过记录、按 label 聚合和排序由 `SliceManifest` 对象完成，最终写 JSON
时调用 `to_dict()`。

错误处理保持现状：对象层不引入第三方 schema 库，不新增业务规则，不改变现有
`FileNotFoundError`、`ValueError` 或文件写入异常的主要失败行为。

## 关键决策

- 只覆盖 `FineFT/datahandler` 中 manifest JSON 相关代码，不合并现有 VAE JSON 对象化变更。
- 输入 `dataset_split_manifest.json` 也对象化，`load_dataset_split_manifest()` 返回
  `DatasetSplitManifest`。
- `build_dataset_manifest()` 和 `run_dataset_generation()` 返回 `DatasetManifest`。
- `slice_model.py` 使用 `SliceManifest` 管理 `slice_manifest.json` 的 contracts、labels 和
  skipped contracts。
- 使用标准库 `dataclass`，不引入 pydantic 或其他运行时 schema 依赖。
- 对外 JSON 文件结构保持兼容，包括字段名、层级、排序和数值类型。
- `to_dict()` 只用于 JSON 写入边界和测试中与 `json.loads(...)` 的兼容性对比。
- 不改变 stage 数据复制、train slice 生成、valid 切片、label 聚合的业务计算逻辑。

## 范围边界

**包含：**
- 新增 `FineFT/datahandler/manifests.py`。
- 为 `dataset_split_manifest.json` 定义 `DatasetSplitManifest` 及相关 stage/contract 对象。
- 为 `dataset_manifest.json` 定义 `DatasetManifest` 及相关 set/contract/slice 对象。
- 为 `slice_manifest.json` 定义 `SliceManifest` 及相关 contract/label/file/skipped 对象。
- 将 `load_dataset_split_manifest()`、`build_dataset_manifest()`、
  `run_dataset_generation()` 的返回值改为 dataclass 对象。
- 将 `commodity_contract_dataset.py` 内部 manifest 访问改为对象属性或对象方法访问。
- 将 `slice_model.py` 的 manifest 读取、更新、聚合、排序和写出改为通过 `SliceManifest`
  完成。
- 更新 focused tests，覆盖对象返回值、属性访问和 JSON 兼容性。

**不包含（本次）：**
- 不重构 `FineFT/RL/DiHFT/VAE` 中的 JSON 输出对象。
- 不重构非 manifest 用途的普通 dict，例如算法临时映射、计数器或 pandas/numpy 中间数据。
- 不改变 `dataset_manifest.json`、`slice_manifest.json`、
  `dataset_split_manifest.json` 的外部字段结构。
- 不新增 JSON 反序列化公共 API，除 manifest 对象从现有 dict/JSON 边界构建所需方法外。
- 不引入第三方数据校验库。
- 不改变 shell 脚本入口、CLI 参数、文件命名或目录布局。

## 验收标准

- [ ] `load_dataset_split_manifest()` 返回 `DatasetSplitManifest`，调用方可通过对象属性访问
  `symbol`、`target_freq`、`sets["train"].contracts`。
- [ ] `build_dataset_manifest()` 返回 `DatasetManifest`，内部不再通过裸 dict key 传递和更新
  dataset manifest。
- [ ] `run_dataset_generation()` 返回 `DatasetManifest`，写出的 `dataset_manifest.json` 与
  `returned.to_dict()` 一致。
- [ ] `write_stage_datasets()`、`rebuild_train_slice_plan()`、`write_train_slices()` 使用
  `DatasetManifest` 对象并保持现有复制、计数和切片行为。
- [ ] `slice_model.py` 使用 `SliceManifest` 更新 contract 记录、skip 记录和 label 聚合。
- [ ] 写出的 `slice_manifest.json` 字段结构与现有输出兼容，并与 `SliceManifest.to_dict()`
  的结果一致。
- [ ] 现有错误场景仍失败，包括 split manifest 缺失、symbol/target_freq 不匹配、
  stage contracts 缺失、state features 缺失或为空、stage feather 缺失或为空、
  train slice index 不连续。
- [ ] 使用 `conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py FineFT/tests/datahandler/test_slice_model.py`
  通过 focused test 验证。
- [ ] 使用 `conda activate finetf && python -m py_compile FineFT/datahandler/manifests.py FineFT/datahandler/commodity_contract_dataset.py FineFT/datahandler/slice_model.py`
  通过语法验证。

## Amendments

### 2026-07-22: typed skipped contract records

原因：build 后复查发现当前对象模型仍在 dataset split 和 dataset manifest 的
`skipped_contracts` 字段中保留 `list[dict]`。该字段属于本次 manifest 对象化边界，继续
使用裸 dict 会留下同类可读性和可维护性问题。

摘要：为 dataset split / dataset output 增加 `DatasetSkippedContract` dataclass，
`DatasetSplitSet.skipped_contracts` 和 `DatasetSetManifest.skipped_contracts` 改为
`list[DatasetSkippedContract]`。JSON 字段结构保持兼容；实现可保留未知审计字段的
round-trip 能力，但调用方不再直接传递 `list[dict]`。
