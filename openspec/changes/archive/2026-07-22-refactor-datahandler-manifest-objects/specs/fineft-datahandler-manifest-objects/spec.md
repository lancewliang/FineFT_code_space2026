## ADDED Requirements

### Requirement: Datahandler manifest data SHALL use dataclass objects internally
系统 SHALL 使用 dataclass 对象表达 `FineFT/datahandler` 中的 split manifest、dataset
manifest 和 valid slice manifest 数据，并在 datahandler 内部通过对象属性或对象方法访问
manifest 数据，而不是通过裸 dict key 访问多层 manifest。

#### Scenario: split manifest 以对象返回和传递
- **WHEN** `load_dataset_split_manifest()` 成功读取并校验 `dataset_split_manifest.json`
- **THEN** 函数 SHALL 返回 `DatasetSplitManifest` 对象
- **AND** 返回对象 SHALL 暴露 `symbol`、`target_freq` 和 `sets` 属性
- **AND** `sets["train"].contracts` 中的每个条目 SHALL 是包含 `contract`、可选 `range`、
  可选 `trading_days` 和可选 `output_row_count` 属性的对象
- **AND** `sets["valid"].skipped_contracts` 中的每个条目 SHALL 是包含 `contract` 和可选
  `reason` 属性的对象，而不是 dict
- **AND** `build_dataset_manifest()` SHALL 接收 `DatasetSplitManifest`，不得依赖
  `split_manifest["sets"]` 这类 dict key 访问 split manifest

#### Scenario: dataset manifest 以对象返回和更新
- **WHEN** `build_dataset_manifest()` 生成 dataset manifest
- **THEN** 函数 SHALL 返回 `DatasetManifest` 对象
- **AND** 返回对象 SHALL 暴露 `symbol`、`target_freq`、`dataset_split_manifest_path`、
  `state_features_source_path`、`state_features_path` 和 `sets` 属性
- **AND** `write_stage_datasets()`、`rebuild_train_slice_plan()` 和 `write_train_slices()`
  SHALL 使用 `DatasetManifest` 对象更新合约行数、集合总行数、slice plan 和 slice 行数
- **AND** `DatasetManifest.sets["valid"].skipped_contracts` SHALL 使用对象列表表达跳过合约
  记录，不得保留 `list[dict]` 作为内部字段类型
- **AND** datahandler 内部 SHALL NOT 使用裸 dict 作为 dataset manifest 在这些函数之间传递

#### Scenario: dataset skipped contracts 以对象传递
- **WHEN** split manifest 的 `sets.valid.skipped_contracts` 包含
  `{"contract": "fu2509", "reason": "no trading days in valid range"}`
- **THEN** `DatasetSplitManifest.sets["valid"].skipped_contracts[0]` SHALL 是
  `DatasetSkippedContract` 对象
- **AND** 该对象 SHALL 暴露 `contract="fu2509"` 和
  `reason="no trading days in valid range"` 属性
- **AND** `build_dataset_manifest()` SHALL 将该 skipped contract 作为对象复制到
  `DatasetManifest.sets["valid"].skipped_contracts`
- **AND** datahandler 内部 SHALL NOT 将 dataset skipped contracts 作为 `list[dict]` 传递

#### Scenario: slice manifest 以对象读取、更新和聚合
- **WHEN** `slice_model.py` 为 valid 合约写出动态标签切片或记录跳过合约
- **THEN** 系统 SHALL 使用 `SliceManifest` 对象读取现有 `slice_manifest.json` 或创建新 manifest
- **AND** 系统 SHALL 通过对象方法替换当前合约记录、移除过期 skip 记录、记录 skip 原因、
  重建 label 聚合视图并稳定排序
- **AND** `slice_model.py` SHALL NOT 在 `_write_slice_manifest()`、
  `_write_skip_manifest()` 或 label 聚合逻辑中手工拼装完整嵌套 manifest dict

### Requirement: Datahandler manifest serialization SHALL preserve JSON contracts
系统 SHALL 只在 JSON 读写边界将 datahandler manifest 对象和 dict 互相转换，并保持现有
manifest JSON 文件结构兼容。

#### Scenario: dataset manifest JSON 结构保持兼容
- **WHEN** `run_dataset_generation()` 写出
  `dataset/{target_freq}/{symbol}/dataset_manifest.json`
- **THEN** JSON SHALL 保持当前兼容结构，包含 `symbol`、`target_freq`、
  `dataset_split_manifest_path`、`state_features_source_path`、`state_features_path` 和
  `sets`
- **AND** 每个 set SHALL 保持 `range`、`contracts`、`skipped_contracts` 和
  `contracts_total_count` 字段语义
- **AND** `skipped_contracts` 写出的 JSON payload SHALL 与输入 split manifest 中的跳过记录
  字段兼容，包括 `contract`、`reason` 和已有额外审计字段
- **AND** train 合约的 `slice_outputs` SHALL 保持 `index`、`contract`、`path`、
  `source_output`、`row_start`、`row_end` 和 `output_row_count` 字段语义
- **AND** 写出的 JSON payload SHALL 等于 `DatasetManifest.to_dict()` 的结果

#### Scenario: slice manifest JSON 结构保持兼容
- **WHEN** `slice_model.py` 写出 `valid/slice_manifest.json`
- **THEN** JSON SHALL 保持当前兼容结构，包含 `valid_path`、`contracts`、`labels` 和
  `skipped_contracts`
- **AND** contract 视角 SHALL 保持每个合约的 `contract`、`processed_path`、`file_count`、
  `total_row_count` 和 `labels`
- **AND** label 视角 SHALL 保持每个 label 的 `label`、`file_count`、`total_row_count` 和
  带 `contract`、`path`、`output_row_count` 的 `files`
- **AND** 写出的 JSON payload SHALL 等于 `SliceManifest.to_dict()` 的结果

#### Scenario: 对象层不改变现有错误处理
- **WHEN** split manifest 校验、stage dataset 写出、train slice 写出或 valid slice manifest
  写出遇到现有错误条件
- **THEN** 系统 SHALL 保持现有失败行为和异常类型
- **AND** dataclass 对象层 SHALL NOT 新增独立业务校验或吞掉底层异常
- **AND** 现有错误消息的主要关键词 SHALL 保持可用于 focused tests 匹配

### Requirement: Datahandler manifest object refactor SHALL be covered by focused tests
系统 SHALL 通过聚焦测试同时验证 datahandler 内部对象接口和外部 JSON 兼容性。

#### Scenario: focused tests assert object return types and attributes
- **WHEN** 执行 datahandler manifest 相关测试
- **THEN** 测试 SHALL 断言 `load_dataset_split_manifest()` 返回 `DatasetSplitManifest`
- **AND** 测试 SHALL 断言 `build_dataset_manifest()` 返回 `DatasetManifest`
- **AND** 测试 SHALL 断言 `run_dataset_generation()` 返回 `DatasetManifest`
- **AND** 测试 SHALL 断言 dataset split 和 dataset output 的 skipped contract 记录可通过
  对象属性访问
- **AND** 测试中针对返回值的业务断言 SHALL 使用对象属性访问

#### Scenario: focused tests assert JSON payload compatibility
- **WHEN** focused tests 读取 `dataset_manifest.json` 或 `slice_manifest.json`
- **THEN** 测试 SHALL 断言 `json.loads(...)` 的结果等于对应 manifest 对象 `to_dict()` 的结果
- **AND** 测试 SHALL 保留关键字段、层级、路径、行数、slice 编号和 label 聚合字段的兼容性断言
- **AND** focused verification SHALL 使用
  `conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py FineFT/tests/datahandler/test_slice_model.py`
