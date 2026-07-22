# Design: refactor-datahandler-manifest-objects

## Context

`FineFT/datahandler` 当前有三类 manifest 数据通过嵌套 dict 生成、读取和传递：

```text
dataset_split_manifest.json
dataset/{target_freq}/{symbol}/dataset_manifest.json
dataset/{target_freq}/{symbol}/valid/slice_manifest.json
```

这些 JSON 的外部结构已经被商品多合约数据集流程和现有测试固定下来。问题在于内部代码
仍在 `commodity_contract_dataset.py` 和 `slice_model.py` 中直接拼装多层 dict，并使用
字符串 key 访问和更新字段。字段层级越深，维护者越难看出哪些字段属于 split 输入、
dataset 输出或 valid slice 输出，也更容易因拼写错误或遗漏字段改变 JSON 契约。

## Decisions

1. 新增 `FineFT/datahandler/manifests.py`，集中定义 datahandler manifest 对象。
2. 使用标准库 `dataclass`，不引入 pydantic 或其他运行时 schema 依赖。
3. 输入 `dataset_split_manifest.json` 也转换为 `DatasetSplitManifest`，不在后续流程传递裸 dict。
4. `build_dataset_manifest()`、`run_dataset_generation()` 返回 `DatasetManifest`。
5. `slice_model.py` 用 `SliceManifest` 读取、更新、聚合、排序和写出 `slice_manifest.json`。
6. `to_dict()` 只用于 JSON 写入边界和测试中的 JSON 兼容断言。
7. JSON 文件字段名、层级、排序和数值类型保持兼容。
8. 对象层不新增业务规则；已有文件存在性、空数据、slice index 连续性和 JSON 写入失败行为保持原状。
9. dataset split 和 dataset output 的 `skipped_contracts` 也使用 dataclass 对象表达，不在
   manifest 对象内部保留 `list[dict]`。

## Module Responsibilities

- `FineFT/datahandler/manifests.py`: split、dataset、slice manifest 的 dataclass 定义、
  `from_dict()` 边界转换、更新方法、排序方法和 `to_dict()` 序列化。
- `FineFT/datahandler/commodity_contract_dataset.py`: 继续负责读取 split JSON、复制 stage
  feather、复制 state features、重建 train slice plan、写 train slices 和写
  `dataset_manifest.json`；内部通过 manifest 对象属性和方法访问数据。
- `FineFT/datahandler/slice_model.py`: 继续负责单合约 valid 动态标签切片；每个合约的
  manifest 更新交给 `SliceManifest`。
- `FineFT/tests/datahandler/test_commodity_contract_dataset.py`: 覆盖 split/dataset manifest
  对象返回值、属性访问、JSON 兼容和既有错误行为。
- `FineFT/tests/datahandler/test_slice_model.py`: 覆盖 slice manifest JSON 兼容和
  `SliceManifest` 聚合行为。

## Data Model

split 输入对象保留现有 JSON 结构：

```python
DatasetSplitManifest(
    symbol="fu",
    target_freq="10min",
    sets={
        "train": DatasetSplitSet(
            range=["2026-01-01", "2026-01-06"],
            contracts=[
                DatasetSplitContract(
                    contract="fu2508",
                    trading_days=["2026-01-01", "2026-01-02"],
                    output_row_count=4,
                    range=None,
                )
            ],
            skipped_contracts=[],
        )
    },
)
```

其中 `DatasetSplitSet.skipped_contracts` 使用 `DatasetSkippedContract` 对象列表表达。
该对象至少暴露 `contract` 和可选 `reason` 属性，并可保留 split manifest 中已有的额外
审计字段，保证 `to_dict()` 后 JSON payload 不漂移。

dataset 输出对象保留 `dataset_manifest.json` 结构：

- `DatasetSliceOutput`: `index`、`contract`、`path`、`source_output`、`row_start`、
  `row_end` 和可选 `output_row_count`
- `DatasetContractManifest`: `contract`、`input_path`、`output_path`、可选 `range`、
  `trading_days`、`output_row_count`、`slice_outputs`
- `DatasetSkippedContract`: `contract`、可选 `reason` 和可选额外审计字段
- `DatasetSetManifest`: `range`、`contracts`、`skipped_contracts`、可选
  `contracts_total_count`
- `DatasetManifest`: `symbol`、`target_freq`、`dataset_split_manifest_path`、
  `state_features_source_path`、`state_features_path`、`sets`

slice 输出对象保留 `slice_manifest.json` 结构：

- `SliceFileManifest`: `path`、`output_row_count` 和可选 `contract`
- `SliceLabelManifest`: `label`、`file_count`、`total_row_count`、`files`
- `SliceContractManifest`: `contract`、`processed_path`、`file_count`、
  `total_row_count`、`labels`
- `SkippedContractManifest`: `contract`、`processed_path`、`reason`、`input_row_count`
- `SliceManifest`: `valid_path`、`contracts`、`labels`、`skipped_contracts`

## Amendments

### 2026-07-22: skipped contracts are typed records

复查发现 `DatasetSplitSet.skipped_contracts` 和 `DatasetSetManifest.skipped_contracts` 仍是
`list[dict]`。这属于 manifest 数据模型自身，不是算法临时映射，因此应纳入当前对象化
重构。新增 `DatasetSkippedContract` 后，split 输入读取、dataset manifest 构建和 JSON
写出都通过对象完成；未知审计字段只在对象内部作为兼容 payload 保留。

## Compatibility

本次是内部接口重构。以下 JSON 文件必须保持兼容：

- `dataset_split_manifest.json` 读取契约
- `dataset_manifest.json` 写出契约
- `slice_manifest.json` 写出契约

对外兼容由两层测试保证：返回对象属性断言证明内部对象化生效，`json.loads(...)` 与
`returned.to_dict()` 或 `SliceManifest.to_dict()` 的一致性断言证明 JSON 结构未漂移。

## Failure Policy

- split manifest 文件不存在时仍失败。
- `symbol` 或 `target_freq` 不匹配时仍失败。
- `sets.<stage>.contracts` 缺失或类型错误时仍失败。
- `state_features.npy` 缺失或为空时仍失败。
- stage feather 缺失或复制后为空时仍失败。
- train slice index 不连续时仍失败。
- JSON 写入失败时仍暴露底层文件写入异常。
