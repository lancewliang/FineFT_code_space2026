## 1. Implementation

- [x] 1.1 Add focused tests for datahandler manifest object return types, attribute access, and JSON payload compatibility. <!-- 已实现: 新增对象返回值、属性访问和 JSON 兼容测试，并确认实现前失败 -->
- [x] 1.2 Add `FineFT/datahandler/manifests.py` with dataclass models for split, dataset, and slice manifests. <!-- 已实现: 新增 split、dataset、slice manifest dataclass 与 to_dict/from_dict/update 方法 -->
- [x] 1.3 Refactor `FineFT/datahandler/commodity_contract_dataset.py` to use `DatasetSplitManifest` and `DatasetManifest` objects internally and at public return boundaries. <!-- 已实现: commodity dataset 流程改为对象返回、属性访问和 to_dict 写出 -->
- [x] 1.4 Refactor `FineFT/datahandler/slice_model.py` to use `SliceManifest` for manifest reads, contract updates, skip updates, label aggregation, sorting, and JSON serialization. <!-- 已实现: slice_model manifest 读写、contract/skip 更新和 label 聚合改由 SliceManifest 完成 -->
- [x] 1.5 Run focused verification for datahandler tests, Python compilation, and OpenSpec validation. <!-- 已实现: focused tests、py_compile、openspec strict 和 diff 范围检查均已运行 -->
- [x] 1.6 Refactor dataset split/output `skipped_contracts` from `list[dict]` to `list[DatasetSkippedContract]` while preserving JSON compatibility. <!-- 已实现: dataset split/output skipped_contracts 改为 DatasetSkippedContract 对象列表并保持 to_dict JSON 兼容 -->
