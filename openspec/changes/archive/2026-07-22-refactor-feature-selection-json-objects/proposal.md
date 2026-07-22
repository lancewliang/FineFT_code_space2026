# refactor-feature-selection-json-objects

## 背景与目标

`data_preprocess/operator_futures/feature_selection` 中多处 JSON 输出目前直接通过
嵌套 dict 拼装，包括 `feature_selection_manifest.json`、
`feature_union_manifest.json`、`ic_window_*.json` 和 `rank_ic_window_*.json`。
这些 JSON 契约散落在多个脚本函数体内，调用方和测试通过字符串 key 访问字段，后续维护时
容易出现字段遗漏、拼写错误或 JSON 结构漂移。

本次重构目标是将该目录内所有 JSON 输出的生成侧改为使用明确的 dataclass 对象表达，让
代码在进程内通过对象属性和对象方法读写 JSON 契约，同时保持现有 JSON 文件结构、字段名、
目录结构、业务处理流程和主要错误行为兼容。

## 用户场景

- 开发者维护多合约 feature selection 流程时，可以通过
  `FeatureSelectionManifest` 理解 train/valid manifest 字段，而不是在嵌套 dict 中
  推断结构。
- 开发者维护合约特征 union 流程时，可以通过 `FeatureUnionManifest` 理解
  per-contract 输入、输出路径和 shape 记录。
- 开发者维护 IC / RankIC 单窗口分数输出时，可以通过 `FeatureScoreWindow` 表达窗口分数，
  同时 `remove_duplicates_feature.py` 等消费者继续读取原有顶层 feature-score JSON 映射。
- 测试既能通过 JSON 文件内容确认外部输出兼容，也能通过返回对象确认内部接口已对象化。

## 设计方向

采用集中 dataclass 契约模块方案。新增
`data_preprocess/operator_futures/feature_selection/manifests.py`，统一定义该目录所有
JSON 输出相关对象。对象只负责表达 JSON 契约、必要的类型归一化、`to_dict()` 序列化和
可选的 JSON 写入边界，不负责 Polars、NumPy、CatBoost 指标计算，也不改变数据处理业务
逻辑。

`muti_contract/pipeline.py` 继续负责加载 split contract 数据、计算多窗口指标、聚合指标、
筛选特征、写出 filtered outputs 和 `feature_selection_manifest.json`。原先拼 dict 的
位置改为实例化 `FeatureSelectionManifest`，`run_feature_selection()` 返回
`FeatureSelectionResult`，调用方通过 `result.manifest.stage`、
`result.manifest.selected_feature_file` 等属性访问。

`contract_feature_union.py` 继续负责加载主力合约摘要、读取每个合约的 state features、
计算 union、可选写出最终 IC_RESULT 数据和 `feature_union_manifest.json`。原先拼 dict 的
位置改为实例化 `FeatureUnionManifest`，`write_contract_feature_union()` 返回
`FeatureUnionResult`，包含 `output_dir` 和 `manifest`。

`ic_correlation.py` 和 `rank_ic_correlation.py` 的窗口分数输出改为使用
`FeatureScoreWindow`。主流程分别返回 `IcCorrelationResult` 和
`RankIcCorrelationResult`，包含原先返回的输出 DataFrame、选中特征、窗口分数对象和输出
目录。`FeatureScoreWindow.to_dict()` 必须返回原有的 `{feature_name: score}` 顶层映射，
确保 `remove_duplicates_feature.py` 等下游读取逻辑不需要改变。

算法内部短生命周期映射可以保留，例如 `contract -> DataFrame`、`contract -> features`、
`filter_results` 等。它们进入 JSON 契约边界或函数返回边界时，由 dataclass 持有并负责
序列化。

## 关键决策

- 覆盖 `data_preprocess/operator_futures/feature_selection` 中所有写 JSON 的位置。
- 使用标准库 `dataclass`，不引入 pydantic 或其他运行时 schema 依赖。
- 外部 JSON 文件结构完全兼容，包括字段名、层级、顶层 list/dict 形状和主要数值类型。
- `ic_window_*.json` 和 `rank_ic_window_*.json` 继续写成顶层 `{feature: score}` 映射。
- 相关流程返回值对象化，不再返回裸 manifest dict；保留原先 DataFrame 返回信息作为
  result 对象字段。
- `to_dict()` 只用于 JSON 写入边界和测试中与 `json.loads(file_text)` 的兼容性对比。
- 不改变 feature selection、feature union、IC、RankIC 的业务计算逻辑。
- 不重构非 JSON 契约用途的普通 dict，例如算法临时映射、计数器或 Polars/NumPy 中间数据。

## 范围边界

**包含：**
- 新增 `data_preprocess/operator_futures/feature_selection/manifests.py`。
- 为 `feature_selection_manifest.json` 定义 `FeatureSelectionManifest` 及相关合约、
  filtered output 记录对象。
- 为 `feature_union_manifest.json` 定义 `FeatureUnionManifest` 及相关 per-contract 输出
  shape/path 记录对象。
- 为 `ic_window_*.json` 和 `rank_ic_window_*.json` 定义 `FeatureScoreWindow`。
- 为 `run_feature_selection()`、`write_contract_feature_union()`、`ic_correlation.main()`、
  `rank_ic_correlation.main()` 定义 result dataclass 返回对象。
- 将 JSON 写入处改为通过 dataclass `to_dict()` 或对象写入方法完成。
- 更新 focused tests，覆盖对象返回值、属性访问和 JSON 兼容性。

**不包含（本次）：**
- 不改变 `feature_selection_manifest.json`、`feature_union_manifest.json`、
  `ic_window_*.json`、`rank_ic_window_*.json` 的外部结构。
- 不重构 `remove_duplicates_feature.py` 的 JSON 读取流程，除非测试暴露必须适配的兼容问题。
- 不重构算法内部短生命周期 dict/list。
- 不新增 JSON 反序列化公共 API，除测试或内部边界所需的轻量方法外。
- 不引入第三方数据校验库。
- 不改变 CLI 参数、文件命名、目录布局或业务计算阈值。

## 验收标准

- [ ] `run_feature_selection()` 返回 `FeatureSelectionResult`，调用方可通过对象属性访问
  `output_dir`、`manifest.stage`、`manifest.selected_feature_file` 或 valid 阶段的
  `manifest.evaluated_feature_file`。
- [ ] 写出的 `feature_selection_manifest.json` 与 `result.manifest.to_dict()` 一致，且
  train/valid 现有字段结构保持兼容。
- [ ] `write_contract_feature_union()` 返回 `FeatureUnionResult`，调用方可通过
  `result.output_dir` 和 `result.manifest.state_feature_count` 等属性访问。
- [ ] 写出的 `feature_union_manifest.json` 与 `result.manifest.to_dict()` 一致，现有字段
  结构保持兼容。
- [ ] `ic_correlation.main()` 返回 `IcCorrelationResult`，包含输出 DataFrame、输出目录、
  选中特征和 `FeatureScoreWindow` 列表。
- [ ] `rank_ic_correlation.main()` 返回 `RankIcCorrelationResult`，包含输出 DataFrame、
  输出目录、选中特征和 `FeatureScoreWindow` 列表。
- [ ] `FeatureScoreWindow.to_dict()` 写出的 `ic_window_*.json` 和 `rank_ic_window_*.json`
  仍是顶层 `{feature: score}` 映射，`remove_duplicates_feature.py` 可按现有逻辑读取。
- [ ] 现有错误场景仍失败，包括 split input 缺失、feature list 缺失或为空、非法数据、
  union 为空、union 特征缺失、输入 feather 缺失等。
- [ ] 使用 `conda activate finetf && pytest data_preprocess/tests/test_commodity_multi_contract_feature_selection.py data_preprocess/tests/test_commodity_feature_pipeline.py data_preprocess/tests/test_feature_selection_polars.py`
  通过 focused test 验证。
- [ ] 使用 `conda activate finetf && python -m py_compile data_preprocess/operator_futures/feature_selection/manifests.py data_preprocess/operator_futures/feature_selection/muti_contract/pipeline.py data_preprocess/operator_futures/feature_selection/contract_feature_union.py data_preprocess/operator_futures/feature_selection/ic_correlation.py data_preprocess/operator_futures/feature_selection/rank_ic_correlation.py`
  通过语法验证。
