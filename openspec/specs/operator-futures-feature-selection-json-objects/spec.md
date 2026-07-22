# operator-futures-feature-selection-json-objects Specification

## Purpose
Define the internal dataclass object contract for operator-futures feature selection JSON outputs while preserving their external JSON file formats.
## Requirements
### Requirement: Operator futures feature selection JSON outputs SHALL use dataclass objects internally
系统 SHALL 使用 dataclass 对象表达
`data_preprocess/operator_futures/feature_selection` 中的 feature selection manifest、
feature union manifest、IC window score 和 Rank IC window score 数据，并在 JSON 生成侧和
函数返回边界通过对象属性或对象方法访问这些数据，而不是通过裸 dict key 访问多层 JSON
payload。

#### Scenario: multi-contract feature selection manifest uses objects
- **WHEN** `run_feature_selection()` 成功写出 `feature_selection_manifest.json`
- **THEN** 函数 SHALL 返回 `FeatureSelectionResult` 对象
- **AND** 返回对象 SHALL 暴露 `output_dir` 和 `manifest` 属性
- **AND** `manifest` SHALL 是 `FeatureSelectionManifest` 对象
- **AND** train 阶段 manifest SHALL 暴露 `stage`、`selected_feature_file`、
  `selected_feature_count`、`selected_features`、`filter_results`、`contracts` 和
  `filtered_outputs` 属性
- **AND** valid 阶段 manifest SHALL 暴露 `stage`、`evaluated_feature_file`、
  `evaluated_feature_count`、`evaluated_features`、`contracts` 和 `report_only` 属性
- **AND** `contracts` 和 `filtered_outputs` 中的每个条目 SHALL 是 dataclass 记录对象，而不是
  dict

#### Scenario: feature union manifest uses objects
- **WHEN** `write_contract_feature_union()` 成功写出 `feature_union_manifest.json`
- **THEN** 函数 SHALL 返回 `FeatureUnionResult` 对象
- **AND** 返回对象 SHALL 暴露 `output_dir` 和 `manifest` 属性
- **AND** `manifest` SHALL 是 `FeatureUnionManifest` 对象
- **AND** manifest SHALL 暴露 `symbol`、`target_freq`、`start_date`、`end_date`、
  `contracts`、`contract_state_feature_paths`、`per_contract_feature_counts`、
  `state_feature_count`、`state_features`、`finalize_filtered_df`、
  `per_contract_output_paths` 和 `per_contract_output_shapes` 属性
- **AND** per-contract output shape SHALL 在对象内部以 dataclass 记录表达，而不是裸
  `dict[str, int]`

#### Scenario: IC and Rank IC window scores use objects
- **WHEN** `ic_correlation.main()` 或 `rank_ic_correlation.main()` 为每个 window 写出
  `ic_window_<window>.json` 或 `rank_ic_window_<window>.json`
- **THEN** 每个窗口分数 SHALL 先由 `FeatureScoreWindow` 对象表达
- **AND** `FeatureScoreWindow` SHALL 暴露 `window_length` 和 `scores` 属性
- **AND** `ic_correlation.main()` SHALL 返回 `IcCorrelationResult` 对象
- **AND** `rank_ic_correlation.main()` SHALL 返回 `RankIcCorrelationResult` 对象
- **AND** 结果对象 SHALL 暴露原先返回的输出 DataFrame、`output_dir`、`selected_features` 和
  `score_windows` 属性

### Requirement: Operator futures feature selection JSON serialization SHALL preserve existing output contracts
系统 SHALL 只在 JSON 写入边界将 feature selection dataclass 对象序列化为 dict，并保持现有
JSON 文件结构完全兼容。

#### Scenario: feature selection manifest JSON structure remains compatible
- **WHEN** `run_feature_selection()` 写出
  `FEATURE_SELECTION/<target_freq>/<symbol>/<stage>/feature_selection_manifest.json`
- **THEN** JSON SHALL 保持当前兼容结构
- **AND** train JSON SHALL 包含 `symbol`、`target_freq`、`stage`、`split_input_dir`、
  `selected_feature_file`、`selected_feature_count`、`selected_features`、
  `windows_list`、`composite_drop_ratio`、`aggregate_metrics_path`、`filter_results`、
  `contracts` 和 `filtered_outputs`
- **AND** valid JSON SHALL 包含 `symbol`、`target_freq`、`stage`、`split_input_dir`、
  `evaluated_feature_file`、`evaluated_feature_count`、`evaluated_features`、
  `windows_list`、`aggregate_metrics_path`、`contracts` 和 `report_only`
- **AND** 写出的 JSON payload SHALL 等于 `FeatureSelectionManifest.to_dict()` 的结果
- **AND** JSON 字段名、字段层级和数值类型 SHALL NOT 因对象化重构而改变

#### Scenario: feature union manifest JSON structure remains compatible
- **WHEN** `write_contract_feature_union()` 写出
  `FEATURE_UNION/<symbol>/<target_freq>/<start>-<end>/feature_union_manifest.json`
- **THEN** JSON SHALL 保持当前兼容结构，包含 `symbol`、`target_freq`、`start_date`、
  `end_date`、`summary_path`、`contracts`、`contract_state_feature_paths`、
  `per_contract_feature_counts`、`state_feature_count`、`state_features`、
  `candidate_source_path`、`all_feature_path`、`ic_result_path`、`finalize_filtered_df`、
  `per_contract_output_paths` 和 `per_contract_output_shapes`
- **AND** `per_contract_output_shapes` JSON SHALL 继续使用 contract 为 key、`rows` 和
  `columns` 为值字段的对象结构
- **AND** 写出的 JSON payload SHALL 等于 `FeatureUnionManifest.to_dict()` 的结果
- **AND** JSON 字段名、字段层级和数值类型 SHALL NOT 因对象化重构而改变

#### Scenario: IC and Rank IC score JSON remains top-level feature-score mapping
- **WHEN** 系统写出 `ic_window_<window>.json` 或 `rank_ic_window_<window>.json`
- **THEN** JSON SHALL 继续是顶层 `{feature_name: score}` 映射
- **AND** JSON SHALL NOT 被包裹为 `{"window_length": "<window>", "scores": "<mapping>"}` 或其他新增层级
- **AND** 写出的 JSON payload SHALL 等于 `FeatureScoreWindow.to_dict()` 的结果
- **AND** `remove_duplicates_feature.py` SHALL 可按现有 `json.load(handle).items()` 逻辑读取窗口分数

#### Scenario: dataclass object layer preserves existing failure behavior
- **WHEN** feature selection 输入、feature list、指标计算、feature union 或 JSON 写出遇到现有错误条件
- **THEN** 系统 SHALL 保持现有失败行为和异常类型
- **AND** dataclass 对象层 SHALL NOT 新增独立业务校验或吞掉底层异常
- **AND** 现有错误消息的主要关键词 SHALL 保持可用于 focused tests 匹配

### Requirement: Operator futures feature selection JSON object refactor SHALL be covered by focused tests
系统 SHALL 通过聚焦测试同时验证 feature selection 内部对象接口和外部 JSON 兼容性。

#### Scenario: focused tests assert object return types and attributes
- **WHEN** 执行 feature selection JSON 对象化相关测试
- **THEN** 测试 SHALL 断言 `run_feature_selection()` 返回 `FeatureSelectionResult`
- **AND** 测试 SHALL 断言 `write_contract_feature_union()` 返回 `FeatureUnionResult`
- **AND** 测试 SHALL 断言 `ic_correlation.main()` 返回 `IcCorrelationResult`
- **AND** 测试 SHALL 断言 `rank_ic_correlation.main()` 返回 `RankIcCorrelationResult`
- **AND** 测试中针对返回值的业务断言 SHALL 使用对象属性访问

#### Scenario: focused tests assert JSON payload compatibility
- **WHEN** focused tests 读取 `feature_selection_manifest.json`、
  `feature_union_manifest.json`、`ic_window_<window>.json` 或
  `rank_ic_window_<window>.json`
- **THEN** 测试 SHALL 断言 `json.loads(file_text)` 的结果等于对应对象 `to_dict()` 的结果
- **AND** 测试 SHALL 保留关键字段、层级、路径、行数、特征数量、窗口分数和顶层映射结构的兼容性断言
- **AND** focused verification SHALL 使用
  `conda activate finetf && pytest data_preprocess/tests/test_commodity_multi_contract_feature_selection.py data_preprocess/tests/test_commodity_feature_pipeline.py data_preprocess/tests/test_feature_selection_polars.py`
