# Design: refactor-feature-selection-json-objects

## Context

`data_preprocess/operator_futures/feature_selection` 当前有四类 JSON 输出由多个脚本直接拼装：

```text
FEATURE_SELECTION/<target_freq>/<symbol>/<stage>/feature_selection_manifest.json
FEATURE_UNION/<symbol>/<target_freq>/<start>-<end>/feature_union_manifest.json
IC_RESULT/<symbol>/<freq>/<start>-<end>/ic_window_<window>.json
IC_RESULT/<symbol>/<freq>/<start>-<end>/rank_ic_window_<window>.json
```

这些 JSON 文件的外部结构已经被后续 feature selection、scale/save、dataset 和测试流程固定。
问题在于生成侧仍在 `muti_contract/pipeline.py`、`contract_feature_union.py`、
`ic_correlation.py` 和 `rank_ic_correlation.py` 中直接拼装 dict，并让调用方通过字符串 key
访问返回 manifest。字段层级越多，维护者越难看出哪些字段属于 JSON 契约，哪些只是算法中间
映射，也更容易因拼写错误或遗漏字段改变输出结构。

## Decisions

1. 新增 `data_preprocess/operator_futures/feature_selection/manifests.py`，集中定义 feature
   selection JSON 输出对象。
2. 使用标准库 `dataclass`，不引入 pydantic 或其他运行时 schema 依赖。
3. 所有写 JSON 的流程先构造对象，再通过 `to_dict()` 或对象写入方法落盘。
4. `run_feature_selection()`、`write_contract_feature_union()`、`ic_correlation.main()` 和
   `rank_ic_correlation.main()` 返回 result dataclass，不再返回裸 manifest dict 或裸
   DataFrame。
5. IC / RankIC 窗口分数内部使用 `FeatureScoreWindow`，但 `to_dict()` 保持顶层
   `{feature: score}` 映射。
6. JSON 文件字段名、层级和主要数值类型保持完全兼容。
7. 对象层只做必要类型归一化，例如路径转字符串、NumPy 标量转 Python 数值；不新增业务规则。
8. 算法内部短生命周期 dict/list 保留，例如 `contract -> DataFrame`、`contract -> features`
   和 `filter_results`，除非这些数据成为 JSON 契约字段或返回对象字段。
9. `remove_duplicates_feature.py` 的读取流程保持不变，因为窗口分数 JSON 外部结构不变。

## Module Responsibilities

- `data_preprocess/operator_futures/feature_selection/manifests.py`: dataclass 定义、
  `to_dict()` 序列化、JSON 写入 helper 和 result 对象。
- `data_preprocess/operator_futures/feature_selection/muti_contract/pipeline.py`: 继续负责多合约
  feature selection 计算和文件输出；在 manifest 写入和返回边界使用
  `FeatureSelectionManifest` / `FeatureSelectionResult`。
- `data_preprocess/operator_futures/feature_selection/contract_feature_union.py`: 继续负责读取合约
  state features、计算 union、可选写出最终 IC_RESULT；在 manifest 写入和返回边界使用
  `FeatureUnionManifest` / `FeatureUnionResult`。
- `data_preprocess/operator_futures/feature_selection/ic_correlation.py`: 继续负责 IC 计算和
  `df.feather`、`df.csv`、`state_features.npy` 输出；窗口 JSON 使用 `FeatureScoreWindow`，
  主流程返回 `IcCorrelationResult`。
- `data_preprocess/operator_futures/feature_selection/rank_ic_correlation.py`: 继续负责 Rank IC
  计算和 `df_rank.feather`、`state_features_rank.npy` 输出；窗口 JSON 使用
  `FeatureScoreWindow`，主流程返回 `RankIcCorrelationResult`。
- `data_preprocess/tests/test_commodity_multi_contract_feature_selection.py`: 覆盖
  `FeatureSelectionResult`、manifest 属性访问和 JSON 兼容。
- `data_preprocess/tests/test_commodity_feature_pipeline.py`: 覆盖 `FeatureUnionResult`、manifest
  属性访问和 JSON 兼容。
- `data_preprocess/tests/test_feature_selection_polars.py`: 覆盖 IC / RankIC result 对象、窗口
  分数对象和窗口 JSON 顶层映射兼容。

## Data Model

多合约 feature selection manifest 保留 train/valid 两种 JSON 结构：

```python
FeatureSelectionManifest(
    symbol="fu",
    target_freq="5min",
    stage="train",
    split_input_dir="PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/5min/fu/train",
    selected_feature_file="PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/train/state_features.npy",
    selected_feature_count=2,
    selected_features=["alpha", "beta"],
    windows_list=[1, 6, 12],
    composite_drop_ratio=0.1,
    aggregate_metrics_path="PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/train/aggregate_metrics.csv",
    filter_results={"Hard Filter": ["alpha"], "Correlation Filter": ["alpha"]},
    contracts=[
        FeatureSelectionContractRecord(
            contract="fu2601",
            input_path="PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/5min/fu/train/fu2601.feather",
            metric_path="PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/train/per_contract/fu2601_metrics.csv",
        )
    ],
    filtered_outputs=[
        FilteredOutputRecord(
            contract="fu2601",
            output_path="PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/train/fu2601/df.feather",
            output_row_count=100,
            output_column_count=12,
        )
    ],
)
```

valid 阶段使用同一个 `FeatureSelectionManifest`，但设置 `evaluated_feature_file`、
`evaluated_feature_count`、`evaluated_features` 和 `report_only=True`，并不设置 train-only
字段。

feature union manifest 保留当前 JSON 字段：

```python
FeatureUnionManifest(
    symbol="fu",
    target_freq="5min",
    start_date="2026-01-01",
    end_date="2026-04-01",
    summary_path="main_contract_summary.json",
    contracts=["fu2601", "fu2605"],
    contract_state_feature_paths={
        "fu2601": "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/fu/fu2601/5min/2026-01-01-2026-04-01/state_features.npy"
    },
    per_contract_feature_counts={"fu2601": 2},
    state_feature_count=3,
    state_features=["alpha", "beta", "gamma"],
    candidate_source_path=None,
    all_feature_path="PREPROCESS_DATASET/commodity-futures/ALL_FEATURE",
    ic_result_path="PREPROCESS_DATASET/commodity-futures/IC_RESULT",
    finalize_filtered_df=False,
    per_contract_output_paths={},
    per_contract_output_shapes={
        "fu2601": ContractOutputShape(rows=2, columns=12),
    },
)
```

窗口分数对象保留顶层映射 JSON：

```python
FeatureScoreWindow(
    window_length=1,
    scores={"feature_a": 0.9, "feature_b": -0.8},
).to_dict()
# {"feature_a": 0.9, "feature_b": -0.8}
```

result 对象用于 Python 调用方，不写入 JSON：

- `FeatureSelectionResult`: `output_dir`、`manifest`
- `FeatureUnionResult`: `output_dir`、`manifest`
- `IcCorrelationResult`: `frame`、`output_dir`、`selected_features`、`score_windows`
- `RankIcCorrelationResult`: `frame`、`output_dir`、`selected_features`、`score_windows`

## Compatibility

本次是内部接口重构。以下 JSON 输出必须保持现有兼容格式：

- `feature_selection_manifest.json`
- `feature_union_manifest.json`
- `ic_window_<window>.json`
- `rank_ic_window_<window>.json`

兼容由两层测试保证：返回对象属性断言证明内部对象化生效，`json.loads(file_text)` 与
`to_dict()` 的一致性断言证明 JSON 结构未漂移。

## Failure Policy

- split input directory 缺失或没有合约 feather 时仍失败。
- valid 阶段 train feature list 缺失或为空时仍失败。
- feature selection 输入存在非法 NaN/Inf 时仍失败。
- feature union 缺少 contract state features 时仍失败。
- finalize union 时 union 为空、ALL_FEATURE 缺失或 union 特征缺失时仍失败。
- IC / RankIC 输入 feather 缺失、非法数据或下游写文件失败时仍暴露原异常。
- JSON 写入失败时仍暴露底层文件写入异常。
