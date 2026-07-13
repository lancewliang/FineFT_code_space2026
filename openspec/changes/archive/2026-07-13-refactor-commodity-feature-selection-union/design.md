# Design: refactor-commodity-feature-selection-union

## Context

商品期货多合约流程已经按 summary 合约循环生成 `ALL_FEATURE`、`IC_RESULT`、`SCALE_SAVE` 和品种级 `FEATURE_UNION`。当前问题出在顺序：每个合约的 `ic_correlation` 会立即用该合约自己的 selected state features 裁剪 `df.feather`，随后 `scale_save` 也立即基于这份合约级特征集保存。等所有合约完成后再运行的 `feature_union` 只能合并已经保存的 feature name，无法恢复各合约 `df.feather` 中被提前裁掉的列。

## Goals

- 让商品期货多合约最终共享同一份 union state feature 列表。
- 保留每个合约过滤后的 `IC_RESULT/{symbol}/{contract}/{target_freq}/{date_range}/df.feather`，供现有 `scale_save` 读取。
- 保持 `scale_save` 的职责为缩放和保存，不让它承担 union、补列或特征降级逻辑。
- 保持 crypto futures 和通用单合约 feature selection 默认输出兼容。
- 让 `fu_full_process.sh` 的阶段顺序清楚表达 candidate、union finalize、scale save 的依赖。

## Non-Goals

- 不修改 IC、Rank IC、CatBoost、Lasso 的特征重要性算法。
- 不修改商品期货特征公式、reward/execution 列定义或时间滚动特征生成逻辑。
- 不修改 `scale_save` 缩放算法和输出格式。
- 不修改 FineFT 训练算法。

## Architecture

### Candidate 阶段

`ic_correlation.py` 增加 candidate-only 模式。该模式仍读取单个合约的 `ALL_FEATURE`，仍执行现有 IC target 计算、窗口相关性、候选阈值过滤和候选内部相关性去重，但只写候选 artifact：

```text
IC_RESULT/{symbol}/{contract}/{target_freq}/{date_range}/state_features_candidate.npy
IC_RESULT/{symbol}/{contract}/{target_freq}/{date_range}/ic_window_*.json
IC_RESULT/{symbol}/{contract}/{target_freq}/{date_range}/correlation.csv
IC_RESULT/{symbol}/{contract}/{target_freq}/{date_range}/candidate_manifest.json
```

candidate-only 模式不写标准 `df.feather` 和标准 `state_features.npy`，避免后续步骤把单合约候选误认为最终 union 特征。

### Union finalize 阶段

`contract_feature_union.py` 扩展为支持从 IC candidate 读取各合约候选特征，并在同一阶段完成两件事：

1. 生成品种级 union：

```text
FEATURE_UNION/{symbol}/{target_freq}/{date_range}/state_features.npy
FEATURE_UNION/{symbol}/{target_freq}/{date_range}/feature_union_manifest.json
```

2. 逐合约回读全量 `ALL_FEATURE`，按 `reward_features + union_state_features` 写标准 `IC_RESULT`：

```text
IC_RESULT/{symbol}/{contract}/{target_freq}/{date_range}/df.feather
IC_RESULT/{symbol}/{contract}/{target_freq}/{date_range}/state_features.npy
```

每个合约标准 `state_features.npy` 与品种级 union 内容一致。

### Shell orchestration

`fu_full_process.sh` 的合约循环内只跑到 `ic_candidate`。所有合约 candidate 完成后，shell 运行一次 `ic_union_finalize`。finalize 成功后，再按 summary 合约列表执行 `scale_save`。旧的独立后置 `feature_union` 阶段不再保留，因为 finalize 已经生成品种级 union 和合约级过滤数据。

## Data Contract

Input:

```text
PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/{symbol}/main_contract_summary.json
PREPROCESS_DATASET/commodity-futures/ALL_FEATURE/{symbol}/{contract}/{target_freq}/{start_date}-{end_date}.feather
PREPROCESS_DATASET/commodity-futures/IC_RESULT/{symbol}/{contract}/{target_freq}/{start_date}-{end_date}/state_features_candidate.npy
```

Final output:

```text
PREPROCESS_DATASET/commodity-futures/FEATURE_UNION/{symbol}/{target_freq}/{start_date}-{end_date}/state_features.npy
PREPROCESS_DATASET/commodity-futures/FEATURE_UNION/{symbol}/{target_freq}/{start_date}-{end_date}/feature_union_manifest.json
PREPROCESS_DATASET/commodity-futures/IC_RESULT/{symbol}/{contract}/{target_freq}/{start_date}-{end_date}/df.feather
PREPROCESS_DATASET/commodity-futures/IC_RESULT/{symbol}/{contract}/{target_freq}/{start_date}-{end_date}/state_features.npy
PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/{symbol}/{contract}/{target_freq}/{start_date}-{end_date}/df.feather
```

## Error Handling

- Missing candidate for any summary contract fails fast.
- Empty union fails fast.
- Missing `mark_price` during candidate calculation fails fast.
- Any union state feature missing from any contract `ALL_FEATURE` fails fast and reports contract plus missing feature names.
- Existing output files may be overwritten for repeatable reruns; unrelated files are not deleted.
- `scale_save` remains fail-fast when expected standard `IC_RESULT` files are absent.

## Testing

- Candidate-only tests verify candidate artifacts are written and final artifacts are not.
- Union finalize tests verify deterministic union order, standard per-contract filtered `df.feather`, identical per-contract `state_features.npy`, and manifest counts.
- Missing-column tests verify finalize fails before writing inconsistent outputs.
- Shell tests verify `fu_full_process.sh` runs candidate before finalize, runs `scale_save` after finalize, and no longer keeps the old separate `feature_union` stage.
- Regression tests cover commodity feature pipeline, commodity main-contract CLI, and feature-selection Polars behavior.
