# Feature Audit 产物、CSV 字段与分析视图

本文按当前仓库源码说明以下三个入口的产物：

- `feature_audit.cli validate-inputs`：只校验配置，不写审计产物。
- `feature_audit.cli run`：生成一次独立多期限审计的原始证据。
- `analyze_audit_results.py`：读取一次审计，生成 `insights/` 派生分析视图；不重新计算标签或 IC。
- `feature_audit.cli compare`：对两个已完成 run 做共同样本比较。

字段定义以写文件的源码为准；配置说明和报告导航作为补充。主要依据是 [`cli.py`](../../data_preprocess/operator_futures/feature_audit/cli.py#L12-L75)、[`runner.py`](../../data_preprocess/operator_futures/feature_audit/runner.py#L328-L352)、[`runner.py`](../../data_preprocess/operator_futures/feature_audit/runner.py#L630-L691)、[`analyze_audit_results.py`](../../data_preprocess/operator_futures/feature_audit/analyze_audit_results.py#L475-L599) 和 [`comparisons.py`](../../data_preprocess/operator_futures/feature_audit/comparisons.py#L55-L155)。

## 1. 目录与期限语义

一次 `run` 的根目录通常是：

```text
analysis_result/DiHFT/feature_horizon_audit/<symbol>/<frequency>/
├── 原始审计 CSV / Parquet / JSON / Markdown
└── insights/                         # analyze_audit_results.py 的输出
```

期限含义由 [`analyze_audit_results.py`](../../data_preprocess/operator_futures/feature_audit/analyze_audit_results.py#L27-L47) 和标签生成逻辑 [`runner.py`](../../data_preprocess/operator_futures/feature_audit/runner.py#L196-L217) 定义：

| 值 | 含义 |
|---|---|
| `1h` / `2h` / `4h` | 按有效交易分钟累计的 60 / 120 / 240 分钟 |
| `today` | 当前交易日 D 收盘 |
| `next_td` | 下一交易日 D+1 收盘 |
| `next_5td` | 第 5 个交易日 D+5 收盘 |
| `next_15td` | 第 15 个交易日 D+15 收盘 |

`return` 是带符号的终点收益率；`abs_return` / `abs_target_return` 是终点收益率的绝对值，不是路径实现波动率。[`runner.py`](../../data_preprocess/operator_futures/feature_audit/runner.py#L292-L326)

## 2. `cli run` 生成的原始产物

### 2.1 文件总览

| 文件 | 是否必有 | 文件描述 |
|---|---:|---|
| `feature_metrics.csv` | 是 | 特征 × 期限 × 目标的主统计表；含合约等权 IC、样本加权 IC、覆盖率、符号一致性和 CI 诊断。 |
| `window_strata_metrics.csv` | 是 | 特征 × 期限 × 目标 × 合约 × 评价窗口的细粒度 IC。 |
| `feature_horizon_corr_matrix.csv` | 是 | 仅 `target_kind=return` 的特征 × 期限有符号 IC 宽表。 |
| `feature_horizon_coverage_matrix.csv` | 是 | 仅 `target_kind=return` 的特征 × 期限覆盖率宽表。 |
| `invalid_reasons_summary.csv` | 是 | 期限 × 标签无效原因的行数汇总；不是特征淘汰原因。 |
| `ci_diagnostics.csv` | 是 | 期限 × 目标的 Bootstrap 有效交易日、区块和 CI 状态诊断。 |
| `matched_horizon_comparisons.csv` | 条件生成 | 同一决策点同时具备 `1h`、`next_td`、`next_5td` 有效标签时的共同样本期限轨迹。 |
| `legacy_selection_comparison.csv` | 条件生成 | 配置提供旧三流 selection manifest 时的一特征一行只读选择注释。 |
| `row_labels.parquet` | 是 | 决策点 × 期限的行级价格标签；异常追查用。 |
| `audit_run_manifest.json` | 是 | 运行版本、品种、频率、配置指纹、行数和完成状态。 |
| `audit_report.md` | 是 | 面向人工阅读的审计报告和产物导航，不承载全量指标。 |
| `_SUCCESS` | 是 | 空的完成标记；不表示统计证据充分。 |

矩阵和报告由 [`report.py`](../../data_preprocess/operator_futures/feature_audit/report.py#L73-L82) 生成；其余原始表由 [`runner.py`](../../data_preprocess/operator_futures/feature_audit/runner.py#L328-L352) 和 [`runner.py`](../../data_preprocess/operator_futures/feature_audit/runner.py#L630-L861) 生成。

### 2.2 原始 CSV 的全部列

#### `feature_metrics.csv`

粒度：`feature + horizon + target_kind`，每个特征、期限、目标一行。

| 列 | 含义 |
|---|---|
| `feature` | 特征名。 |
| `horizon` | 预测期限。 |
| `target_kind` | `return` 或 `abs_return`。 |
| `legacy_stream_membership` | 旧三流归属，如 `slope_vae`、`vol_vae`、`rl_decision`，可逗号拼接；无旧清单时通常为 `unassigned`。不参与当前审计准入。 |
| `spearman_corr` | 主报告 IC；当前等同 `contract_equal_spearman`。 |
| `contract_equal_spearman` | 先按合约计算 Spearman，再对有效合约等权平均。 |
| `sample_weighted_spearman` | 按各合约有效样本数加权的 Spearman 平均，用作敏感性对照。 |
| `ci_lower`, `ci_upper` | 基准交易日移动区块 Bootstrap 的 CI 下、上界。 |
| `ci_lower_2x`, `ci_upper_2x` | 区块长度乘以敏感性倍数后的 CI 下、上界。 |
| `sign_consistency` | 各有效合约 IC 符号与主汇总符号一致的比例；合约不足时为空。 |
| `eligible_contracts_count` | 可用于合约级 IC 的有效合约数。 |
| `valid_pairs` | 当前 `horizon + target_kind` 的有效标签行数，即 `len(valid_h)`；它是标签层面的有效对数，不是逐特征重新清洗非有限值后的保证有效对数。 |
| `total_pairs` | 当前期限参与覆盖率分母的样本对数量。 |
| `coverage_ratio` | `valid_pairs / total_pairs`，标签覆盖率，范围通常为 [0,1]。 |
| `null_reason` | 主 IC 或符号一致性为空时的原因，如 `INSUFFICIENT_PAIRS`、`INSUFFICIENT_CONTRACTS_FOR_SIGN_CONSISTENCY`、`UNDEFINED`；null 不能解释成 0。 |
| `ci_status` | 基准 CI 状态；源码可能写 `AVAILABLE_EXPLORATORY`、`INSUFFICIENT_TRADING_DAYS`、`INSUFFICIENT_EFFECTIVE_BLOCKS`、`UNDEFINED_REPLICATES`，有效对少于 3 时为 `INSUFFICIENT_PAIRS`。 |
| `ci_status_2x` | 2× 区块敏感性 CI 状态，判定逻辑同上但使用敏感性区块长度/门槛。 |
| `ci_valid_trading_days` | 用于基准 CI 的有效交易日数。 |
| `ci_block_length` | 基准移动区块长度，单位为交易日。 |
| `ci_block_length_2x` | 敏感性区块长度。 |
| `ci_effective_blocks` | 基准有效区块数，源码为 `num_valid_days // block_length`。 |
| `ci_effective_blocks_2x` | 敏感性有效区块数，源码为对应有效交易日数除以 2×区块长度的整数除法结果。 |
| `ci_min_effective_blocks` | 基准允许计算 CI 所需的最小有效区块数。 |
| `ci_min_effective_blocks_2x` | 敏感性分析所需的最小有效区块数。 |
| `ci_null_reason` | 基准 CI 为空的原因。 |
| `ci_null_reason_2x` | 敏感性 CI 为空的原因。 |

这些列的实际 schema 在 [`runner.py`](../../data_preprocess/operator_futures/feature_audit/runner.py#L660-L691)；合约等权、样本加权和符号一致性的计算语义见 [`runner.py`](../../data_preprocess/operator_futures/feature_audit/runner.py#L490-L540) 及 [`CONFIG_GUIDE.md`](../../data_preprocess/operator_futures/feature_audit/CONFIG_GUIDE.md#L250-L268)。Spearman 采用平均秩；少于 3 对、常数序列或不可定义时保持 null 并记录原因，而不是写成测得的 0【`data_preprocess/operator_futures/feature_audit/metrics.py:L8-L27`】。

#### `window_strata_metrics.csv`

粒度：`feature + horizon + target_kind + contract + window_id`。

| 列 | 含义 |
|---|---|
| `feature` | 特征名。 |
| `horizon` | 预测期限。 |
| `target_kind` | `return` 或 `abs_return`。 |
| `contract` | 真实交割合约代码。 |
| `window_id` | 滚动评价窗口 ID。 |
| `is_historical_reference` | 是否是历史参考窗口；分析稳定性时应排除 `true`。 |
| `spearman_corr` | 该合约、该窗口内的 Spearman IC；不可定义时为空。 |
| `valid_pairs` | 该合约、该窗口内的有效样本对数。 |
| `null_reason` | 该窗口 IC 为空的原因，如 `INSUFFICIENT_PAIRS` 或 `UNDEFINED`。 |

#### `invalid_reasons_summary.csv`

| 列 | 含义 |
|---|---|
| `horizon` | 预测期限。 |
| `invalid_reason` | 行级标签失效原因。当前定义包括 `ZERO_LENGTH_HORIZON`、`STALE_PRICE`、`MISSING_TARGET_PRICE`、`CONTRACT_EXPIRED`、`UNVERIFIED_TIMING`、`NON_FINITE_PRICE`、`INSUFFICIENT_TAIL`，以及窗口边界净化使用的 `UNMATURED_AT_WINDOW_BOUNDARY`【`data_preprocess/operator_futures/feature_audit/contracts.py:L23-L32`】【`data_preprocess/operator_futures/feature_audit/runner.py:L221-L309`】。 |
| `count` | 该期限、该原因的无效行数。 |

失效标签来自 `row_labels.parquet`，按 [`runner.py`](../../data_preprocess/operator_futures/feature_audit/runner.py#L345-L352) 聚合；它和旧选择器的 `drop_reasons` 是两套不同证据。

#### `ci_diagnostics.csv`

粒度：`horizon + target_kind`。

| 列 | 含义 |
|---|---|
| `horizon` | 预测期限。 |
| `target_kind` | 目标类型。 |
| `num_valid_days` | 用于 CI 的有效交易日数。 |
| `block_length` | 基准区块长度。 |
| `block_length_2x` | 2× 敏感性区块长度。 |
| `effective_blocks` | 基准有效区块数。 |
| `effective_blocks_2x` | 敏感性有效区块数。 |
| `min_effective_blocks` | 基准最小有效区块门槛。 |
| `min_effective_blocks_2x` | 敏感性最小有效区块门槛。 |
| `ci_status` | 基准 CI 状态，如 `AVAILABLE_EXPLORATORY`、`INSUFFICIENT_TRADING_DAYS`、`INSUFFICIENT_EFFECTIVE_BLOCKS`、`UNDEFINED_REPLICATES` 或 `INSUFFICIENT_PAIRS`。 |
| `ci_status_2x` | 敏感性 CI 状态。 |
| `null_reason` | 基准 CI 为空的原因。 |
| `null_reason_2x` | 敏感性 CI 为空的原因。 |

schema 见 [`runner.py`](../../data_preprocess/operator_futures/feature_audit/runner.py#L642-L658)。CI 缺失不应直接解释成特征无效；先读这些诊断字段。

#### `feature_horizon_corr_matrix.csv` 与 `feature_horizon_coverage_matrix.csv`

两者都有一个固定列 `feature`，以及按本次配置实际出现的期限动态生成的列，例如 `1h`、`2h`、`4h`、`today`、`next_td`、`next_5td`、`next_15td`。前者的单元格是 `return` 的 `spearman_corr`，后者是 `return` 的 `coverage_ratio`。它们不包含 `abs_return`，也不含合约/窗口明细；写法见 [`report.py`](../../data_preprocess/operator_futures/feature_audit/report.py#L73-L82)。

#### `matched_horizon_comparisons.csv`

仅当本次标签中同时存在 `1h`、`next_td`、`next_5td` 时生成。粒度：`feature + target_kind`，共同样本固定为同一 `contract + timestamp + trading_day`。

| 列 | 含义 |
|---|---|
| `feature` | 特征名。 |
| `target_kind` | `return` 或 `abs_return`。 |
| `legacy_stream_membership` | 旧三流归属注释。 |
| `matched_samples` | 三个期限共同有效的决策点数。 |
| `sample_retention_ratio` | 共同决策点数 / 全部审计行数。 |
| `ic_common_1h` | 共同样本上的 1h IC。 |
| `ic_common_1day` | 共同样本上的 D+1 收盘 IC。 |
| `ic_common_5day` | 共同样本上的 D+5 收盘 IC。 |
| `delta_1h_to_1day` | `ic_common_1day - ic_common_1h`。 |
| `delta_1day_to_5day` | `ic_common_5day - ic_common_1day`。 |
| `delta_1h_to_5day` | `ic_common_5day - ic_common_1h`。 |
| `trajectory_category` | 规则化轨迹标签。判定顺序为：缺任一共同 IC=`INSUFFICIENT_COMMON_DATA`；三者最大绝对值 `<0.03`=`NEGLIGIBLE_SIGNAL`；同号且 `|1h|≥|1d|≥|5d|`、`|1h|≥0.05`=`CLEAN_DECAY`；同号且 `|1h|≤|1d|≤|5d|`、`|5d|≥0.05`=`CLEAN_EMERGENCE`；1d 严格大于两端且至少 0.05=`DAY_RESONANCE`；同号、三者至少 0.05 且最大/最小绝对值比 `<1.6`=`ALL_PERSISTENT`；中间值同时低于两端=`VALLEY_INTERMEDIATE`；不同号=`SIGN_FLIP_INSTABILITY`；其余=`MIXED_IRREGULAR`【`data_preprocess/operator_futures/feature_audit/runner.py:L22-L65`】。 |

当前实际列名是 `legacy_stream_membership`；部分 ADR 文字曾写成 `selected_streams`，应以 [`runner.py`](../../data_preprocess/operator_futures/feature_audit/runner.py#L807-L837) 为准。

#### `legacy_selection_comparison.csv`

仅在 `AuditConfig.legacy_selection_manifest` 存在时由 run 写出。一行一个候选特征：

| 列 | 含义 |
|---|---|
| `feature` | 特征名。 |
| `legacy_stream_membership` | 所属旧流；多流用逗号连接；`rejected` 表示有淘汰记录但未入选。 |
| `is_selected` | 是否在旧清单中入选；不完整旧清单时可为空。 |
| `selected_streams` | JSON 字符串数组。 |
| `drop_reasons` | JSON 字符串数组，每项含 `scope` 和 `reason`。 |
| `drop_reason_status` | `recorded` 或 `unknown`。 `unknown` 不是统计失败。 |
| `source_manifest` | 旧选择清单路径。 |
| `source_sha256` | 旧选择清单 SHA-256。 |

来源实现见 [`legacy_selection.py`](../../data_preprocess/operator_futures/feature_audit/legacy_selection.py#L22-L60)。

### 2.3 非 CSV 原始产物的字段/内容

`row_labels.parquet` 的实际列为：

`contract`, `timestamp`, `trading_day`, `horizon`, `window_id`, `is_historical_reference`, `start_price`, `end_price`, `target_return`, `abs_target_return`, `valid_mask`, `invalid_reason`。其中价格是同一真实合约的起止参考价，`valid_mask=true` 才能进入统计。[`runner.py`](../../data_preprocess/operator_futures/feature_audit/runner.py#L328-L343)

`audit_run_manifest.json` 当前至少写入：`schema_version`, `protocol_version`, `config_fingerprint`, `symbol`, `frequency`, `horizons`, `status`, `qualified_features_count`, `total_rows_evaluated`, `completed_at`, `matched_horizon_status`, `ci_diagnostics_status`；有旧选择清单时另有 `legacy_selection_source.path`、`legacy_selection_source.sha256`，有共同样本时另有 `matched_samples`、`sample_retention_ratio`。[`runner.py`](../../data_preprocess/operator_futures/feature_audit/runner.py#L696-L716)

`audit_report.md` 是阅读导航；`_SUCCESS` 只是空文件完成标记。报告对 `row_labels.parquet` 使用的 `label_status` 说法已落后于当前实现，实际字段是 `valid_mask` 与 `invalid_reason`。[`report.py`](../../data_preprocess/operator_futures/feature_audit/report.py#L149-L190)

## 3. `analyze_audit_results.py` 生成的 `insights/` 产物

分析脚本读取五个必需 CSV：`feature_metrics.csv`、`window_strata_metrics.csv`、两个矩阵和 `invalid_reasons_summary.csv`；`matched_horizon_comparisons.csv` 与 `audit_run_manifest.json` 可选。[`analyze_audit_results.py`](../../data_preprocess/operator_futures/feature_audit/analyze_audit_results.py#L108-L135)

### 3.1 文件总览

| 文件 | 文件描述 |
|---|---|
| `feature_evidence.csv` | 最重要的长表：原始 `feature_metrics` 加 selection 注释和 CI 可读状态。 |
| `legacy_selection_comparison.csv` | 分析阶段使用的选择注释副本；可能来自原 run、显式 retrospective manifest，或全部 `unknown`。 |
| `feature_dynamics_classification.csv` | 每个特征按 1h 与 15TD 两个端点做的探索性动态分类，并附各期限 IC/覆盖率；`description` 是源码内置解释文本，不是统计检验结论。 |
| `ci_evidence_summary.csv` | `horizon × target_kind × ci_status × ci_status_2x` 的指标行计数。 |
| `selection_dynamics_summary.csv` | `category × selection_status × drop_reason_status` 的特征计数。 |
| `top_return_<h>_features.csv` | 某期限 `target_kind=return`，按绝对 IC 降序取前 `top-k` 的展示摘录。 |
| `top_volatility_<h>_features.csv` | 某期限 `target_kind=abs_return`，按绝对 IC 降序取前 `top-k`；名称中的 volatility 不是路径波动率。 |
| `top_short_horizon_features.csv` | `top_return_1h_features.csv` 的兼容别名。 |
| `top_long_horizon_features.csv` | `top_return_next_15td_features.csv` 的兼容别名。 |
| `top_volatility_15td_features.csv` | `top_volatility_next_15td_features.csv` 的兼容别名。 |
| `analysis_manifest.json` | 分析选择来源、输入文件指纹、期限和展示阈值。 |
| `executive_summary.md` | 执行摘要和三个优先阅读入口。 |

默认七个期限会产生 14 个 `top_*_<h>_features.csv`，再按条件产生 3 个兼容别名；若 `--horizons` 只给部分期限，数量相应减少。[`analyze_audit_results.py`](../../data_preprocess/operator_futures/feature_audit/analyze_audit_results.py#L554-L599)

### 3.2 派生 CSV 的全部列

#### `feature_evidence.csv`

它保留 `feature_metrics.csv` 的 28 列，并追加 selection 证据列：

`feature`, `horizon`, `target_kind`, `legacy_stream_membership`, `spearman_corr`, `contract_equal_spearman`, `sample_weighted_spearman`, `ci_lower`, `ci_upper`, `ci_lower_2x`, `ci_upper_2x`, `sign_consistency`, `eligible_contracts_count`, `valid_pairs`, `total_pairs`, `coverage_ratio`, `null_reason`, `ci_status`, `ci_status_2x`, `ci_valid_trading_days`, `ci_block_length`, `ci_block_length_2x`, `ci_effective_blocks`, `ci_effective_blocks_2x`, `ci_min_effective_blocks`, `ci_min_effective_blocks_2x`, `ci_null_reason`, `ci_null_reason_2x`, `is_selected`, `selected_streams`, `drop_reasons`, `drop_reason_status`, `selection_status`。

若 selection 来源包含溯源信息，还会有 `source_manifest`、`source_sha256`。`ci_status` / `ci_status_2x` 会被分析脚本规范化：有合法区间时为 `AVAILABLE_EXPLORATORY`；无区间且上游没有原因时为 `MISSING_CI_REASON_UNRECORDED`；原状态为 `OK` 但区间非法时为 `INVALID_CI_ARTIFACT`。[`analyze_audit_results.py`](../../data_preprocess/operator_futures/feature_audit/analyze_audit_results.py#L138-L199)

#### `legacy_selection_comparison.csv`

基本列为：`feature`, `legacy_stream_membership`, `is_selected`, `selected_streams`, `drop_reasons`, `drop_reason_status`, `selection_status`。如果输入副本/重建结果带来源，还包括 `source_manifest`, `source_sha256`。`selection_status` 是 `selected`、`not_selected` 或 `unknown`。[`analyze_audit_results.py`](../../data_preprocess/operator_futures/feature_audit/analyze_audit_results.py#L138-L184)

#### `feature_dynamics_classification.csv`

固定列：`feature`, `category`, `description`；随后按输入矩阵实际存在的期限动态追加：

- `ic_<h>`：`feature_horizon_corr_matrix.csv` 中该期限的 `return` IC；
- `cov_<h>`：`feature_horizon_coverage_matrix.csv` 中该期限的 `return` 覆盖率。

最后附选择证据列：`legacy_stream_membership`、`is_selected`、`selected_streams`、`drop_reasons`、`drop_reason_status`、`selection_status`，以及有来源时的 `source_manifest`、`source_sha256`。`category` 只比较 `1h` 和 `next_15td` 两个端点：缺任一端点为 `INSUFFICIENT_HORIZON_DATA`；`abs(ic_1h) >= 0.15` 且超过 `abs(ic_next_15td)` 两倍为 `FAST_DECAY_MICROSTRUCTURE`；长端 `>=0.18` 且超过短端 1.8 倍为 `SLOW_EMERGING_TREND`；两端绝对值均 `>=0.10` 且同号为 `ENDPOINT_SIGN_ALIGNED`；两端都 `<0.05` 为 `NEGLIGIBLE_SIGNAL`；其余为 `INTERMEDIATE_MIXED`。它不是全期限单调性、显著性或自动选入结论【`data_preprocess/operator_futures/feature_audit/analyze_audit_results.py:L260-L306`】。

#### `ci_evidence_summary.csv`

列为：`horizon`, `target_kind`, `ci_status`, `ci_status_2x`, `metric_rows`。`metric_rows` 是指标行计数，不是独立样本数。[`analyze_audit_results.py`](../../data_preprocess/operator_futures/feature_audit/analyze_audit_results.py#L556-L564)

#### `selection_dynamics_summary.csv`

列为：`category`, `selection_status`, `drop_reason_status`, `feature_count`。它是计数交叉表，不是新的统计检验。[`analyze_audit_results.py`](../../data_preprocess/operator_futures/feature_audit/analyze_audit_results.py#L562-L565)

#### `top_return_<h>_features.csv` / `top_volatility_<h>_features.csv`

两类文件的列集合相同：

1. `feature_evidence.csv` 的列；
2. 窗口稳定性派生列：`win_mean`, `win_std`, `win_min`, `win_max`, `win_count`, `win_sign_consistency`, `win_temporal_ir`；
3. `absolute_ic`：`abs(spearman_corr)`，仅用于排序，原始 `spearman_corr` 的正负号仍保留。

实际行先排除硬编码的伪影候选 `ask1_price`、`bid1_price`、`index_price`、`UpperLimitPrice`、`LowerLimitPrice`，再要求 `sign_consistency >= --min-sign-consistency`、`coverage_ratio >= --min-coverage`，按 `absolute_ic` 降序取 `--top-k`。这些是展示筛选条件，不是上游特征选择门禁。[`analyze_audit_results.py`](../../data_preprocess/operator_futures/feature_audit/analyze_audit_results.py#L341-L365)

#### `analysis_manifest.json`

当前包含（部分字段为条件字段）：

- `selection_source`：注释来源模式；有选择证据时才有路径和 SHA-256；无证据时通常只有 `mode=unavailable`；
- `evaluation_window_count`, `contract_count`, `metric_rows`；
- `ci_status_counts`, `ci_status_2x_counts`；
- `matched_horizon_status`, `matched_horizon_path`，以及共同样本可用时的 `matched_samples`, `sample_retention_ratio`；
- `display_parameters`：`horizons`, `top_k`, `min_sign_consistency`, `min_coverage`, `ranking`；
- `source_artifacts`：上游输入文件名到 SHA-256 的映射。

写入逻辑见 [`analyze_audit_results.py`](../../data_preprocess/operator_futures/feature_audit/analyze_audit_results.py#L519-L565)。

## 4. `cli compare` 生成的产物

`compare` 要求两个目录含有 `audit_run_manifest.json`，且其中的 `protocol_version` 和 `symbol` 一致；源码并不额外检查 `_SUCCESS` 或 manifest 的 `status`。它只保留两个 `row_labels.parquet` 中都 `valid_mask=true` 且键 `contract + timestamp + trading_day + horizon` 相同的样本。[`comparisons.py`](../../data_preprocess/operator_futures/feature_audit/comparisons.py#L56-L84)

| 文件 | 描述 |
|---|---|
| `comparison_metrics.csv` | 按期限的共同样本目标收益比较。 |
| `audit_comparison_manifest.json` | 比较协议、A/B 路径、状态和共同样本数。 |
| `comparison_report.md` | 人工阅读的比较报告。 |

`comparison_metrics.csv` 的全部列：

| 列 | 含义 |
|---|---|
| `horizon` | 匹配到的预测期限。 |
| `target_kind` | 当前实现固定写为 `return`。 |
| `matched_samples` | 该期限共同有效样本数。 |
| `mean_paired_difference` | `mean(target_return_A - target_return_B)`。 |
| `target_correlation` | A 与 B 的目标收益 Spearman 相关。 |

比较 manifest 当前字段为：`schema_version`, `protocol_version`, `comparison_type`, `status`, `run_a`, `run_b`, `matched_samples_count`, `completed_at`。没有共同样本时仍写 CSV/manifest/report，但状态为 `NO_COMMON_SAMPLES`。[`comparisons.py`](../../data_preprocess/operator_futures/feature_audit/comparisons.py#L98-L155)

## 5. 如何构建适合自己的分析数据视图

### 5.1 推荐的主视图：`feature_evidence.csv`

把它当作“特征 × 期限 × 目标”的事实表；不要直接把 `top_*` 当成最终选择表。一个保守的方向收益候选视图可以是：

```python
import pandas as pd

e = pd.read_csv(".../insights/feature_evidence.csv")

directional = e[
    (e["target_kind"] == "return")
    & (e["horizon"] == "next_td")
    & e["spearman_corr"].notna()
    & (e["coverage_ratio"] >= 0.50)
    & (e["eligible_contracts_count"] >= 3)
    & (e["sign_consistency"] >= 0.80)
    & e["selection_status"].isin(["selected", "not_selected", "unknown"])
].copy()
directional["ic_abs"] = directional["spearman_corr"].abs()
```

这一步只得到“可进入复核”的视图。`sign_consistency` 或 `coverage_ratio` 为空时不要把空值当成 0 直接淘汰；先看 `null_reason` 和 `eligible_contracts_count`。默认 `top_*` 展示筛选正是 `sign_consistency >= 0.80`、`coverage_ratio >= 0.50`，但这些参数可由命令行改变。[`analyze_audit_results.py`](../../data_preprocess/operator_futures/feature_audit/analyze_audit_results.py#L93-L104)

### 5.2 用窗口表做“稳健性视图”

同一特征的全局 IC 高，不代表每个合约、每个时间窗都稳定。建议：

1. 从 `window_strata_metrics.csv` 过滤 `is_historical_reference == False`；
2. 过滤 `spearman_corr.notna()`，并按 `feature + horizon + target_kind + window_id` 查看合约分布；
3. 要求 `valid_pairs` 达到你设定的最低样本量；
4. 检查窗口 IC 是否变号、`win_min` 是否极端、以及 `win_sign_consistency` / `win_temporal_ir`；这七个汇总列已经在 `top_*` 表中提供，也可以自己从窗口表重算。

注意：分析脚本的窗口稳定性计算已经排除历史参考期，并先在窗口内对合约 IC 求均值，再跨窗口计算均值、标准差、最小值、最大值、符号一致率和 temporal IR。[`analyze_audit_results.py`](../../data_preprocess/operator_futures/feature_audit/analyze_audit_results.py#L238-L280)

### 5.3 用共同样本表做跨期限视图

如果问题是“信号随期限衰减还是涌现”，优先使用 `matched_horizon_comparisons.csv`：

```python
m = pd.read_csv(".../matched_horizon_comparisons.csv")
view = m[
    (m["target_kind"] == "return")
    & m["trajectory_category"].isin([
        "CLEAN_DECAY", "CLEAN_EMERGENCE", "ALL_PERSISTENT"
    ])
]
```

不要直接用 `feature_horizon_corr_matrix.csv` 比较期限优劣，因为各期限原生有效样本不同；共同样本表至少把 1h、D+1、D+5 放到同一决策点集合中。它仍然不能证明利润、因果或排除市场阶段效应。

### 5.4 用 `abs_return` 做幅度关联视图

若研究的是“特征与未来终点收益幅度是否相关”，筛 `target_kind == "abs_return"`，可以使用 `top_volatility_<h>_features.csv` 作为起点。但必须把它命名为“终点绝对收益关联”或“幅度关联”，不要解释成路径实现波动率；若要研究真实路径波动率，需要另建标签和产物。

### 5.5 把选择证据和统计证据分开

用 `legacy_selection_comparison.csv` 连接三流历史状态：

- `selection_status == selected`：旧清单中入选；
- `not_selected`：明确未入选；
- `unknown`：没有足够选择来源证据；
- `drop_reason_status == unknown`：没有记录明确淘汰原因，不能推断为“IC 不合格”。

因此，一个适合人工复核的宽视图通常是：`feature_evidence` 左连接 `window_strata_metrics` 的稳定性汇总，再左连接 `matched_horizon_comparisons` 的共同样本轨迹，最后保留 `legacy_selection_comparison` 的来源和原因列。统计证据、标签质量和旧选择 provenance 不要混成一个布尔字段。

### 5.6 先检查标签质量再看特征

先读 `invalid_reasons_summary.csv` 和 `ci_diagnostics.csv`：若某期限大量 `STALE_PRICE`、`MISSING_TARGET_PRICE`、`CONTRACT_EXPIRED` 或 `UNMATURED_AT_WINDOW_BOUNDARY`，该期限的覆盖率和 IC 应先解释为数据/边界问题。若 CI 是 `INSUFFICIENT_EFFECTIVE_BLOCKS`，这是证据不足状态，不等价于特征无效。`row_labels.parquet` 只在需要定位具体合约、时间、起止价格或无效原因时下钻。

## 6. 建议的最终分析视图分层

| 视图 | 主表 | 必要过滤/连接 | 适合回答的问题 |
|---|---|---|---|
| 方向候选 | `feature_evidence.csv` | `target_kind=return`、指定 `horizon`、非空 IC、覆盖率和合约数门槛 | 哪些特征与未来方向收益有关？ |
| 幅度候选 | `feature_evidence.csv` 或 `top_volatility_<h>_features.csv` | `target_kind=abs_return`；明确标注为终点幅度 | 哪些特征与未来收益幅度有关？ |
| 稳健性 | `window_strata_metrics.csv` | 排除历史参考期，分合约/窗口检查变号和样本量 | 关系是否跨时间和合约稳定？ |
| 期限动力学 | `matched_horizon_comparisons.csv` + 两个矩阵 | 优先共同样本；结合 `matched_samples` 和 `sample_retention_ratio` | 短期信号是否衰减、涌现或翻转？ |
| 选择复核 | `legacy_selection_comparison.csv` + `feature_evidence.csv` | 按流展开 `drop_reasons`，保留 `unknown` | 历史三流为什么保留/淘汰？ |
| 标签质量 | `invalid_reasons_summary.csv` + `ci_diagnostics.csv` + `row_labels.parquet` | 先看无效原因和 CI 状态，再定位行级样本 | 结论是否受数据覆盖或边界影响？ |

## 7. 运行命令

按仓库要求在 `finetf` 环境执行：

```bash
conda activate finetf
python -m data_preprocess.operator_futures.feature_audit.cli validate-inputs \
  --config data_preprocess/operator_futures/feature_audit/configs/10min/audit_fu_10min_template.yaml

python -m data_preprocess.operator_futures.feature_audit.cli run \
  --config data_preprocess/operator_futures/feature_audit/configs/10min/audit_fu_10min_template.yaml

python -m data_preprocess.operator_futures.feature_audit.analyze_audit_results \
  --audit-dir analysis_result/DiHFT/feature_horizon_audit/fu/10min \
  --config data_preprocess/operator_futures/feature_audit/configs/10min/audit_fu_10min_template.yaml

python -m data_preprocess.operator_futures.feature_audit.cli compare \
  --manifest data_preprocess/operator_futures/feature_audit/configs/comparison_fu_10min_vs_30min_template.yaml
```

`--config` 或 `--legacy-selection-manifest` 在分析阶段只提供旧选择注释来源；不会重算标签/IC，也不会自动修改三流清单。两者互斥；传 `--config` 时还会校验其中的 `symbol/frequency` 是否与审计 manifest 匹配。[`analyze_audit_results.py`](../../data_preprocess/operator_futures/feature_audit/analyze_audit_results.py#L485-L518)

## 8. 运行时注意事项

`analyze_audit_results.py` 使用 `mkdir(..., exist_ok=True)`，不会清理输出目录。缩小 `--horizons` 后，目录中上一次运行遗留的 `top_*` CSV 仍可能存在；读取时应以本次 `analysis_manifest.json` 的 `display_parameters.horizons` 为准，或使用新的空输出目录。[`analyze_audit_results.py`](../../data_preprocess/operator_futures/feature_audit/analyze_audit_results.py#L475-L479)
