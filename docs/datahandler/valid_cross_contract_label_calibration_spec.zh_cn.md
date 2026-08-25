# Valid 多合约统一 Label 定标 Spec

Triage: `ready-for-agent`

## Problem Statement

FineFT 当前对 valid 阶段的每个合约分别运行市场动态分片，并由每个合约独立拟合最终 Label 阈值；下游却会把不同合约的同名 Label 数据合并用于 VAE 跨合约训练。这种“逐合约定标、跨合约合并”的组合没有共同口径：相同 Label 编号来自不同阈值，不能确认它们属于同一套划分规则。

现有百分比斜率已经消除了不同绝对价格单位的影响，因此问题不是重新缩放 State Feature，而是把最终 Label 阈值的拟合范围从单个合约提升到整个 valid 集合。同时，Label 只应作为无语义动态 Label，不能承载方向、幅度或涨跌停含义，也不能驱动 Label 方向语义相关的 Agent 选择、路由或动作约束。

## Solution

将 Valid 动态切片改为 valid 数据集级原子构建。所有合约仍独立执行既有 Butterworth、turning-point 和 slice-and-merge；合并过程继续使用合约内临时 quantile Label。所有未跳过合约完成分片后，系统汇总每个最终市场动态片段的带符号百分比斜率，以每个片段一票的方式，在 `i / dynamic_number` 全局 Segment Quantile 位置拟合一套全合约共享阈值，再把同一套阈值应用到所有合约。

系统不平衡合约或 Label 样本比例，不按片段长度或合约重新加权，也不按各合约波动率再次标准化。涨跌停及接近涨跌停行情与普通行情共同进入既有 `dynamic_number` 个无语义动态 Label。构建通过临时位置完成并原子发布；可跳过行数不足的合约，其他数据或分片错误使整个构建失败并保留上一代完整产物。

### Volatility Labeling Extension

生产构建同时支持 `labeling_method="volatility"`。该模式保留与 slope 模式相同的合约内 turning-point 与 slice-and-merge 边界，但将每个最终片段的 score 定义为片段内 `bid1_price` 对数收益率的非年化总体标准差（`ddof=0`），并乘以 100 表示为百分比。所有合约的 volatility score 以每个片段一票的方式，在 `i / dynamic_number` 全局 Segment Quantile 位置拟合共享阈值。

两种方法原子发布到独立目录 `valid/slope/` 与 `valid/volatility/`，各自包含 processed 文件、合约 Label 目录和 `slice_manifest.json`；重建一种方法不得删除或覆盖另一种方法。volatility 模式的 Label 按 segment volatility 从低到高排序。`labeling_method="quantile"` 与 `DTW` 仍不是生产 final labeling method。

## User Stories

1. As a FineFT 数据工程师, I want all valid contracts to share one final slope threshold set, so that identically numbered Labels use one calibration rule.
2. As a FineFT 数据工程师, I want single-contract and multi-contract valid datasets to execute the same build rule, so that contract count does not introduce a hidden compatibility branch.
3. As a Stage II 研究人员, I want each contract to retain independent turning-point and slice-and-merge processing, so that market dynamic segments never cross contract boundaries.
4. As a Stage II 研究人员, I want contract-local temporary quantile Labels to remain internal to each merge round, so that the existing merge constraint behavior is preserved.
5. As a Stage II 研究人员, I want only final Labels to use Cross-contract Label Calibration, so that temporary merge mechanics and published Label identity remain distinct.
6. As a quantitative researcher, I want segment scores expressed as signed percentage slope relative to segment start price, so that contracts with different price units remain comparable.
7. As a quantitative researcher, I want the percentage-slope sign preserved, so that the implementation does not accidentally replace the existing score with an absolute-value return.
8. As a quantitative researcher, I want no per-contract volatility normalization, so that a given percentage move has the same score across contracts.
9. As a quantitative researcher, I want every final market dynamic segment to contribute one equally weighted score, so that the calibration preserves the current segment-level statistical unit.
10. As a quantitative researcher, I want no segment-length weighting, so that long segments do not receive more threshold influence than short segments.
11. As a quantitative researcher, I want no contract-level reweighting, so that contracts with more final segments naturally contribute more observations.
12. As a quantitative researcher, I want final shared thresholds fitted at `i / dynamic_number` quantiles of all final segment scores, so that each final Label receives a comparable number of pooled segments when slope values are distinct.
13. As a quantitative researcher, I want the final threshold method recorded separately from the slope score method, so that global Segment Quantile calibration is not confused with contract-local temporary quantile Labels.
14. As a pipeline operator, I want production Cross-contract Label Calibration to accept the complete valid directory, so that the system can discover all calibration inputs before publishing outputs.
15. As a pipeline operator, I want single-contract execution to remain diagnostic-only, so that it cannot partially overwrite official Label outputs or the Slice Manifest.
16. As a pipeline operator, I want a changed valid contract set to trigger a complete rebuild, so that one published generation never mixes old and new thresholds.
17. As a pipeline operator, I want the build published atomically, so that downstream jobs observe either the previous complete generation or the new complete generation.
18. As a pipeline operator, I want contracts with insufficient filtering rows recorded as Skipped Contracts, so that valid data from other contracts can still be calibrated.
19. As a pipeline operator, I want Skipped Contracts excluded from threshold fitting, so that absent segment scores cannot distort calibration accounting.
20. As a pipeline operator, I want missing columns, non-finite values, and segmentation failures to abort the build, so that structurally invalid input cannot produce a partial official dataset.
21. As a pipeline operator, I want a zero-final-segment build to fail, so that the system never publishes thresholds without calibration observations.
22. As a pipeline operator, I want a one-to-four-segment build to use the existing small-sample fallback, so that small valid datasets remain supported.
23. As a VAE trainer, I want a contract to be allowed to have no data for some Labels, so that shared thresholds do not force artificial per-contract proportions.
24. As a VAE trainer, I want empty contract Labels omitted from NumPy array generation, so that downstream training never receives empty training arrays.
25. As a manifest consumer, I want every contract to list all configured Labels, including zero-file and zero-row entries, so that a legal empty Label is distinguishable from a missing pipeline result.
26. As a manifest consumer, I want the Slice Manifest to record participating contracts, skipped contracts, total final segments, shared thresholds, dynamic count, labeling method, and slope statistics, so that calibration is auditable and reproducible.
27. As a manifest consumer, I want contract and Label aggregate row/file counts rebuilt from the same published generation, so that both manifest views reconcile with disk outputs.
28. As a Stage II researcher, I want Labels treated as opaque identifiers, so that no trading direction, magnitude, or price-limit meaning is inferred from a Label number.
29. As an Agent evaluator, I want Label-direction-dependent selection removed, so that Agent performance is evaluated without assuming semantics that the Label contract does not provide.
30. As a routing developer, I want routing and action constraints to use observed market, performance, and risk state rather than Label direction, so that arbitrary Label numbering cannot impose a trading direction.
31. As a market-data researcher, I want limit-price and near-limit-price rows retained in ordinary slicing, so that these valid market observations are not discarded.
32. As a market-data researcher, I want no dedicated limit-up or limit-down Labels, so that output cardinality remains exactly `dynamic_number`.
33. As an RL researcher, I want sliced Feather State Features to remain byte-equivalent in value to upstream Scale Save output, so that Label calibration does not create a second model-input scaler.
34. As an RL researcher, I want the upstream train-only RobustScaler unchanged, so that existing train/valid/test feature-scale guarantees remain intact.
35. As a VAE trainer, I want existing cross-contract same-Label materialization to consume all non-empty contract arrays, so that shared final thresholds support the current training layout.
36. As a pipeline operator, I want downstream rebuilds to remain an orchestration responsibility, so that this change does not introduce calibration-ID compatibility gates.
37. As a maintainer, I want `labeling_method="quantile"` and final `DTW` Label requests rejected by the production directory build, so that changing the shared slope threshold method is not confused with changing the final segment score or clustering method.
38. As a maintainer, I want contract-local temporary `quantile` retained only for the merge constraint and global Segment Quantile used only for final shared slope thresholds, so that the two calibration scopes remain distinct.
39. As a maintainer, I want stale contract and Label outputs removed only during successful atomic publication, so that deleted inputs do not leave misleading artifacts and failed builds do not destroy the previous generation.
40. As a reviewer, I want all input rows from non-skipped contracts accounted for exactly once across output segments, so that no Label boundary drops or duplicates market data.
41. As a quantitative researcher, I want an optional volatility segment score based on non-annualized population standard deviation of log returns, so that the same turning-point slices can be grouped by realized variability.
42. As a quantitative researcher, I want volatility scores to remain invariant under multiplicative price-unit changes, so that differently priced contracts share one calibration scale.
43. As a pipeline operator, I want slope and volatility outputs stored under method-specific directories, so that both generations can coexist in one dataset.
44. As a VAE data operator, I want to select which method-specific valid directory to consume, so that downstream arrays are built from the intended Label definition.

## Implementation Decisions

- The production unit is the complete valid dataset directory, not an individual contract file.
- The production workflow is two-phase: first discover final market dynamic segments independently for every contract; then fit and apply one shared final Label calibration.
- Butterworth filtering, turning-point detection, minimum-length handling, DTW-based neighbor distance, merge rounds, and the contract-local temporary quantile merge constraint retain their current behavior.
- Temporary merge Labels are not persisted, are not written to the Slice Manifest, and are never reused as final Labels.
- In slope mode, the final segment score remains the signed percentage slope: segment price slope divided by segment start price and multiplied by 100.
- In volatility mode, the final segment score is `100 * std(diff(log(bid1_price)), ddof=0)` within each existing final segment; it is not annualized.
- Both score definitions are price-unit invariant. Per-contract volatility, IQR, MAD, standard-deviation, or rank normalization is not added.
- Each final segment contributes one score with weight one. Segment length, row count, and contract identity do not change calibration weight.
- Production final labeling supports `slope` and `volatility`. `labeling_method="quantile"` and final `DTW` modes fail before official outputs are mutated.
- Production shared thresholds support an explicit `global_segment_quantile` threshold method using `i / dynamic_number` pooled-segment quantiles. Every final segment has weight one; row count and contract identity do not change the quantiles.
- The legacy equal-width slope threshold method remains available for pipelines that have not explicitly migrated.
- Volatility labeling defaults to `global_segment_quantile`; slope labeling retains the legacy equal-width default unless explicitly configured otherwise.
- The small-sample fallback remains valid when at least one final segment exists. A build with no final segments fails.
- The same algorithm applies when the valid set contains one contract or many; no single-contract production special case is introduced.
- The official build discovers all contract-level Valid Feature files before processing and deterministically identifies each contract from the dataset contract record or file identity.
- Contracts that cannot satisfy filtering because of insufficient rows become Skipped Contracts and are excluded from calibration.
- Missing required columns, non-finite required inputs, merge/label exceptions, invalid final labels, and output-accounting mismatches abort the complete build.
- Official outputs are first written beneath a staging location. The method-specific Slice Manifest and all contract Label directories are published together only after validation succeeds.
- Published generations are isolated under `valid/<labeling_method>/`; publishing one method leaves other method directories unchanged.
- A successful build replaces the complete previous Label generation, including stale contract directories. A failed build leaves the previous generation unchanged.
- Every configured Label appears in each non-skipped contract manifest record. Contract-empty Labels use `file_count=0`, `total_row_count=0`, and an empty file list.
- Aggregate Label records also include all configured Labels, including zero-count Labels when no contract contains data for them.
- Empty Label directories may exist for layout compatibility, but empty Feather or NumPy files must not be created.
- Slice Manifest calibration data records fit scope, participating and skipped contracts, final segment count, `dynamic_number`, `labeling_method`, segmentation and score methods, threshold method, threshold weighting, quantile levels, shared thresholds, and descriptive statistics for the pooled segment scores.
- Slice Manifest output accounting must reconcile input rows, contract output rows, Label output rows, and segment file rows.
- Limit-price and near-limit-price observations remain ordinary input rows. The build creates exactly `dynamic_number` final Labels and no dedicated directional or price-limit Labels.
- Unsemantic Dynamic Labels carry no trading-control contract. Agent selection, routing, and action guards must not infer direction or allowed actions from Label number.
- PnL-, market-state-, execution-, and risk-based controls remain available; only Label-direction-derived constraints are invalidated.
- Sliced Feather State Feature values are copied from the existing dataset inputs without a new normalization pass.
- The upstream train-only Scale Save implementation and Scale Manifest are unchanged.
- VAE data creation continues to materialize non-empty arrays by contract and Label. It skips Contract-empty Labels without treating the contract as failed.
- Cross-contract VAE materialization may continue to concatenate available arrays with the same final Label.
- No calibration identifier is required in VAE Training Manifest, Selection Manifest, Potential Model metadata, or runtime loading.
- Commodity data-handler orchestration invokes the directory-level production build once instead of looping over contract files.
- A single-contract diagnostic interface may remain, but it cannot update official Label directories or the official Slice Manifest.

## Testing Decisions

- The primary test seam is the directory-level production build. Tests should invoke the same interface used by commodity data-handler orchestration and assert only published files, manifest data, exit status, and preservation/replacement behavior.
- One integration fixture should contain at least two contracts with deliberately different price levels and slope distributions. It should prove that the manifest contains one threshold set and that every contract is labeled using that set.
- A multiplicative price-scale scenario should run equivalent paths whose prices differ by a constant factor and assert identical segment boundaries, percentage slopes within tolerance, and final Labels.
- A volatility scenario should verify non-negative, multiplicative-price-scale-invariant segment scores, global Segment Quantile thresholds, and method-specific publication coexistence.
- A row-conservation scenario should assert that every row from each non-skipped input appears exactly once across its output segment files, including limit-price and near-limit-price rows.
- A Contract-empty Label scenario should assert explicit zero-count manifest records and the absence of empty downstream arrays.
- A Skipped Contract scenario should include a contract shorter than the filtering requirement and assert that it is excluded from calibration while other contracts publish successfully.
- An atomic-failure scenario should start with an existing valid generation, inject a malformed contract, assert a non-zero build result, and verify that the previous manifest and Label outputs remain unchanged.
- A stale-output scenario should remove a contract from the valid input set, run a successful build, and assert that its previous Label directory is absent from the new generation.
- A single-contract scenario and an equivalent one-member valid directory scenario should follow the same production rule and produce the same segment boundaries and Labels.
- A limit-price scenario should assert that no extra Label indices are created and that all limit-related rows remain in the ordinary output set.
- A State Feature preservation scenario should compare every selected State Feature value before and after slicing and assert exact equality, while allowing Label/output metadata to differ.
- A manifest-accounting scenario should reconcile contract totals, Label totals, file counts, row counts, empty Labels, and skipped contracts against actual outputs.
- Focused calibration tests should cover global Segment Quantile thresholds, equal segment weighting, duplicate thresholds, shared cross-contract application, and small segment pools.
- Unsupported `labeling_method="quantile"` and final `DTW` production requests should fail before staging publication.
- Good tests should verify externally meaningful behavior rather than implementation calls, private helper structure, or exact internal object decomposition.
- Existing Valid Dynamic Slice tests provide prior art for contract-scoped outputs, row preservation, final-segment writing, insufficient-row skipping, and small slope segment counts.
- Existing commodity dataset tests provide prior art for directory layout, production-script wiring, manifest accounting, and State Feature copying.
- Existing VAE cross-contract tests provide prior art for missing contract Labels, non-empty array discovery, feature-dimension validation, and merged training manifests.
- Test commands must run in the `finetf` conda environment. The focused suite should cover datahandler slice, commodity contract dataset, and VAE cross-contract behavior before the broader repository suite.

## Out of Scope

- Replacing Butterworth filtering, turning-point detection, slice-and-merge, DTW neighbor distance, or merge-round behavior.
- Row-weighted, contract-weighted, or contract-local final Label quantiles.
- Forcing equal Label proportions within each contract.
- Per-contract volatility, IQR, MAD, standard-deviation, or rank normalization.
- Final production support for `labeling_method="quantile"` or `DTW` labeling.
- Re-scaling State Features in the sliced Feather outputs.
- Changing Feature Selection, Scale Save, Scale Manifest, or train/valid/test split behavior.
- Adding a calibration ID or enforcing compatibility across VAE, Selection Manifest, Potential Model, or runtime artifacts.
- Redesigning VAE architecture, OOD scoring, Meta Router scoring, PnL memory, or circuit-breaker behavior.
- Guaranteeing that every contract or every Label contains data.
- Assigning direction, price-limit, or cross-method semantics to Label numbers. Volatility mode only guarantees its documented low-to-high volatility ordering within that method's calibration.
- Publishing or updating a GitHub Issue; this Spec is stored locally at the user's request.

## Further Notes

- ADR-0008 establishes that Dynamic Labels are semantic-free identifiers and supersedes the earlier Label-direction semantic-guard decision.
- ADR-0009 establishes shared cross-contract slope thresholds, contract-local temporary quantile Labels, equal segment weighting, atomic valid-dataset rebuilds, manifest audit requirements, and the absence of downstream calibration-ID enforcement.
- ADR-0011 extends production final labeling with method-isolated volatility outputs and supersedes ADR-0009/0010 only where they restricted production final scores to slope.
- Existing datasets containing dedicated limit-up or limit-down Label directories are stale relative to this Spec and must be fully rebuilt before evaluation.
- The highest-value regression signal is the directory-level production build because it covers contract discovery, per-contract segmentation, pooled calibration, output publication, Slice Manifest accounting, and downstream layout in one seam.
