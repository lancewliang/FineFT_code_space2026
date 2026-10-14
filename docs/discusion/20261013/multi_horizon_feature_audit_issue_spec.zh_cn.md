## Problem Statement

研究者希望知道原生 10min、30min 特征对未来 1 小时、2 小时、4 小时、Today、Next TradingDay、Next 5 TradingDays、Next 15 TradingDays 的预测关联，同时保持已有三流特征选择、训练输入及历史实验结果不变。

现有短期限标签主要按行情行数偏移，6 根 Bar 在不同频率下代表不同时间跨度；休市、夜盘、缺失 Bar 和部分 Bar 会进一步改变它的含义。只评价已经入选的特征，也无法发现被原短期门禁排除但可能具有长期关联的候选。

长期标签还会触及换月、到期、缺失行情、评价窗口边界及最终 Test。大量重叠 Bar 与同日相关合约容易造成虚假独立样本量；历史数据的交易日、价格时点和特征可用时刻也需要核验。研究者需要一份能够解释这些边界、覆盖率和不确定性的完整审计，而不是另一个会自动改写选择结果的筛选器。

## Solution

交付独立的多预测期限特征审计能力：只读消费 Stage 6 全部审计合格候选，分别处理原生 10min 和 30min，在统一交易日历下构造七期限的同合约方向收益与终点绝对收益标签，输出可复现的完整指标、热图、覆盖率、无效原因和配对比较。

小时期限按有效交易时间累计；日级终点为当前所属 TradingDay 的 D、D+1、D+5、D+15 日盘收盘。原三流入选状态和剔除证据仅用于解释。首期以 fu 两个频率完成真实数据验收，缺少的数据通过独立准备流程补齐。审计不会改变原选择器，也不是原流水线继续运行的前置条件。

## User Stories

1. As a quantitative researcher, I want to audit native 10min and 30min features, so that I can understand both data frequencies without changing existing feature definitions.
2. As a quantitative researcher, I want seven explicit prediction horizons, so that short-term and multi-day associations can be compared consistently.
3. As a quantitative researcher, I want hourly horizons measured in active trading time, so that lunch breaks, overnight closures and holidays do not silently change their meaning.
4. As a quantitative researcher, I want Today tied to the exchange TradingDay, so that night-session observations are assigned to the correct business day.
5. As a quantitative researcher, I want NextTD, Next5TD and Next15TD to end at D+1, D+5 and D+15 closes, so that counting conventions remain unambiguous.
6. As a quantitative researcher, I want every horizon return to begin at the current decision time, so that cumulative future outcomes share a clear starting point.
7. As a quantitative researcher, I want each label to retain one actual contract, so that rolling between contracts cannot create artificial predictive returns.
8. As a quantitative researcher, I want signed and absolute terminal returns reported separately, so that directional association is distinguished from the magnitude of the terminal move.
9. As a quantitative researcher, I want all audit-eligible candidates evaluated, so that potentially useful omitted features remain visible.
10. As a quantitative researcher, I want legacy selection and rejection evidence attached to candidates, so that findings can be understood in the context of the existing three streams.
11. As a quantitative researcher, I want old statistical gates and stream-specific blacklists excluded from audit admission, so that the audit can examine their blind spots.
12. As a data engineer, I want identity columns, future labels, duplicate merge columns and execution fields excluded from candidate discovery, so that numerical metadata is not mistaken for a feature.
13. As a data engineer, I want feature availability verified against source evidence, so that future information cannot enter an earlier decision.
14. As a data engineer, I want source timestamps, timezones and TradingDay mappings validated, so that filenames and naive calendar dates cannot silently define labels.
15. As a data engineer, I want partial bars and observations available during closures handled conservatively, so that features are never backdated into the past.
16. As a quantitative researcher, I want zero-length Today labels marked invalid, so that a completed trading day does not create artificial zero-return observations.
17. As a quantitative researcher, I want reference prices taken from verified fine-grained snapshots, so that exact endpoints are not replaced by the next coarse bar.
18. As a quantitative researcher, I want one declared reference-price rule and freshness limit, so that results are not mixed across undisclosed price conventions.
19. As a data engineer, I want quote conflicts, stale prices and missing endpoints explained, so that excluded labels can be traced to their causes.
20. As a quantitative researcher, I want missing long-horizon observations and contract expiry reflected in coverage, so that missing evidence is not interpreted as zero predictive association.
21. As a quantitative researcher, I want each horizon evaluated on its own valid samples, so that short horizons do not unnecessarily inherit every long-horizon exclusion.
22. As a quantitative researcher, I want matched valid samples for horizon comparisons, so that differences in data coverage are separated from differences in association.
23. As a quantitative researcher, I want cross-frequency comparisons aligned to the same contracts and decision times, so that the comparison uses genuinely shared observations.
24. As a quantitative researcher, I want native lookback differences disclosed, so that same-named features are not incorrectly treated as equal-duration experiments.
25. As a quantitative researcher, I want chronological development windows, so that the stability of associations across time is visible.
26. As a quantitative researcher, I want labels purged by their actual maturity times, so that training or evaluation boundaries cannot borrow future outcomes.
27. As a research owner, I want final Test data sealed during development, so that long labels cannot consume it as a convenient extension of the development period.
28. As a quantitative researcher, I want standard Spearman handling of tied ranks and valid-pair masks, so that statistics remain numerically meaningful.
29. As a quantitative researcher, I want undefined statistics represented explicitly, so that constants or insufficient samples are not presented as measured zero correlations.
30. As a quantitative researcher, I want equal-contract summaries and sample-weighted sensitivity results, so that large contracts do not silently dominate the main result.
31. As a quantitative researcher, I want joint TradingDay block uncertainty estimates, so that overlapping labels and same-day contract dependence are respected.
32. As a quantitative researcher, I want shared resampling for matched comparisons, so that uncertainty in paired differences is calculated coherently.
33. As a quantitative researcher, I want block-length sensitivity and insufficient-evidence statuses, so that fragile confidence intervals are not mistaken for reliable conclusions.
34. As a quantitative researcher, I want all attempted candidates and horizons retained, so that selective presentation cannot conceal the scale of the search.
35. As a quantitative researcher, I want complete feature-by-horizon tables and heatmaps, so that I can inspect patterns without a forced cross-horizon score.
36. As a quantitative researcher, I want Today results stratified by remaining active trading time, so that near-close observations are distinguished from long remaining sessions.
37. As a quantitative researcher, I want positive and negative associations preserved, so that potentially meaningful directionality is not hidden by absolute-value ranking.
38. As a research owner, I want manifests linking inputs, calendars, protocols and output versions, so that an audit can be reproduced and compared responsibly.
39. As a pipeline maintainer, I want an independent audit entry point, so that audits can run without training models or invoking the old selector.
40. As a pipeline maintainer, I want source files and legacy artifacts unchanged after an audit, so that exploratory research cannot alter production research baselines.
41. As a data engineer, I want missing 30min inputs prepared in isolation, so that creating audit inputs does not overwrite shared summaries or trigger selection and scaling.
42. As a pipeline maintainer, I want invalid contracts to fail clearly and local data gaps to remain traceable, so that failures are neither silently ignored nor reported as successful empty runs.
43. As a research owner, I want both frequencies tested on real fu data, so that claimed support is backed by actual integration evidence.
44. As a pipeline maintainer, I want measured runtime and memory usage with bounded data loading, so that the audit can be operated within known resource limits.
45. As an implementing agent, I want one main behavioral test boundary and explicit acceptance criteria, so that I can implement the feature without coupling tests to private internals.

## Implementation Decisions

1. **Independent audit boundary.** Introduce one externally visible audit application with input validation, audit execution and comparison operations. Internally separate input adaptation, candidate qualification, calendar/session resolution, reference-price lookup, target/label construction, maturity guards, statistics and reporting. Prefer existing verified pure functions and read-only interfaces. Do not initialize legacy orchestration that writes outputs. No change to original three-stream policies, lists, order, scaling, checkpoints or model inputs.

2. **Separate data preparation.** Audit execution consumes already prepared inputs. Missing 30min Stage 1–6 inputs are prepared through a separate, explicitly isolated operation. Do not invoke the existing full pipeline, which proceeds unconditionally to selection and scaling, and do not overwrite shared summaries. Record any upstream defect requiring separate repair instead of silently rewriting historical 10min data.

3. **Calendar and identity contracts.** Require a versioned, instrument-specific historical exchange calendar with TradingDay, session boundaries and exceptions. Validate raw InstrumentID, TradingDay and actual event timestamps; filenames and observed dates do not define the calendar. Use an explicit Asia/Shanghai timezone. Static session schedules alone are insufficient. Require the same real contract at both ends of a label, even when main-contract identity changes.

4. **Horizon semantics.** The hourly horizons accumulate exactly 60, 120 and 240 active trading minutes, skipping closures and permitting session/day crossings. Missing observations do not stop the clock. Today ends at D day-session close; NextTD, Next5TD and Next15TD end at D+1, D+5 and D+15 closes. All begin at the decision time. Ending exactly on a session close must not advance another session. The historical six-row return, if displayed, is separately named legacy_6bar and is not an eighth primary horizon or a substitute for 1h.

5. **Candidate qualification.** Evaluate all causally available numerical feature candidates, excluding identifiers, execution/reward fields, future labels, duplicate merge columns and nonnumerical fields. Preserve exclusion reasons, mandatory-context status, original selection membership and evidenced legacy rejection reasons; unknown evidence remains unknown. Do not reuse old IC, PSI, directional or stream-blacklist admission criteria. Causal restrictions still apply even when recorded in an old blacklist. Constants and non-finite observations receive explicit validity masks and statuses.

6. **Native-frequency semantics.** Keep original integer lookback windows at both frequencies. Record their actual historical coverage and transformation definitions; nominal minutes are auxiliary information. Same names do not establish equal physical lookbacks. Cross-frequency comparisons require compatible feature meaning and exact common contract/decision-time/evaluation-date keys, without nearest-neighbor matching or interpolation.

7. **Decision-time availability.** Only admit features with availability supported by generation rules and data-version evidence. The earliest allowed decision is no earlier than both verified availability and the conservative source-bar label. Availability during a closure moves forward to the next open. When multiple source bars map to the same decision key, retain the latest already-available source snapshot and preserve provenance; an unresolved ordering conflict rejects that key. Apply column-level masks when features have different availability. Do not infer early availability solely from the last observed quote.

8. **Reference-price protocol.** At both start and target, use the latest valid snapshot at or before the intended time, from the same contract and session, carrying a finite positive LastPrice and no more than 60 seconds of snapshot age. No future lookup, interpolation, automatic midprice fallback or cross-session carry. Record separate start/end observation times and ages. Snapshot freshness does not establish last-trade freshness or executable fillability. Conflicting same-time quotes without a reliable sequence are invalid.

9. **Labels and maturity.** Signed terminal return is the endpoint reference price divided by the start reference price, minus one; absolute terminal return is its absolute value. Record target endpoint separately from price observation time and actual availability evidence. Maturity cannot precede the intended endpoint, even if the as-of price was observed earlier. Missing prices, expiry, insufficient tail data, stale quotes and unverified timing create reasoned invalid labels, never zeros, shorter horizons or another contract's prices. Today with no remaining future interval is ZERO_LENGTH_HORIZON; other horizons remain independently eligible.

10. **Evaluation schedule.** Use the first six calendar months as initial historical reference, followed by chronological three-calendar-month evaluation windows. Anchor offsets at the earliest allowed development TradingDay, align boundaries forward to the appropriate TradingDay session start, retain a short final window and split further at the original Train/Valid boundary. Persist one window table shared by both frequencies. Historical reference rows are excluded from evaluation summaries. Any fitted bins or interpretive rules use only already-mature historical samples and declare fit scope.

11. **Boundary purification and Test sealing.** Use the same business-time boundaries across contracts and frequencies, including preceding-calendar-date night sessions. History used for a window must mature strictly before its start; labels reported within that window must mature strictly before its end. Every development label must mature strictly before final Test starts. Do not read final Test prices to complete development labels. Existing legacy membership uses information from historical selection and is retrospective context, not proof of independently selected out-of-sample features.

12. **Metric definition.** Calculate signed and absolute-return Spearman correlations on each feature/target group's actual valid pairs with average ranks for ties. Recompute ranks after masking. Fewer than three pairs, constant feature/target or undefined denominators produce null with reasons; three pairs is only a mathematical calculation floor. Keep statistics by frequency, feature, horizon, target kind, contract and evaluation window. For primary summaries, pool only legitimate evaluation rows within each contract, recompute its statistic, then average eligible contract statistics equally. Also report sample-weighted sensitivity and the full time-window series. At least three eligible contracts are required for sign-consistency summaries.

13. **Sample-comparison contracts.** Output each horizon's native valid-sample metrics and denominators. Also calculate pairwise matched-horizon and matched-frequency comparisons using genuinely common valid observations; do not force all seven horizons into the 15TD intersection. Preserve positive/negative signs. If there is no comparable sample, report that explicitly. Matched comparisons still do not isolate frequency as the sole causal difference when native lookback durations differ.

14. **Uncertainty protocol.** Use joint moving blocks on the full TradingDay calendar, preserving within-day contract/frequency/horizon dependence and missing masks. Recompute the Bar-level target statistic for each resample; matched differences use paired resampling. Defaults are 2,000 replicates, seed 42, pointwise 95% percentile intervals, and block length equal to the greater of 20 TradingDays or the largest relevant label's TradingDay span. Report twice-length sensitivity. Do not compress missing dates into artificial continuity.

15. **Evidence sufficiency.** Report both nominal coverage and unique TradingDays with valid observations. The floor of valid-day count divided by block length is a conservative operational block-count diagnostic, not an estimate of truly independent observations. Below 20 such blocks, show descriptive statistics and INSUFFICIENT_DATA instead of a confidence or stability-pass claim. Record undefined bootstrap replicates; if the required valid statistics cannot be produced, output null with reasons without silently shortening blocks or changing methods. Apply this separately to the base and doubled-length analysis. Undefined variation ratios remain UNDEFINED.

16. **Exploratory interpretation.** Retain all candidates and attempts. The pointwise intervals are not adjusted for multiple comparisons and must be described as exploratory; do not claim automatic statistical significance or produce pass lists. Distinguish a CI spanning zero, insufficient samples and poor quote coverage. No cross-horizon composite score, automatic Top-N list, mandatory deletion or automatic feature promotion. A complete table may be sorted by a requested metric while retaining its exploratory context.

17. **Output contracts.** Produce independently versioned run and comparison manifests; candidate catalogs; row-level label provenance; complete feature/horizon metrics; contract, window and remaining-time strata; invalid-reason summaries; legacy-selection comparisons; matched comparisons; heatmaps; and a Markdown report. Preserve row keys, source-bar labels, event/availability/decision times, TradingDay, prices, quote ages, maturity, split/window, status and reason. Where availability differs by feature, preserve an associated feature/group availability table. Include raw and effective denominators, valid rows/dates/contracts and null reasons.

18. **Reproducibility and publication.** Record schema/protocol versions, code/config/input/calendar fingerprints, legacy list source and ordering, exact read boundaries, exclusions, random seed, numerical method, hardware and measured resources. New protocol values create new versions, not overwritten runs. Comparison manifests link compatible completed runs. Publish audit outputs only after contract checks, with a distinct completion marker; failures retain diagnostics without posing as completed reports. Reject output destinations that can overwrite inputs or legacy artifacts.

19. **Error and resource handling.** Missing or conflicting global calendar, identity or split contracts block a run before statistics. Feature-level causal failures and local missing prices are traceable exclusions with denominators; a result with no eligible evidence cannot be called a successful statistical audit. Use projected/chunked reads, reference-price indexing and bounded batches. Numerical correctness, tie handling and determinism take priority over accelerator reuse. Measure runtime and peak memory on declared hardware instead of promising unmeasured throughput.

20. **Acceptance scope.** The first release must demonstrate real fu data for both 10min and 30min across all seven horizons, including native and matched comparisons and long-label invalidity. Parameterized interfaces may support other instruments, but they are not declared validated without their own evidence. Python execution uses the project's finetf Conda environment. This issue requests implementation of the accepted design; the design and this issue do not imply that the functionality already exists.

## Testing Decisions

- **Primary seam: one independent audit application boundary.** Test validation, run and comparison behavior through the public entry, with fixture input data, calendar and configuration, then inspect exit status, published labels/metrics/manifests/reports, failure diagnostics and unchanged source/legacy fingerprints. Do not assert private helper calls, class structure, thread counts or a particular internal decomposition. The user explicitly confirmed this seam during issue preparation: “符合，按此发布”.
- **Prefer existing testing practice.** The repository already has subprocess-based independent feature-validation entry tests that verify command arguments, custom report destinations and absence of the main preprocessing flow. Commodity CLI tests build temporary raw snapshot/summary inputs. Dataset split tests assert contract and merged outputs, global boundaries, missing planned files and empty split failures. Comparison-report tests exercise timestamp alignment, tolerance handling and reported failures. Reuse those fixture and behavioral patterns rather than introducing another cross-project test framework.
- **Narrow numerical and time checks only where valuable.** Supplement the main seam with table-driven tests for calendar endpoints, availability mapping and standard Spearman/paired-statistic reference values. Existing predictive-audit tests provide noise and anti-causality scenarios, but their internal ranking behavior and FDR gates are not the new module's statistical oracle. Do not import the old selector as the test boundary.
- **Calendar scenarios.** Both frequencies must produce the same intended active-time endpoints; cover lunch breaks, night sessions, holidays/weekends, partial bars, temporary session changes, exact session ends, D+1/D+5/D+15 counting and missing observations that must not pause the trading clock.
- **Identity, availability and prices.** Cover same-contract labels across source partitions, a main-contract change without identity change, expiry/missing tails, closure-time decision mapping, duplicate source ordering, zero-length Today, future or cross-session quotes, 60-second freshness boundaries, positive/finite prices, conflict diagnostics and per-feature availability masks.
- **Leakage controls.** Verify maturity-based exclusions at every evaluation boundary, including night-session boundaries. Change isolated fixture Test prices and assert identical development outputs; verify the audit's allowed read set does not include final Test. Do not mutate actual sealed Test data to perform this test.
- **Statistical contracts.** Assert average-tie Spearman against a trusted reference; valid-pair re-ranking; null statuses for constants and insufficient pairs; contract-equal and sample-weighted results; distinct native/matched denominators; exactly paired frequency/horizon comparisons; and no fabricated comparison without common observations.
- **Uncertainty contracts.** Use deterministic fixtures to verify joint TradingDay block draws retain cross-contract and paired structure, preserve missing calendar slots, produce reproducible results for the recorded seed, and expose both block-length analyses. Verify insufficient effective block counts and undefined resamples lead to explicit statuses rather than zeros or improvised methods. Tests should not require a random CI to cross an arbitrary boundary on every run.
- **Isolation and failure behavior.** Verify audit runs neither invoke legacy selection/scaling nor change their configuration, outputs or input fingerprints; the old workflow still runs without an audit. Verify a failed run cannot publish a successful completion marker or overwrite an existing run. Verify independent 30min preparation cannot modify shared summaries.
- **Real integration and acceptance evidence.** After isolated inputs are prepared, run the full public audit for fu at both frequencies, inspect all seven target definitions, produce comparable run manifests and a paired comparison report, and record timings and peak resources. Synthetic fixtures and a frequency option alone do not satisfy real-data acceptance. Missing historical-calendar or causal-timing evidence remains an explicit implementation blocker, never fabricated coverage.

## Out of Scope

- Changing existing three-stream selection thresholds, blacklists, feature order/lists, scaling, checkpoints, training inputs or historical experiments.
- Feeding audit labels or rankings into online state, automatically deleting/promoting features, or making the original pipeline depend on a successful audit.
- Rebuilding feature families to equalize physical lookback windows across frequencies.
- Continuous-main-contract or roll-adjusted targets, executable fills, fees/slippage, net PnL, trading win rates or profitability guarantees.
- Path realized volatility, maximum favorable/adverse excursion, model-use attribution or strategy-level profit contribution.
- RL/VAE retraining, supervised prediction baselines, calibration, feature-group strategy ablation or optimizing on final Test.
- An automatic significance/FDR pass gate, a cross-horizon total score or an automatic best-feature list.
- Claiming verified operation for instruments other than fu or unmeasured performance targets.
- Silently repairing or regenerating existing historical inputs when upstream timing or provenance defects are found.

## Further Notes

- This spec synthesizes three completed design rounds on 2026-10-13. All 16 business/research decisions were accepted. The development baseline is v1.0.0; the frozen default parameters above are project conventions, not exchange requirements or statistical guarantees.
- The domain vocabulary is Multi-Horizon Feature Audit, Feature Data Frequency, Prediction Horizon, Active Trading Time, TradingDay, Same-Contract Horizon Return, Audit-Eligible Candidate, Matched Valid Sample Comparison, Audit Reference Price and Quote Age. Respect the accepted time-semantics and independent-audit ADRs and the existing three-stream architecture; the audit does not supersede the selector.
- Verified prerequisites: 10min Stage 6 data exists locally; expected 30min Stage 6 outputs do not. Sampled data lacks explicit TradingDay and feature availability, has naive timestamps and duplicate columns. Raw snapshots provide identity, TradingDay, event time and LastPrice, but complete coverage is unverified. A versioned historical exchange calendar has not been found. Some materialized bar and price-time fields require provenance checks against their generation version.
- The existing full preprocessing runner has no discovered stop-at-Stage-6 selector and proceeds into original selection/scaling. Isolated input preparation is therefore substantive delivery work, not a command alias for the old full pipeline.
- Current split metadata describes Train from 2023-01-03 to 2024-07-23, Valid to 2025-06-30, and Test to 2026-02-06, using half-open business-date intervals. These are snapshot facts; actual runs consume the versioned split contract and proper session-time boundaries instead of hardcoding dates.
- Historical feature counts such as 566 candidates or a 165-feature union describe particular artifacts. Do not hardcode them, reconcile the recorded 12-versus-13 volatility-list discrepancy through provenance, and use unknown when prior rejection evidence is absent.
- Local design documents and ADR updates were not committed or pushed as part of this synthesis. This issue body is self-contained so that implementation does not depend on unpublished local document changes; bring the accepted documentation into the eventual implementation changeset with the same decisions.
- Apply the ready-for-agent triage label. Readiness means the specification is settled; it does not claim that missing inputs, calendar evidence or implementation work have already been completed.
