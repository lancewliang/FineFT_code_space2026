# 三模块高层门控与校准交易效用准入规格

Status: ready-for-agent

## Problem Statement

当前高层门控把输入是否可信、市场属于哪个二维状态槽位以及候选交易是否值得执行混在同一决策中。VAE 似然和模式清晰度能够描述状态匹配程度，却不能证明一个候选动作在手续费、滑点、冲击成本和尾部风险之后仍有正效用。旧门控因此可能拒绝有价值的交易，也可能放行状态匹配但经济价值为负的交易。

现有 Q 值是带奖励塑形的折扣回报，不是真实扣费净 PnL；不同槽位的 Q 网络也不是针对同一候选动作训练的同质集成，不能用跨槽位均值和方差冒充期望收益与认知不确定性。同时，valid 和 test 数据有限，无法再永久切出校准集，但校准和门禁仍必须基于严格未见证据。研究人员还需要并行保留旧门控和新门控，确保两套代码可以独立运行、统一审计，并由用户在未来自行决定是否删除旧门控。

## Solution

新增一套与 legacy 物理隔离的 `tri_module` 高层门控栈，并通过单一顶层参数选择运行哪一套。新栈按决策权拆成三个串行模块：Module 1 只做确定性 VAE OOD 与数据完整性拒识；Module 2 只输出最匹配的二维状态槽位及连续路由证据；Candidate Resolver 将槽位显式映射为候选；Module 3 使用独立监督 Utility Model 对所有风险增加动作做交易效用准入。

Utility Model 直接学习固定视界内、相对拒绝基线的真实扣费净 PnL、下行 CVaR、最大不利偏移和盈利概率。连续效用使用 Isotonic Regression 校准，盈利概率使用 Platt scaling 校准。valid 采用两个外层 Fold 的严格扩展窗口 Prequential；每折只使用此前历史重新生成 valid 派生资产、Agent Selection、模型和校准器。最终 test 在全部资产冻结后只运行一次。

新旧门控共享同一版本化审计 envelope。legacy 不依赖新模型，新模型异常不得静默回退到 legacy。新门控只有在对应 `(slot, direction)` 具有充分未见证据、通过显著性和单调性门禁，且候选效用保守下界大于零时才允许增加风险，否则主动保持或回到不增加风险的状态。

## User Stories

1. As a strategy researcher, I want input validity, regime routing, and trading admission to be separate decisions, so that each conclusion has one clear source of authority.
2. As a strategy researcher, I want to select legacy or tri-module gating with one top-level parameter, so that experiments cannot accidentally mix both stacks.
3. As a strategy researcher, I want legacy behavior preserved as an isolated stack, so that historical baselines remain reproducible.
4. As a strategy researcher, I want tri-module runs to fail closed when their assets are invalid, so that they never silently inherit legacy behavior.
5. As a strategy researcher, I want missing tri-module assets to have no effect on legacy runs, so that the new experiment cannot destabilize the old one.
6. As a model reviewer, I want the selected gate stack recorded in every run, so that results can be attributed correctly.
7. As a risk researcher, I want Module 1 to evaluate only OOD and data integrity, so that economic judgments do not contaminate input-validity defense.
8. As a risk researcher, I want VAE OOD scoring to use deterministic latent means, so that identical inputs produce identical hard safety decisions.
9. As a risk researcher, I want OOD thresholds derived from train-ID empirical likelihoods, so that valid and test cannot redefine the reference distribution.
10. As a risk researcher, I want either VAE axis to trigger degradation when all of its slots are extreme, so that one failed market view cannot be masked by the other.
11. As a risk researcher, I want data gaps and missing execution fields detected separately from VAE OOD, so that operational failures have truthful reason codes.
12. As an operator, I want new-risk orders and persistence cancelled when Module 1 enters degradation, so that stale intent cannot continue increasing exposure.
13. As an operator, I want OOD exits to be best-effort under actual executable conditions, so that the system never pretends a missing market produced an instant fill.
14. As an operator, I want hysteretic OOD recovery, so that the stack does not oscillate between defense and trading at the threshold boundary.
15. As an auditor, I want OOD enter, stay, and recover events distinguished, so that a prolonged incident can be reconstructed.
16. As a strategy researcher, I want VAE weights, train scaler, and train-ID likelihood reference frozen across both outer Folds, so that the OOD detector is not repeatedly retuned to valid.
17. As a strategy researcher, I want VAE training separated from test analysis, so that test remains untouched until the final run.
18. As a routing researcher, I want Module 2 to return only a two-dimensional regime slot and continuous evidence, so that routing confidence cannot become an undeclared trade veto.
19. As a routing researcher, I want ambiguous regimes still routed to their best slot, so that Module 3 can judge the candidate on economic evidence.
20. As a strategy researcher, I want a distinct Candidate Resolver, so that slot selection and candidate availability remain observable steps.
21. As a strategy researcher, I want empty slots represented as `NullCandidate`, so that absence of a strategy is not mistaken for a rejected trade.
22. As a utility-model researcher, I want VAE and routing evidence available as continuous features, so that they can inform utility without acquiring hidden hard-veto authority.
23. As a utility-model researcher, I want labels based on real after-cost economics, so that the gate learns the outcome it is expected to protect.
24. As a utility-model researcher, I want reward shaping and discounted Q values excluded from labels, so that training targets retain direct monetary meaning.
25. As a utility-model researcher, I want all objectives expressed in bps, so that mean return and tail losses share a stable unit across contracts.
26. As an execution researcher, I want commission, slippage, and impact charged exactly once, so that the gate does not double-count friction.
27. As a risk researcher, I want tail utility to penalize both lower-tail CVaR and maximum adverse excursion, so that smooth small gains cannot hide severe path risk.
28. As a strategy researcher, I want shadow counterfactual episodes for every available risk-increasing candidate, so that training is not limited to actions admitted by the current gate.
29. As an auditor, I want counterfactual fills kept out of the real Trade and Account records, so that hypothetical labels never appear as executed facts.
30. As a strategy researcher, I want incomplete H-step tails and Fold-crossing episodes discarded, so that labels never use unavailable or leaked future data.
31. As a utility-model researcher, I want one shared model bundle per market and frequency with slot and direction features, so that evidence can be shared without granting unsupported local admission.
32. As a utility-model researcher, I want separate predictions for mean net PnL, lower-tail loss, MAE, and win probability, so that each component can be audited independently.
33. As a utility-model researcher, I want CatBoost preferred behind a replaceable interface, so that the first implementation is practical without hard-coding the model family forever.
34. As an operator, I want Utility Model training to have an independent entry point and shell launcher, so that it can be scheduled without coupling to VAE training.
35. As an operator, I want runtime assets loaded only through an explicit manifest, so that a run cannot silently pick an unrelated latest model.
36. As a model reviewer, I want every model, calibrator, feature schema, threshold, Fold, and data source hashed or versioned, so that a decision can be reproduced.
37. As a strategy researcher, I want valid evaluated with exactly two outer expanding-window Folds, so that limited data is not fragmented into four small evaluation blocks.
38. As a strategy researcher, I want the first valid period used only as initial history, so that Fold 1 begins with a usable training prefix.
39. As a strategy researcher, I want every outer Fold to rebuild all valid-derived selection assets from its historical prefix, so that Agent Selection cannot see its evaluation block.
40. As a calibration researcher, I want inner prequential OOF predictions, so that Isotonic and Platt never train on in-sample scores.
41. As a calibration researcher, I want each eligible post-warmup valid observation used in at most one outer unseen block, so that evidence is not counted repeatedly.
42. As a calibration researcher, I want embargo of at least H and intact position episodes at every boundary, so that adjacent Fold leakage is controlled.
43. As a strategy researcher, I want the final raw model retrained on all valid data, so that frozen production assets use all available pre-test history.
44. As a calibration researcher, I want the final calibrator trained only on accumulated leakage-free OOF evidence, so that the full-valid model cannot calibrate itself with training predictions.
45. As a strategy researcher, I want test run only once after all assets are frozen, so that it remains a final evaluation rather than another tuning set.
46. As a strategy researcher, I want any post-Fold-1 protocol change to invalidate both outer results, so that retrospective rule changes cannot survive in the evidence set.
47. As a model reviewer, I want Isotonic Regression to calibrate continuous utility, so that larger calibrated scores have an interpretable economic ordering.
48. As a model reviewer, I want Platt scaling to calibrate win probability for diagnostics, so that probability audits do not rely on raw classifier scores.
49. As a model reviewer, I want win probability excluded from online hard vetoes, so that a redundant threshold cannot contradict calibrated economic utility.
50. As a risk researcher, I want eligibility evaluated separately for each `(slot, direction)`, so that unsupported local channels cannot borrow a global pass.
51. As a risk researcher, I want one-sided block bootstrap and FDR control, so that correlated contract-day observations and multiple channel tests receive explicit protection.
52. As a risk researcher, I want minimum episode, quintile, contract, and TradingDay support enforced, so that sparse channels fail safely.
53. As a risk researcher, I want insufficient channels disabled without merged bins or relaxed thresholds, so that safety standards remain precommitted.
54. As a model reviewer, I want after-cost win rate and payoff ratio to rise strictly across utility quintiles, so that high scores demonstrate better realized trading quality.
55. As a model reviewer, I want monotonicity checked with rank correlation and a bootstrap trend lower bound, so that point estimates alone cannot pass the gate.
56. As a model reviewer, I want undefined payoff ratios treated as insufficient evidence, so that empty loss groups cannot create infinite apparent quality.
57. As a strategy researcher, I want tail-penalty coefficients selected only from a small preregistered nonzero grid, so that test-driven tuning cannot dilute risk aversion.
58. As a strategy researcher, I want candidate admission based on a calibrated utility lower bound above zero, so that estimated net benefit conservatively exceeds the friction already present in the label.
59. As a trader, I want rejected opens to remain Flat, so that an unprofitable candidate cannot create exposure.
60. As a trader, I want rejected same-direction adds to preserve my current position without increasing it, so that a rejection does not force an unrelated exit.
61. As a trader, I want rejected reversals to close the old direction and remain Flat, so that rejection cannot leave the old risk active or open the new risk.
62. As a developer, I want Flat resolved symbolically from the action space, so that action numbering changes do not corrupt safety behavior.
63. As an auditor, I want candidate action, denied baseline, and final action recorded together, so that every utility intervention is explainable.
64. As an auditor, I want utility rejection recorded as a decision rejection rather than an exchange rejection, so that policy and execution failures remain distinct.
65. As an auditor, I want legacy and tri-module runs to share one versioned audit envelope, so that comparisons do not require two unrelated data models.
66. As an auditor, I want tri-module-only fields nullable under legacy, so that compatibility does not force legacy to execute new candidate inference.
67. As a strategy researcher, I want test comparisons to remain advisory, so that no result can automatically delete legacy or switch the default stack.
68. As a developer, I want the full feature verified through the public training/evaluation and high-level routing contracts, so that tests assert observable behavior rather than private model internals.

## Implementation Decisions

- ADR-0062 is the architectural authority for this work. ADR-0031 remains the description of the retained legacy stack but is superseded for new design decisions.
- The system exposes exactly two high-level gate-stack choices: `legacy` and `tri_module`.
- The initial default remains `legacy`. Changing the default is a separate user decision and is not inferred from evaluation results.
- A run selects one stack at construction time. Stack implementations do not invoke each other.
- Both stacks adapt to one common runtime decision and audit envelope. Fields that have no legacy meaning are nullable rather than synthesized.
- Tri-module construction validates its explicit artifact manifest, feature schema, model hashes, calibrators, thresholds, and data lineage before trading. Validation failure is fail-closed and never falls back to legacy.
- Module 1 owns only OOD and input-validity decisions. Module 2 owns only regime-slot selection. Module 3 owns only economic admission of risk increases.
- Module 1 uses deterministic VAE latent means and per-axis train-ID empirical likelihood quantiles.
- Module 1 enters degradation when either axis has no slot above the 0.005 best-slot threshold or when a data-integrity check fails.
- Data-integrity failures have reason codes distinct from VAE OOD failures.
- Degradation blocks opens, same-direction adds, and reversals; cancels outstanding new-risk intent; and attempts safe reduction or closure only when executable market state exists.
- Recovery requires both axes at or above 0.01 and clean input for `max(VAE rolling window, H)` consecutive valid decisions.
- Module 1 reports distinct enter, stay, and recover lifecycle events.
- VAE weights, train scaler, and train-ID likelihood reference remain frozen for both valid outer Folds and the final test.
- VAE inference becomes deterministic for hard OOD decisions. Test OOD analysis is removed from any automatic training completion path.
- Module 2 returns the two-dimensional slot plus continuous routing evidence. It does not reject trades on a mode-margin threshold.
- Candidate resolution is a separate observable step. An empty selected slot yields `NullCandidate`.
- Module 3 may consume VAE and routing evidence as continuous features but has no model-external confidence veto beyond the declared Module 1 safety state and channel eligibility.
- Module 3 reviews every risk increase: open, same-direction add, and reversal.
- A rejected open resolves to symbolic Flat. A rejected add resolves to the current position. A rejected reversal closes the old side and does not open the new side.
- Existing hard stops and circuit breakers retain priority over normal admission and holding behavior.
- Offline training constructs shadow counterfactual candidate episodes for all available risk-increasing actions, while online inference scores only the actual candidate proposed by the selected policy.
- Each label compares the approved transition with its denied-action baseline over H equal to action persistence. The fu 10min protocol fixes H to 3.
- Labels use real economic quantities in bps and exclude shaped reward, limit reward, discounted Q values, and cross-slot Q dispersion.
- Net labels subtract commission, slippage, and impact exactly once. Any 1.5-times friction stress is applied during label construction and is not deducted again online.
- Incomplete H-step episodes, episodes that cross Fold boundaries, and episodes without trustworthy execution inputs are excluded.
- Counterfactual transactions remain training evidence only and never enter real Prediction, Trade, or Account ledgers as executed events.
- Utility combines predicted mean net PnL with positive loss measures for 5% lower-tail CVaR and H-window MAE.
- CVaR and MAE coefficients are global, nonzero, and selected only from `{0.25, 0.5, 1, 2}` using inner prequential evidence. Risk and monotonicity constraints are applied before ranking candidates by conservative net utility.
- The Utility Model is one shared bundle per market and frequency. Slot, direction, state, candidate, routing, and permitted microstructure context are features; weighting prevents dominant contracts or channels from monopolizing training.
- The bundle contains a mean-net-PnL regressor, a multiple-lower-quantile tail model used to approximate 5% CVaR, an MAE regressor, and a win-probability classifier.
- All model heads share one frozen feature contract, Fold contract, candidate-label contract, and artifact manifest.
- CatBoost is the preferred first model family behind a replaceable estimator interface.
- Utility training has its own programmatic entry point and dedicated shell launcher. VAE training remains separate and gains no duplicate launcher.
- Runtime requires an explicit manifest reference and never resolves an implicit latest artifact.
- The manifest records stack identity, data ranges, TradingDay Fold boundaries, embargo, H, features, cost model, Selector, all model heads, calibrators, residual distribution, coefficient choices, seeds, support counts, eligibility results, code version, and hashes.
- The valid initial-history interval is 2024-07-23 through 2024-10-31 and produces no outer score.
- Outer Fold 1 evaluates 2024-11-01 through 2025-02-28. Outer Fold 2 evaluates 2025-03-01 through 2025-06-27.
- Fold boundaries align to the official TradingDay field. Natural dates identify the intended ranges but never override actual session ownership.
- Each boundary applies an embargo of at least H and keeps position episodes intact.
- Each outer Fold rebuilds every valid-derived statistic, slice, Agent Selection result, Utility Model, and calibrator from the preceding historical prefix only.
- Inner prequential/cross-fit creates leakage-free raw scores for hyperparameter choice and calibration. Two outer Folds do not restrict the number of inner fits required to produce OOF evidence.
- Each eligible post-warmup valid sample appears in at most one outer unseen evaluation block.
- After both outer evaluations, the final Selector and raw Utility Model are fit using all valid history.
- The final calibrator uses only accumulated leakage-free OOF/unseen predictions and outcomes; it never consumes in-sample scores from the final full-valid model.
- The final production fit is not counted as a third outer evaluation Fold.
- Feature schema, estimator families and search spaces, coefficient grid, quintile method, insufficient-evidence policy, significance rules, monotonicity rules, OOD thresholds, and recovery rules are frozen before Fold 1.
- Any later change to those frozen protocol elements invalidates prior outer evidence and requires a complete rerun beginning at Fold 1.
- All final assets are frozen before the single test run covering 2025-06-30 through 2026-02-05.
- Test never selects models, hyperparameters, calibrators, thresholds, default stack, or legacy removal.
- Isotonic Regression maps raw continuous utility to calibrated utility. Platt scaling calibrates after-cost win probability.
- Calibrated win probability is diagnostic and participates in monotonicity reporting but is not an online hard threshold.
- Offline channel eligibility is evaluated independently for every `(slot, direction)` using aggregated outer unseen evidence.
- Eligibility uses a one-sided 95% bootstrap whose resampling block is `contract × TradingDay`, with FDR control across channel tests.
- A channel requires at least 150 complete candidate episodes, at least 30 episodes per utility quintile, at least 3 contracts, and at least 20 effective TradingDay blocks.
- An insufficient channel is disabled without bin merging, threshold relaxation, or substitution of a global pass.
- For each channel, utility quintiles require strictly increasing after-cost win rate and payoff ratio, Spearman correlation of at least 0.80, and a positive one-sided bootstrap lower bound for the overall trend slope.
- Payoff ratio is mean positive net PnL divided by the absolute mean negative net PnL. A quintile without adequate winners and losers is insufficient and cannot produce an infinite passing ratio.
- Fold-level monotonicity tables are diagnostic; the pooled channel-level outer unseen result is the eligibility gate.
- Online conservative utility is the calibrated estimate plus the fifth percentile of strict OOF residuals.
- A risk increase is admitted only when Module 1 is healthy, the local channel is eligible, every artifact check passes, and the conservative utility is greater than zero.
- Online admission does not subtract friction again because the target and residual evidence are already after cost.
- The common audit envelope includes stack identity, OOD lifecycle and reason, routing slot and evidence, candidate, denied baseline, final action, raw and calibrated utility, conservative lower bound, calibrated win probability, channel eligibility, rejection stage and reason, artifact identity, and links to the existing audit evidence chain.
- A utility rejection is a decision-layer event, not an exchange rejection. Hypothetical label fills are never written as real execution facts.
- No code automatically deletes legacy, changes the default stack, or interprets test superiority as authorization to do either.

## Testing Decisions

- The primary online test seam is the public high-level routing decision contract exercised through the existing fake network, fake environment, and compact multi-step fixtures. Tests observe selected stack, OOD lifecycle, routed slot, candidate, final action, position transition, and audit envelope.
- The primary offline test seam is the public Utility Model training/evaluation command and its published manifest plus prequential audit report. Tests provide a compact chronological multi-contract dataset and assert asset lineage, Fold isolation, OOF-only calibration, eligibility, and fail-closed publication.
- Tests assert external decisions, artifacts, and audit records rather than CatBoost internals, private helper calls, or the numerical layout of model objects.
- Existing gating-strategy tests are prior art for legacy behavior, strategy construction, thresholds, and normalized route evidence.
- Existing VAE routing action-persistence, reversal, non-main defense, hard-stop, circuit-breaker, and graceful-exit tests are prior art for position-transition semantics.
- Existing precomputed-quantile and hysteresis gating tests are prior art for deterministic threshold state and enter/stay/recover behavior.
- Existing two-dimensional Agent Selector and evidence-based Selector tests are prior art for prefix-scoped selection manifests and evidence failures.
- Existing daily trading audit and end-to-end audit evidence-chain tests are prior art for schema versioning, deterministic linkage, nullable fields, and decision-versus-execution attribution.
- A stack-isolation scenario runs identical input once under legacy and once under tri-module, verifies that each stack uses only its own dependencies, and confirms both produce the shared envelope.
- A missing or mismatched tri-module artifact scenario proves fail-closed behavior and confirms that no legacy object is constructed as fallback.
- A legacy-without-utility-assets scenario proves that the legacy stack still runs unchanged.
- Deterministic OOD scenarios repeat identical inputs and require identical decisions, quantiles, reason codes, and lifecycle transitions.
- Per-axis OOD scenarios prove that either failed axis enters degradation and that ambiguous but in-distribution routing does not.
- Data-integrity scenarios cover gaps, missing required fields, non-monotonic event order, and unavailable execution state with distinct reason codes.
- OOD recovery scenarios prove the 0.005 entry threshold, 0.01 recovery threshold, clean K-step requirement, and reset of the recovery counter after a dirty step.
- Unexecutable OOD scenarios verify that no synthetic fill or false Trade Record is created.
- Router scenarios prove that the best two-dimensional slot is returned even when the top-two margin is small and that routing never emits an admission decision.
- Candidate Resolver scenarios cover valid strategy mapping and `NullCandidate` without hidden route changes.
- Admission transition scenarios cover open, same-direction add, and reversal for both approved and rejected decisions, including symbolic Flat resolution.
- Priority scenarios verify that hard stop and circuit breaker behavior still overrides normal utility admission and holding.
- Label tests reconcile gross PnL, commission, slippage, impact, denied baseline, CVaR, and MAE in bps and prove that each cost appears once.
- Label tests reject incomplete H-step tails, Fold-crossing episodes, and data-invalid episodes.
- Counterfactual tests prove that all available risk-increasing candidates receive labels while no hypothetical execution enters the real audit ledger.
- Model-bundle contract tests verify the four required outputs, common feature schema, deterministic manifest identity, and rejection of incompatible heads.
- Two-Fold schedule tests assert the exact official TradingDay ranges, expanding historical prefixes, embargo, and one-time assignment of eligible evaluation samples.
- Leakage tests place sentinel observations in future blocks and prove they cannot influence the current Fold's Selector, features, models, calibration, residual distribution, or thresholds.
- Calibration tests prove Isotonic and Platt receive only OOF inputs and that the final full-valid model's training predictions are excluded.
- Protocol-freeze tests change one registered schema, threshold, grid, or Fold field and require prior outer evidence to be rejected as incompatible.
- Test-once workflow tests reject any attempt to update frozen assets from test outcomes.
- Eligibility tests cover passing channels, insufficient episode count, insufficient quintile count, insufficient contracts, insufficient TradingDay blocks, bootstrap failure, FDR failure, and monotonicity failure.
- Payoff-ratio tests cover ordinary winners and losers, zero or missing loss groups, and non-finite values; undefined ratios must never pass.
- Monotonicity tests cover strict pointwise ordering, Spearman threshold, positive trend lower bound, and the distinction between diagnostic Fold tables and pooled eligibility.
- Conservative-utility tests verify the fifth-percentile residual adjustment, strict greater-than-zero admission rule, and absence of online double cost subtraction.
- Audit compatibility tests serialize equivalent legacy and tri-module runs, verify one schema version, and require tri-module-only legacy fields to remain null rather than fabricated.
- Decision-attribution tests verify utility rejection is not an exchange rejection and that candidate, denied baseline, final action, channel status, and artifact identity remain joinable to Prediction, Trade, and Account records.
- A compact end-to-end tracer bullet trains on the initial prefix, evaluates Fold 1, expands and evaluates Fold 2, builds final assets, and performs one frozen test inference without examining private estimator state.

## Out of Scope

- Deleting the legacy gate stack.
- Automatically switching the default stack based on valid or test performance.
- Running both gate stacks inside one live decision path or combining their votes.
- Implementing or training models as part of this specification-writing task.
- Redesigning the low-level trading policy or its reward function.
- Treating cross-slot Q values as a same-candidate uncertainty ensemble.
- Adding a hard win-probability admission threshold.
- Retuning VAE weights in each valid Fold.
- Using test to choose features, selectors, estimators, coefficients, calibrators, thresholds, eligibility, or protocol rules.
- Guaranteeing that every admitted trade is profitable or eliminating all losing trades.
- Guaranteeing immediate OOD liquidation when executable market data is unavailable.
- Weakening evidence standards to fill every slot/direction channel.
- Reconstructing missing utility evidence for legacy runs.
- Shared-wallet portfolio allocation, broker connectivity, or real-exchange fill guarantees.
- Extending the initial protocol beyond fu 10min before the market/frequency-specific manifest and evidence requirements are defined.

## Further Notes

The two outer Folds are evaluation Folds, not a cap on model fitting. Inner fits remain necessary to produce leakage-free OOF predictions for selection and calibration, and the final full-valid production fit is not a third outer evaluation.

The VAE does not adopt the Utility Model's full prequential retraining cycle. Its weights and train-derived reference assets stay frozen; only deterministic inference, data-integrity handling, valid-derived downstream assets, and strict separation of test analysis must change.

The resulting test comparison is evidence for a later human decision. Even if tri-module performs better, this specification intentionally leaves legacy removal and default-stack migration to a separate explicit request.
