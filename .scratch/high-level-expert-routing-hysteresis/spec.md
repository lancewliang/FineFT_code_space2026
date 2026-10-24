# 高层专家路由迟滞实施规格

Status: ready-for-agent
Date: 2026-10-24
Decision: [ADR-0062](../../docs/adr/0062-stateful-expert-routing-hysteresis-and-frozen-score-calibration.md)
Design: [详细设计与 Q1–Q11 共识](../../docs/train3/high_level_routing_hysteresis_design.zh_cn.md)

## Problem Statement

研究人员希望高层路由对继续使用当前专家采用较宽松条件，对切换过去采用更严格条件，减少临界分数造成的专家反复切换，同时保留专家对最新行情的响应。

目标项目现有双轴门控只看各轴最高分，并按是否持仓决定宽松阈值；普通动作持续性还会跳过策略推断。它们不能表达“当前专家仍有效就每步重新决策”，也不能防止一个已经失效的专家被其他专家的高分间接保活。旧研究工作树的单轴示例与本项目的二维 Slot、风险分层和空仓降级槽位不兼容。

本规格基于已经完成的 Q1–Q11 访谈，不重新访谈，不把文档生成当作已完成实现。

## Solution

新增显式可选的 expert_hysteresis 门控策略，围绕当前专家身份保存状态。当前专家对应双轴评分达到保持门槛时继续每步决策；新候选达到更高进入门槛，并在发生改变的轴足够领先、连续确认后接管。当前专家任一轴失效立即撤销资格，没有完成确认的替代者则规则接管。

初次启用、规则恢复、专家切换共用 K；恢复不额外要求领先第二名。候选仍连续合格时，其证据可跨普通专家失效延续。两轴 Slot 原子切换，保留既有硬风控、空槽 No New Risk、缓存、轻量调参和最终回放约束。

## User Stories

1. As a strategy researcher, I want an explicit expert-hysteresis strategy, so that historical routing baselines remain reproducible.
2. As a strategy researcher, I want the active expert identified independently of the current position, so that Flat output does not erase expert ownership.
3. As a strategy researcher, I want the retained expert to evaluate each fresh observation, so that route stability does not freeze an old trade action.
4. As a strategy researcher, I want both slope and volatility evidence checked for the active Slot, so that another expert cannot conceal its loss of applicability.
5. As a strategy researcher, I want lower holding thresholds and higher entry thresholds, so that continued use and new control require different evidence.
6. As a strategy researcher, I want an expert invalidated on the first smoothed-score breach, so that candidate confirmation does not keep an invalid expert alive.
7. As a strategy researcher, I want a candidate to clear both entry thresholds, so that a relatively best but unsuitable expert cannot take control.
8. As a strategy researcher, I want each changed axis to lead its incumbent counterpart by Δ, so that small score fluctuations do not replace the current Slot.
9. As a strategy researcher, I want unchanged axes exempted from self-comparison, so that a legitimate one-axis transition can complete.
10. As a strategy researcher, I want one candidate Slot to qualify for K consecutive decisions, so that unrelated experts cannot pool evidence.
11. As a strategy researcher, I want Slot changes to be atomic, so that independently confirmed axes cannot synthesize an unconfirmed expert.
12. As a strategy researcher, I want candidate changes and interruptions to restart confirmation, so that old evidence is never reused incorrectly.
13. As a strategy researcher, I want valid candidate evidence retained across ordinary incumbent invalidation, so that loss of the old expert does not impose an artificial delay.
14. As a strategy researcher, I want a fully confirmed candidate to take over on the same invalidation step, so that a redundant rule-only step is unnecessary.
15. As a strategy researcher, I want the rule branch used when no expert is ready, so that the system does not continue an invalid expert.
16. As a strategy researcher, I want entry and recovery to require no runner-up margin, so that close but suitable regimes can regain control after persistent evidence.
17. As a strategy researcher, I want first entry, recovery, and switching to share K, so that the initial search space remains manageable.
18. As a strategy researcher, I want the active expert retained when tied for best, so that ties cannot trigger replacement.
19. As a strategy researcher, I want an existing tied-best candidate preferred, so that index ordering cannot repeatedly destroy its confirmation.
20. As a strategy researcher, I want deterministic ordering when no tied candidate exists, so that identical replay inputs produce identical routes.
21. As a risk reviewer, I want empty-model Slots to retain No New Risk semantics, so that hysteresis cannot turn an unavailable expert into permission to trade.
22. As a risk reviewer, I want hard stops, cooldowns, circuit breakers, and non-main restrictions preserved, so that route memory cannot bypass risk control.
23. As a risk reviewer, I want no stale expert inference in an empty-model interval, so that graceful holding is not confused with expert authorization.
24. As a backtest operator, I want invalid scores and incompatible Slot mappings to stop the run, so that scoring failures cannot masquerade as low confidence.
25. As a backtest operator, I want route state reset between independent contract runs, so that one contract cannot influence another's confirmation.
26. As an experiment owner, I want VAE scoring models, windows, and decay fixed during hysteresis calibration, so that the comparison isolates routing changes.
27. As an experiment owner, I want calibration periods independent of model fitting and expert selection, so that parameter selection has a defensible temporal boundary.
28. As an experiment owner, I want test parameters frozen, so that final performance is evaluated without further selection.
29. As an experiment owner, I want quantile caches reused without sharing mutable route state, so that memory savings do not cause cross-run contamination.
30. As an experiment owner, I want lean trials and one complete best-trial replay, so that calibration preserves the existing resource model.
31. As an auditor, I want every routing event linked to its Decision ID and genuine prediction provenance, so that route changes can be separated from execution changes.
32. As an auditor, I want holding, confirmation, replacement, invalidation, and rule causes distinguished, so that replay explains each chosen action.
33. As an evaluator, I want switching, delay, friction, net results, and drawdown reported together, so that reduced switching is not mistaken for guaranteed improvement.
34. As a developer, I want tests at the existing public router boundary, so that requirements are verified through observable decisions and actions.

## Implementation Decisions

- Add a separately selectable expert_hysteresis strategy through the existing gating factory and public decision contract. Retain absolute and hierarchical baseline behavior and their existing configuration meanings.
- The state belongs to one independent contract rollout. It contains active Slot identity, pending candidate identity, candidate count, stable axis/Slot identities, and a decision reason. Shared read-only quantile data never carries this mutable state.
- The target architecture remains dual-axis. A Slot is the existing ordered volatility/slope pair from the Selection Manifest; do not introduce a single-VAE expert list or invent a joint profitability score.
- For each axis define H as the holding threshold and E as the entry threshold, with 0 <= H < E <= 1. Reuse the existing exit-ratio expression H_axis = exit_ratio × E_axis, requiring 0 < exit_ratio < 1 for the new strategy. Persist both configured and derived values.
- Validate public inputs as nonempty finite quantile vectors in [0, 1], with stable axis lengths, label ordering and Slot mapping throughout the independent rollout. Invalid input raises an explicit error; no silent clipping, NaN fallback or continuation as an ordinary defensive result.
- Evaluate current applicability using the active Slot's own two scores. Both must be at least their H. A score below H on either axis revokes the active expert on that decision step, independently of other experts' scores and independently of the account being Flat.
- Build the highest-score Slot set from the Cartesian product of the two axes' maximum-score sets. Prefer the valid active Slot, then the existing candidate if still a member, then deterministic Slot order. Do not switch tied active experts.
- An ordinary replacement candidate must clear both E thresholds. Every changed axis must strictly exceed its active counterpart and meet the same nonnegative Δ. An unchanged axis does not compare against itself, but must still meet its entry requirement.
- Without a valid incumbent, first entry or recovery only requires both entry thresholds and continuity of the same highest-score Slot. Do not apply a runner-up margin gate from the hierarchical strategy to this path.
- First entry, recovery, and replacement use the same positive integer K. Count only consecutive eligible decision steps of the same candidate Slot; a different eligible candidate starts at 1 and an ineligible step clears the pending evidence.
- Ordinary incumbent invalidation does not itself clear still-valid candidate evidence. If the same candidate completes K on that step, it takes control immediately without a forced intermediate rule action.
- Completed confirmation clears the pending candidate state. If confirmation is incomplete, retain a valid incumbent; otherwise route to the target project's existing rule branch.
- Exact threshold equality is inclusive: score equal to H remains valid, score equal to E can qualify, and a positive lead equal to Δ qualifies. Δ = 0 still requires strict superiority. Numerically boundary-focused tests must verify the chosen comparison contract.
- Retaining an ordinary model Slot must query the selected expert with the latest observation, true previous action, account state and available-action mask on every eligible step. A nonzero legacy persistence counter must not skip that query or force repetition of the previous micro action in this strategy.
- This requirement partially supersedes legacy nondefensive inference-skipping only for the new strategy. It does not retrain low-level models, rewrite Reward Shaping, or change DP Q tables or physical account balances.
- A selected dominant empty_model Slot is not an eligible ordinary expert. Preserve its existing No New Risk and finite graceful-hold-exit behavior; never fit a second-best model merely to fill the Slot, run a stale expert during the interval, or refresh the empty-slot hold buffer indefinitely.
- Hard stops, circuit breakers, directional cooldown, unavailable-action handling and non-main restrictions retain their existing priority. No K-step wait delays their required action. Risk/empty-slot blocking does not count as a qualifying candidate step and cannot silently join confirmation streaks across the interruption.
- Independent rollout initialization and reset clear route state. Existing score warm-up remains; after warm-up the initial candidate must complete normal entry confirmation.
- Calibration freezes VAE and specialist versions, reference score distributions, both scoring windows, and both decay parameters. Search both axis entry thresholds, the exit ratio, Δ and K; record the corresponding H/E values.
- Split time chronologically between fitting/selection, calibration, and final evaluation. Reuse of a fitting or selection segment for calibration must be corrected before treating a result as independent. Final test outcomes cannot update chosen parameters.
- Parameter parsing, trial records, best-parameter replay and final evaluation must preserve strategy identity/version, fixed scoring settings, all thresholds, Δ and K without reverting to defaults.
- Reuse quantile caching, lean trials and best-trial replay. Calibration may suppress large histories, while the best-trial replay persists complete routing/audit evidence.
- Extend existing routing attribution using the same Decision ID and Prediction Record rules. Record before/after active Slot, candidate, count before/after update, the just-completed candidate when count resets, per-axis scores, thresholds, event reason, genuine inference source and final action.
- Distinguish observed dominant Slot from active expert and defensive state in the audit output. Do not count an expert switch as a trade, fabricate a Prediction Record during a rule-only step, or charge an additional switching fee outside actual simulated execution.
- Persist candidate ordering and experiment identities so that fixed input replay is deterministic. Score generation variability must be controlled with the same cache or a frozen scoring protocol when comparing router variants.
- This spec is the behavioral reference for a separately initiated future implementation task. The current authorization remains documentation only: no Python migration, training, calibration, test run or live deployment.

## Testing Decisions

- The main business test seam is the existing public high-level get_action entry point. Use fixed dual-axis score sequences, a controlled account and changing fake specialist outputs; assert the selected expert, returned action and public audit artifacts. This follows the already discussed fixed-score and end-to-end validation plan.
- Prefer the existing router, action-persistence, gating, non-main-defense, quantile-cache and final-result fixtures. Do not introduce a new private-helper test seam merely to check counters.
- A retained expert must emit updated actions when the fake policy's output changes across observations, even with a nonzero legacy persistence setting. Expert ID alone is insufficient proof of Q1.
- Cover one-axis replacement, two-axis replacement, both-axis entry qualification, per-changed-axis Δ, subthreshold current scores hidden by a high rival, exact threshold equality and strict nonreplacement on ties.
- Replay interrupted and changing candidates, K-th-step handover, ordinary invalidation at K-1, direct same-step takeover, and rule-only recovery. Assert that candidates and axes cannot pool confirmation evidence.
- Cover all tied-best cases: active tied-best, an existing higher-ID candidate tied by a newly best lower-ID Slot, and a tie without previous state. Assert externally stable selection.
- Test initial warm-up followed by K-step entry and state reset between independent contract rollouts.
- Cover empty-model Flat and held positions, finite graceful-hold-exit, no stale ordinary inference, hard-stop and circuit-breaker preemption, cooldown and non-main restrictions. Route memory must never broaden available risk.
- Invalid-input cases cover empty arrays, NaN/Inf, out-of-range values, changed axis dimensions or Slot identity mapping, and invalid configuration. Assert an explicit failed run rather than a plausible continued return.
- Use public tuning and final-evaluation preparation/replay entry points for the necessary configuration integration seam. Assert that scoring windows and decay remain fixed, all new policy parameters are actually sampled, and chosen values survive export and reload.
- Compare cached and the same frozen scoring protocol through the router; vary only the source of identical quantile inputs, not stochastic model outputs.
- Retain historical strategies' regression scenarios as baseline coverage. New strategy fixtures must assert genuine inference rather than copying legacy ACTION_PERSISTENCE expectations.
- Backtest comparisons report replacement and rule transitions separately, short-term switchbacks, confirmation waiting in steps and time, rule occupancy, real turnover, fees, net metrics and drawdown. Reduced switching alone is not a success criterion.
- Implementation acceptance requires the observable state/risk/replay contracts above. Numeric performance tolerances and final calibrated values require actual independent evidence; this spec does not invent thresholds or guarantee returns.

## Out of Scope

- Copying the obsolete single-axis Python router from the old worktree into this project.
- Replacing or reinterpreting the historical absolute and hierarchical strategies.
- Changing low-level model training, the action space, turnover Reward Shaping, DP expert tables, transaction fees or account accounting.
- Relaxing Selection Manifest evidence requirements, filling empty Slots with inferior models, or changing ADR-0061's No New Risk behavior.
- Weakening hard stops, circuit breakers, cooldowns, non-main restrictions or execution masks.
- Defining new joint class probabilities, multiplying raw scores into a profitability estimate, or mixing uncalibrated specialist Q values.
- New fast/slow degradation classifiers or specialist-performance retraining triggers; this feature is expert-selection hysteresis.
- Live order execution, deployment, large training jobs or running calibration as part of this documentation update.
- Claiming optimal numeric settings or improved future profits without validation.
- Generating implementation tickets or an unrelated OpenSpec change in place of the project's requested local to-spec publication.

## Further Notes

ADR-0062 explicitly scopes the updates to ADR-0031, ADR-0045 and ADR-0049. Their historic benchmark behavior and unrelated risk/reward decisions remain documented rather than erased. ADR-0046 caching, lean tuning and best-trial replay remain applicable; ADR-0047/0057/0060/0061 risk and contract semantics remain constraints.

The single-axis H/E/Δ/K discussion has been projected onto existing per-axis thresholds and one atomic Slot identity. The per-axis exit ratio already used by the target can express the new H/E relationship without adding an arbitrary joint score. This adaptation and its acceptance examples are recorded in the linked design.

Suggested implementation order is a public-router tracer scenario proving fresh decisions under retained expert identity, then the complete state/risk cases, then parameter freeze/calibration/replay integration, then independent retrospective evaluation. These are implementation boundaries, not generated tickets.

Formal parameter values, calibration outcome and performance are not yet available. The ready-for-agent label means the behavioral implementation is specified, not that calibration has already succeeded or that live use is approved.
