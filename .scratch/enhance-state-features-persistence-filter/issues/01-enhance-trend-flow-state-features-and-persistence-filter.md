# 01 — 增强趋势/资金流 State Feature 并加入微观返还率持续性过滤

**What to build:** 为商品期货 Feature Engineering 补充非绝对价格形式的 VWAP slope、EMA slope、ADX/DMI 和 CVD slope candidate State Feature，并在 Feature Selection 中加入微观返还率族 Persistence Filter，减少衰减过快的噪声翻仓输入。

**Blocked by:** None — can start immediately

**Status:** ready-for-agent

## Problem Statement

当前商品期货 Feature Engineering 缺少若干趋势强度、趋势斜率和资金流持续性类网络输入特征：ADX/DMI、显式 EMA slope、CVD slope，以及 10min 级别 96/192 bar VWAP slope。同时，Feature Selection 主要依赖 IC、RankIC、Importance、Sharpe 和相关性过滤，缺少对信号持续性的约束，导致衰减过快的 2-bar 微观返还率特征可能进入 State Feature，驱动噪声翻仓。

已有代码中还存在数学语义重复风险：若特征公式在有效区间等价、仅列名或窗口标签不同，不应重复增加或选择。新增特征也必须遵守 No Absolute Price Rule，不能把 raw close、raw VWAP、raw EMA 或绝对价差这类价格水平直接作为网络输入。

## Solution

补充非绝对价格形式的趋势、动量和资金流 State Feature，并在 Feature Selection 中加入 Persistence Filter，用自相关或方向半衰期识别衰减过快的微观返还率族特征。

新增特征只表达结构、斜率、相对变化、趋势强度或资金流方向持续性，不增加绝对价格特征。标准 MACD 不作为新增目标，因为现有 EMA spread/velocity 语义已经覆盖同一信号族，应避免数学语义重复。

## User Stories

1. As a feature engineer, I want 96/192 bar VWAP slope features, so that the agent can observe medium-horizon VWAP trend without receiving raw VWAP levels.
2. As a feature engineer, I want 96/192 bar EMA slope features, so that the agent can observe trend slope without receiving raw EMA levels.
3. As a feature engineer, I want ADX/DMI features, so that the agent can observe trend strength and directional movement.
4. As a feature engineer, I want 96/192 bar CVD slope features, so that the agent can observe accumulated order-flow direction without relying on raw cumulative levels.
5. As a researcher, I want mathematically duplicate candidate State Features removed or canonicalized, so that Feature Selection does not reward aliases of the same formula.
6. As a researcher, I want standard MACD excluded when an equivalent EMA spread signal family already exists, so that the feature set stays semantically compact.
7. As a researcher, I want No Absolute Price Rule enforced for new features, so that models do not learn contract price scale artifacts.
8. As a researcher, I want Feature Selection to record autocorrelation or half-life diagnostics, so that fast-decaying signals are visible in the manifest.
9. As a researcher, I want a Persistence Filter for one-step micro-return families, so that noise-driven turnover features can be rejected before final State Feature selection.
10. As a researcher, I want the persistence threshold applied conservatively, so that valid mean-reversion signals are not accidentally removed.
11. As a pipeline operator, I want mandatory State Feature behavior preserved, so that required environment inputs are not filtered out by the new rule.
12. As a pipeline operator, I want Feature Selection Manifest to describe the new persistence configuration, so that runs are reproducible.
13. As a pipeline operator, I want selected features to remain compatible with Scale Save, so that downstream training artifacts keep working.
14. As an RL practitioner, I want fewer fast-decay micro-return features, so that the policy is less likely to churn positions on noise.
15. As an RL practitioner, I want medium-horizon slope features, so that the policy can distinguish persistent trend from local bounce.
16. As an RL practitioner, I want order-flow slope rather than raw cumulative flow, so that the observation is more stationary.
17. As a maintainer, I want tests at existing high-level seams, so that behavior is protected without overfitting implementation details.
18. As a maintainer, I want expected feature columns updated only for canonical new features, so that validation remains meaningful.
19. As a maintainer, I want old mathematically duplicated return aliases handled deliberately, so that backward compatibility tradeoffs are explicit.
20. As a future agent, I want the issue to name out-of-scope absolute-price additions clearly, so that implementation does not drift.

## Implementation Decisions

- Add 10min-compatible VWAP slope features for 96 and 192 bars.
- Add EMA slope features for 96 and 192 bars.
- Add DMI directional features and ADX trend-strength feature with the canonical 14-bar period unless existing project conventions require a different default.
- Add CVD slope features for 96 and 192 bars using signed trade volume derived from existing trade direction estimation.
- Do not add raw VWAP, raw EMA, raw close/open/high/low, absolute spread, or other absolute price-level State Features.
- Do not add standard MACD as a separate network input when the existing EMA spread/velocity signal family already covers that mathematical semantics.
- Apply Feature Semantic Deduplication: formulas equivalent over the valid interval should have one canonical candidate implementation.
- Add Persistence Filter to Feature Selection after hard filtering and before stability/composite/correlation filtering.
- Scope hard persistence rejection to one-step micro-return families first; other features should record diagnostics but not be rejected solely by autocorrelation.
- Use directional half-life or autocorrelation metrics aggregated across contracts, with a conservative default such as minimum half-life of 1.0 bar for targeted micro-return families.
- Feature Selection Manifest should record persistence diagnostics, threshold configuration, filtered features, and whether a feature was diagnostic-only or actively filtered.
- Mandatory State Features should continue to bypass optional feature filtering unless explicitly made invalid by schema or data-quality checks.
- Any change to duplicated return horizon semantics should be treated as a compatibility-sensitive decision and documented separately if implementation changes existing column contracts.

## Testing Decisions

- Highest feature-generation seam: run the existing enhanced State Feature generation path on deterministic synthetic OHLCV/order-flow input and assert that the requested canonical columns are emitted, finite after warm-up, and comply with No Absolute Price Rule.
- Highest Feature Selection seam: run the existing selection pipeline on a small deterministic candidate set with synthetic metrics and persistence profiles, then assert that fast-decay one-step micro-return features are filtered and diagnostic-only features are retained.
- Manifest behavior should be tested externally by inspecting produced manifest fields, not by asserting private helper internals.
- Semantic deduplication should be tested through observable selected/candidate feature names, ensuring equivalent aliases do not both survive.
- Existing validation fixtures for expected columns should be updated to include only canonical new features.
- Regression tests should preserve mandatory State Feature passthrough behavior.
- Tests should avoid using real market data unless an existing integration fixture already does so.

## Out of Scope

- Adding absolute price-level features such as raw close, raw VWAP, raw EMA, or raw price spreads.
- Adding standard MACD as a duplicate of the existing EMA spread/velocity signal family.
- Broadly filtering all mean-reversion features by positive autocorrelation.
- Changing reward/execution columns.
- Reworking the entire Feature Selection scoring model.
- Refactoring unrelated feature families.
- Changing downstream RL model architecture.

## Further Notes

The domain glossary now includes Feature Semantic Deduplication and a generalized No Absolute Price Rule. The implementation should preserve those terms: new candidate State Features should be normalized, relative, slope-based, return-based, or otherwise stationary enough to avoid encoding contract price scale directly.
