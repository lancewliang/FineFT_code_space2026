# Spec: Front-Loaded Filtering, Distribution Drift Gate, and Decentralized Multi-Contract Feature Selection

- **Triage Label**: `ready-for-agent`
- **Related Research**: `docs/research/multi_contract_feature_engineering_and_selection_ood_remediation_report.md`
- **Related ADR**: `docs/adr/0035-front-loaded-filtering-distribution-drift-gate-and-decentralized-multi-contract-feature-selection.md`
- **Target Subsystems**: Time Operator Feature Generation (`time_operator`), Multi-Contract Feature Selection Pipeline (`muti_contract`), Correlation Filtering (`cor_util`), Market State Regime Audit (`regime_audit`)

---

## Problem Statement

In the commodity multi-contract feature engineering and selection pipeline, features selected across training contracts frequently exhibit severe out-of-distribution (OOD) likelihood degradation when deployed out-of-sample in validation and testing contracts. This triggers downstream variational autoencoder (VAE) Gaussian negative log-likelihood (NLL) explosions, forcing high-level routing policies into perpetual defensive shutdown (holding defensive cash over 98% of the time, diluting annualized returns down to ~0.5%).

Empirical investigation has identified four critical architectural vulnerabilities in the current workflow:

1. **Catastrophic Blacklist Timing Inversion**:
   Feature blacklists are applied *after* correlation filtering. High-fitting, non-stationary blacklisted features (e.g. nominal price pivots, long-period momentum) achieve high composite ranking and eliminate valid, stationary non-blacklisted features during greedy correlation deduplication. Subsequently, 112 out of 150 selected features (74.7%) are dropped by the blacklist, leaving the final candidate feature set severely depleted.
2. **Defunct Stability Filter & Zero Directional Consistency**:
   The stability constraint `RankIC_Std <= 1.0` is mathematically inert because RankIC is bounded in $[-1, 1]$, allowing 682 out of 687 features (99.3%) to pass. Furthermore, the selector lacks cross-contract directional sign consistency checks, allowing spurious signals with opposing predictive directions across contracts to pass.
3. **Absence of Distribution Drift Screening**:
   The pipeline selects features based solely on in-sample return prediction (IC, RankIC, CatBoost importance) without testing whether feature distributions are homogeneous across contracts. Features that drift across contracts in training inevitably explode into OOD events on out-of-sample test splits.
4. **Simpson's Paradox in Multi-Contract Correlation**:
   The correlation filter vertically concatenates raw, unscaled DataFrames from different contracts with disparate baseline price levels (e.g. 2,500 vs. 3,500), creating spurious cross-contract level-shift correlations that distort feature deduplication.
5. **Physical Window Horizon Mismatch**:
   For 10-minute bars, feature generation includes multi-day and multi-week windows ($w=96, 192, 240$, spanning 16 to 40 trading hours) that accumulate macroeconomic multi-year trend drifts, contaminating short-horizon models.

---

## Solution

We overhaul Phase 1 (Stop Bleeding: Steps 1 & 2) and Phase 2 (Distribution & Stability Gates: Steps 3 to 6) through a cohesive, six-step architectural remediation:

1. **Physical Window Truncation (Step 1)**:
   Restrict time operator generation windows for 10-minute sampling frequency to intraday spans ($w \le 48$, $\le 8$ trading hours). Multi-day and multi-week windows ($w \ge 96$) are purged from 10-minute generation and default selection windows, eliminating multi-year macroeconomic drift.
2. **Front-Loaded Blacklist Filtering (Step 2)**:
   Purge blacklisted features and ablation patterns from the candidate feature universe immediately upon initialization, prior to computing metrics, fitting CatBoost regressors, or running correlation deduplication. This eliminates the "borrowed knife" phenomenon and saves ~30% in training runtime.
3. **Multi-Contract Distribution Drift Gate (Step 3)**:
   Introduce a dedicated distribution stability audit calculating pairwise Population Stability Index ($\text{PSI}$) and Kolmogorov-Smirnov statistics across all training contracts on pooled quantile bins. Enforce a hard gate requiring $\overline{\text{PSI}} \le 0.10$ and $\text{PSI}_{\max} \le 0.25$, with an automated safety fallback guard ensuring at least 20 features survive.
4. **Cross-Contract Sign Consistency & RankIC IR Gates (Step 4)**:
   Require directional sign consistency ($\text{SignConsistency} \ge 0.75$, meaning at least 11 of 14 contracts share the same RankIC direction), enforce an active RankIC Information Ratio constraint ($IR_{\text{RankIC}} \ge 0.40$), and raise minimum absolute IC from 0.01 to 0.02.
5. **Contract-Normalized Decentralized Correlation & OOD-Aware Priority (Step 5)**:
   Replace naive vertical concatenation with intra-contract correlation matrix averaging weighted by contract sample sizes. Re-weight composite score priority ($0.40 \cdot \text{Rank}(1/\overline{\text{PSI}}) + 0.35 \cdot \text{Rank}(|\overline{RankIC}|) + 0.25 \cdot \text{Rank}(\text{CatBoost\_Imp})$) so that greedy deduplication at threshold 0.70 actively retains drift-resistant, stationary features over high-risk signals.
6. **Regime Audit Anchor Cross-Regime Variance Bounding (Step 6)**:
   Require conditionally retained market state anchors in regime audit to satisfy cross-regime variance stability ($\sigma^2_{\text{extreme}} / \sigma^2_{\text{neutral}} \le 3.0$) in addition to 90% LCB significance, preventing variance explosions in extreme market states.

---

## User Stories

1. As a quantitative pipeline operator, I want time operator feature generation for 10-minute sampling to constrain windows to intraday spans ($w \le 48$), so that feature generation does not waste compute on multi-day macroeconomic metrics that are later blacklisted.
2. As a feature engineering developer, I want deterministic lifecycle indicators (e.g. `contract_life_remaining_ratio`, `cm_*_open_interest_share_*`) to be strictly segregated from state feature candidates into execution/reward schemas, so that calendar time progress is not mistaken for a stationary trading signal.
3. As a machine learning researcher, I want the feature selection pipeline to purge blacklisted features immediately at candidate pool initialization, so that CatBoost models and IC metrics are computed only on viable features.
4. As an algorithmic trader, I want correlation deduplication to evaluate only unblacklisted features, so that non-stationary blacklisted features cannot eliminate genuine stationary features before being dropped themselves.
5. As a risk management engineer, I want the feature selection pipeline to measure pairwise Population Stability Index ($\text{PSI}$) across all participating training contracts, so that features with structural distribution shifts across contracts are proactively identified.
6. As a machine learning engineer, I want a hard distribution gate enforcing $\overline{\text{PSI}} \le 0.10$ and $\text{PSI}_{\max} \le 0.25$, so that features with significant cross-contract drift are rejected prior to predictive scoring.
7. As a pipeline reliability engineer, I want a safety fallback guard that gracefully relaxes the PSI threshold if fewer than 20 features survive, so that unexpected market regime shifts do not abort the preprocessing pipeline.
8. As a quant researcher, I want distribution audit metrics (including mean PSI, max pairwise PSI, and KS statistics) persisted to a CSV artifact, so that I can inspect the empirical stability distribution of all candidate features.
9. As a reinforcement learning researcher, I want features to demonstrate at least 75% directional sign consistency across training contracts, so that state signals do not reverse their directional relationship with future returns between contracts.
10. As a portfolio manager, I want the stability filter to enforce a minimum RankIC Information Ratio of 0.40, so that noisy features with high variance relative to their mean predictive power are eliminated.
11. As a quant developer, I want the minimum absolute IC threshold raised from 0.01 to 0.02, so that economically and statistically insignificant features are filtered out early.
12. As a statistical modeling engineer, I want multi-contract correlation to be computed within each contract and sample-weighted rather than vertically concatenated, so that Simpson's paradox and baseline price level differences do not produce spurious correlations.
13. As a VAE modeling researcher, I want the greedy correlation filter priority to weight inverse PSI by 40%, so that when two features are collinear, the algorithm consistently preserves the more stationary, drift-resistant feature.
14. As an RL state space architect, I want market state regime anchors evaluated in regime audit to satisfy an extreme-to-neutral variance ratio of $\le 3.0$, so that conditionally retained anchors do not inject variance explosions into downstream state vectors during extreme market regimes.
15. As a pipeline maintainer, I want `feature_selection_manifest.json` to record distribution drift audit parameters, sign consistency thresholds, and anchor variance statistics, so that every feature selection run is fully reproducible and auditable.
16. As an automated test developer, I want comprehensive unit tests covering PSI calculation, sign consistency filtering, contract-normalized correlation matrices, and anchor variance bounding, so that regressions are caught automatically in CI.
17. As a system operator, I want all modifications to preserve backward compatibility for downstream feather/csv dataset schemas and CLI interfaces, so that Stage 2 data preparation and RL training scripts require zero breaking changes.

---

## Implementation Decisions

### 1. Architectural Placement and Seams
- **Primary Integration Seam**:
  The primary entry point and integration seam is `run_feature_selection` in the multi-contract feature selection module. This single interface orchestrates candidate ingestion, front-loaded blacklist pruning, distribution drift auditing, predictive stability filtering, contract-normalized correlation deduplication, and conditional anchor retention.
- **Component Audit Module**:
  Create a dedicated distribution audit module within the multi-contract package implementing `audit_distribution_drift`. This module is pure-functional, receiving multi-contract Polars DataFrames and candidate feature names, and returning per-feature stability metrics and surviving feature lists.
- **Correlation Utility Refactoring**:
  Update the correlation selection utility to accept either a precomputed contract-weighted correlation matrix or compute it via contract-wise sample-weighted aggregation, completely removing vertical DataFrame concatenation of unscaled series.

### 2. Algorithmic Specifications

#### A. Front-Loaded Candidate Ingestion
- Immediately after reading contract frames, intersect the candidate feature universe with `feature_blacklist` and `feature_ablation_patterns`.
- Remove all blacklisted and ablated columns from `candidate_universe`.
- Record `Feature Blacklist Dropped` and `Feature Ablation Dropped` directly into the selection manifest at the initial ingestion gate.
- Evaluate metric frames (CatBoost, IC, RankIC, Sharpe, Permutation Importance) strictly on the pruned `candidate_universe`.

#### B. Distribution Drift Audit Engine
- For each candidate feature, calculate its global empirical distribution by pooling across mature observations of all participating training contracts.
- Determine 10 quantile bin edges $[q_0, q_1, \dots, q_{10}]$ from the pooled data.
- For every pair of contracts $(C_i, C_j)$, compute the empirical bin probability vectors $P_i$ and $P_j$:
  $$\text{PSI}(C_i, C_j) = \sum_{k=1}^{10} (P_{i,k} - P_{j,k}) \cdot \ln\left(\frac{P_{i,k} + \epsilon}{P_{j,k} + \epsilon}\right)$$
  where $\epsilon = 10^{-6}$.
- Compute mean pairwise $\overline{\text{PSI}}(f)$ and maximum pairwise $\text{PSI}_{\max}(f)$.
- Candidate feature passes the distribution gate if and only if:
  $$\overline{\text{PSI}}(f) \le \text{max\_mean\_psi} \ (0.10) \quad \text{and} \quad \text{PSI}_{\max}(f) \le \text{max\_pair\_psi} \ (0.25)$$
- If the count of surviving features is less than `min_drift_survivors` (default: 20), trigger the safety fallback guard: relax thresholds to $\overline{\text{PSI}} \le 0.15$ and log a diagnostic warning while continuing the pipeline.

#### C. Cross-Contract Sign Consistency and RankIC Information Ratio
- Compute directional sign consistency across contracts on the primary decision window:
  $$\text{SignConsistency}(f) = \frac{\max(\sum_{c=1}^N \mathbb{I}(RankIC_c > 0), \sum_{c=1}^N \mathbb{I}(RankIC_c < 0))}{N}$$
- Reject any feature with $\text{SignConsistency}(f) < 0.75$.
- Compute RankIC Information Ratio across contracts:
  $$IR_{\text{RankIC}}(f) = \frac{|\overline{RankIC}(f)|}{\text{std}(RankIC) + 10^{-6}}$$
- Enforce $IR_{\text{RankIC}}(f) \ge 0.40$ (replacing the legacy inactive `RankIC_Std <= 1.0`).
- Enforce $|\overline{RankIC}(f)| \ge 0.02$ (elevated from 0.01).

#### D. Contract-Normalized Correlation Matrix & Priority
- For each training contract $C_c$, calculate the intra-contract correlation matrix $R_c = \text{Corr}(X_c)$ across candidate features passing previous gates.
- Aggregate matrices using sample-length weights $w_c = T_c / \sum_{i=1}^N T_i$:
  $$\bar{R} = \sum_{c=1}^N w_c R_c$$
- Compute composite priority ranking:
  $$\text{Priority}(f) = 0.40 \cdot \text{Rank}\left(\frac{1}{\overline{\text{PSI}}(f) + 10^{-4}}\right) + 0.35 \cdot \text{Rank}(|\overline{RankIC}(f)|) + 0.25 \cdot \text{Rank}(\text{CatBoost\_Imp}(f))$$
- Execute greedy elimination on $\bar{R}$ with threshold $0.70$ in descending order of $\text{Priority}(f)$.

#### E. Regime Audit Anchor Variance Bounding
- In market state regime audit, for each conditionally retained anchor candidate passing 90% LCB in target bins, compute variance in extreme regime bins ($\sigma^2_{\text{extreme}}$) versus neutral bins ($\sigma^2_{\text{neutral}}$).
- Disallow retention if $\sigma^2_{\text{extreme}} / \sigma^2_{\text{neutral}} > 3.0$.

---

## Testing Decisions

### What Makes a Good Test
- **Behavioral and Contract-Driven**: Tests must verify external input-output contracts (filtered feature subsets, generated manifest fields, artifact existence, and mathematical invariants) rather than inspecting private implementation trivia.
- **Deterministic and Fast**: Tests use synthetic multi-contract Polars DataFrames with controlled price paths, known drifts, and seeded distributions, running entirely in-memory within seconds.
- **Failure Mode Coverage**: Tests must explicitly verify gate rejection behaviors (e.g. rejection when PSI exceeds 0.10, rejection when sign consistency is below 75%, rejection when anchor variance explodes).

### Target Test Suites
1. `test_distribution_audit.py`:
   - Validates that two identical distributions yield $\text{PSI} \approx 0.0$.
   - Validates that artificially shifted mean or variance across contracts generates high PSI and is correctly rejected by the gate.
   - Validates safety fallback relaxation when too few features survive.
2. `test_commodity_multi_contract_feature_selection.py`:
   - Updates existing end-to-end tests to verify that front-loaded blacklists prevent blacklisted features from ever entering metric frames or correlation filters.
   - Verifies that sign consistency rejects alternating sign features.
   - Verifies that contract-normalized correlation does not produce spurious collinearity between contracts with disparate price baselines.
3. `test_regime_audit_and_anchor_retention.py`:
   - Verifies that extreme-to-neutral variance ratio gating prevents volatile anchors from conditional retention.

### Prior Art
- `data_preprocess/tests/test_commodity_multi_contract_feature_selection.py`: Existing end-to-end test harnesses for multi-contract pipeline execution.
- `data_preprocess/tests/test_regime_audit_and_anchor_retention.py`: Existing regime audit and conditional anchor unit tests.

---

## Out of Scope

- **Phase 3 (Dual-Stream Decoupling)**: Splitting downstream outputs into `vae_features.npy` versus `rl_features.npy` is deferred to Phase 3.
- **Phase 3 (Adaptive Scale Save)**: Refactoring `muti_contract_scale_save.py` with rolling Z-scores and hyperbolic tangent soft clamping is deferred to Phase 3.
- **Model Architecture Alterations**: No changes to VAE decoder architectures, RL Qnet heads, or high-level heuristic routing algorithms.
- **Historical Raw Data Downscaling**: Re-running raw downscaling from tick archives is not required; changes begin from feature engineering time operators onward.

---

## Further Notes

- **CLI Compatibility**: All existing CLI flags (`--symbol`, `--target_freq`, `--stage`, `--orderbook_depth`, `--feature_blacklist`) remain fully backward-compatible. New parameters (`--max_mean_psi`, `--min_sign_consistency`, `--min_rank_ic_ir`) provide sensible, conservative defaults matching this specification.
- **Downstream Safety**: The primary output artifact `state_features.npy` retains identical numpy array format, guaranteeing seamless ingestion by Stage 2 data preparation scripts.
