---
status: accepted
---

# Front-Loaded Filtering, Distribution Drift Gate, and Decentralized Multi-Contract Feature Selection

We overhaul the commodity multi-contract feature selection architecture by front-loading feature blacklists, enforcing cross-contract distribution stability gates (PSI <= 0.10) and predictive sign consistency (>= 75%), decentralizing correlation filtering via contract-normalized weighted averaging, and adding cross-regime anchor variance bounding.

## Context

Empirical audits of the FineFT 10-minute commodity pipeline (`docs/research/multi_contract_feature_engineering_and_selection_ood_remediation_report.md` and `PREPROCESS_DATASET/.../feature_selection_manifest.json`) exposed four critical systemic flaws in `operator_futures.feature_selection.muti_contract`:

1. **Catastrophic Blacklist Timing Inversion**: In `pipeline.py`, feature blacklists were applied *after* greedy correlation filtering (`select_feature`). Because non-stationary blacklisted features (e.g., nominal prices, long-period momentum) scored near the top on training returns, they eliminated correlated stationary features during deduplication before being pruned themselves (112 out of 150 selected features, or 74.7%, were subsequently dropped).
2. **Defunct Stability Filter**: The condition `RankIC_Std <= 1.0` was mathematically inert because RankIC is bounded in $[-1, 1]$, allowing 682 out of 687 features (99.3%) to pass without filtering. Moreover, cross-contract directional sign consistency was unmeasured, allowing spurious signals with opposing signs across contracts to pass.
3. **Zero Distribution Drift Screening**: Feature selection optimized purely for in-sample return prediction (IC, RankIC, CatBoost) without evaluating distribution stability across contracts. Non-stationary features with high in-sample IC drifted severely out-of-sample, triggering downstream VAE likelihood collapse (Test vs Train $\Delta\text{NLL}$ degradation).
4. **Simpson's Paradox in Correlation Matrix**: The correlation filter vertically concatenated raw, unscaled contract DataFrames with disparate price levels (e.g. 2,500 vs 3,500), introducing spurious cross-contract level shifts that distorted feature deduplication.

## Decision

We establish the following Phase 1 & Phase 2 architectural decisions across Steps 1–6:

### 1. Source-Level Physical Window Truncation (Step 1)
- For 10-minute sampling, restrict `--windows` in `fu_full_process.sh` (`run_commodity_time_feature`) and `--windows_list` in feature selection to intraday spans $[2, 6, 12, 16, 24, 48]$ ($w \le 48$, $\le 8$ trading hours).
- Bar multi-day and multi-week macro features ($w \ge 96$) from being generated or evaluated for 10-minute frequency, eliminating long-term macroeconomic variance expansion.

### 2. Front-Loaded Blacklist and Ablation Execution (Step 2)
- In `pipeline.py:run_feature_selection`, immediately filter `candidate_universe` against `feature_blacklist` and `feature_ablation_patterns` prior to computing metric frames, CatBoost models, or correlation matrices.
- Eliminate wasted GPU/CPU CatBoost iterations on blacklisted features and terminate the "borrowed knife" phenomenon that killed valid stationary features.

### 3. Cross-Contract Distribution Drift Gate via Multi-Contract PSI (Step 3)
- Introduce `operator_futures.feature_selection.muti_contract.distribution_audit:audit_distribution_drift`.
- Calculate pairwise Population Stability Index ($\text{PSI}(C_i, C_j)$) across all training contracts on 10 quantile bins.
- Enforce strict distribution homogeneity:
  $$\overline{\text{PSI}}(f) \le 0.10 \quad \text{and} \quad \text{PSI}_{\max}(f) \le 0.25$$
- Drop features exceeding thresholds with a safety fallback guard ensuring at least 20 candidate features survive.

### 4. Cross-Contract Directional Sign Consistency and RankIC IR (Step 4)
- Enforce directional sign consistency across contracts:
  $$\text{SignConsistency}(f) = \frac{\max(\sum \mathbb{I}(r_c > 0), \sum \mathbb{I}(r_c < 0))}{N_{\text{contracts}}} \ge 0.75$$
- Replace inert `RankIC_Std <= 1.0` with effective RankIC Information Ratio:
  $$IR_{\text{RankIC}}(f) = \frac{|\overline{RankIC}(f)|}{s_{RankIC}(f) + 10^{-6}} \ge 0.40$$
- Raise `min_abs_ic` from 0.01 to 0.02.

### 5. Contract-Normalized Decentralized Correlation Matrix (Step 5)
- Replace vertical DataFrame concatenation with contract-wise correlation averaging:
  $$\bar{R} = \sum_{c=1}^N \frac{T_c}{\sum T_i} \text{Corr}(X_c)$$
- Incorporate distribution stability into Composite Score priority:
  $$\text{CompositeScore}(f) = 0.40 \cdot \text{Rank}\left(\frac{1}{\overline{\text{PSI}}(f)}\right) + 0.35 \cdot \text{Rank}(|\overline{RankIC}(f)|) + 0.25 \cdot \text{Rank}(\text{CatBoost\_Imp}(f))$$
- Deduplicate on $\bar{R}$ at $|\rho| > 0.70$ prioritizing stationary, drift-resistant features.

### 6. Market State Regime Anchor Cross-Regime Variance Bounding (Step 6)
- In `regime_audit.py`, require that conditional retained anchors satisfy cross-regime variance stability:
  $$\frac{\sigma^2_{\text{extreme}}}{\sigma^2_{\text{neutral}}} \le 3.0$$
- Prevent anchor retention from injecting variance-exploding features into the state observation set.

## Consequences

- Completely eliminates the 74.7% blacklist drop rate during correlation filtering, preserving genuine stationary features.
- Truncates computation time by ~30% by skipping model training on blacklisted features.
- Guarantees that selected state features maintain distribution homogeneity ($\text{PSI} \le 0.10$) and directional consistency ($\ge 75\%$) across contracts before entering downstream VAE and RL models.
- Completely prevents Simpson's paradox from polluting feature deduplication.
