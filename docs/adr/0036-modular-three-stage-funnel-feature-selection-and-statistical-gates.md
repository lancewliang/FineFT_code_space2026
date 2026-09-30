---
status: accepted
---

# Modular Three-Stage Funnel Feature Selection and Advanced Statistical Gates

We restructure `operator_futures.feature_selection.muti_contract` from a monolithic God pipeline into a decoupled three-stage hierarchical funnel composed of deep modules, resequence statistical gates ahead of machine learning regressors, focus CatBoost on the single target decision window with purged embargo gaps, upgrade correlation deduplication to Spearman rank clustering, enforce within-contract ADF unit-root stationarity, and preserve raw split data immutability.

## Context

Following the Phase 1 & 2 remediation (ADR-0035, commit `4fe40e0`), our code and statistical audits (`docs/research/feature_selection_pipeline_gap_and_flow_optimization_research.md`) identified major architectural bottlenecks in `pipeline.py` (860 lines):

1. **Computational Cost Inversion**: CatBoost was trained across 6 time horizons ($w \in [1, 2, 6, 12, 24, 48]$) for all 14 contracts ($14 	imes 6 = 84$ models) on the full candidate pool (~800–1000 features), accounting for >85% of execution time. Subsequently, cheap linear and rank filters (`Hard RankIC >= 0.02`, `SignConsistency >= 0.75`, `RankIC_IR >= 0.40`) eliminated 70%+ of these features. Training heavy nonlinear models on features failing basic linear and directional criteria was computationally wasteful.
2. **Early Stopping Label Overlap Leakage**: In `metrics.py`, contiguous 80/20 train/eval splitting leaked future price trajectory across the split boundary for forward return horizons $w$, distorting early stopping.
3. **Pre-Selection Data Hygiene Gap**: Outlier clipping occurred in Stage 4 (`scale_save`) after feature selection. Raw outliers distorted Pearson correlation matrices in `cor_util.py` (leverage points) and compressed quantile binning in `distribution_audit.py`.
4. **Missing Financial & Statistical Gates**: Multi-contract PSI only verified cross-sectional marginal homogeneity, failing to detect temporal non-stationarity (unit roots / random walks) within contracts. Multicollinearity was bounded only pairwise ($|r| > 0.70$), leaving linear combinations unpruned ($	ext{VIF} > 25$).
5. **Monolithic Code Coupling**: `pipeline.py` mixed I/O, data validation, metrics computation, regex ablation, filter ladders, persistence diagnostics, and manifest formatting without internal seams, making isolated unit testing impossible.

## Decision

We establish the following architectural decisions:

### 1. Two-Phase Progressive Deepening Strategy
- **Phase 2.5 (Code Architecture & Funnel Resequencing)**:
  - Decompose `pipeline.py` into dedicated deep modules (`types.py`, `io_manager.py`, `data_hygiene.py`, `predictive_audit.py`, `nonlinear_scoring.py`, `orthogonal_dedup.py`).
  - Resequence the pipeline into a Three-Stage Funnel: execute fast vectorized statistical gates *before* CatBoost fitting.
  - Focus CatBoost training strictly on `--target_decision_window` (default: 6), eliminating redundant multi-window training.
  - Insert a Purged Embargo Gap equal to horizon $w$ between CatBoost train and eval pools.
  - Upgrade correlation deduplication to Spearman rank correlation.
  - Maintain 100% backward-compatible adapter interfaces and green test suites.
- **Phase 3.5 (Advanced Statistical & Financial Gates)**:
  - Mount Near-Zero Variance (NZV), within-contract ADF stationarity testing with automated fallback safeguarding, Ward hierarchical clustering, and False Discovery Rate (FDR) control.

### 2. Three-Stage Funnel Architecture
```
Stage 1: Vectorized Fast Filters (O(1) ~ O(N))
  └── Near-Zero Variance -> Blacklist/Ablation -> Pre-Winsorization -> PSI Drift -> ADF Stationarity -> Vectorized RankIC / Sign / IR Gates
       │
       ▼ (Surviving features: ~150-200)
Stage 2: Target-Horizon Nonlinear Scoring (O(ML))
  └── Single Decision Horizon (w=6) CatBoost with Purged Embargo Gap -> Anti-Causality Screen -> Composite Priority Scoring
       │
       ▼ (Surviving features: ~100)
Stage 3: Orthogonal Representation & Regime Audit
  └── Spearman Rank Correlation + Ward Hierarchical Clustering (VIF <= 10.0) -> Regime Audit Variance Ratio Gating
       │
       ▼ (Final selected features: ~50-70)
```

### 3. Modular Deepening & Seam Specifications
- **`types.py`**: Immutable configuration dataclasses (`DataHygieneConfig`, `StationarityAuditConfig`, `PredictiveAuditConfig`, `NonlinearScoringConfig`, `OrthogonalDedupConfig`, `FeatureSelectionPipelineConfig`) and standardized output contract `PipelineStepResult`.
- **`data_hygiene.py`**: Encapsulates zero-variance, mode frequency, and $5	imes	ext{IQR}$ Winsorization.
- **`stationarity_audit.py`**: Encapsulates Augmented Dickey-Fuller (ADF) unit-root testing ($p < 0.05$ on $\ge 70\%$ of contracts with a 25-feature fallback floor) and persistence half-life / SAR constraints.
- **`predictive_audit.py`**: Computes vectorized forward returns, IC, RankIC, cross-contract Sign Consistency, RankIC IR, and FDR correction.
- **`nonlinear_scoring.py`**: Trains purged-embargo CatBoost on the decision window and computes composite priority:
  $$	ext{Priority}(f) = 0.40 \cdot 	ext{Rank}\left(rac{1}{\overline{	ext{PSI}}(f)}
ight) + 0.35 \cdot 	ext{Rank}(|\overline{	ext{RankIC}}(f)|) + 0.25 \cdot 	ext{Rank}(	ext{CatBoost\_Imp}(f))$$
- **`orthogonal_dedup.py`**: Computes contract-normalized Spearman rank correlation and performs Ward hierarchical clustering with `--dedup_method {cluster, greedy}` compatibility.
- **`io_manager.py`**: Handles Polars IPC I/O, schema checks, and manifest generation.
- **`pipeline.py`**: Reduced to a pure coordinator (< 120 lines). `run_feature_selection` serves as a backward-compatible adapter.

### 4. Data Hygiene Scope & Immutability Guarantee
- Pre-selection Winsorization operates strictly in-memory on loaded Polars DataFrames during feature selection. Source feather files in `SPLIT-TRAIN-VALID-TEST` remain strictly immutable raw data. Stage 4 `scale_save` retains sole ownership over downstream feature scaling.

### 5. Detailed Execution Parameters and Compatibility Guarantees
- **Multi-Window Diagnostic Profiles**: Multi-window vectorized metrics (`IC`, `RankIC`, `Sharpe`) continue to be calculated across all evaluation windows ($w \in [1, 2, 6, 12, 24, 48]$) to maintain visibility into factor half-life decay curves in `{contract}_metrics.csv` and `aggregate_metrics.csv`. `CatBoost Importance` is evaluated exclusively on the target decision window ($w=6$), with non-decision window rows populated with 0.0.
- **Dynamic Cluster Capacity Constraint**: In `orthogonal_dedup.py`, Ward hierarchical clustering enforces a dynamic target capacity $K \in [50, 70]$. If initial distance cutting yields $> 70$ clusters, branches are merged hierarchically to $K=70$; if $< 50$ clusters are produced, intra-cluster selection retains the top-2 scoring features per cluster, stabilizing downstream VAE and RL state dimensions.
- **Forward Outpost Sampling**: Forward boundary distribution drift ($	ext{PSI}_{	ext{forward}}$) reads exclusively the first chronological validation contract ($C_{	ext{valid\_1}}$) as an outpost. Only feature columns are loaded without reference prices or returns, precluding lookahead leakage.
- **Public Adapter Compatibility**: `pipeline.py:run_feature_selection` retains full signature and keyword-argument backward compatibility, dynamically constructing a `FeatureSelectionPipelineConfig` when legacy parameters are passed.

## Consequences

- Reduces feature selection compute time by 75%–85% through feature pruning before ML fitting and single-window CatBoost focus.
- Eliminates label overlap leakage in CatBoost early stopping via purged embargo gaps.
- Replaces brittle bivariate greedy correlation with robust rank-based hierarchical clustering, resolving multi-collinear cliques.
- Modular decomposition establishes clean seams and test surfaces, enabling isolated, sub-second in-memory unit tests for every filter step.
- Preserves full CLI and pipeline backward compatibility.
