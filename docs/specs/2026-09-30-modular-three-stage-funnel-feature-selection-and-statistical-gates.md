# Spec: Modular Three-Stage Funnel Feature Selection and Advanced Statistical Gates

- **Triage Label**: `ready-for-agent`
- **Related Research**: `docs/research/feature_selection_pipeline_gap_and_flow_optimization_research.md`
- **Related ADR**: `docs/adr/0036-modular-three-stage-funnel-feature-selection-and-statistical-gates.md`
- **Target Subsystems**: Multi-Contract Feature Selection Pipeline, Statistical Stationarity Gates, Nonlinear Regressor Scoring, Orthogonal Clustering Deduplication, Pipeline I/O & Manifest Infrastructure

---

## Problem Statement

Following the Phase 1 & 2 remediation, the multi-contract feature selection system successfully eliminated catastrophic post-correlation blacklist drops, Simpson's paradox in cross-contract correlation, and exploding regime anchors. However, empirical profiling and architectural code auditing have exposed critical structural bottlenecks in the existing pipeline:

1. **Computational Cost Inversion**: The pipeline fits heavy gradient-boosted decision trees across 6 evaluation windows for all training contracts on the raw candidate pool (~800–1000 features) before applying fast linear and rank filters. This consumes over 85% of total feature selection compute time, only for cheap subsequent statistical gates (Hard RankIC, Sign Consistency, and Stability IR) to instantly discard more than 70% of those features.
2. **Label Overlap Leakage in Model Evaluation**: In time series regressor early stopping, contiguous 80/20 train/eval splitting leaks forward return labels across the split boundary for multi-bar horizons, causing early stopping to overfit to unpurged future price movements.
3. **Outlier Distortion Before Selection**: Robust outlier clipping is deferred to downstream scaling after feature selection has concluded. Unclipped raw outliers distort covariance matrices (inducing leverage points that create spurious collinearity) and squash quantile binning during population stability index evaluation.
4. **Absence of Temporal Stationarity and High-Order Multicollinearity Controls**: Cross-contract distribution drift checks only evaluate cross-sectional marginal homogeneity, failing to test whether time series within each contract contain unit roots or random walk drifts. Additionally, bivariate greedy correlation pruning misses high-order linear combinations across three or more features, leaving condition numbers ill-conditioned.
5. **Monolithic Code Coupling**: The primary pipeline script spans nearly 900 lines and combines disk I/O, data validation, metrics computation, regex ablation, filter ladders, persistence diagnostics, and manifest formatting without internal seams, preventing modular extension and isolated unit testing.

---

## Solution

We restructure the multi-contract feature selection subsystem into an explicit **Three-Stage Hierarchical Funnel** composed of decoupled, highly cohesive deep modules:

1. **Funnel Resequencing**: Front-load vectorized $O(1) \sim O(N)$ statistical gates (Near-Zero Variance, In-Memory Winsorization, Distribution Drift PSI, ADF Unit-Root Stationarity, and Vectorized RankIC / Sign Consistency / IR) *before* invoking any machine learning models.
2. **Target-Horizon Regressor Fitting with Purged Embargo**: Restrict gradient-boosted decision tree fitting strictly to the target decision horizon (e.g. 6 bars) rather than across 6 redundant horizons, and enforce a purged embargo buffer equal to the forward horizon between train and evaluation pools to eliminate lookahead leakage.
3. **Pre-Selection Data Hygiene**: Apply in-memory double-sided Winsorization during Stage 1 while preserving the absolute immutability of raw split files on disk.
4. **Orthogonal Representation via Rank Clustering**: Replace greedy bivariate correlation pruning with contract-normalized Spearman rank correlation and Ward minimum-variance hierarchical clustering constrained by dynamic capacity ($K \in [50, 70]$) and variance inflation factor checks.
5. **Modular Architectural Deepening**: Decompose the monolithic pipeline into dedicated deep modules presenting minimal interfaces, while preserving full backward compatibility for existing callers through an adapter pattern.

---

## User Stories

1. As a quantitative researcher, I want cheap statistical filters (RankIC, Sign Consistency, and Stability IR) evaluated before running machine learning regressors, so that feature selection completes in minutes rather than hours.
2. As a reinforcement learning engineer, I want the feature selector to enforce Augmented Dickey-Fuller unit-root stationarity on candidate series, so that downstream policy networks receive strictly stationary Markovian state observations.
3. As an RL agent, I want state observations to have a bounded action turnover rate and minimum half-life, so that policy execution does not generate excessive microstructural churn that is destroyed by transaction friction.
4. As a machine learning engineer, I want gradient-boosted decision tree early stopping to include a purged embargo gap matching the forward return horizon, so that early stopping decisions are untainted by future label overlap.
5. As a quantitative researcher, I want gradient-boosted decision tree feature importance evaluated exclusively on the primary decision window, so that model compute is focused directly on the trading execution horizon.
6. As a data engineer, I want multi-contract raw feature data in storage splits to remain strictly immutable during feature selection, so that downstream modules always access uncorrupted source files.
7. As a quantitative researcher, I want pre-selection Winsorization applied in-memory during feature selection, so that fat-tailed outliers cannot distort correlation matrices or quantile bins.
8. As a risk controller, I want a forward boundary distribution drift gate comparing training distribution against the earliest validation contract outpost, so that macroscopically drifting features are blocked prior to live agent training.
9. As a data engineer, I want the forward boundary drift gate to read strictly feature columns without reference prices or returns, so that zero forward return leakage is introduced into the selection process.
10. As a portfolio manager, I want collinear features deduplicated via Spearman rank correlation and Ward hierarchical clustering rather than greedy elimination, so that feature selection is invariant to monotonic transformations and independent of feature evaluation order.
11. As a reinforcement learning practitioner, I want hierarchical clustering to enforce a dynamic feature capacity between 50 and 70 features, so that downstream representation networks receive consistent, stable dimensionality regardless of commodity volatility regimes.
12. As a machine learning engineer, I want retained feature clusters checked for Variance Inflation Factors below 10.0, so that high-order multicollinearity cliques do not destabilize linear and neural network weights.
13. As a researcher, I want candidate features evaluated under Benjamini-Hochberg False Discovery Rate control, so that multiple-testing data snooping false discoveries are bounded below 5%.
14. As a developer, I want near-zero variance and quasi-constant features rejected early in the pipeline, so that degenerate tree splits and division-by-zero scaling blowups are prevented.
15. As a QA engineer, I want each filtering step implemented in its own deep module behind a standardized result interface, so that every filter can be unit-tested in isolation using fast in-memory synthetic DataFrames.
16. As a systems engineer, I want the primary feature selection pipeline entrypoint to retain full backward compatibility with existing command-line arguments and shell scripts, so that automated daily preprocessing jobs continue running without disruption.
17. As an ML researcher, I want factor diagnostic matrices to retain multi-window IC and RankIC decay profiles, so that signal decay curves remain visible even when nonlinear tree fitting is restricted to the decision horizon.
18. As a risk controller, I want stationarity testing to include an automated fallback safeguard ensuring at least 25 features survive, so that unexpected market volatility cannot starve the feature pool.

---

## Implementation Decisions

### 1. Phased Delivery Roadmap
The implementation is partitioned into two clear releases:
- **Phase 2.5 (Code Architecture & Funnel Resequencing)**: Extract monolithic pipeline logic into dedicated deep modules, resequence the Three-Stage Funnel (moving statistical gates before regressor fitting), focus CatBoost on the decision window with purged embargo gaps, and upgrade correlation deduplication to Spearman rank correlation. All existing test suites must pass without regression.
- **Phase 3.5 (Advanced Statistical & Financial Gates)**: Implement Near-Zero Variance filtering, ADF unit-root stationarity with fallback safeguards, Ward hierarchical clustering, Forward Outpost PSI, and False Discovery Rate control.

### 2. Three-Stage Funnel Resequencing
The selection pipeline executes sequentially through three distinct stages:
- **Stage 1 (Vectorized Fast Filters, $O(1) \sim O(N)$)**:
  - Near-zero variance and mode-frequency filtering.
  - Front-loaded feature blacklist and regex ablation.
  - In-memory 5-sigma / $5	imes	ext{IQR}$ Winsorization.
  - Pairwise multi-contract distribution drift gate (PSI $\le 0.10$, $	ext{PSI}_{\max} \le 0.25$).
  - Forward boundary drift gate against the earliest validation outpost ($	ext{PSI}_{	ext{forward}} \le 0.15$).
  - Within-contract ADF unit-root stationarity testing ($p_{	ext{ADF}} < 0.05$ on $\ge 70\%$ of contracts, with relaxation to $p < 0.10$ and minimum 25 survivor floor).
  - Persistence half-life ($	au_{1/2} \ge 2.0$ bars) and sign alternation rate ($	ext{SAR} \le 0.40$).
  - Vectorized forward returns and RankIC / Sign Consistency ($\ge 0.75$) / RankIC IR ($\ge 0.40$) / FDR control ($Q \le 0.05$).
- **Stage 2 (Target-Horizon Nonlinear Scoring, $O(ML)$)**:
  - Evaluated strictly on the surviving subset (~150–200 features).
  - Fit CatBoostRegressor exclusively on the designated target decision window (default: 6 bars).
  - Enforce a Purged Embargo Gap equal to horizon $w$ between the 80% train split and 20% early stopping eval split.
  - Compute Composite Priority Score:
    $$	ext{Priority}(f) = 0.40 \cdot 	ext{Rank}\left(rac{1}{\overline{	ext{PSI}}(f)}ight) + 0.35 \cdot 	ext{Rank}(|\overline{	ext{RankIC}}(f)|) + 0.25 \cdot 	ext{Rank}(	ext{CatBoost\_Imp}(f))$$
  - Discard the bottom 10% lowest-scoring candidates.
- **Stage 3 (Orthogonal Representation & Regime Audit)**:
  - Compute contract-normalized weighted Spearman rank correlation matrix.
  - Perform Ward minimum-variance hierarchical clustering to group features into semantic clusters, enforcing dynamic capacity $K \in [50, 70]$. Select the highest-priority feature per cluster.
  - Verify Variance Inflation Factor $	ext{VIF} \le 10.0$ on selected representatives.
  - Run market state regime audit and enforce cross-regime variance bounding ($\sigma^2_{	ext{extreme}} / \sigma^2_{	ext{neutral}} \le 3.0$) on conditionally retained anchors.

### 3. Modular Architecture and Seam Definitions
The pipeline package is decomposed into high-cohesion deep modules:
- **`types`**: Encapsulates immutable configuration dataclasses and the standardized step result contract:
  ```python
  @dataclass(frozen=True)
  class PipelineStepResult:
      step_name: str
      surviving_features: list[str]
      dropped_features: list[str]
      audit_metrics_df: pl.DataFrame | None = None
      diagnostics: dict[str, float | str | bool | list[str]] = field(default_factory=dict)
  ```
- **`data_hygiene`**: Encapsulates zero-variance checks ($\sigma^2 > 10^{-6}$), mode-frequency thresholds ($< 0.98$), feature blacklist/ablation, and in-memory Winsorization ($5	imes	ext{IQR}$).
- **`distribution_audit`**: Encapsulates pairwise multi-contract PSI, Kolmogorov-Smirnov statistics, adaptive zero-isolated quantile binning, and forward boundary outpost drift testing.
- **`stationarity_audit`**: Encapsulates Augmented Dickey-Fuller tests across contracts, fallback relaxation, lag-1 autocorrelation half-life, and sign alternation rate checks.
- **`predictive_audit`**: Encapsulates vectorized multi-window forward returns, IC, RankIC, cross-contract Sign Consistency, RankIC IR, anti-causality anomaly screening, and Benjamini-Hochberg FDR correction.
- **`nonlinear_scoring`**: Encapsulates purged-embargo CatBoost fitting on the target decision window and composite priority scoring.
- **`orthogonal_dedup`**: Encapsulates contract-normalized Spearman rank correlation, Ward hierarchical clustering, dynamic capacity calibration ($50 \le K \le 70$), and VIF checks, while supporting a legacy greedy deduplication fallback flag (`--dedup_method`).
- **`io_manager`**: Encapsulates Polars IPC frame loading, schema validation, filtered dataset writing, and manifest serialization.
- **`pipeline`**: High-level coordinator reduced to under 120 lines, implementing `run_feature_selection` as an adapter accepting legacy keyword arguments or a unified pipeline configuration dataclass.

### 4. Diagnostic Metric Retention
`metrics.csv` and `aggregate_metrics.csv` retain all evaluation windows ($w \in [1, 2, 6, 12, 24, 48]$) for vectorized linear metrics (`IC`, `RankIC`, `Sharpe`) to preserve signal decay profiling. For `CatBoost Importance`, non-decision windows are populated with 0.0, and the decision window holds the genuine model importance.

---

## Testing Decisions

### What Makes a Good Test
- **Pure Behavioral Verification**: Tests must verify inputs, outputs, invariant constraints, filtered subsets, and manifest records without probing private functions or implementation minutiae.
- **Sub-Second Fast Execution**: Individual module tests must run entirely in memory using synthetic Polars DataFrames with deterministically seeded price and indicator paths, completing within tens of milliseconds.
- **Direct Seam Testing ("Interface is the Test Surface")**: Each newly extracted deep module must have a corresponding dedicated unit test exercising its public interface directly.
- **Failure Mode & Boundary Guard Coverage**: Explicit test cases must verify boundary rejections (e.g. rejection of random walk unit roots, rejection of quasi-constant columns, rejection of label-overlap leakage, and trigger of stationarity fallback relaxation).

### Target Test Suites
1. **`test_data_hygiene.py`**:
   - Verifies rejection of columns with variance $< 10^{-6}$.
   - Verifies rejection of quasi-constant columns where mode frequency $\ge 0.98$.
   - Verifies that in-memory Winsorization clamps extreme spikes to $5	imes	ext{IQR}$ without mutating source files.
2. **`test_stationarity_audit.py`**:
   - Verifies that a stationary mean-reverting series ($I(0)$) passes the ADF gate ($p < 0.05$).
   - Verifies that a cumulative random walk series ($I(1)$) is rejected.
   - Verifies that the automated fallback safeguard relaxes threshold when surviving feature count drops below 25.
3. **`test_predictive_audit.py`**:
   - Verifies vectorized RankIC, Sign Consistency, and IR calculation across multiple synthetic contract frames.
   - Verifies that anti-causal coincident price indicators with extreme IC ($|IC| > 0.30$) are caught.
   - Verifies Benjamini-Hochberg FDR control on synthetic p-value arrays.
4. **`test_nonlinear_scoring.py`**:
   - Verifies that CatBoost trains only on the designated decision window.
   - Verifies that the purged embargo gap of $w$ bars separates train and eval pools.
   - Verifies composite priority ranking and bottom 10% truncation.
5. **`test_orthogonal_dedup.py`**:
   - Verifies that Spearman rank correlation is invariant to monotonic nonlinear shifts.
   - Verifies that Ward hierarchical clustering collapses collinear cliques into single representatives within capacity $K \in [50, 70]$.
   - Verifies backward-compatible operation under `--dedup_method greedy`.
6. **`test_commodity_multi_contract_feature_selection.py` (Regression Harness)**:
   - Verifies that the slimmed-down `pipeline.py` orchestrator and its backward-compatible adapter pass all 33 existing end-to-end tests without regression.

### Prior Art
- `data_preprocess/tests/test_commodity_multi_contract_feature_selection.py`: Existing end-to-end multi-contract integration harness.
- `data_preprocess/tests/test_distribution_audit.py`: Reference pattern for distribution drift tests.
- `data_preprocess/tests/test_regime_audit_and_anchor_retention.py`: Reference pattern for regime audit variance ratio tests.

---

## Out of Scope

- **Phase 3 Dual-Stream Decoupling**: Partitioning feature selection outputs into separate VAE representation features (`vae_features.npy`) and RL policy decision features (`rl_features.npy`) is deferred to Phase 3.
- **Stage 4 Scale Save Modifications**: Modifying `muti_contract_scale_save.py` (e.g. rolling z-scores or hyperbolic tangent soft clamping) is deferred to Phase 3.
- **Model Architecture Adjustments**: Modifying Q-network architectures, VAE decoder heads, or heuristic routing policies is out of scope.
- **Raw Tick Downscaling Operators**: Re-running raw historical downscaling from tick archives is not required.

---

## Further Notes

- **CLI Compatibility**: All existing CLI flags and environment variables in `fu_full_process.sh` (`--symbol`, `--target_freq`, `--stage`, `--orderbook_depth`, `--feature_blacklist`, `--target_decision_window`) remain fully compatible.
- **Artifact Compatibility**: Downstream consumers (`state_features.npy`, `df.feather`, `feature_selection_manifest.json`) maintain unchanged schemas and directory structures, ensuring seamless downstream execution.
