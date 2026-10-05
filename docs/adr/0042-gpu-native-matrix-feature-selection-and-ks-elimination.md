---
status: accepted
---

# GPU-Native Matrix Feature Selection and KS Elimination Architecture

We replace legacy Python scalar loops, unvectorized SciPy exact tests, and redundant validation CatBoost models in the multi-contract feature selection pipeline (`feature_selection_train` and `feature_selection_valid`) with a GPU-native PyTorch GEMM matrix architecture and fast distribution drift gating, compressing end-to-end execution latency from ~24 minutes to under 40 seconds (over 35x speedup) with zero backward-compatibility baggage.

## Context

Empirical audits of `fu_10min_2023-01-01_2026-03-01_feature_selection_train.log` (21 min 32 s) and `feature_selection_valid.log` (2 min 31 s) revealed four compounding bottlenecks in `operator_futures.feature_selection.muti_contract`:

1. **Unvectorized SciPy Exact KS-Test Bottleneck (80%+ of Train Latency)**:
   In `distribution_audit.py:_compute_pairwise_psi_and_ks`, evaluating candidate features across 14 contracts involved $951 \text{ features} \times 91 \text{ contract pairs} = 86,541$ invocations of `scipy.stats.ks_2samp`. Because `ks_2samp` defaults to `method='auto'`, continuous financial arrays ($N \approx 5,000$) triggered the exact two-sample Smirnov algorithm (`_attempt_exact_2kssamp` via dynamic programming C-extension `_compute_outer_prob_inside_method`), taking 15~65 ms per call (5.9 s per feature). Crucially, inspection of `distribution_audit.py:250-265` confirmed that `max_ks_d` and `min_ks_p` were **never used in any feature gating or truncation logic**; only PSI was filtered against. The 17-minute KS calculation was purely writing unused columns into `distribution_audit_metrics.csv`.
2. **Legacy Multi-Window Validation CatBoost Fitting (83% of Valid Latency)**:
   In ADR-0029, training-stage CatBoost fitting was restricted to the single target decision window (`w_dec=1`, 14 models, ~21 s total). However, `_run_validation_stage` still invoked the legacy `calculate_metric_frame`, which trained CatBoost regressors across all 7 forecast windows on 12 contracts ($12 \times 7 = 84$ models, consuming 126 s). Because validation is strictly `report_only=True` and performs zero feature pruning, fitting 84 CatBoost models was completely redundant.
3. **Scalar Metric Loops and Target Re-sorting in Predictive Audit**:
   In `predictive_audit.py:execute_predictive_audit`, a triple loop ($14 \text{ contracts} \times 7 \text{ windows} \times 951 \text{ features} = 93,198$ iterations) extracted Polars series individually and recalculated `np.argsort(np.argsort(target))` 951 times per window, while evaluating unreferenced `Permutation Importance` and `Sharpe` metrics in scalar Python code.
4. **Nested Quadrifold Loop in Multi-Regime Audit**:
   In `regime_audit.py:audit_regimes`, a 4-fold nested loop ($3 \times 3 \text{ bins} \times 7 \text{ windows} \times 679 \text{ features} \times 14 \text{ contracts} \approx 600,000$ iterations) repeatedly sliced individual contract series with Boolean masks and executed scalar correlation functions.

## Decision

We establish the **GPU-Native Matrix Feature Selection Architecture** with zero backward-compatibility baggage:

### 1. Fast Distribution Drift Gate with KS Elimination
In `data_preprocess/operator_futures/feature_selection/muti_contract/distribution_audit.py`:
- Completely bypass `scipy.stats.ks_2samp` during pipeline distribution drift auditing.
- Populate `max_ks_d` with `0.0` and `min_ks_p` with `1.0` as static placeholders in the DataFrame schema and CSV output to preserve metric table structure without the 17-minute computational penalty.
- Retain pure quantile binning and vectorized pairwise Population Stability Index ($\text{PSI}$) computation, reducing distribution audit time from ~17 minutes to **< 3 seconds**.

### 2. Elimination of Redundant Validation CatBoost Models
In `data_preprocess/operator_futures/feature_selection/muti_contract/pipeline.py:_run_validation_stage`:
- Eliminate all CatBoost regressor training in validation stage. Assign `CatBoost Importance` directly to `0.0` across all windows and contracts.
- Retain exact distribution drift, RankIC, and market state regime stability reporting, compressing validation stage latency from 2.5 minutes to **< 5 seconds**.

### 3. GPU-Native Batch Matrix RankIC and Predictive Audit (`TorchMatrixRankIC`)
In `data_preprocess/operator_futures/feature_selection/muti_contract/predictive_audit.py`:
- Ingest contract feature matrix $X \in \mathbb{R}^{N \times D}$ directly onto GPU (`torch.cuda`, FP32, ~30MB VRAM per contract).
- Vectorized double-argsort ranking along dimension 0 in parallel:
  $$\text{Rank}(X) = \text{argsort}(\text{argsort}(X, \text{dim}=0), \text{dim}=0)$$
- Column-wise normalization (Z-score) of feature ranks $\tilde{X}$ and multi-window return ranks $\tilde{Y} \in \mathbb{R}^{N \times W}$.
- Single-step General Matrix Multiply (GEMM) computing RankIC across all $D$ features and all $W=7$ forecast windows simultaneously:
  $$\text{RankIC}_{\text{Matrix}} = \frac{1}{N} \tilde{X}^\top \tilde{Y} \in \mathbb{R}^{D \times W}$$
- Compute $\text{VolRankIC}_{\text{Matrix}}$ in parallel using $\tilde{Y}_{\text{vol}} = \text{Rank}(|Y|)$.
- Benchmarked at **11.13 ms** for 1,311 features $\times$ 7 windows on RTX 4070 Ti SUPER (155 ms total for all 14 contracts), replacing the 180-second scalar loop with an **1,150x speedup**.

### 4. GPU-Masked Multi-Regime Matrix Correlation
In `data_preprocess/operator_futures/feature_selection/muti_contract/regime_audit.py`:
- Invert the loop hierarchy: slice 2D feature matrices by regime bin masks on GPU:
  $$X_{\text{bin}} = X[\text{mask}], \quad Y_{\text{bin}} = Y[\text{mask}]$$
- Compute batch matrix-vector correlations across all 9 market state bins simultaneously on GPU, compressing regime audit from ~80 seconds to **< 4 seconds**.

## Consequences

- **End-to-End Latency**:
  - `feature_selection_train`: Compresses from **21.5 minutes (1,292 s)** to **~30-35 seconds** (~40x speedup).
  - `feature_selection_valid`: Compresses from **2.5 minutes (151 s)** to **~3-5 seconds** (~30x speedup).
  - Total pipeline elapsed time drops from **~24 minutes to ~35-40 seconds**.
- **Hardware Footprint**:
  - VRAM utilization: Maximum peak batch memory < 100 MB on `RTX 4070 Ti SUPER 16GB` (< 1% GPU memory), leaving ample headroom for concurrent training tasks.
- **Zero Compatibility Shims**:
  - No speculative CPU fallback trees or legacy adapters; directly replaces scalar implementations with typed PyTorch CUDA matrix operations.
- **Contract & Downstream Invariance**:
  - File artifacts (`rl_state_features.npy`, `vae_slope_state_features.npy`, `vae_volatility_state_features.npy`, `state_features.npy`, `feature_selection_manifest.json`) and output directories remain 100% compliant with downstream consumers (`scale_save`, `vae_data_creation`, `low_level_train`).
- **Test Compatibility**:
  - Preserves unit test coverage across `test_commodity_multi_contract_feature_selection.py`, `test_distribution_audit.py`, and `test_predictive_audit.py`.
