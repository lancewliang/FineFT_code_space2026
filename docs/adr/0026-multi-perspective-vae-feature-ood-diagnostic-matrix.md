---
status: accepted
---

# Multi-Perspective VAE Feature OOD Diagnostic Matrix

We replace the single-baseline (`valid`) VAE feature OOD reporting logic in `FineFT/analysis/feature/vae_feature_ood_analysis.py` with a three-perspective diagnostic matrix anchored on `train` as the true in-distribution training baseline, eliminating diagnostic direction inversion.

## Context

In ADR-0018, feature-level closed-form Gaussian negative log-likelihood (NLL) decomposition was introduced with `valid` as the reference baseline because VAE representation models were originally trained on `valid`.

Subsequently, ADR-0021 adapted VAE training data generation to dynamically slice and train directly on `train` contracts, resolving validation data leakage and expanding regime coverage. However, `FineFT/analysis/feature/vae_feature_ood_analysis.py` remained hardcoded with `valid` as the sole reference baseline ($\Delta \text{NLL}_{\text{test}} = \text{NLL}_{\text{test}} - \text{NLL}_{\text{valid}}$, $\Delta \text{NLL}_{\text{train}} = \text{NLL}_{\text{train}} - \text{NLL}_{\text{valid}}$).

This created critical architectural and diagnostic problems:
1. **Diagnostic Direction Inversion**: Because VAE weights are optimized on `train`, `train` is the true in-distribution ground truth. If the validation split exhibits macro drift or regime shift, measuring `train - valid` inverted the diagnostic direction, reporting negative deltas or falsely misattributing valid drift to the training set.
2. **Confounded Test OOD Diagnosis**: Measuring `test` exclusively against `valid` confounded validation set idiosyncrasies with out-of-sample test OOD collapse.
3. **Single Sorting View Limitations**: A single CSV table sorted exclusively by `delta_nll_test_vs_valid` prevented practitioners from directly inspecting which features degrade between training and validation versus between training and testing.

## Decision

Following the architectural grilling session, we establish the following decisions:

### 1. Three-Perspective Diagnostic Matrix
The OOD diagnostic report decomposes into three complementary, self-contained perspective tables:
- **Table 1 (`valid_vs_train`)**:
  - Reference Baseline: `train`
  - Evaluation Target: `valid`
  - Formula: $\Delta \text{NLL} = \text{NLL}_{\text{valid}} - \text{NLL}_{\text{train}}$
  - Purpose: Quantifies distribution shift and likelihood degradation introduced when transitioning from training data to the validation split.
- **Table 2 (`test_vs_train`) [Primary OOD Benchmark]**:
  - Reference Baseline: `train`
  - Evaluation Target: `test`
  - Formula: $\Delta \text{NLL} = \text{NLL}_{\text{test}} - \text{NLL}_{\text{train}}$
  - Purpose: Quantifies true out-of-distribution likelihood collapse directly against the VAE's training distribution.
- **Table 3 (`test_vs_valid`)**:
  - Reference Baseline: `valid`
  - Evaluation Target: `test`
  - Formula: $\Delta \text{NLL} = \text{NLL}_{\text{test}} - \text{NLL}_{\text{valid}}$
  - Purpose: Quantifies generalization degradation between validation and out-of-sample testing, matching the historical ADR-0018 metric.

### 2. Standardized Directionality and Sign Convention
Across all comparisons:
$$\Delta \text{NLL} = \text{NLL}_{\text{target}} - \text{NLL}_{\text{ref}}$$
A positive $\Delta \text{NLL}$ strictly signifies that the target distribution has higher negative log-likelihood (worse VAE fit / more OOD) than the reference baseline. Each perspective table is independently sorted descending by its own $\Delta \text{NLL}$.

### 3. Self-Contained Statistical Drift Pairing
Each perspective calculates empirical distribution drift metrics (`mean_shift`, `var_ratio`, reference/target mean and standard deviation) strictly between its designated reference and target datasets using `calculate_distribution_drift(ref_df, target_df)`.

### 4. Artifact Persistence Layout
`analyze_feature_vae_ood` persists:
1. `feature_ood_valid_vs_train.csv`: Sorted by `delta_nll_valid_vs_train` descending.
2. `feature_ood_test_vs_train.csv`: Sorted by `delta_nll_test_vs_train` descending.
3. `feature_ood_test_vs_valid.csv`: Sorted by `delta_nll_test_vs_valid` descending.
4. `feature_ood_summary.csv`: Master consolidated table containing all perspective columns, sorted primarily by `delta_nll_test_vs_train` descending.

### 5. Multi-Perspective Terminal Presentation
The CLI report prints three sequential, formatted tables (configurable via `--top_k`, default 10) representing the top degradation features for `valid_vs_train`, `test_vs_train`, and `test_vs_valid`.

### 6. Per-Contract Baseline Alignment
When `--per_contract` is enabled, test contracts are evaluated primarily against the `train` baseline ($\Delta \text{NLL}_{\text{contract vs train}}$), while retaining $\Delta \text{NLL}_{\text{contract vs valid}}$ columns for cross-validation comparison.
