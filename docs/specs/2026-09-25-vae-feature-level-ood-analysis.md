# Spec: VAE Feature-Level Out-Of-Distribution (OOD) Analysis Pipeline

## Problem Statement

In hierarchical reinforcement learning for commodity futures trading, high-level routing networks rely on dual-axis Variational Autoencoders (VAEs) to identify market dynamics regimes (such as trend slope and volatility) and trigger defensive flat-position safeguards when out-of-distribution (OOD) market states occur.

During out-of-sample backtesting on unseen evaluation contracts, VAE reconstruction log-likelihoods often experience catastrophic collapse (dropping from typical in-distribution levels of +130 down to -2,000 ~ -14,000). This triggers the high-level policy's conservative gating condition across every single evaluation timestep, causing total trading paralysis (zero trades opened, zero returns extracted, despite achieving positive returns during validation).

Quant researchers and reinforcement learning engineers currently lack an automated diagnostic tool to pinpoint and quantify exactly which market state features among hundreds of engineered indicators (such as term structure spreads, rolling volatility, orderbook flow imbalances, and calendar cycles) drive this VAE likelihood degradation between the in-distribution validation baseline and evaluation datasets. Without exact, mathematically sound feature-level loss attribution, researchers cannot determine whether the OOD behavior is caused by legitimate structural regime shifts, static scaler normalization failures, or non-stationary features that should be pruned or redesigned.

## Solution

Build a dedicated, standalone feature-level VAE OOD analysis engine and CLI tool. Taking the validation dataset (`valid`) as the canonical in-distribution reference baseline (since VAE networks and regime threshold boundaries are calibrated on this split), the tool systematically analyzes both the evaluation test dataset (`test`) and the historical training dataset (`train`) against this reference.

The engine decomposes the overall VAE reconstruction loss into feature-wise Gaussian Negative Log-Likelihood (NLL) contributions for every state feature:
$$\text{NLL}_j = 0.5 \cdot \left(\frac{x_j - \mu_j}{\sigma_j}\right)^2 + \log \sigma_j + 0.5 \ln(2\pi)$$

It aggregates these feature-level metrics across all dual axes (`slope` and `volatility`) and across all regime clustering models (`label_0`, `label_1`, `label_2`). It pairs the resulting likelihood deltas with empirical distribution drift metrics (such as standardized mean shift, variance inflation ratios, and extreme quantile boundary violations).

The output provides structured diagnostic CSV reports, ranked contribution tables, terminal summary views, and visual comparison charts (horizontal bar charts of top OOD contributors and distribution shift plots) to empower researchers to make data-driven feature selection, scaling, and threshold adjustments.

## User Stories

1. As a quant researcher, I want to execute an automated feature-level VAE OOD analysis command on any commodity dataset, so that I can immediately identify the primary features responsible for out-of-distribution rejections.
2. As a reinforcement learning engineer, I want the analysis tool to take `valid` as the reference in-distribution baseline, so that the computed likelihood shifts accurately reflect the distribution on which VAE models were trained and calibrated.
3. As a strategy developer, I want to evaluate both `test vs valid` and `train vs valid` feature shifts, so that I can understand how features evolved across historical training, validation calibration, and out-of-sample testing periods.
4. As a machine learning researcher, I want the tool to compute the exact per-feature Gaussian NLL contribution ($\text{NLL}_j$), so that I have a mathematically exact breakdown of the VAE loss function rather than a heuristic proxy.
5. As an algorithmic trader, I want to see a ranked percentage contribution ($\text{Contrib}\%$) for each feature relative to the total NLL increase, so that I can focus engineering effort on the highest-impact drifting features.
6. As a risk manager, I want the tool to evaluate and aggregate across both dual VAE axes (`slope` and `volatility`), so that I can detect whether an OOD event is driven by volatility regime models or trend slope models.
7. As a quant researcher, I want to inspect individual regime cluster models (`label_0`, `label_1`, `label_2`), so that I can verify whether feature drift affects specific market regimes or applies uniformly across all states.
8. As a data engineer, I want the tool to report standard distribution drift statistics (mean shift, variance ratio, and min/max extremes) alongside NLL deltas, so that I can understand the underlying statistical behavior causing the reconstruction failure.
9. As a feature engineer, I want to identify which non-stationary features (such as term structure price spreads or moving average distances) have drifted in mean level across contract years, so that I can replace them with stationary rolling differentials.
10. As a data pipeline maintainer, I want to detect extreme outlier values and clipping issues in test datasets, so that I can identify features that exceed training set scaling bounds.
11. As a quant researcher, I want the option to run contract-level breakdown analysis (`--per_contract`), so that I can distinguish between universal market-wide regime shifts and idiosyncratic liquidity anomalies on individual far-month contracts.
12. As a backtest analyst, I want the tool to output a clean, formatted Markdown/ASCII table to the terminal, so that I can immediately review the top 15-20 OOD features without opening external files.
13. As a strategy researcher, I want comprehensive summary CSV files exported to a standardized results directory, so that I can import the metrics into downstream notebooks or automated regression pipelines.
14. As a research lead, I want the tool to generate visual plots showing top feature NLL contribution bars and probability density comparisons, so that I can communicate findings clearly in research presentations and documentation.
15. As a developer, I want the tool to support configurable sample batch sizes and GPU/CPU device placement, so that the analysis executes quickly on both development laptops and GPU servers.
16. As an automated testing agent, I want the analysis module to expose a clean, decoupled Python interface, so that unit and integration tests can verify feature decomposition logic on synthetic data without touching large disk datasets.
17. As a project maintainer, I want standardized CLI parameter flags matching existing project conventions (`--base_path`, `--dataset_name`, `--experiment_name`), so that new scripts integrate seamlessly with existing bash runner workflows.

## Implementation Decisions

- **Architectural Separation**: The analysis pipeline is implemented as a standalone diagnostic module, decoupled from live RL training and environment execution loops. It consumes exported dataset artifacts and pre-trained VAE checkpoints as read-only inputs.
- **Reference In-Distribution Baseline**: The validation split (`valid`) is established as the canonical In-Distribution (ID) reference for all $\Delta \text{NLL}$ computations, matching the domain contract where VAE regime labeling, threshold calibration, and training materialization are centered on validation dynamics.
- **Dual Comparison Scope**: The pipeline evaluates two distinct directional comparisons:
  - `Test vs Valid`: Diagnoses extrapolation failures and OOD triggers during out-of-sample forward testing.
  - `Train vs Valid`: Diagnoses distributional coverage and covariate shift between historical agent training and validation calibration.
- **Exact Per-Feature Loss Decomposition**: The analysis directly extracts the decoded mean ($\mu$) and variance ($\sigma^2$) vectors from the VAE decoder outputs for each sample $x$, computing the closed-form Gaussian Negative Log-Likelihood contribution per feature dimension:
  $$\text{NLL}_j = 0.5 \cdot \left(\frac{x_j - \mu_j}{\sigma_j}\right)^2 + \log \sigma_j + 0.5 \ln(2\pi)$$
- **Hierarchical Aggregation Levels**:
  - Global Composite: Arithmetic average of $\Delta \text{NLL}_j$ across all six VAE models (two axes $\times$ three regimes), serving as the primary ranking key.
  - Axis-Level: Separate rankings for the `slope` axis and `volatility` axis.
  - Regime-Level: Explicit per-label column preservation in the exported CSV.
- **Statistical Drift Companion Metrics**: For each feature, the engine computes:
  - Standardized Mean Shift: $\frac{|\mu_{\text{target}} - \mu_{\text{valid}}|}{\sigma_{\text{valid}}}$
  - Variance Ratio: $\frac{\sigma^2_{\text{target}}}{\sigma^2_{\text{valid}}}$
  - Support Envelope: Target minimum and maximum relative to validation boundaries.
- **Contract Aggregation**: By default, data from all contracts in a split are pooled using uniform stratified sampling to produce representative dataset-level metrics. An optional contract-level drill-down mode computes independent metrics per contract.
- **Output Artifacts**:
  - Main tabular summary: `feature_ood_summary.csv` containing all features, their ranks, multi-level $\Delta \text{NLL}$ values, and drift statistics.
  - Visualization artifacts: `top_ood_features.png` (bar chart of top contributors) and `top_feature_distributions.png` (density comparison across train, valid, and test).
  - Console report: Top 15 features printed in a clean tabular layout upon completion.

## Testing Decisions

- **Testing Philosophy**: Tests must evaluate external behavior rather than internal implementation details. A high-quality test feeds known inputs into the module and asserts mathematical consistency, ranking correctness, and artifact creation.
- **Single Highest Testing Seam**: The highest seam is an end-to-end execution of the analysis pipeline function using a lightweight synthetic test harness.
  - The test generates small synthetic DataFrames for `train`, `valid`, and `test` with a known set of synthetic features (including one intentionally injected mean-shifted feature and one variance-inflated feature).
  - It constructs a minimal dummy VAE network matching the interface of the project's model.
  - It runs the complete feature OOD diagnostic function and asserts:
    1. The output summary DataFrame contains exactly one row per feature.
    2. The artificially shifted feature is deterministically identified as the top contributor (Rank 1) with the highest $\Delta \text{NLL}$.
    3. The sum of per-feature NLL contributions equals the total sample NLL difference.
    4. Required artifact files (CSV and summaries) are produced at the expected destination paths.
- **Prior Art**: Follows patterns established in `FineFT/tests/rl/test_commodity_vae_cross_contract.py` (which verifies cross-contract VAE artifact ingestion) and `FineFT/tests/datahandler/test_valid_cross_contract_label_calibration.py`.

## Out of Scope

- Modifying the VAE architecture, hyperparameter space, or re-training VAE model weights.
- Modifying the online trading environment or live high-level routing logic in simulation loops.
- Modifying low-level Q-network architectures or action selection mechanisms.
- Altering existing data preprocessing pipelines or raw tick feature generation scripts.

## Further Notes

- Relates to ADR-0005 (`label-agent-selection-meta-routing`), ADR-0010 (`global-segment-quantile-label-thresholds`), and ADR-0011 (`volatility-labeling-and-method-isolated-outputs`).
- This analysis directly explains the 0.000000 return rate observed during the fuel oil (`fu`) 10-minute parallel final result backtest, where dual cross-month spread features accounted for over 90% of the VAE likelihood collapse.
