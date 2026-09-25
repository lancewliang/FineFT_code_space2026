---
status: accepted
---

# VAE Feature-Level Out-Of-Distribution (OOD) Analysis

We establish a dedicated diagnostic analysis engine and pipeline (`FineFT/analysis/feature/vae_feature_ood_analysis.py`) to quantify per-feature contributions to VAE out-of-distribution (OOD) likelihood degradation across `test` and `train` datasets against the canonical `valid` in-distribution baseline.

We chose closed-form feature-level Gaussian Negative Log-Likelihood (NLL) decomposition:
$$\text{NLL}_j = 0.5 \cdot \left(\frac{x_j - \mu_j}{\sigma_j}\right)^2 + \log \sigma_j + 0.5 \ln(2\pi)$$
over heuristic feature perturbation or marginal statistical divergence tests alone because it mathematically matches the exact objective evaluated by the VAE decoder. Furthermore, we aggregate across both dual VAE axes (`slope` and `volatility`) and three regime clusters, pairing NLL deltas with empirical distribution shift statistics (standardized mean shift and variance ratios) to distinguish between structural price spread shifts, liquidity flow imbalances, and seasonal calendar anomalies.
