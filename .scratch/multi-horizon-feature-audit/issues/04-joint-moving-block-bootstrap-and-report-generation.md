# 04: Joint Moving-Block Bootstrap Uncertainty Protocol and Audit Report Generation

**What to build:** Deliver uncertainty estimation and automated exploratory audit reporting for all candidate features across the seven horizons. Implement joint moving-block bootstrap resampling on the full TradingDay calendar (2,000 resamples, fixed seed 42, 95% percentile intervals, block length $\ge \max(20\text{TD}, \text{max label span})$, and 2x double-length sensitivity analysis). Enforce sample sufficiency rules (`INSUFFICIENT_DATA` status when effective block count is under 20). Attach retrospective legacy three-stream selection/rejection annotations from Stage 7/8 for context without gating admission. Render publication-ready exploratory Markdown audit reports (`audit_report.md`) and feature-by-horizon correlation heatmaps without automatic FDR significance claims or composite scores.

**Blocked by:** 03: Temporal Evaluation Walk-Forward, Boundary Maturity Purification, and Test Sealing

**Status:** completed

- [x] Joint moving-block bootstrap resamples blocks of contiguous TradingDays simultaneously across all contracts, features, and horizons, preserving cross-sectional dependencies and missing observation masks.
- [x] Bootstrap computes 95% percentile confidence intervals with 2,000 resamples (seed 42) for base block length ($\ge \max(20\text{TD}, \text{max horizon span})$) and a 2x double-length sensitivity block length.
- [x] Evidence sufficiency rules evaluate effective block count ($\lfloor \text{valid days} / \text{block length} \rfloor$); runs with fewer than 20 effective blocks output descriptive statistics and status `INSUFFICIENT_DATA` rather than claims of stability.
- [x] Legacy three-stream selection history (Slope VAE, Volatility VAE, RL decision, or rejection reasons) is ingested and attached to candidate output rows as informative annotations without filtering candidates.
- [x] Markdown audit report (`audit_report.md`) is automatically generated, presenting candidate correlation profiles, window stability, coverage ratios, and explicit exploratory disclaimers (no automatic significance gate, no composite score).
- [x] Correlation matrix heatmaps and coverage distributions across features and horizons are exported as machine-readable CSV/JSON artifacts.
- [x] Automated tests verify deterministic reproducibility under fixed seeds and proper error handling for under-sampled or non-varying candidate series.
