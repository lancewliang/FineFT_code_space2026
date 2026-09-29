# 03: Multi-Contract Distribution Drift Gate via Pairwise PSI and KS Tests

**What to build:** Build a dedicated distribution stability audit module `operator_futures.feature_selection.muti_contract.distribution_audit.py` that computes pairwise Population Stability Index ($\text{PSI}$) and Kolmogorov-Smirnov statistics across all training contracts on pooled quantile bins, enforcing strict distribution homogeneity ($\overline{\text{PSI}} \le 0.10$, $\text{PSI}_{\max} \le 0.25$) with a safety fallback guard.

**Blocked by:** 02: Front-Loaded Feature Blacklist Execution in Multi-Contract Selection Pipeline

**Status:** closed

- [x] Implement `audit_distribution_drift(frames, feature_universe, num_bins=10, max_mean_psi=0.10, max_pair_psi=0.25)` computing pairwise PSI and KS test across participating training contracts.
- [x] Enforce hard distribution gate rejecting features with $\overline{\text{PSI}} > 0.10$ or $\text{PSI}_{\max} > 0.25$.
- [x] Incorporate safety guard: if fewer than a configurable minimum (e.g. 20) candidate features pass, log diagnostics and gracefully relax threshold with explicit warning.
- [x] Persist `distribution_audit_metrics.csv` containing per-feature $\overline{\text{PSI}}$, $\text{PSI}_{\max}$, and KS statistics in the stage output directory.
- [x] Comprehensive unit tests verifying deterministic PSI calculation and drift rejection behavior on synthetic multi-contract frames.
