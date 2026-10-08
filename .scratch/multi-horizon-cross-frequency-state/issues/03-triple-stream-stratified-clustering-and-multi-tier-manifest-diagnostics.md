# 03: Triple-Stream Stratified Clustering and Multi-Tier Manifest Diagnostics

**What to build:** Agglomerative correlation clustering deduplication operates within scale tiers to enforce explicit capacity quotas across streams (RL: 85~100 micro, 35~45 meso, 10~15 macro; Slope VAE: 0 micro, 8~10 meso, 4~6 macro; Volatility VAE: 0~1 micro, 6~8 meso, 4~5 macro). The feature selection manifest (`FeatureSelectionManifest`) and audit records (`StreamAuditRecord`) log comprehensive multi-tier diagnostics (`tier_breakdown`), capturing candidate counts, gate drops, survivors, quotas, and human-readable process documentation for complete reproducibility.

**Blocked by:** 02: Triple-Stream Multi-Horizon Predictive Funnel and Scale Classifier

**Status:** ready-for-agent

- [x] Implement stratified correlation-based clustering deduplication in `pipeline.py`:
  - Partition surviving features by tier before clustering.
  - Apply correlation thresholds: 0.80 for RL, 0.65 for Slope VAE, 0.60 for Volatility VAE.
  - Enforce bounded capacity quotas per tier:
    - RL Decision Stream: Micro 85~100, Meso 35~45, Macro 10~15 (total $135 \sim 160$)
    - Slope VAE Stream: Micro 0, Meso 8~10, Macro 4~6 (total $12 \sim 16$)
    - Volatility VAE Stream: Micro 0~1, Meso 6~8, Macro 4~5 (total $10 \sim 14$)
- [x] Upgrade `StreamAuditRecord` in `data_preprocess/operator_futures/feature_selection/manifests.py` to record `tier_breakdown` for each scale tier:
  - `forward_horizons`, `decision_horizon`, and evaluation thresholds
  - `candidates`, `anti_causal_dropped`, `rank_ic_dropped`, `sign_consistency_dropped`, `stability_ir_dropped`
  - `survivors`, `selected_quota`, and `selected` count
- [x] Add `process_documentation` to `FeatureSelectionManifest` detailing the tier partitioning, tier-matched predictive funnel, stratified deduplication, and rationale across all three streams.
- [x] Add unit tests verifying that `FeatureSelectionManifest` serializes and deserializes the full multi-tier diagnostic payload, and that macro features are preserved according to assigned quotas.
