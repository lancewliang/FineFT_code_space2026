# 03: Temporal Evaluation Walk-Forward, Boundary Maturity Purification, and Test Sealing

**What to build:** Implement chronological evaluation scheduling and boundary maturity purification across multiple walk-forward time windows, accompanied by strict isolation of the final Test dataset. The evaluation schedule reserves an initial 6-calendar-month historical reference window, followed by rolling 3-calendar-month evaluation windows aligned to session starts, split at the historical Train/Valid boundary, and retaining any partial final window. Enforce strict label maturity purification so that evaluation rows and historical baselines only include labels that have fully matured before window boundaries. Guard the final Test dataset with hard read-isolation policies to guarantee zero data leakage into development audits.

**Blocked by:** 01: Minimal Tracer Bullet: Audit Validation & End-to-End Single-Horizon Run, 02: Full Seven-Horizon Semantics, Session Transitions, and Label Invalidation

**Status:** completed

- [x] Evaluation schedule generates standardized window definitions: 6-month historical warmup reference plus chronological 3-month evaluation windows forward-aligned to trading day session starts.
- [x] Boundary partitioning strictly splits evaluation windows at the original Train/Valid partition boundary while retaining any partial final development window.
- [x] Label maturity purification ensures any observation admitted into an evaluation window matures strictly before that window's boundary, and historical reference data matures strictly before the evaluation window starts.
- [x] Final Test dataset sealing prevents the audit application from reading Test partition prices or labels during development runs; changing fixture Test prices produces identical development audit outputs.
- [x] Historical reference observations are excluded from primary evaluation performance summaries.
- [x] Metrics are computed per evaluation window and per contract, yielding contract-equal primary aggregates and sample-weighted sensitivity aggregates across all valid development windows.
- [x] Sign-consistency metrics across contracts are computed only when at least 3 eligible contracts have valid statistics, outputting null with explicit reasons otherwise.
- [x] Automated tests verify boundary purification on long-horizon labels (e.g. Next15TD) spanning evaluation window and dataset split boundaries.
