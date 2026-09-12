# 02: Train Dataset Full-Contract Regime Injection and Slicing

**What to build:** An integrated training dataset generation pipeline that fits turning-point macro segments and annotates 2D regime grid IDs across full continuous contract files before slicing into chunks. The pipeline independently calibrates tercile thresholds on training contracts, saves calibration metadata to the training directory, and produces fixed-length continuous training slices that inherit unbroken macro regime labels without boundary distortion.

**Blocked by:** 01: Core 2D Regime Calibration Engine

**Status:** ready-for-agent

- [ ] Training dataset generation runs 2D regime calibration on full continuous contract files prior to chunk slicing
- [ ] Calibration terciles are fitted independently across all participating training contracts with scope recorded as `train_all_contracts`
- [ ] Calibration results and threshold metadata are persisted to `regime_thresholds.json` under the training directory
- [ ] Full continuous contract files have `regime_grid_id` (and constituent dimension labels) written directly into their columns
- [ ] Fixed-length continuous slice generation (`chunk_length=3200`) produces slice files that naturally inherit unbroken row-level `regime_grid_id` values
- [ ] Missing required baseline price column (`mark_price`) fails fast with a clear error rather than attempting speculative fallbacks
- [ ] End-to-end dataset generation pipeline tests verify artifact creation, column presence, and slice continuity
