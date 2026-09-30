# 02: Dual-Branch Feature Selection and Union Persistence

**What to build:** Implement dual-branch filtering, scoring, clustering, and persistence logic in the multi-contract feature selection pipeline. Compute shared candidate metrics and fit the target-horizon CatBoost model exactly once across the wide candidate pool, branch surviving candidates through independent VAE and RL filter profiles, enforce physical isolation on mandatory features, unblock microstructural Alpha signals (OFI and short returns) for RL, and persist `vae_state_features.npy`, `rl_state_features.npy`, and the mathematical union `state_features.npy` alongside the dual-stream manifest.

**Blocked by:** 01: Dual-Stream Profiles, Manifest Schema and Artifact Constants

**Status:** completed

**Deliverables (交付物):**
- Core selection engine: Dual-branch execution logic in `operator_futures.feature_selection.muti_contract.pipeline`
- IO persistence engine: Multi-artifact writer in `operator_futures.feature_selection.muti_contract.io_manager`
- Physical output artifacts:
  - `vae_state_features.npy` ($12 \sim 18$ dimensions of hyper-stationary regime indicators)
  - `rl_state_features.npy` ($50 \sim 65$ dimensions of Alpha-rich decision indicators)
  - `state_features.npy` (exact set union $\mathcal{S}_{\text{union}} = \mathcal{S}_{\text{vae}} \cup \mathcal{S}_{\text{rl}}$)
  - `feature_selection_manifest.json` with top-level `stream_mode: "dual"` and dedicated stream audit blocks
  - Filtered contract `.feather` files containing reward columns plus union state features
- End-to-end test suite: Integration tests in `data_preprocess/tests/test_commodity_multi_contract_feature_selection.py` asserting artifact creation, dimension constraints, signal preservation, and union consistency

- [x] Stage 1 vectorized filters and Stage 2 CatBoost model fitting execute exactly once on the shared candidate pool without redundant compute.
- [x] VAE branch evaluates `DEFAULT_VAE_PROFILE`, applying strict PSI ($\le 0.10$), correlation cap ($r \le 0.65$), persistence noise dropping, and regex filtering on mandatory features to retain only time topology (`base_time_*`, `time_*`, `trading_minute_*`).
- [x] RL branch evaluates `DEFAULT_RL_PROFILE`, applying relaxed PSI ($\le 0.25$), correlation cap ($r \le 0.80$), disabling persistence noise dropping (retaining `level5_ofi_weighted_norm` and `log_return_1/2/6`), and preserving all 17 mandatory features.
- [x] Pipeline writes `vae_state_features.npy`, `rl_state_features.npy`, and `state_features.npy` to the train stage output directory.
- [x] Output `state_features.npy` strictly equals the mathematical union $\mathcal{S}_{\text{vae}} \cup \mathcal{S}_{\text{rl}}$.
- [x] `feature_selection_manifest.json` writes valid dual-stream audit blocks detailing per-stream candidate counts and filter drops.
- [x] All tests pass cleanly under `pytest data_preprocess/tests/test_commodity_multi_contract_feature_selection.py`.
