# 04: Unified Scale-Save on Union State Features

**What to build:** Validate and ensure `data_preprocess.operator_futures.scale_describe_save.muti_contract_scale_save` scales the Dual-Stream Union State Features (`state_features.npy`) in a single pass into a unified wide dataset (`df.feather`), applying adaptive intraday rolling Z-score and hyperbolic tangent $\tanh(z/M)$ soft-saturation, and verify that downstream RL state loaders cleanly perform zero-copy view slicing on `rl_state_features.npy` without missing column or NaN errors.

**Blocked by:** 02: Dual-Branch Feature Selection and Union Persistence

**Status:** completed

**Deliverables (交付物):**
- Scaling pipeline integration: Validated execution in `muti_contract_scale_save.py` with union state features
- Standardized wide dataset artifacts:
  - Scaled contract tables (`df.feather`) containing scaled union features and reward columns
  - Standardized feature lists (`state_features.npy`, `rl_state_features.npy`, `vae_state_features.npy`) copied to scaled destination
- Zero-copy view slicing validation: Verified environment state reader slicing `df[rl_state_features]`
- Verification test suite: Tests in `data_preprocess/tests/test_commodity_scale_save.py` validating scaling and dual-stream slicing

- [x] `muti_contract_scale_save.py` loads `state_features.npy` (Union) and preflight-validates all contracts without missing column exceptions.
- [x] Scaling applies adaptive intraday rolling Z-score and $\tanh(z/M)$ soft saturation across all non-passthrough features in the union set.
- [x] Scaled `df.feather` contains zero NaN values across both `vae_state_features` and `rl_state_features` columns.
- [x] Slicing `df[rl_state_features]` yields a valid 2D array matching the low-level agent observation specification.
- [x] Tests pass cleanly with zero regressions.
