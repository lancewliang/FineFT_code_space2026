# 03: Dynamic VAE Data Generation and Model Adaptation

**What to build:** Refactor VAE data preparation in `FineFT.datahandler.vae_data_creation` to prioritize loading `vae_state_features.npy` over `state_features.npy`, dynamically derive model input dimensions (`INPUT_DIM = len(vae_state_features)`) instead of hardcoding dimensions, and ensure VAE training arrays are generated with the exact compact dimension ($12 \sim 18$) of the VAE regime feature stream.

**Blocked by:** 02: Dual-Branch Feature Selection and Union Persistence

**Status:** completed

**Deliverables (交付物):**
- VAE data loader: Dynamic feature resolution logic in `FineFT/datahandler/vae_data_creation.py`
- Artifact ingestion: Canonical constant usage via `ArtifactNames.VAE_STATE_FEATURES_NPY`
- Physical data artifacts:
  - VAE training numpy arrays (`<label>.npy`) with column count equal to `len(vae_state_features)`
  - VAE test evaluation numpy arrays (`test_<contract>.npy`) with column count equal to `len(vae_state_features)`
- Test suite: Unit tests in `FineFT/tests/` or `data_preprocess/tests/` validating dynamic input dimension resolution and array shape consistency

- [x] `make_data` in `FineFT/datahandler/vae_data_creation.py` strictly requires `vae_state_features.npy` and fails fast if missing, with legacy `state_features.npy` fallback purged.
- [x] VAE network configuration and dataset creation derive input dimensions dynamically as `INPUT_DIM = len(state_features)`.
- [x] Generated `test_<contract>.npy` and label array `.npy` files match the exact column count of the selected VAE stream ($12 \sim 18$ dims).
- [x] Unit tests pass verifying array shape and fail-fast when VAE state features are missing.
