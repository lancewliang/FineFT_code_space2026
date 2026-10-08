# 05: High-Level VAE Routing and Low-Level RL State Space End-to-End Ingestion

**What to build:** The downstream training and inference pipelines cleanly ingest the updated multi-horizon feature vectors. Slope VAE ($12 \sim 16$ dims, 0% micro) and Volatility VAE ($10 \sim 14$ dims) perform stable latent inference without microstructural jitter. The Low-Level RL dataset loader (`commodity_contract_dataset.py`) and single-stream flat Q-network (`FineFT/model/low_level.py:Qnet`) ingest the updated $135 \sim 160$ dimension state vector and execute forward passes without architectural modification or tensor shape errors.

**Blocked by:** 04: Triple-Stream Pipeline Orchestration and Multi-Contract Scaling Adaptation

**Status:** ready-for-agent

- [x] Verify that `FineFT.RL.DiHFT.high_level.vae_routing_util` ingests the updated Meso/Macro feature lists from `vae_slope_state_features.npy` and `vae_volatility_state_features.npy`.
- [x] Verify that high-level VAE latent inference produces stable Bull/Bear and High/Low vol classifications across multiday bull runs without spurious regime flapping during minor intraday pullbacks.
- [x] Verify that `FineFT.datahandler.commodity_contract_dataset` cleanly loads `rl_state_features.npy` and maps multi-horizon features into observation tensors.
- [x] Verify that `FineFT.model.low_level.Qnet` initializes and executes flat forward passes with input dimension $N_{\text{states}} \in [135, 160]$ with zero architectural modifications.
- [x] Add integration tests in `FineFT/tests/rl/test_vae_routing_final_result.py` and `FineFT/tests/datahandler/test_commodity_contract_dataset.py` asserting tensor shape alignment and end-to-end forward pass execution.
