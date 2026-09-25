---
status: accepted
---

# Train Dynamic Slicing and VAE Training Source Adaptation

We establish a unified data generation and calibration topology that slices the `train` dataset by slope and volatility, and adapts VAE model training to ingest dynamic slices from `train` while preserving `valid` slicing for Stage II agent evaluation and selection.

## Context

Prior to this decision, `FineFT/script/data/commodity_data_handler_10min_fu.sh` executed turning-point segmentation and dynamic slicing exclusively on the validation set (`valid`) via `valid_cross_contract_label_calibration.py`. Subsequently, `vae_data_creation.py` hardcoded reading from `valid`, generating `VAE_data/<labeling_method>/<contract>/label_<k>.npy`. Downstream `RL/DiHFT/VAE/main.py` merged these contract arrays into `VAE_data/train/<labeling_method>/label_<k>.npy` and trained the VAE representation network on them.

This introduced several structural and methodological issues:
1. **Validation Data Leakage**: The validation set is intended as an unbiased out-of-sample testbed for Stage II Agent selection (`FineFT_two_dimensional_agent_selector.py`) and Optuna macro action routing hyperparameter search (`vae_routing_optuna.py`). Training the VAE representation network directly on the validation set violated machine learning discipline and biased the state density estimation.
2. **Sample Volume and Regime Coverage Discrepancy**: The training split contains 14 commodity contracts (spanning 2023-2024), offering a broader distribution of market regimes than the 12 validation contracts. Training VAE on historical training data aligns the representation model with the exact market dynamics encountered by Stage I low-level reinforcement learning policies.
3. **Stale Directory Overwrite Hazards**: `merge_vae_train.py:contract_dirs()` scans all non-reserved directory items in `VAE_data/<labeling_method>/`. If validation contracts previously occupied that directory, switching sources without directory pruning would silently merge both `train` and `valid` contracts into the VAE training set.

## Decision

1. **Dual-Track Dynamic Slicing for Train and Valid**:
   Both `train` and `valid` splits are dynamically sliced along both `slope` and `volatility` dimensions:
   - `valid` dynamic slicing is strictly retained under `valid/slope` and `valid/volatility` to satisfy downstream Stage II Agent Selector contracts.
   - `train` dynamic slicing is introduced under `train/slope` and `train/volatility`, storing continuous market dynamic segments (`df_*.feather`).
   - The continuous RL rollout chunks under `train/slice/` (`chunk_length=5000+2`) remain completely untouched and isolated from turning-point slices.

2. **Self-Contained Threshold Pooling on Train**:
   `train` dynamic slicing invokes `valid_cross_contract_label_calibration.py` with `--data_dir dataset/.../train` and `--threshold_method global_segment_quantile`. This independently pools segment scores across all `train` contracts and calculates $[1/3, 2/3]$ quantiles. Because this mirrors the quantile parameters used in `commodity_contract_dataset.py`, the calculated thresholds are mathematically identical while maintaining self-contained `slice_manifest.json` metadata for the training split.

3. **Source Split Parameterization in VAE Data Creation**:
   Refactor `FineFT/datahandler/vae_data_creation.py` to accept `--source_split {train, valid}` (defaulting to `"train"`), allowing flexible selection between training and validation data sources for VAE generation.

4. **Automated Stale Contract Cleanup in VAE Data Creation**:
   `vae_data_creation.py` must cleanly prune existing non-reserved contract subdirectories under `VAE_data/<labeling_method>/` prior to populating new arrays. This prevents catastrophic contract mixing when switching data sources.

5. **Transparent Downstream VAE Ingestion**:
   `merge_vae_train.py` and `RL/DiHFT/VAE/main.py` consume `VAE_data/<labeling_method>/<contract>/label_<k>.npy` transparently without modification. Switching the VAE training data source is handled entirely at the data preparation layer.
