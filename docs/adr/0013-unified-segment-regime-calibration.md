# ADR-0013: Unified Segment-Level Regime Calibration for Train and Valid Datasets

Date: 2026-09-12

Status: Accepted (Supersedes Decision 4 of ADR-0012)

## Context

ADR-0012 established a 9-grid Regime-Stratified Replay Buffer and Rotating Epoch Curriculum to eliminate behavioral homogenization during Stage I diverse training. However, Decision 4 of ADR-0012 introduced a "Dual-Track Regime Granularity" where training data utilized a 48-bar causal rolling window while validation data utilized a macroeconomic turning-point segment-and-merge algorithm (`slice_and_merge`).

This dual-track approach resulted in:
1. Regime Definition Disconnect: Step-level rolling noise caused rapid micro-regime oscillations during training that did not correspond to the macro market waves evaluated during Stage II.
2. Code Duplication and Divergence: Stage I relied on `FineFT/RL/util/calibrate_regime_thresholds.py` rolling calculations, while Stage II relied on `FineFT/datahandler/valid_cross_contract_label_calibration.py`, creating parallel calibration pipelines.
3. Training Ground Truth Alignment: Since `regime_grid_id` is only passed in the environment `info` dictionary for experience replay routing and epoch curriculum scheduling (and is never part of the agent observation state vector $S_t$), assigning macro segment labels to historical training data introduces zero future price leakage into the policy network.

## Decision

1. **Unify Slicing and Regime Algorithm Under Turning-Point Macro Segmentation**:
   Retire the 48-bar causal rolling window. Both training and validation datasets calculate market regimes using `datahandler`'s `Worker(slice_and_merge)` bilateral filtering, turning-point identification, and DTW segment merging.

2. **Row-Level Feature Annotation on Continuous Long Slices**:
   Training datasets retain their continuous long-slice structure (`chunk_length=3200` slices, e.g. `df_0.feather` ... `df_N.feather`). The 3×3 regime label is injected as a row-level column (`regime_grid_id`) into each step. This preserves continuous multi-step discounted returns (12-step N-step return) and position episode tracking in the trading environment without artificial file boundary truncations.

3. **Two-Dimensional Joint 3×3 Regime Grid Topology**:
   The calibration pipeline simultaneously computes both slope (`signed_percentage_slope`) and volatility (`segment_log_return_volatility`) dimensions, mapping each time step to `slope_label \in {0, 1, 2}`, `volatility_label \in {0, 1, 2}`, and synthesizing `regime_grid_id = volatility_label * 3 + slope_label \in [0, 8]`.

4. **Symmetric Independent Calibration Scopes**:
   Training and validation sets calibrate quantile thresholds independently on their respective contract pools:
   - Training set: `fit_scope = "train_all_contracts"`, persisting to `train/regime_thresholds.json`.
   - Validation set: `fit_scope = "valid_all_contracts"`, persisting to `valid/slope/slice_manifest.json` and `valid/volatility/slice_manifest.json`.

5. **Strict Enforcement of the Negative Slope Boundary Invariant**:
   All calibrations across both training and validation must enforce $T_{\text{slope}}[0] < 0.0$ and $T_{\text{slope}}[1] > 0.0$. If the empirical lower slope tercile is non-negative or the upper tercile non-positive, calibration must fail fast with an error.

6. **Decoupled 2D Calibration Core Engine**:
   Extract pure computation (turning-point extraction, 2D score pooling, quantile fitting, invariant checking, row-level mapping) into `FineFT/datahandler/regime_calibration_engine.py`. Dataset generation pipelines (`commodity_contract_dataset.py`) and validation slicing pipelines (`valid_cross_contract_label_calibration.py`) delegate directly to this shared engine.

7. **Validation File Structure Backward Compatibility**:
   Validation slicing maintains its dual-track directory structure (`valid/slope` and `valid/volatility`) to ensure 100% backward compatibility with downstream Stage II selection scripts (`FineFT_two_dimensional_agent_selector.py`), while embedding `regime_grid_id` in all slice files.

8. **Strict Single Price Column Contract**:
   Baseline price column resolution defaults to `mark_price`. Missing required price columns must fail fast rather than performing speculative fallbacks.
