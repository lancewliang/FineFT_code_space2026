# Spec: 9-Grid Regime-Stratified Replay Buffer and Rotating Epoch Curriculum with Unified Macro-Segment Calibration

Related Documents:
- Research Design Report: [docs/research/nine_grid_regime_stratified_buffer_design.md](/home/lanceliang/opt/aiwork/FineFT_code_space2026/docs/research/nine_grid_regime_stratified_buffer_design.md:1)
- Architectural Decision Records: [docs/adr/0012-regime-stratified-buffer-and-rotating-curriculum.md](/home/lanceliang/opt/aiwork/FineFT_code_space2026/docs/adr/0012-regime-stratified-buffer-and-rotating-curriculum.md:1), [docs/adr/0013-unified-segment-regime-calibration.md](/home/lanceliang/opt/aiwork/FineFT_code_space2026/docs/adr/0013-unified-segment-regime-calibration.md:1)
- Domain Glossary: [CONTEXT.md](/home/lanceliang/opt/aiwork/FineFT_code_space2026/CONTEXT.md:328)
- Downstream 2D Agent Selector: [FineFT/analysis/pick_agent/FineFT_two_dimensional_agent_selector.py](/home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/analysis/pick_agent/FineFT_two_dimensional_agent_selector.py:98)

Triage Label: `ready-for-agent`

---

## Problem Statement

During Stage I diverse training in the FineFT reinforcement learning pipeline, all transitions gathered by parallel rollout workers across different contracts and market regimes are pooled into a single, monolithic global replay buffer (`buffer_diverse`). Because financial time series are heavily dominated by common market states (e.g. low-volatility range-bound periods) and relatively starved of rare states (e.g. low-volatility directional bear trends), uniform mini-batch sampling causes gradients to reflect only the statistical average of the market. Opposing directional signals (long and short) frequently collide within the same mini-batch, muting specialized directional responses.

As a result, sub-agents and policy checkpoints experience severe behavioral homogenization across training epochs. When evaluated in Stage II by the 2D Agent Selector across orthogonal volatility and slope slots, all candidate checkpoints fail to deliver positive returns on challenging or scarce market regimes (such as low-volatility downward trends), causing slots to systematically fall back to `empty_model`.

Furthermore, a critical definition disconnect previously existed between training and evaluation: Stage I relied on a 48-bar causal rolling window producing noisy step-level classifications, while Stage II evaluation slices were constructed from macroeconomic turning-point segments (`slice_and_merge`). This divergence broke alignment between training experience routing and downstream evaluation regimes, while duplicating regime calibration logic across separate modules.

## Solution

Unify the regime classification and slicing algorithm across training and validation pipelines using datahandler's macroeconomic turning-point segment-and-merge algorithm as the canonical reference, and decompose the monolithic replay buffer into a 9-grid (3x3 volatility x slope) Regime-Stratified Replay Buffer (`RegimeStratifiedReplayBuffer`).

A shared 2D calibration core engine extracts macro turning points and simultaneously scores both slope and volatility dimensions. Training datasets retain their continuous long-slice structure (`chunk_length=3200`) to preserve continuous 12-step discounted return calculation and position episode trajectories, with the 3x3 `regime_grid_id` (0..8) injected at the row level prior to chunk slicing. Training and validation datasets independently calibrate their tercile thresholds on their respective contract sets, enforcing strict negative lower slope invariants ($T_{slope}[0] < 0.0$ and $T_{slope}[1] > 0.0$).

To eliminate cross-regime and cross-contract sequence corruption during N-step return accumulation, compute 12-step discounted returns strictly along continuous single-episode rollout trajectories on the worker side *before* routing transitions into buffers, labeling each transition by its initial decision step's regime.

To induce massive policy divergence across checkpoints, introduce a Rotating Epoch Curriculum that trains exclusively on discrete directional regime subsets in blocks of at least 3 epochs per phase (Phase 0: Downtrend, Phase 1: Range/Flat, Phase 2: Uptrend), drawing balanced sample quotas across active queues.

## User Stories

1. As an RL research engineer, I want the replay buffer to be partitioned into 9 isolated regime queues based on 3x3 orthogonal volatility and slope coordinates, so that rare market dynamics are not overwritten by high-frequency market noise.
2. As an RL research engineer, I want each regime queue to have an independent FIFO eviction boundary, so that an influx of high-volatility samples cannot evict low-volatility samples from memory.
3. As a quantitative developer, I want N-step discounted returns to be calculated strictly along continuous rollout trajectories before buffer insertion, so that transitions crossing regime boundaries do not stitch together chronologically unrelated steps.
4. As a quantitative developer, I want the trailing steps of an episode (fewer than N steps remaining) to be accumulated via truncated discounting with terminal flags set, so that terminal position unwinds are retained rather than discarded.
5. As an RL practitioner, I want each completed N-step transition to be routed to a queue based on the regime at the initial decision step, so that policy actions are evaluated against the market regime that motivated them.
6. As a strategy researcher, I want training epochs to rotate through distinct directional market regimes in separate phases, so that checkpoints specialize deeply into directional expertise without opposing gradient interference.
7. As a strategy researcher, I want downtrend (bear) regimes and uptrend (bull) regimes to be trained in strictly separate curriculum phases, so that long and short gradient updates do not neutralize each other within the same training block.
8. As a training system operator, I want each curriculum phase to execute for a block of at least 3 consecutive epochs before switching, so that network weights undergo sufficient gradient iterations to adapt to that market dynamic.
9. As an RL researcher, I want the curriculum to follow a 3-phase cycle (Phase 0: Downtrend grids [0, 3, 6], Phase 1: Range/Flat grids [1, 4, 7], Phase 2: Uptrend grids [2, 5, 8]), so that distinct checkpoints emerge across training epochs.
10. As a training pipeline designer, I want the stratified sampler to allocate balanced batch quotas across active regime queues, so that scarce regimes contribute equally to model updates during active phases.
11. As a robust system designer, I want the stratified sampler to fall back gracefully to sampling with replacement when an active queue contains fewer samples than its assigned quota, so that early training epochs do not crash from cold-start scarcity.
12. As a data pipeline engineer, I want training and validation datasets to share a single, canonical turning-point segment-and-merge algorithm, so that training experience routing perfectly aligns with downstream evaluation market regimes.
13. As a systems architect, I want the core 2D turning-point extraction, quantile fitting, and row-level label mapping logic to be abstracted into a dedicated datahandler engine module, so that training and validation pipelines reuse the exact same numerical implementation without duplication.
14. As a quantitative developer, I want the training pipeline to fit macro turning points and inject row-level 3x3 `regime_grid_id` on full continuous contract files before slicing into chunks, so that continuous price trends are not corrupted by artificial chunk boundaries.
15. As an RL practitioner, I want training datasets to retain continuous fixed-length slices (`chunk_length=3200`) with row-level regime annotations, so that multi-step returns and position holding dynamics are uninterrupted.
16. As a data pipeline engineer, I want the training pipeline to independently calibrate its tercile thresholds on all participating training contracts (`fit_scope="train_all_contracts"`) and persist the result to `train/regime_thresholds.json`.
17. As an evaluation engineer, I want the validation pipeline to independently calibrate its tercile thresholds on all participating validation contracts (`fit_scope="valid_all_contracts"`) and persist manifests to validation subdirectories.
18. As a domain specialist, I want the calibration engine to strictly enforce that the lower slope threshold is negative ($T_{slope}[0] < 0.0$) and the upper slope threshold positive ($T_{slope}[1] > 0.0$) across all datasets, failing fast if empirical distributions violate economic trend semantics.
19. As a system architect, I want regime coordinates in code, manifests, and CLI options to remain purely numerical and semantic-free (e.g. grid IDs 0..8, vol_bin 0..2, slope_bin 0..2), preserving mathematical purity.
20. As an environment developer, I want the trading environment to expose `regime_grid_id` directly in its step and reset `info` dictionaries with zero runtime calculation overhead.
21. As a downstream evaluation analyst, I want validation dataset outputs to retain their `valid/slope` and `valid/volatility` folder hierarchies while embedding 2D regime columns, so that the Stage II 2D Agent Selector runs without modifications.
22. As an engineer upholding fail-fast principles, I want price column resolution to default strictly to `mark_price` and fail fast when missing, eliminating speculative fallbacks and hidden schema inconsistencies.
23. As an ML engineer, I want the stratified buffer to support content and semantic-key deduplication with TD-error priority replacement per grid, so that high-information samples displace low-information duplicates within each regime.
24. As a performance engineer, I want `RegimeStratifiedReplayBuffer` to extract pre-stacked contiguous PyTorch tensors for each grid, so that GPU sampling remains zero-copy and eliminates element-wise array stacking overhead.
25. As a Stage II selection analyst, I want checkpoints produced by rotating curriculum epochs to exhibit diverse Q-value landscapes and action preferences, so that the 2D selector finds winning models across all 9 slots without defaulting to empty models.
26. As an automated testing agent, I want high-level execution seams covering core calibration, dataset slice generation, valid dataset generation, and end-to-end curriculum rotation, ensuring tests verify observable contracts rather than private implementation details.

## Implementation Decisions

### Modular Separation and 2D Calibration Core Engine
- Implement a dedicated, side-effect-free calibration engine in datahandler (`regime_calibration_engine.py`) encapsulating:
  - Turning-point identification via bilateral filtering (`Worker(slice_and_merge)`) and DTW segment merging.
  - Calculation of segment slope (`100 * coef_ / price_start`) and segment volatility (`100 * std(diff(log(prices)), ddof=0)`).
  - Cross-contract pooling of segment scores and quantile threshold fitting with strict invariant enforcement.
  - Row-level mapping from segments and thresholds to `slope_label \in {0, 1, 2}`, `volatility_label \in {0, 1, 2}`, and synthesized `regime_grid_id = volatility_label * 3 + slope_label \in [0, 8]`.
  - Application of limit-up and limit-down boundary rules per existing project conventions.
- Both dataset generation (`commodity_contract_dataset.py`) and validation calibration (`valid_cross_contract_label_calibration.py`) delegate directly to this core engine.

### Continuous Long Slices with Injected Row-Level Regime Annotations
- Training data is NOT physically fragmented into short segment files. Slices remain continuous 3200-step time series (`chunk_length=3200` plus `early_stop=320`).
- Segmentation and calibration execute on full contract files (`train/<contract>.feather`) during dataset creation before chunk slicing. Slices produced by `write_train_slices` inherit row-level `regime_grid_id` values seamlessly.
- Baseline price column resolution is strictly configured to `mark_price`, failing fast if the column is absent.

### Independent Symmetric Calibration Scopes
- Training datasets calibrate tercile thresholds across all participating training contracts with `fit_scope="train_all_contracts"`, persisting to `train/regime_thresholds.json`.
- Validation datasets calibrate tercile thresholds across all participating validation contracts with `fit_scope="valid_all_contracts"`, persisting to `valid/slope/slice_manifest.json` and `valid/volatility/slice_manifest.json`.
- Both scopes enforce the invariant $T_{slope}[0] < 0.0$ and $T_{slope}[1] > 0.0$.

### Validation File Contract Compatibility
- Validation datasets maintain dual-track physical directory structures (`valid/slope/<contract>/label_*/df_*.feather` and `valid/volatility/<contract>/label_*/df_*.feather`).
- Generated slice files contain row-level `slope_label`, `volatility_label`, and `regime_grid_id` columns, enabling full backward compatibility with the existing Stage II 2D Agent Selector (`FineFT_two_dimensional_agent_selector.py`).

### Trajectory-First N-Step Return Accumulation
- Rollout workers compute 12-step discounted returns locally on the continuous episode trajectory before transmitting completed transition records to the main process.
- For steps $t$ where $t + N > T$, the discounted sum is accumulated over the remaining $T - t$ steps, and the transition is marked with `done = True`.
- Transitions carry the initial step's `regime_grid_id` (derived from the macro turning-point segment), which governs buffer routing.

### Buffer Topology and Per-Grid Eviction
- The stratified buffer instantiates 9 isolated underlying queue structures, each sized to $\lfloor \text{total\_capacity} / 9 \rfloor$.
- Underlying queues use single-step configuration because multi-step discounting has already been resolved along continuous trajectories.
- Each queue maintains an independent deduplication tracker mapping semantic decision keys to buffer indices and TD-errors, performing in-place replacement when an incoming transition has a higher TD-error.
- Overwriting within one queue follows a circular FIFO policy and has strictly zero impact on the occupancy or contents of the other 8 queues.

### Rotating Epoch Curriculum Specification
- Training epochs follow a 3-phase pure directional cycle parameterized by a configurable block size of at least 3 epochs per phase:
  - Phase 0 (Downtrend / Short): Active queues `[0, 3, 6]` (low, mid, high volatility bear states).
  - Phase 1 (Range / Flat / Neutral): Active queues `[1, 4, 7]` (low, mid, high volatility flat states).
  - Phase 2 (Uptrend / Long): Active queues `[2, 5, 8]` (low, mid, high volatility bull states).
- Phase index is determined by `(epoch // block_epochs) % 3`.

### Stratified Sampling Allocation
- For a batch size $B$ and $K$ active queues, each active queue is assigned an integer quota $b = B // K$, with remainder $B \% K$ distributed across the initial queues.
- Samples are drawn without replacement when queue size $\ge$ quota, and dynamically with replacement (accompanied by a diagnostic warning) when queue size < quota.
- Sub-batches are stacked into continuous tensors along dimension 0, ensuring consistent mini-batch shapes across all training updates.

## Testing Decisions

### Good Test Philosophy
- Tests must assert observable external contracts: correct 2D turning-point extraction, invariant enforcement on slope boundaries, full row coverage with valid `regime_grid_id \in [0, 8]`, proper continuous slice generation, accurate queue routing, exact sampling quotas, and multi-epoch curriculum rotation.
- Tests must avoid mocking internal deque states or asserting private implementation structures.

### Execution Seams Under Test
1. **Core 2D Calibration Engine Seam (`regime_calibration_engine.py`)**: Tested on synthetic multi-contract price series for simultaneous slope and volatility segmentation, tercile threshold calibration, invariant rejection ($T_{slope}[0] \ge 0$), limit state overrides, and complete row-level label mapping.
2. **Dataset Generation Pipeline Seam (`commodity_contract_dataset.py`)**: Tested for end-to-end dataset generation (`run_dataset_generation` / `write_train_slices`), verifying full contract annotation, `regime_thresholds.json` emission, and continuous slice inheritance of `regime_grid_id`.
3. **Validation Pipeline Seam (`valid_cross_contract_label_calibration.py`)**: Tested for directory-level atomic calibration, emission of dual-track `valid/slope` and `valid/volatility` hierarchies, independent validation thresholding, and slice column verification.
4. **Environment & Stratified Replay Buffer Seam (`commodity_env.py` + `RegimeStratifiedReplayBuffer`)**: Tested for step-level exposure of `info["regime_grid_id"]`, trajectory-first 12-step return accumulation, and exact routing into the matching 9-grid queue.
5. **Parallel Diverse Training Seam (`run_parallel_diverse_training`)**: Tested for multi-epoch 3-phase curriculum rotation, balanced batch extraction across active queues, and checkpoint production without crashes.

### Prior Art
- `FineFT/tests/datahandler/test_valid_cross_contract_label_calibration.py`: Baseline for atomic cross-contract turning-point calibration testing.
- `FineFT/tests/datahandler/test_commodity_contract_dataset.py`: Baseline for train dataset slicing and plan execution testing.
- `FineFT/tests/rl/test_regime_stratified_replay_buffer.py`: Baseline for 9-grid routing and independent capacity eviction testing.
- `FineFT/tests/rl/test_regime_curriculum_diverse_train.py`: Baseline for 3-phase rotating curriculum scheduling testing.

## Out of Scope

- Modifying downstream Stage II 2D Agent Selector evaluation logic or converting Stage II validation slices from `valid/slope` and `valid/volatility` to a 9-folder structure (backward compatibility is strictly preserved).
- Modifying the VAE architecture or Stage III Meta Router routing mechanisms.
- Altering Algorithm 2 partial loss weighting formulas or sub-agent 1-to-1 hard binding in Phase 1.

## Further Notes

- By unifying training set regime labeling with datahandler's turning-point segment-and-merge algorithm, Stage I experience stratification mirrors the ground truth market waves evaluated in Stage II.
- Because `regime_grid_id` is an offline experience routing attribute in `info` and is never present in `state_features.npy`, macro-segment annotation introduces zero future information leakage into the RL policy network.
