# Spec: 9-Grid Regime-Stratified Replay Buffer and Rotating Epoch Curriculum

Related Documents:
- Research Design Report: [docs/research/nine_grid_regime_stratified_buffer_design.md](/home/lanceliang/opt/aiwork/FineFT_code_space2026/docs/research/nine_grid_regime_stratified_buffer_design.md:1)
- Architectural Decision Record: [docs/adr/0012-regime-stratified-buffer-and-rotating-curriculum.md](/home/lanceliang/opt/aiwork/FineFT_code_space2026/docs/adr/0012-regime-stratified-buffer-and-rotating-curriculum.md:1)
- Domain Glossary: [CONTEXT.md](/home/lanceliang/opt/aiwork/FineFT_code_space2026/CONTEXT.md:328)
- Downstream 2D Agent Selector: [FineFT/analysis/pick_agent/FineFT_two_dimensional_agent_selector.py](/home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/analysis/pick_agent/FineFT_two_dimensional_agent_selector.py:98)

Triage Label: `ready-for-agent`

---

## Problem Statement

During Stage I diverse training in the FineFT reinforcement learning pipeline, all transitions gathered by parallel rollout workers across different contracts and market regimes are pooled into a single, monolithic global replay buffer (`buffer_diverse`). Because financial time series are heavily dominated by common market states (e.g. low-volatility range-bound periods) and relatively starved of rare states (e.g. low-volatility directional bear trends), uniform mini-batch sampling causes gradients to reflect only the statistical average of the market. Opposing directional signals (long and short) frequently collide within the same mini-batch, muting specialized directional responses.

As a result, sub-agents and policy checkpoints experience severe behavioral homogenization across training epochs. When evaluated in Stage II by the 2D Agent Selector across orthogonal volatility and slope slots, all candidate checkpoints fail to deliver positive returns on challenging or scarce market regimes (such as low-volatility downward trends), causing slots to systematically fall back to `empty_model`.

## Solution

Decompose the monolithic replay buffer into a 9-grid (3x3 volatility x slope) Regime-Stratified Replay Buffer (`RegimeStratifiedReplayBuffer`) implemented in a dedicated module, where each regime maintains an independent queue, capacity quota, and FIFO eviction boundary.

To eliminate cross-regime and cross-contract sequence corruption during N-step return accumulation, compute 12-step discounted returns strictly along continuous single-episode rollout trajectories on the worker side *before* routing transitions into buffers, labeling each transition by its initial decision step's regime.

To induce massive policy divergence across checkpoints, introduce a Rotating Epoch Curriculum that trains exclusively on discrete directional regime subsets in blocks of at least 3 epochs per phase (Phase 0: Downtrend, Phase 1: Range/Flat, Phase 2: Uptrend), drawing balanced sample quotas across active queues. Enforce by construction that the lower slope threshold separating downtrend and flat regimes is strictly negative, ensuring down-market training never ingests non-negative price paths.

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
12. As a data pipeline engineer, I want an automatic regime threshold calibrator that computes pooled terciles from training slices, so that binning thresholds adapt objectively to different commodity datasets.
13. As a domain specialist, I want the regime calibrator to strictly enforce that the lower slope threshold is negative ($T_{slope}[0] < 0$), so that downtrend queues never contain flat or positive price trajectories.
14. As a domain specialist, I want the regime calibrator to strictly enforce that the upper slope threshold is positive ($T_{slope}[1] > 0$), so that slope 0 strictly resides inside the range-bound regime.
15. As a system architect, I want regime coordinates in code, manifests, and CLI options to remain purely numerical and semantic-free (e.g. grid IDs 0..8, vol_bin 0..2, slope_bin 0..2), so that mathematical purity is preserved in Stage I.
16. As an environment developer, I want the trading environment to expose `regime_grid_id` directly in its step and reset `info` dictionaries with zero runtime calculation overhead, so that parallel rollout performance is maximized.
17. As an ML engineer, I want the stratified buffer to support content and semantic-key deduplication with TD-error priority replacement per grid, so that high-information samples displace low-information duplicates within each regime.
18. As a performance engineer, I want `RegimeStratifiedReplayBuffer` to extract pre-stacked contiguous PyTorch tensors for each grid, so that GPU sampling remains zero-copy and eliminates element-wise array stacking overhead.
19. As a Stage II selection analyst, I want checkpoints produced by rotating curriculum epochs to exhibit diverse Q-value landscapes and action preferences, so that the 2D selector finds winning models across all 9 slots without defaulting to empty models.
20. As an automated testing agent, I want high-level execution seams that test end-to-end multi-epoch curriculum rotation and queue quota balance, so that regression tests verify actual training behavior rather than private implementation details.
21. As a data pipeline engineer, I want `calibrate_regime_thresholds.py` to support an `--apply` CLI flag to materialize `regime_grid_id` directly into `df_*.feather` slice files on disk, so that environments read precomputed labels with zero runtime overhead.
22. As a dataset pipeline maintainer, I want `commodity_contract_dataset.py:write_train_slices` to automatically inject `regime_grid_id` when generating train slices, so that future dataset generation cycles produce materialized slices out-of-the-box.

## Implementation Decisions

### Modular Separation
- `RegimeStratifiedReplayBuffer` must be implemented in its own dedicated module outside the monolithic `replay_buffer_DQN.py` file, keeping responsibilities decoupled and testable in isolation.
- The regime calibrator and threshold persistence logic must reside in a dedicated utility module with a standard JSON manifest schema.

### Trajectory-First N-Step Return Accumulation
- Rollout workers compute 12-step discounted returns locally on the continuous episode trajectory before transmitting completed transition records to the main process.
- For steps $t$ where $t + N > T$, the discounted sum is accumulated over the remaining $T - t$ steps, and the transition is marked with `done = True`.
- Transitions carry the initial step's `regime_grid_id` (determined by 48-bar causal rolling calculations on prices), which governs buffer routing.

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

### Threshold Calibration and Slice Materialization Pipeline
- Calibrator fits pooled terciles (33.333% and 66.667% quantiles) over causal 48-bar rolling slope and volatility arrays pooled across all training slices.
- The calibrator enforces the invariant $T_{slope}[0] < 0.0$ and $T_{slope}[1] > 0.0$. If a dataset yields a non-negative lower slope quantile, calibration must fail fast with an informative error rather than silently accepting corrupted directional definitions.
- **Physical Slice Materialization**: The calibrator CLI (`python -m RL.util.calibrate_regime_thresholds --slice_dir <dir> --apply`) writes `regime_thresholds.json` and updates all `df_*.feather` files on disk with the `regime_grid_id` column.
- **Dataset Pipeline Integration**: `commodity_contract_dataset.py:write_train_slices` automatically reads `regime_thresholds.json` and injects `regime_grid_id` into each sliced feather before saving.
- **Diagnostics Cache Ingestion**: `pretrain_qtable_diagnostics.py` automatically injects `regime_grid_id` via `ensure_regime_grid_id_column` when loading training DataFrames if the disk slice lacks the column.

## Testing Decisions

### Good Test Philosophy
- Tests must assert observable external contracts: correct routing into isolated queues, proper multi-step discounted values, exact active-queue sampling quotas, strict invariant enforcement, and epoch-to-phase transitions.
- Tests must avoid mocking internal deque states or asserting private implementation structures.

### Modules Under Test
1. **Regime Calibrator Module**: Tested for correct tercile computation on multi-slice data, accurate boundary mapping, rejection of invalid invariants ($T_{slope}[0] \ge 0$), and JSON manifest generation.
2. **Stratified Replay Buffer Module**: Tested for 9-grid routing accuracy, per-grid independent FIFO capacity saturation, TD-error priority deduplication, and zero cross-queue leakage.
3. **Trajectory Accumulator**: Tested for mathematical exactness of 12-step discounted returns on synthetic trajectories, terminal lookahead handling, and tail-step truncation.
4. **Stratified Stacked Sampler**: Tested for balanced quota extraction across diverse active queue combinations, replacement fallback on starved queues, and correct tensor output shapes.
5. **Parallel Diverse Training Loop**: Tested at the high-level seam (`run_parallel_diverse_training`) to verify multi-epoch curriculum scheduling, worker integration, and checkpoint generation without crashes.

### Prior Art
- `FineFT/tests/rl/test_semantic_prioritized_buffer.py`: Baseline for semantic key deduplication and TD-error priority replacement testing.
- `FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py`: Baseline for diverse training phase execution, stacked sampler integration, and worker communication testing.

## Out of Scope

- Modifying downstream Stage II 2D Agent Selector evaluation logic or converting Stage II validation slices to step-level causal rolling formats (ADR-0012 dual-track coexistence remains in force).
- Modifying the VAE architecture or Stage III Meta Router routing mechanisms.
- Altering Algorithm 2 partial loss weighting formulas or sub-agent 1-to-1 hard binding in Phase 1 (reserved for future exploratory phases if rotating curriculum alone proves insufficient).

## Further Notes

- The 9-grid empirical distribution measured on `fu 30min` confirms that all 9 grids have robust representation, with the scarcest grid (`v0, s0`) containing 994 transitions across 71 runs and 13 contracts.
- This specification fulfills the design established in research document `nine_grid_regime_stratified_buffer_design.md` and ratified by `ADR-0012`.
