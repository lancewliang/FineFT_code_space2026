# ADR-0012: Regime-Stratified Replay Buffer and Rotating Curriculum

Date: 2026-09-12

Status: Accepted

## Context

Stage I diverse training suffered from policy homogenization across sub-agents and checkpoints because all transitions were pooled into a single shared replay buffer (`buffer_diverse`). Mini-batches drawn uniformly reflected average market statistics, muting specialized responses on scarce regimes (such as low-volatility bear runs). In downstream Stage II selection, candidates failed to show positive performance across extreme grids, causing slots to fall back to `empty_model`.

## Decision

1. **3×3 Regime Stratified Topology**: Decompose the monolithic buffer into 9 isolated queues (`RegimeStratifiedReplayBuffer`), indexed by `(vol_bin, slope_bin) \in {0, 1, 2} \times {0, 1, 2}`. Each queue possesses an independent capacity quota and FIFO eviction boundary to guarantee that frequent regimes cannot overwrite scarce regime samples.
2. **Trajectory-First N-Step Accumulation**: Calculate 12-step discounted returns strictly along continuous single-episode rollout trajectories *before* transition routing. Each completed transition is indexed and stored by its initial decision step's regime label (`info_t["regime_grid_id"]`), eliminating chronological corruption across disjoint regime visits.
3. **Rotating Epoch Curriculum**: Induce cross-epoch policy divergence by activating discrete regime subsets per epoch rather than forcing sub-agent 1-to-1 hard binding in Phase 1. Crucially, Uptrend (slope_bin=2, grids [2, 5, 8]), Downtrend (slope_bin=0, grids [0, 3, 6]), and Flat/Range (slope_bin=1, grids [1, 4, 7]) are trained in strictly separated phases in a 3-phase pure directional rotation (Phase 0: Downtrend [0, 3, 6], Phase 1: Flat [1, 4, 7], Phase 2: Uptrend [2, 5, 8]). Each phase executes for a block of at least 3 epochs before switching, ensuring the policy has sufficient gradient iterations to deeply adapt to each directional regime. Mini-batches sample uniformly across active queues to correct natural market sample skew.
4. **Dual-Track Regime Granularity**: Use 48-bar causal rolling calculations in `base_env` for Step-level Causal Regime Labels during Stage I collection, while preserving segment-level macroeconomic slice evaluation in Stage II to maintain validation integrity and cross-experiment comparability.
5. **Semantic-Free Coordinates and Negative Slope Boundary Invariant**: Reject subjective market regime labels in code and configurations; all queues, thresholds, and curriculum schedules strictly use numerical coordinates `grid_id \in [0, 8]` and `(vol_bin, slope_bin)`. An auto-calibrator generates `regime_thresholds.json` via pooled terciles, but enforces the strict invariant that the lower slope boundary must be negative ($T_{\text{slope}}[0] < 0$) and the upper boundary positive ($T_{\text{slope}}[1] > 0$) so that slope 0 strictly resides inside `slope_bin = 1` (flat) and `slope_bin = 0` (down) never admits non-negative trajectories.
