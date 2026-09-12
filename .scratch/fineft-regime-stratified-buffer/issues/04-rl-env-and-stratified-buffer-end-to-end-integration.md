# 04: RL Environment and Stratified Buffer End-to-End Integration

**What to build:** An end-to-end integration between the trading environment, rollout workers, and the 9-grid regime-stratified replay buffer under the new macro segment labels. The environment exposes `regime_grid_id` directly in its step and reset information dictionaries, rollout workers compute 12-step discounted returns along continuous trajectories tagged by the initial decision step's macro regime, and the stratified buffer accurately routes transitions into 9 isolated queues for balanced 3-phase curriculum sampling.

**Blocked by:** 02: Train Dataset Full-Contract Regime Injection and Slicing

**Status:** ready-for-agent

- [ ] Trading environment loads slices containing `regime_grid_id` and exposes the current step's grid ID in reset and step `info` dictionaries with zero runtime calculation overhead
- [ ] Continuous trajectory 12-step return accumulator assigns the decision step's `regime_grid_id` to completed transitions
- [ ] Stratified replay buffer routes incoming transitions into the designated 9 isolated queues according to their macro `regime_grid_id`
- [ ] 3-phase rotating curriculum (Phase 0: Downtrend [0, 3, 6], Phase 1: Flat [1, 4, 7], Phase 2: Uptrend [2, 5, 8]) successfully executes balanced sampling across active queues over multi-epoch runs
- [ ] End-to-end rollout and buffer tests verify non-empty queue populations and balanced sampling without crashes
