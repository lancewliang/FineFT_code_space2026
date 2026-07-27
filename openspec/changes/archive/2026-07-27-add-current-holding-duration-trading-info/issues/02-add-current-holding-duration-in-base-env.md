# 02 — Add Current Holding Duration In Base Env

**What to build:** Upgrade the Futures Trading Environment contract so every observable `trading_info` contains `current_holding_duration_norm` as the fourth Trading Process Feature. The duration is counted in env steps from the actual position state: flat is 0, a non-zero initial position starts at 1, opening from flat starts at 1, same-direction holding/add/reduce increments without reset, closing resets to 0, and reversing starts the new direction at 1.

**Blocked by:** 01 — Prefactor Trading Process Feature Contract.

**Status:** ready-for-agent

- [x] `trading_info` exposes four fields in the documented order, ending with `current_holding_duration_norm`.
- [x] The default normalization window is 180 env steps.
- [x] `current_holding_duration_norm` is calculated as `min(current_holding_duration / holding_duration_norm_steps, 1.0)`.
- [x] Invalid normalization windows fail fast instead of creating an environment with invalid duration values.
- [x] Focused environment tests cover flat reset, non-zero reset, open, hold, same-direction add, same-direction reduce, close, reverse, and clipping to 1.0.
