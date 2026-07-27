# 05 — Update Stage I Training And Replay Buffer Flow

**What to build:** Make Stage I low-level training collect, store, sample, and train with the four-field Trading Process Feature from the environment. Replay buffers should preserve `trading_info` as environment data and should not synthesize or append holding duration themselves.

**Blocked by:** 02 — Add Current Holding Duration In Base Env; 04 — Upgrade Low-Level Model Trading Info Dimension.

**Status:** ready-for-agent

- [x] Stage I low-level training paths construct low-level models with the upgraded trading info dimension.
- [x] Serial and parallel training flows pass environment-produced four-field `trading_info` into model updates.
- [x] Replay buffer sample paths preserve current and next-state `trading_info` with last dimension 4.
- [x] Focused replay buffer and training-path tests are updated to the four-field contract.
- [x] No training path manually appends `current_holding_duration_norm` outside the environment contract.
