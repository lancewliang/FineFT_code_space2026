# 04 — Upgrade Low-Level Model Trading Info Dimension

**What to build:** Upgrade the low-level Q network contract so default low-level models consume the four-field Trading Process Feature produced by the environment. This is a breaking model-input change: old three-field low-level checkpoints do not need compatibility loading.

**Blocked by:** 02 — Add Current Holding Duration In Base Env.

**Status:** ready-for-agent

- [x] Default low-level Q network construction accepts four-field `trading_info`.
- [x] Default ensemble low-level Q network construction accepts four-field `trading_info`.
- [x] Model factory helpers use the upgraded default contract.
- [x] Focused model tests assert four-field inputs work by default.
- [x] No automatic migration, padding, or compatibility shim is added for old three-field checkpoints.
