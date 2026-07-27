# 03 — Propagate Holding Duration Config Through Env Constructors

**What to build:** Make the holding duration normalization window configurable through the environment construction paths that create Futures Trading Environments, while keeping the default behavior at 180 env steps. All environment variants should surface the same four-field Trading Process Feature contract.

**Blocked by:** 02 — Add Current Holding Duration In Base Env.

**Status:** ready-for-agent

- [x] Environment construction paths accept `holding_duration_norm_steps` without requiring every caller to set it.
- [x] The default value remains 180 wherever no explicit value is provided.
- [x] Demo, commodity, aggregate, and base initiation flows expose compatible four-field `trading_info`.
- [x] Focused smoke tests or constructor tests verify the default and an overridden normalization window.
