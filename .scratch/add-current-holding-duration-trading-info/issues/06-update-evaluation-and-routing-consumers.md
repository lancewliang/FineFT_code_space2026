# 06 — Update Evaluation And Routing Consumers

**What to build:** Make low-level evaluation, aggregate environment usage, and high-level routing consumers use the upgraded four-field low-level model contract. A newly trained four-field low-level model should be loadable and usable along these inference paths.

**Blocked by:** 03 — Propagate Holding Duration Config Through Env Constructors; 04 — Upgrade Low-Level Model Trading Info Dimension.

**Status:** ready-for-agent

- [ ] Low-level evaluation paths construct and call models with four-field `trading_info`.
- [ ] Aggregate environment and routing consumers no longer hard-code the old three-field dimension.
- [ ] Focused inference-path tests or smoke checks cover the upgraded contract.
- [ ] Old three-field checkpoint loading is allowed to fail with a shape mismatch and is not silently adapted.
