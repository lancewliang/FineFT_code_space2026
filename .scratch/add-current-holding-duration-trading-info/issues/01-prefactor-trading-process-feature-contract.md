# 01 — Prefactor Trading Process Feature Contract

**What to build:** Make the existing Trading Process Feature contract easier to update by centralizing its field order, dimension, and zero-value shape behind the environment-owned contract. This ticket should preserve the current three-field behavior while removing scattered assumptions that would make the four-field upgrade brittle.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] Trading Process Feature field order remains the single source of truth for its array dimension.
- [ ] Reset, normal step, terminal, and liquidation paths produce zero-value `trading_info` through the same contract-aware helper.
- [ ] Existing environment and low-level model focused tests still pass with the current three-field behavior.
- [ ] No low-level model or training behavior is changed in this prefactor.
