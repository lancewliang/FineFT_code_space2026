# 05: Cleanup Obsolete Regime Code and Legacy Artifacts

**What to build:** Deletion and removal of all dead code, obsolete implementations, and legacy components superseded by the unified 2D macro-segment calibration pipeline. This removes the legacy 48-bar causal rolling calculations, cleans up redundant single-dimension slice-and-merge helpers, eliminates unused imports and dead test fixtures, and ensures no backward-compatibility baggage remains in the codebase.

**Blocked by:** 02: Train Dataset Full-Contract Regime Injection and Slicing, 03: Validation Pipeline Refactor and Dual-Track Directory Compatibility, 04: RL Environment and Stratified Buffer End-to-End Integration

**Status:** ready-for-agent

- [ ] Legacy 48-bar causal rolling slope and volatility calculation functions are removed or replaced with thin delegation to the core 2D engine
- [ ] Obsolete single-dimension slicing helpers and unused intermediate functions in datahandler are deleted
- [ ] Superseded test cases targeting dead causal rolling code are removed or migrated to the new core engine contract
- [ ] Codebase conforms to fail-fast guidelines with zero legacy compatibility shims or fallback branches
- [ ] Full regression test suite across datahandler, environment, and RL training passes with all tests green
