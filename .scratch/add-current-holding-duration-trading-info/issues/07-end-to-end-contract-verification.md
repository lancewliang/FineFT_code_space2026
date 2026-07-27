# 07 — End-To-End Contract Verification

**What to build:** Verify the complete upgraded Trading Process Feature contract from environment output through low-level model consumers, and prepare the OpenSpec change to be marked complete once implementation lands.

**Blocked by:** 01 — Prefactor Trading Process Feature Contract; 02 — Add Current Holding Duration In Base Env; 03 — Propagate Holding Duration Config Through Env Constructors; 04 — Upgrade Low-Level Model Trading Info Dimension; 05 — Update Stage I Training And Replay Buffer Flow; 06 — Update Evaluation And Routing Consumers.

**Status:** ready-for-agent

- [ ] Focused environment, low-level model, replay buffer, training, and inference tests pass in the project conda environment.
- [ ] Changed Python entry points compile successfully.
- [ ] OpenSpec strict validation passes for `add-current-holding-duration-trading-info`.
- [ ] Any remaining broad-suite failures are documented with whether they are related to this contract upgrade.
- [ ] The OpenSpec task list is updated to reflect completed implementation and verification work.
