# 01: Targeted Sub-Network Rollout Inference

**What to build:** Refactor exploration action selection in `FineFT/RL/DiHFT/low_level/parallel_diverse_train.py` (`DfRolloutWorkerRunner._act`). Replace full-ensemble evaluation (`self.model(...)`, which computes all 13 sub-networks in `ensemble_Qnet` and discards 12) with targeted single-subnetwork forward propagation on `self.model.qnet_list[context_index]`. Deliver identical action and Q-value outputs with a 92.3% reduction in forward computation, dropping rollout inference time by over 10x.

**Blocked by:** None (can start immediately)

**Status:** completed

- [x] `DfRolloutWorkerRunner._act` calls `self.model.qnet_list[context_index]` directly, outputting masked Q-values of shape `(1, N_ACTIONS)` without iterating through irrelevant subnets.
- [x] Action selection (`argmax`) and chosen Q-value remain mathematically identical to the baseline ensemble output for any given `context_index`.
- [x] Action persistence logic (`action_persistence`) continues to correctly preserve non-flat actions across consecutive timesteps.
- [x] Unit tests verify forward inference output equivalence between targeted sub-network invocation and full ensemble invocation across randomized state and trading context inputs.
