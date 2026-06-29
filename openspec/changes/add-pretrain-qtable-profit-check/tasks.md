## 1. Sample plan and qtable cache implementation

- [ ] 1.0 Complete sample plan and qtable cache implementation.
- [ ] 1.1 Add focused helper methods inside `FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py` for sample plan generation, qtable creation/caching, initial state setup, and DP path reward evaluation.
- [ ] 1.2 Update `DQN.train()` to build `sample_plan` and `q_table_cache` before `for sample in range(self.num_sample)`.
- [ ] 1.3 Update the training loop to read `df_index` and `initial_action` from `sample_plan[sample]`.
- [ ] 1.4 Update the pretrain branch to reuse `q_table_cache[df_index]` when creating `self.perfection_action_list`.
- [ ] 1.5 Print or log per-sample qtable diagnostics including `sample_index`, `df_index`, `initial_action`, `episode_reward_sum`, and `profitable`.

## 2. Verification

- [ ] 2.0 Complete verification.
- [ ] 2.1 Run `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`.
- [ ] 2.2 If local training data is available, run a small smoke command with `--num_sample 2 --pretrain_epoch 1` or the nearest existing script equivalent and confirm qtable diagnostics appear before the sample loop.
- [ ] 2.3 Run `openspec validate add-pretrain-qtable-profit-check --strict`.
