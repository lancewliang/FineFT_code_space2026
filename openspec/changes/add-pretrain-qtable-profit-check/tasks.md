## 1. Sample plan and qtable cache implementation

- [x] 1.0 Complete sample plan, qtable cache, multiprocessing, and CSV diagnostics implementation. <!-- 已实现: 独立诊断模块、多进程 qtable cache、sample 级 CSV 导出、训练脚本接入 -->
- [x] 1.1 Create `FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py` for sample plan generation, qtable multiprocessing/caching, initial state helpers, DP path reward evaluation, and CSV export. <!-- 已实现: pretrain_qtable_diagnostics.py -->
- [x] 1.2 Update `DQN.train()` to build `sample_plan`, precompute qtables with multiprocessing, and export sample diagnostics before `for sample in range(self.num_sample)`. <!-- 已实现: train 前调用 prepare_pretrain_qtable_diagnostics -->
- [x] 1.3 Update the training loop to read `df_index` and `initial_action` from `sample_plan[sample]`. <!-- 已实现: loop 使用 sample_plan -->
- [x] 1.4 Update the pretrain branch to reuse `q_table_cache[df_index]` when creating `self.perfection_action_list`. <!-- 已实现: q_table_cache[df_index] -->
- [x] 1.5 Print or log per-sample qtable diagnostics including `sample_index`, `df_index`, `initial_action`, `episode_reward_sum`, and `profitable`. <!-- 已实现: logger + print sample 诊断 -->
- [x] 1.6 Write one diagnostic CSV per sample under `self.model_path/qtable_diagnostics/`, including OHLCV, `mark_price`, slippage, commission rate, actions, step reward, and cumulative profit. <!-- 已实现: sample_*.csv 明细 -->

## 2. Verification

- [ ] 2.0 Complete verification.
- [x] 2.1 Run `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py`. <!-- 已执行: 当前环境无 conda，使用 python3 -m py_compile 通过 -->
- [ ] 2.2 If local training data is available, run a small smoke command with `--num_sample 2 --pretrain_epoch 1` or the nearest existing script equivalent and confirm qtable diagnostics appear before the sample loop and one CSV is generated per sample.
- [x] 2.3 Run `openspec validate add-pretrain-qtable-profit-check --strict`. <!-- 已实现: OpenSpec strict valid -->
