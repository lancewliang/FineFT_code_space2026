## 1. Implementation

- [x] 1.1 Extend `Base_Env` Trading Process Feature tests to expect four-field `trading_info` and cover flat reset, non-zero reset, open, hold, add, reduce, close, reverse, and normalization clipping.
- [x] 1.2 Update `Base_Env` to maintain Current Holding Duration, expose `current_holding_duration_norm`, validate `holding_duration_norm_steps`, and remove hard-coded three-field zero `trading_info` arrays.
- [x] 1.3 Pass `holding_duration_norm_steps` through base/demo/commodity/aggregate environment constructors and initiation helpers where they construct `Base_Env`.
- [x] 1.4 Update `model.low_level` Qnet, ensemble_Qnet, and selected-qnet factory defaults to use four-field `trading_info`.
- [x] 1.5 Update Stage I low-level training and parallel training instantiation paths so low-level model dimensions match the environment `trading_info` contract.
- [x] 1.6 Update low-level testing, aggregate env, high-level routing, and related focused tests so model loading and inference use the four-field `trading_info` contract.
- [x] 1.7 Update replay-buffer tests and fixtures to use four-field `trading_info`.
- [x] 1.8 Update or add focused tests that assert old three-field assumptions are removed from the affected low-level model and environment seams.

## 2. Verification

- [x] 2.1 Run `source ~/miniconda3/etc/profile.d/conda.sh && conda activate finetf && pytest FineFT/tests/env/test_trading_process_features.py FineFT/tests/rl/test_qnet_trading_info.py FineFT/tests/rl/test_replay_buffer_trading_info.py -q`.
- [x] 2.2 Run focused tests for low-level test-agent and Stage I training helpers that are changed by the implementation.
- [x] 2.3 Run `source ~/miniconda3/etc/profile.d/conda.sh && conda activate finetf && python -m py_compile FineFT/env/env_class/base_env.py FineFT/model/low_level.py FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py`.
- [x] 2.4 Run `openspec validate add-current-holding-duration-trading-info --strict`.

## 3. Rollback Notes

- [ ] 3.1 If the feature must be rolled back before retraining low-level checkpoints, restore `TRADING_INFO_KEYS` and low-level `TRADING_INFO_DIM` to the previous three-field contract and rerun the focused env/model/replay-buffer tests.
