## 1. Implementation

- [ ] 1.1 Extend trading wallet-change results to expose true `commission_fee_step`, `realized_pnl_step`, and `slippage_step` while preserving existing fee semantics.
- [ ] 1.2 Update `Base_Env` and `Simple_Env` callers to consume the explicit wallet-change result and expose/reset per-step and cumulative execution metrics.
- [ ] 1.3 Update `test_agent_index.py` aggregate result collection to include `df_path` and write default `analysis_result.csv` with JSON array fields while preserving `analysis_result.npy`.
- [ ] 1.4 Add `--save_trading_detail_csv` to `test_agent_index.py` and write `trading_action_detail_epoch_<epoch_num>.csv` only when the flag is provided.
- [ ] 1.5 Build detail CSV rows with context, optional OHLCV fields, action target state, actual pre/post execution state, action-change counts, trade counts, execution economics, and account value columns.
- [ ] 1.6 Add focused unit tests for aggregate CSV output, detail CSV opt-in behavior, execution metric exposure, and action-change versus actual-trade counting.

## 2. Verification

- [ ] 2.1 Run focused pytest for `FineFT/tests/rl/test_test_agent_index.py` and relevant environment tests.
- [ ] 2.2 Run `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/test_agent_index.py FineFT/env/env_class/base_env.py FineFT/env/env_class/simple_env.py FineFT/env/env_class/futures_util.py`.
- [ ] 2.3 Run `openspec validate add-test-agent-csv-outputs --strict`.

