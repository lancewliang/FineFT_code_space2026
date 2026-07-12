## 1. Implementation

- [x] 1.1 Extend trading wallet-change results to expose true `commission_fee_step`, `realized_pnl_step`, and `slippage_step` while preserving existing fee semantics. <!-- 已实现: WalletChangeResult exposes named execution metrics while preserving tuple access. -->
- [x] 1.2 Update `Base_Env` and `Simple_Env` callers to consume the explicit wallet-change result and expose/reset per-step and cumulative execution metrics. <!-- 已实现: Env step/reset info exposes per-step and cumulative execution metrics. -->
- [x] 1.3 Update `test_agent_index.py` aggregate result collection to include `df_path` and write default `analysis_result.csv` with JSON array fields while preserving `analysis_result.npy`. <!-- 已实现: Aggregate npy is preserved and analysis_result.csv writes JSON array columns. -->
- [x] 1.4 Add `--save_trading_detail_csv` to `test_agent_index.py` and write `trading_action_detail_epoch_<epoch_num>.csv` only when the flag is provided. <!-- 已实现: Detail CSV is opt-in and named by epoch. -->
- [x] 1.5 Build detail CSV rows with context, optional OHLCV fields, action target state, actual pre/post execution state, action-change counts, trade counts, execution economics, and account value columns. <!-- 已实现: Detail CSV rows include context, market fields, execution metrics, counts, and account values. -->
- [x] 1.6 Add focused unit tests for aggregate CSV output, detail CSV opt-in behavior, execution metric exposure, and action-change versus actual-trade counting. <!-- 已实现: Focused RL/env tests cover aggregate CSV, detail opt-in, metrics, and counting boundaries. -->
- [x] 1.7 Change aggregate and detail CSV headers to English/Chinese bilingual names while preserving row values and JSON array cells. <!-- 已实现: Aggregate and detail CSV writers emit 英文/中文 bilingual headers while preserving values. -->
- [x] 1.8 Align CSV header tests and close verification with the current Chinese semantic headers. <!-- 已实现: Tests and close verification now match the current Chinese semantic headers. -->

## 2. Verification

- [x] 2.1 Run focused pytest for `FineFT/tests/rl/test_test_agent_index.py` and relevant environment tests. <!-- 已实现: Combined focused pytest passed in finetf with PYTHONPATH=FineFT. -->
- [x] 2.2 Run `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/test_agent_index.py FineFT/env/env_class/base_env.py FineFT/env/env_class/simple_env.py FineFT/env/env_class/futures_util.py`. <!-- 已实现: py_compile passed under finetf. -->
- [x] 2.3 Run `openspec validate add-test-agent-csv-outputs --strict`. <!-- 已实现: Strict OpenSpec validation passed and changed files were checked. -->
