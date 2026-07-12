# Close Issues: add-test-agent-csv-outputs

## 2026-07-12 Close Attempt

### Verification Commands

- `source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_test_agent_index.py FineFT/tests/env/test_commodity_env.py -q`
  - Result: failed, 2 failed and 7 passed.
- `source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/test_agent_index.py FineFT/env/env_class/base_env.py FineFT/env/env_class/simple_env.py FineFT/env/env_class/futures_util.py`
  - Result: passed.
- `openspec validate add-test-agent-csv-outputs --strict`
  - Result: passed.

## CRITICAL

- Focused pytest failed because generated CSV headers are Chinese-only while the tests and amended spec expect English/Chinese bilingual headers.
  - `FineFT/tests/rl/test_test_agent_index.py::test_weighted_trader_passes_order_book_depth_to_base_env`
    - Expected aggregate headers such as `label/标签`, `initial_action/初始动作`, `bin_index/分箱索引`.
    - Actual aggregate headers include Chinese-only names such as `标签`, `初始动作`, `分箱索引`.
  - `FineFT/tests/rl/test_test_agent_index.py::test_trading_detail_csv_records_actions_trades_and_execution_metrics`
    - Expected detail headers such as `label/标签`, `df_path/数据文件`, `timestamp/时间戳`, `target_position/目标仓位`.
    - Actual detail headers include Chinese-only names such as `标签`, `数据文件`, `时间戳`, `目标仓位`.

## Required Follow-Up

- 2026-07-12 update: 用户确认当前中文-only 表头才是正确含义。已通过 amend 将规格改为当前中文语义表头。后续应使用 `/sddflow build` 更新测试断言，然后重新执行 close。

## Resolution

- Updated `FineFT/tests/rl/test_test_agent_index.py` to match the amended Chinese semantic header requirement.
- Re-ran focused RL tests: `5 passed in 1.68s`.
- Re-ran combined focused RL/env tests: `9 passed in 1.79s`.
- Re-ran py_compile for modified runtime files: passed.
- Re-ran `openspec validate add-test-agent-csv-outputs --strict`: passed.
- Re-run close after focused pytest passes.

## 2026-07-12 Close Re-run

### Verification Commands

- `source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate finetf && PYTHONPATH=FineFT pytest -q`
  - Result: collection failed before running tests.
  - Collection errors:
    - `FineFT/RL/base/ncqrdqn_test.py` imports `NCQRDQN`, which is not exported by `FineFT/model/low_level.py`.
    - `FineFT/tests/datahandler/test_vae_data_creation.py` imports `FineFT.datahandler`, but `FineFT` is not importable as a package in this invocation.
    - `FineFT/tests/env/test_env.py` reads `/data2/mlqin/HFT4Ind2/dataset/BNBUSDT/train/df_0.feather`, which is not present in this environment.
- `source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_test_agent_index.py FineFT/tests/env/test_commodity_env.py -q`
  - Result: passed, `9 passed in 1.74s`.
- `source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_test_agent_index.py -q`
  - Result: passed, `5 passed in 1.69s`.
- `source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/env/test_commodity_env.py -q`
  - Result: passed, `4 passed in 0.37s`.
- `source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/test_agent_index.py FineFT/env/env_class/base_env.py FineFT/env/env_class/simple_env.py FineFT/env/env_class/futures_util.py`
  - Result: passed.
- `openspec validate add-test-agent-csv-outputs --strict`
  - Result: passed.

### Close Assessment

- CRITICAL issues from the earlier close attempt are resolved for this change.
- Root-level `pytest -q` remains blocked by pre-existing collection/import/data issues outside this change's scope.
- Focused tests covering this change pass.
