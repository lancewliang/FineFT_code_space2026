# 实现计划：add-test-agent-csv-outputs

## 来源
- 提案：openspec/changes/add-test-agent-csv-outputs/proposal.md
- 设计：openspec/changes/add-test-agent-csv-outputs/design.md
- 规格：openspec/changes/add-test-agent-csv-outputs/specs/
- 任务：openspec/changes/add-test-agent-csv-outputs/tasks.md

## 实现步骤

### Task 1: Extend trading wallet-change results to expose true `commission_fee_step`, `realized_pnl_step`, and `slippage_step` while preserving existing fee semantics.
- [ ] **任务完成**（与 superpowers plan `Task 1`、`tasks.md` 对应条目同步勾选）
- 目标：让 `FineFT/env/env_class/futures_util.py` 的 wallet-change 路径显式返回每步真实手续费、已实现利润和滑点，同时保留旧调用方需要的六元组语义。
- 改动文件：`FineFT/env/env_class/futures_util.py`、`FineFT/tests/env/test_commodity_env.py`
- 验证方式：运行 focused wallet-change 测试，确认买卖费率语义不变，且新字段能读到真实 `commission_fee_step`、`realized_pnl_step`、`slippage_step`。

### Task 2: Update `Base_Env` and `Simple_Env` callers to consume the explicit wallet-change result and expose/reset per-step and cumulative execution metrics.
- [ ] **任务完成**（与 superpowers plan `Task 2`、`tasks.md` 对应条目同步勾选）
- 目标：让 env 在 `reset()` 后清零 execution metrics，并在每次 `step()` 后通过 `info` 暴露 step/cumulative metrics。
- 改动文件：`FineFT/env/env_class/base_env.py`、`FineFT/env/env_class/simple_env.py`、相关 env 测试
- 验证方式：运行 env focused tests，确认 `info` 中有执行指标，累计值等于 step 指标累计，旧 fee-rate 行为不变。

### Task 3: Update `test_agent_index.py` aggregate result collection to include `df_path` and write default `analysis_result.csv` with JSON array fields while preserving `analysis_result.npy`.
- [ ] **任务完成**（与 superpowers plan `Task 3`、`tasks.md` 对应条目同步勾选）
- 目标：为汇总结果补充 `df_path`，默认生成同目录 `analysis_result.csv`，列表字段用 JSON 数组字符串。
- 改动文件：`FineFT/RL/DiHFT/low_level/test_agent_index.py`、`FineFT/tests/rl/test_test_agent_index.py`
- 验证方式：运行 low-level test-agent focused tests，确认 npy 仍存在，CSV 存在且字段和 JSON 列表正确。

### Task 4: Add `--save_trading_detail_csv` to `test_agent_index.py` and write `trading_action_detail_epoch_<epoch_num>.csv` only when the flag is provided.
- [ ] **任务完成**（与 superpowers plan `Task 4`、`tasks.md` 对应条目同步勾选）
- 目标：增加明细 CSV 开关，默认不输出大文件，显式传参时按 epoch 命名输出。
- 改动文件：`FineFT/RL/DiHFT/low_level/test_agent_index.py`、`FineFT/tests/rl/test_test_agent_index.py`
- 验证方式：测试不传开关不生成明细 CSV，传开关生成 `trading_action_detail_epoch_<epoch_num>.csv`。

### Task 5: Build detail CSV rows with context, optional OHLCV fields, action target state, actual pre/post execution state, action-change counts, trade counts, execution economics, and account value columns.
- [ ] **任务完成**（与 superpowers plan `Task 5`、`tasks.md` 对应条目同步勾选）
- 目标：实现每时间步明细行构造，覆盖规格要求字段和计数口径。
- 改动文件：`FineFT/RL/DiHFT/low_level/test_agent_index.py`、`FineFT/tests/rl/test_test_agent_index.py`
- 验证方式：使用 fake env 覆盖 action id 变化但真实仓位不变，以及真实仓位变化两种路径，断言计数、费用、利润、滑点和账户价值字段。

### Task 6: Add focused unit tests for aggregate CSV output, detail CSV opt-in behavior, execution metric exposure, and action-change versus actual-trade counting.
- [ ] **任务完成**（与 superpowers plan `Task 6`、`tasks.md` 对应条目同步勾选）
- 目标：补齐回归测试，使 CSV 输出和 env execution metrics 的边界可验证。
- 改动文件：`FineFT/tests/rl/test_test_agent_index.py`、`FineFT/tests/env/test_commodity_env.py`
- 验证方式：运行新增/更新的 focused pytest，确认失败用例先覆盖需求，再由实现通过。

### Task 7: Run focused pytest for `FineFT/tests/rl/test_test_agent_index.py` and relevant environment tests.
- [ ] **任务完成**（与 superpowers plan `Task 7`、`tasks.md` 对应条目同步勾选）
- 目标：执行变更直接相关测试。
- 改动文件：无代码改动；验证命令执行
- 验证方式：`conda activate finetf && pytest FineFT/tests/rl/test_test_agent_index.py FineFT/tests/env/test_commodity_env.py -q` 通过。

### Task 8: Run `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/test_agent_index.py FineFT/env/env_class/base_env.py FineFT/env/env_class/simple_env.py FineFT/env/env_class/futures_util.py`.
- [ ] **任务完成**（与 superpowers plan `Task 8`、`tasks.md` 对应条目同步勾选）
- 目标：确认修改文件语法有效。
- 改动文件：无代码改动；验证命令执行
- 验证方式：`conda activate finetf && python -m py_compile ...` 成功返回 0。

### Task 9: Run `openspec validate add-test-agent-csv-outputs --strict`.
- [ ] **任务完成**（与 superpowers plan `Task 9`、`tasks.md` 对应条目同步勾选）
- 目标：确认 OpenSpec 变更仍严格有效。
- 改动文件：无代码改动；验证命令执行
- 验证方式：`openspec validate add-test-agent-csv-outputs --strict` 输出 `Change 'add-test-agent-csv-outputs' is valid`。
