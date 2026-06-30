# 实现计划：add-pretrain-qtable-profit-check

## 来源
- 提案：openspec/changes/add-pretrain-qtable-profit-check/proposal.md
- 设计：openspec/changes/add-pretrain-qtable-profit-check/design.md
- 规格：openspec/changes/add-pretrain-qtable-profit-check/specs/fineft-stage-i-pretrain/spec.md
- 任务：openspec/changes/add-pretrain-qtable-profit-check/tasks.md

## Amendments

### 2026-06-30: 独立诊断模块、多进程 qtable 计算与 sample CSV 导出
- 原因：用户要求更严格地按 sample 独立验证；生成 qtable 后导出 CSV；qtable 计算使用多进程；qtable 诊断代码拆为独立 Python 文件。
- 影响规格：openspec/changes/add-pretrain-qtable-profit-check/specs/fineft-stage-i-pretrain/spec.md
- 影响设计：openspec/changes/add-pretrain-qtable-profit-check/design.md
- 影响任务：tasks.md 1.0-1.6、2.1-2.2

## 实现步骤

### Task 1: Sample plan and qtable cache implementation
- [x] **任务完成**（与 superpowers plan `Task 1`、`tasks.md` 对应条目同步勾选）
- 目标：新增 `pretrain_qtable_diagnostics.py`，在训练循环前预生成 `sample_plan`，对唯一 `df_index` 使用多进程预计算并缓存 qtable，按 sample 独立回放 DP 专家路径，打印诊断并导出每个 sample 的 CSV 明细；训练循环改为复用 plan 和 cache。
- 改动文件：`FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py`、`FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`
- 验证方式：代码中 `for sample in range(self.num_sample)` 前调用独立模块构建 plan/cache/diagnostics；循环内不再随机抽取 `df_index` 和 `initial_action`；pretrain 分支使用 `q_table_cache[df_index]`；诊断日志包含 `sample_index`、`df_index`、`initial_action`、`episode_reward_sum`、`profitable`；`self.model_path/qtable_diagnostics/` 下按 sample 生成 CSV，CSV 包含 OHLCV、`mark_price`、滑点、费率、动作、单步 reward 和累计利润。

### Task 2: Verification
- [ ] **任务完成**（与 superpowers plan `Task 2`、`tasks.md` 对应条目同步勾选）
- 目标：确认训练脚本和独立诊断模块语法有效，OpenSpec 严格校验通过；如果本地训练数据可用，用小参数训练命令观察训练循环前的 qtable 盈利诊断和 CSV 导出。
- 改动文件：无代码改动；验证 `FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`、`FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py` 和 OpenSpec 文档。
- 验证方式：运行 `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py`；运行 `openspec validate add-pretrain-qtable-profit-check --strict`；若数据可用，运行 `--num_sample 2 --pretrain_epoch 1` smoke 命令并确认诊断输出和每个 sample 的 CSV 文件。
