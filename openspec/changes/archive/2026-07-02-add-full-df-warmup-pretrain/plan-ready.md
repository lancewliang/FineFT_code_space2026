# 实现计划：add-full-df-warmup-pretrain

## 来源
- 提案：openspec/changes/add-full-df-warmup-pretrain/proposal.md
- 设计：openspec/changes/add-full-df-warmup-pretrain/design.md
- 规格：openspec/changes/add-full-df-warmup-pretrain/specs/fineft-stage-i-pretrain/spec.md
- 任务：openspec/changes/add-full-df-warmup-pretrain/tasks.md

## 实现步骤

### Task 1: Full-df warmup implementation
- [x] **任务完成**（与 superpowers plan `Task 1`、`tasks.md` 对应条目同步勾选）
- 目标：在 `weight_advantage_pretrain.py` 的 sample loop 前新增默认启用的 full-df warmup 阶段。该阶段遍历所有 `df_index`，固定空仓 initial action，复用/补齐 qtable 和 df cache，采集专家/规则轨迹并调用 `update_pretrain()` 直接更新网络。
- 改动文件：`FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`、`FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py`
- 验证方式：focused tests 断言默认参数、关闭开关、空仓 action 解析、cache 缺失补齐和每个 df 只 warmup 一次；代码检查确认 full-df warmup 位于 `for sample in range(self.num_sample)` 前，且 warmup update 复用 `update_pretrain()`。

### Task 2: Verification
- [x] **任务完成**（与 superpowers plan `Task 2`、`tasks.md` 对应条目同步勾选）
- 目标：补充并运行验证，确认 OpenSpec、focused tests 和 Python 语法检查通过；如本地数据可用，可执行小参数 smoke 确认日志顺序。
- 改动文件：`FineFT/tests/rl/test_pretrain_qtable_diagnostics.py`、`FineFT/tests/rl/test_weight_advantage_pretrain_logging.py`、OpenSpec 文档 checkbox 状态（build 阶段完成后同步）
- 验证方式：运行 `conda activate finetf && pytest FineFT/tests/rl/test_pretrain_qtable_diagnostics.py FineFT/tests/rl/test_weight_advantage_pretrain_logging.py -q`；运行 `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py`；运行 `openspec validate add-full-df-warmup-pretrain --strict`。
