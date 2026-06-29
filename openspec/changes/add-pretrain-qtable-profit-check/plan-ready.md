# 实现计划：add-pretrain-qtable-profit-check

## 来源
- 提案：openspec/changes/add-pretrain-qtable-profit-check/proposal.md
- 设计：无（OpenSpec 判定无需）
- 规格：openspec/changes/add-pretrain-qtable-profit-check/specs/fineft-stage-i-pretrain/spec.md
- 任务：openspec/changes/add-pretrain-qtable-profit-check/tasks.md

## 实现步骤

### Task 1: Sample plan and qtable cache implementation
- [ ] **任务完成**（与 superpowers plan `Task 1`、`tasks.md` 对应条目同步勾选）
- 目标：在 `Weighted_Contexts_DQN.train()` 的 sample 循环前预生成本次训练的 `sample_plan`，按唯一 `df_index` 预计算并缓存 qtable，对每个 sample 回放 DP 专家路径并输出盈利诊断，训练循环改为复用 plan 和 cache。
- 改动文件：`FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`
- 验证方式：代码中 `for sample in range(self.num_sample)` 前存在 plan/cache 构建逻辑；循环内不再随机抽取 `df_index` 和 `initial_action`；pretrain 分支使用 `q_table_cache[df_index]`；诊断日志包含 `sample_index`、`df_index`、`initial_action`、`episode_reward_sum`、`profitable`。

### Task 2: Verification
- [ ] **任务完成**（与 superpowers plan `Task 2`、`tasks.md` 对应条目同步勾选）
- 目标：确认训练脚本语法有效，OpenSpec 严格校验通过；如果本地训练数据可用，用小参数训练命令观察训练循环前的 qtable 盈利诊断。
- 改动文件：无代码改动；验证 `FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py` 和 OpenSpec 文档。
- 验证方式：运行 `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`；运行 `openspec validate add-pretrain-qtable-profit-check --strict`；若数据可用，运行 `--num_sample 2 --pretrain_epoch 1` smoke 命令并确认诊断输出。
