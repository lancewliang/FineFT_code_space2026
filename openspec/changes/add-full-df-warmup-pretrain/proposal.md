# add-full-df-warmup-pretrain

## 背景与目标

`FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py` 当前只在前 `pretrain_epoch` 个 sample 中执行专家/规则预训练。进入多样性训练后，未经过专家 warmup 的训练分块会由当前 context policy 通过 epsilon-greedy 采集轨迹。对于高杠杆期货任务，这可能让早期 transition 偏离专家行为，导致多样性训练把策略学歪。

目标是在多样性训练开始前，先覆盖所有训练分块 `df_*.feather` 做一次专家 warmup。每个 `df_index` 只 warmup 一次，并固定使用空仓初始动作。warmup 阶段应直接调用现有 `update_pretrain()` 更新网络参数，而不是只预填 replay buffer。完成后再进入原 sample 训练循环。

## 用户场景

- 训练低层智能体时，维护者希望每个训练分块在进入 diverse training 前都至少经过一次专家/规则轨迹 warmup。
- 维护者希望避免为所有初始仓位重复 warmup，只用空仓初始动作控制成本。
- 维护者希望保留关闭 full-df warmup 的能力，用于消融实验或恢复旧训练路径。

## 设计方向

采用 **训练前 full-df warmup + 原 sample 训练循环**。

在 `DQN.train()` 中、进入 `for sample in range(self.num_sample)` 前：

1. 继续准备现有 replay buffer、`qtable_kwargs`、`env_kwargs`、`sample_plan` 和 qtable diagnostics。
2. 如果 full-df warmup 启用，确保 `df_0` 到 `df_{total_df_index_length-1}` 的 df 和 qtable 都在 cache 中。已由 diagnostics 计算过的 df 应复用，避免重复读取和重复计算 qtable。
3. 定位空仓对应的 `initial_action`。优先根据现有 `position_list` / `position_choices` 的动作语义查找 position 为 0 的索引，不硬编码动作编号。
4. 遍历每个 `df_index`：
   - 使用该 df、空仓 initial action 和现有 env 初始化逻辑创建 demo env；
   - 使用该 df 的 qtable 生成 DP expert action path；
   - 跑现有 4 种预训练策略：专家最优、最大多仓、最大空仓、空仓；
   - transition 写入 `buffer_pretrain`；
   - 满足现有 pretrain update 条件时调用 `update_pretrain()`，直接更新网络参数；
   - 记录每个 df 的累计 reward、最终余额、收益率和更新次数。
5. full-df warmup 完成后进入原 sample loop。
6. `pretrain_epoch` 参数保留，但默认改为 `0`。默认流程是在 full-df warmup 后直接进入 diverse training；用户显式设置 `pretrain_epoch > 0` 时，继续按旧逻辑追加 sample-level pretrain。

新增 CLI 开关默认启用 full-df warmup，并提供关闭路径，例如 `--full_df_warmup` / `--no_full_df_warmup`，具体命名在规格阶段确定。

## 关键决策

- 每个 `df_index` 只 warmup 一次，不覆盖所有 `df_index × initial_action` 组合。
- full-df warmup 固定使用空仓初始动作。
- full-df warmup 直接更新网络参数，不只是预填 replay buffer。
- full-df warmup 默认启用，并提供关闭开关支持消融。
- `pretrain_epoch` 保留兼容旧行为，但默认改为 `0`。
- qtable/df cache 尽量复用现有 diagnostics 模块能力；网络训练副作用保留在 `weight_advantage_pretrain.py`。
- df 读取失败、qtable 计算失败、空仓动作定位失败应 fail-fast；专家路径累计收益不盈利只记录 warning，不中断训练。

## 范围边界

**包含：**

- 修改 `weight_advantage_pretrain.py` 的训练流程，在 sample loop 前新增 full-df warmup 阶段。
- 新增默认启用的 full-df warmup CLI 开关和关闭开关。
- 将 `pretrain_epoch` 默认值改为 `0`，但保留用户显式设置后的旧 sample-level pretrain 行为。
- 每个训练 df 固定空仓初始动作 warmup 一次。
- full-df warmup 阶段复用现有 `update_pretrain()`、`act_multi_styles_pretrain()`、demo env 和 qtable action path 逻辑。
- 记录 full-df warmup 的 per-df 日志和总体摘要。
- 增加 focused tests 或 smoke 验证，覆盖默认启用、关闭开关、空仓动作选择和每个 df 一次 warmup。

**不包含（本次）：**

- 不为每个初始仓位都执行 warmup。
- 不改变 `create_optimal_q_table_from_df()` 的 qtable 计算逻辑。
- 不改变 `get_dp_action_from_qtable()` 的专家路径生成逻辑。
- 不修改 Stage II/Stage III、VAE routing、agent filtering 或其他 baseline 脚本。
- 不要求 full-df warmup 为每个 df 额外导出诊断 CSV；已有 sample diagnostics 保持原职责。
- 不在专家路径亏损时跳过 df 或停止训练。

## 验收标准

- [ ] 默认训练流程会在 diverse training 前执行 full-df warmup。
- [ ] full-df warmup 遍历所有 `df_index`，每个 df 只 warmup 一次。
- [ ] full-df warmup 固定使用空仓初始动作，且空仓动作不是硬编码编号。
- [ ] full-df warmup 会调用 `update_pretrain()` 更新网络参数。
- [ ] `pretrain_epoch` 默认值为 `0`，但显式设置大于 0 时仍保留旧 sample-level pretrain 行为。
- [ ] 存在 CLI 关闭开关，可跳过 full-df warmup 并恢复旧训练路径。
- [ ] qtable/df cache 对 diagnostics 已计算过的 df 进行复用，避免重复计算。
- [ ] df 读取失败、qtable 计算失败或空仓动作定位失败会停止训练并给出明确错误。
- [ ] 专家路径累计收益不盈利只输出 warning，不中断训练。
- [ ] `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py` 通过。
