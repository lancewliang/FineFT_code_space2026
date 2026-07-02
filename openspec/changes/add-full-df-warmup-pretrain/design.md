# add-full-df-warmup-pretrain 设计

## 背景

当前 Stage I 训练只在 `sample < pretrain_epoch` 时执行专家/规则预训练。进入多样性训练后，尚未经过专家 warmup 的 `df_index` 会由 context policy 通过 epsilon-greedy 采集轨迹。对于高杠杆期货任务，这会提高早期 transition 偏离专家行为的风险。

本变更在 sample 训练循环前新增 full-df warmup 阶段：每个训练分块 `df_index` 固定空仓初始动作 warmup 一次，并直接更新网络参数。完成后再进入 sample loop。

## 设计决策

### 独立训练前 warmup 阶段

`weight_advantage_pretrain.py` 在 `prepare_pretrain_qtable_diagnostics(...)` 之后、`for sample in range(self.num_sample)` 之前执行 full-df warmup。该阶段遍历 `range(self.total_df_index_length)`，确保每个 `df_index` 的 df 和 qtable 都存在于 cache 中，然后使用空仓初始动作创建 demo env 并采集专家/规则轨迹。

full-df warmup 直接复用现有 `update_pretrain()`、`act_multi_styles_pretrain()`、`buffer_pretrain` 和 TensorBoard loss 记录逻辑，避免新增第二套预训练 loss。

### 缓存边界

`pretrain_qtable_diagnostics.py` 继续负责 qtable/df cache 构建与诊断数据准备，不承担网络训练副作用。为了支持 full-df warmup，可以补充一个 cache 扩展 helper：给定目标 `df_index` 集合和已有 cache，只计算缺失的 df/qtable 并合并返回。

`weight_advantage_pretrain.py` 负责调用该 helper，并负责所有会更新网络参数的 warmup loop。

### 空仓动作定位

空仓初始动作不得硬编码为固定数字。实现应根据现有动作语义定位 position 为 0 的 action index，例如使用 `map_action_to_position_leverage(...)` 遍历 `range(self.position_choices)`，找到映射后 `position == 0` 的 action。找不到时 fail-fast。

### 默认行为与兼容性

full-df warmup 默认启用，并提供 CLI 关闭开关用于消融或恢复旧路径。`pretrain_epoch` 保留，但默认改为 `0`。因此默认训练流程为：

1. qtable diagnostics/cache
2. full-df warmup
3. diverse training

用户显式设置 `--pretrain_epoch > 0` 时，full-df warmup 后仍按旧逻辑追加 sample-level pretrain。

## 取舍

- 每个 `df_index` 只 warmup 一次，控制成本；不覆盖所有初始仓位组合。
- full-df warmup 直接训练网络，能在 diverse training 前校准策略；代价是训练启动前耗时增加。
- full-df warmup 不额外导出每个 df 的 CSV，避免输出量膨胀；日志记录 per-df 和总体摘要即可。
- 专家路径亏损只 warning，不中断训练；数据读取、qtable 计算、空仓动作定位等基础错误 fail-fast。
