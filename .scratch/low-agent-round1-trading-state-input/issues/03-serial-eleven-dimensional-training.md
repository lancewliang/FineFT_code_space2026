# 03: 接通模型与串行训练闭环

**What to build:** 消费 08 的真实十一维环境观测，贯通 Qnet/ensemble、串行采集、CPU replay 与真实 learner 更新，并保存可严格校验的模型及 replay 产物。

**Blocked by:** 01 — 集中观测与行情辅助数据契约；08 — 实现环境十一维观测与退出生命周期

**Status:** resolved

**Implementation:** 未实施。此前代码改动已全部回退；本轮只重切任务，不启动代码修改。ready-for-agent 表示规格可实施，不表示依赖已完成。

## 文件范围

FineFT/model/low_level.py、串行预训练/多样化训练实际入口、replay_buffer_DQN.py、CPU replay 与序列化/加载入口。

## 验收标准

- [ ] 与 08 在同一集成变更中激活目标十一维契约；Qnet、ensemble、重建辅助入口统一宽度，时间、previous_action、mask 和 RL 市场特征保持独立。
- [ ] 串行 CLI 显式接收 --exit_price_scale_floor 5（本次 fu 实验，元/吨）并传到环境/工厂；缺失、非正或非有限参数失败，记录实验来源，区分交易所 tick_size。
- [ ] 实际 rollout 传递当前和 n-step 终点完整十一维观测；CPU replay 入池/采样严格校验，不补零、不截断、不以行情坐标重建退出历史。
- [ ] 保留原 n-step 累计、gamma**n_step 和终止掩码；全十一维参与模型可见内容身份，四后续上限、FIFO、随机轮换及 float32/正负零语义不变。
- [ ] 模型/checkpoint、预训练 replay 快照和正式加载接入 01 元数据接口；校验 schema、顺序、维度、窗口、256、除数 5、实际下限和来源、持仓时长参数及行情统计契约。拒绝旧 4/18 维和无版本旧 replay。
- [ ] 裸 state_dict 至少校验 fc_trading 输入宽度，不能代替正式元数据；确定性权重证明新增字段影响 Qnet 和 ensemble 输出。
- [ ] 以真实环境采集、CPU replay 入池/采样、项目真实 learner update 完成短程串行闭环，损失有限且参数改变；覆盖 Demo 继承路径。旧 DP 仍为辅助基线，不扩维或重算算法。

## 职责边界

不重复实现 02 的统计、07 的数据透传或 08 的事件规则；不提前承担并行/GPU 和独立路由入口。共享变更造成待迁移入口失败须记录，不能宣称全入口完成。

共同遵循 [spec](../spec.md) 第 4–6 节与 [实施依赖总览](../implementation-plan.md)。Python/pytest 使用 `conda activate finetf`。不更改已确认范围，不启动正式重训。

## Comments

- 2026-11-01：按确认后的行情统计预计算、Q8 A 与 exit_price_scale_floor 命名重新切分职责；当前验收项全部未完成。

- 2026-11-01（回退后状态校正）：用户确认本功能所有代码改动已回退；重新打开本票，全部验收项恢复未完成。下方记录只描述回退前工作，不能证明当前源码已实现或验证通过。

### 回退前历史记录（不作为当前验收证据）

- 2026-11-01：Base_Env、Demo 环境、串行/并行预训练入口、Qnet/ensemble 和 CPU replay 已统一使用 `trading_info.v2` 的 11 维 float32 向量；串行入口显式保存 `trading_observation.json`。
- 2026-11-01：完整退出只在实际仓位方向从非零变为零时登记，并使用当次真实关闭金额/数量计算成交均价；终止路径改为保留真实观测，不再无条件清零。
- 2026-11-01：模型前向、replay 入池和产物加载均拒绝旧宽度或非有限输入；模型层输入宽度变化测试证明第 11 维可影响输出并可完成参数更新。
- 验证：`python -m pytest tests/model/test_low_level_trading_observation.py tests/rl/test_trading_info_replay_contract.py tests/env/test_causal_price_scale_preparation.py tests/env/test_trading_observation_lifecycle_v2.py -q`（conda 环境 `finetf`）通过。

## Comments

- 2026-11-01：完成 Qnet/ensemble、串行预训练入口、CPU replay 与真实 learner 更新的 `trading_info.v2` 接入。模型前向严格要求 `(B, 11)` 且拒绝非有限值；replay 当前/下一观测均要求显式 11 维，不再补零。串行 CLI 使用 `--exit_price_scale_floor`，环境工厂透传有效行情辅助列、尺度下限和 18 行 warm-up 起点。
- 2026-11-01：模型保存与加载校验 `fc_trading` 输入宽度和 `trading_observation.json`；预训练 replay 快照升级为 `pretrain_replay.v2`，嵌入完整观测元数据并拒绝无版本/旧宽度快照。
- 验证：在 `conda activate finetf` 环境执行 `python -m pytest tests/model/test_low_level_trading_observation.py tests/rl/test_trading_info_replay_contract.py tests/rl/test_qnet_trading_info.py tests/rl/test_replay_buffer_trading_info.py tests/rl/test_weight_advantage_pretrain_paper.py -q`，结果 `18 passed`。
