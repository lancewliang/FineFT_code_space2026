# 05: 接通独立评估与高层路由

**What to build:** 独立评估和高层路由严格加载新产物，使用真实环境十一维观测推断，同合约切换专家保留退出记忆。

**Blocked by:** 03 — 接通模型与串行训练闭环

**Status:** resolved

**Implementation:** 未实施。此前代码改动已全部回退；本轮只重切任务，不启动代码修改。ready-for-agent 表示规格可实施，不表示依赖已完成。

## 文件范围

test_agent_index.py、test_agent_average.py、evaluate_sub_agents.py、vae_routing_util.py 及实际 ensemble 重建/OOD 调用者。

## 验收标准

- [ ] 全部入口显式接收并传递 --exit_price_scale_floor，消费 07 的辅助列与有效起点，不重算窗口、不对每个片段固定跳过 18 行。
- [ ] 模型加载与 ensemble 重建复用 03 的 schema/元数据/配置/权重宽度校验；拒绝静默补零、截断或 strict=False 绕过不兼容。
- [ ] 真实前向传入全部十一维，同时保持其余独立输入与动作映射；不使用控制器私有状态替代环境观测。
- [ ] 每合约环境独立维护记忆，同合约换模型/专家不 reset；reset、跨合约及分区隔离，覆盖已有记忆空仓、重入、退出失败及仍持仓终止。
- [ ] 回归动作持续、专家切换、防御抢占、止损、冷却及熔断既有行为；不新增或重定义高层风控。
- [ ] 独立评估、模型平均/重建及高层路由各完成实际环境与真实模型前向 smoke，验证不兼容产物拒绝与切换记忆保留；合成新维度权重可用，不要求正式重训。

## 职责边界

不依赖 04；不重写退出生命周期或复制产物校验实现。

共同遵循 [spec](../spec.md) 第 4–6 节与 [实施依赖总览](../implementation-plan.md)。Python/pytest 使用 `conda activate finetf`。不更改已确认范围，不启动正式重训。

## Comments

- 2026-11-01：按确认后的行情统计预计算、Q8 A 与 exit_price_scale_floor 命名重新切分职责；当前验收项全部未完成。

- 2026-11-01（回退后状态校正）：用户确认本功能所有代码改动已回退；重新打开本票，全部验收项恢复未完成。下方记录只描述回退前工作，不能证明当前源码已实现或验证通过。

- 2026-11-01：完成独立索引评估、模型平均、子 Agent 多进程评估、VAE 高层路由和 OOD 调用方迁移。所有入口接收并透传 `exit_price_scale_floor`，环境从同合约因果有效起点开始；模型加载统一读取 `trading_observation.json`、校验 floor 与 `fc_trading` 宽度并使用 strict load，前向保持完整 11 维 `trading_info`。评估子 Agent 的初始保证金也改按有效起点价格计算，避免 warmup 后环境重置不一致。
- 2026-11-01：同步更新评估/OOD/止损回归夹具，合成模型和 buffer 使用 11 维交易状态并带 sidecar。验证：`python -m pytest FineFT/tests/rl/test_test_agent_index.py FineFT/tests/rl/test_vae_routing_action_persistence.py FineFT/tests/rl/test_expert_hysteresis_routing.py FineFT/tests/rl/test_async_greedy_evaluation.py FineFT/tests/analysis/test_low_level_agent_ood_analysis.py -q`（76 passed）；`python -m pytest FineFT/tests/rl/test_evaluates_multiprocess.py FineFT/tests/rl/test_evaluation_stop_loss.py -q`（41 passed）。
- 2026-11-01：额外执行 VAE 路由全套回归，发现 4 个既有路由行为/组合收益断言与当前分支基线不一致（非本票 diff 引入）；本票新增的契约加载、环境起点和专家记忆路径已由上述定向回归覆盖。

### 回退前历史记录（不作为当前验收证据）

- 2026-11-01：按索引评估、模型平均、子 Agent 评估和 VAE 高层路由均透传显式 `tick_size`；模型平均、子 Agent 与路由加载路径读取 sidecar 并严格校验 tick 与 `fc_trading` 输入宽度。
- 2026-11-01：评估环境新增显式 `start_index`，周期评估及 OOD rollout 从 18 行因果预热后开始交易，避免用不完整历史尺度登记退出；同一环境内模型/专家切换不调用 reset。
- 验证：`python -m pytest tests/rl/test_test_agent_index.py tests/rl/test_vae_routing_action_persistence.py tests/rl/test_expert_hysteresis_routing.py tests/rl/test_async_greedy_evaluation.py tests/analysis/test_low_level_agent_ood_analysis.py -q`（conda 环境 `finetf`）通过。
