# 0045. 高层 VAE 路由动作持续性与防御优先抢占架构

为消除高层 VAE 路由（`vae_routing_optuna.py`、`vae_routing_util.py`）与低层强化学习策略（`parallel_diverse_train.py`、`test_agent_index.py`）在持仓持续性上的语义割裂，并在降低微观交易换手磨损与 CPU 推理开销的同时保障极致风控响应，我们决定在高层路由中引入具备防御抢占（Defensive Preemption）特性的动作持续性（Action Persistence）机制，并沉淀逐帧决策归因审计轨迹。

## Status

accepted

## Context

在目前的 FineFT 交易系统中：
1. **训练与评测语义脱节**：低层模型训练和单 Agent 测试统一采用 `--action_persistence 3`（非 Flat 持仓连续维持 3 步 = 30 分钟），但高层 VAE 路由的 Optuna 搜参脚本（`vae_optuna_fu_10.sh`）与最终评测脚本此前未接入该参数，导致 Optuna 在“每步频繁换手”的错误假设下探索门控超参数，脱离低层真实执行特征；
2. **缺乏事后审计归因**：历史回测仅导出离散动作值 `micro_action_history.npy`，事后无法区分连续相同的持仓动作是由策略网络主动推断输出、由持续性机制物理锁定，还是由高层防御机制强行平仓。

## Considered Options

- **选项 1：完全复刻低层外层锁定，高层防御无法打断**：将 `action_persistence` 仅置于环境最外层，持续期内彻底跳过包括高层 VAE 门控在内的全部计算。
  *缺陷*：当市场出现极端 OOD 漂移或进入非主力合约需要紧急平仓时，系统因处于持仓锁定期而无法及时防御，存在重大风险暴露。
- **选项 2：仅对低层 Q 网络输出进行同 Slot 滞回**：若高层路由在网格临界点发生微小振荡（例如从 Slot 0 切换到 Slot 1），立即打断持续性重新推断。
  *缺陷*：在真实市场中，临界点微小波动会导致持续性名存实亡，换手率与手续费大幅飙升。
- **选项 3：具备高层防御抢占的动作持续性 + 决策归因双重沉淀（选中）**：
  1. **防御优先抢占（Defensive Preemption）**：高层检测到 OOD 漂移、非主力合约防御或空模型槽位时，拥有最高控制权，立即打断持仓持续锁定（`remaining_persist = 0`）并强行执行规则平仓；
  2. **非防御跨 Slot 锁定**：只要高层维持在非防御状态，跨 Slot 的路由切换不破坏底层持仓锁定，跳过神经网络前向推理；
  3. **空仓不锁定**：Flat 动作（0 仓位）不触发持续性，每步重新评估；
  4. **全链路统一**：CLI 参数默认值统一为 3，并在 `vae_optuna_fu_10.sh` 与 `final_result_fu_10.sh` 中显式透传；
  5. **双重审计归因**：落盘 `action_decision_reason_history.npy` 逐帧记录决策根因代码，并在 `trading_info.npy` 中输出聚合诊断指标（如 `skip_inference_ratio`）。

## Consequences

- **正面收益**：
  - 高层 Optuna 寻优环境与低层策略执行语义 100% 严格对齐；
  - 维持非 Flat 动作时跳过 60% 以上的高层/低层神经网络前向推断，大幅加速 Optuna 调优与回测速度；
  - 极端行情与非主力合约下防御平仓零延迟生效，坚守风控底线；
  - 评测导出文件具备完全的可解释性与事后溯源能力。
- **权衡与约束**：
  - 结果目录新增 `action_decision_reason_history.npy` 存储，需确保与 `micro_action_history.npy` 严格逐帧同维。
