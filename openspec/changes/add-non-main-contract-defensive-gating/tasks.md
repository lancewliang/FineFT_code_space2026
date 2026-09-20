# Tasks: 增加非主力与次主力合约门控防御与退化消融开关 (add-non-main-contract-defensive-gating)

## 1. 核心门控防御与参数解析 (Core Implementation)
- [x] 1.1 在 `FineFT/RL/DiHFT/high_level/vae_routing_util.py` 的 `parser` 中添加 `--enable_non_main_contract_defense` 参数（默认 `False`）。
- [x] 1.2 在 `FineFT/RL/DiHFT/high_level/vae_routing_util.py` 的 `vae_risk_aware_routing` 类中添加 `enable_non_main_contract_defense` 与 `role_tier_index` 属性，并在 `__init__` 中基于 `prev_day_contract_role_tier` 解析索引。
- [x] 1.3 在 `reconfigure_routing` 中同步更新 `enable_non_main_contract_defense` 参数，确保 Optuna 跨 Trial 复用实例时配置有效。
- [x] 1.4 在 `get_action` 决策入口增加防御拦截：当开关启用、特征存在且角色档位小于 0.5 时，直接执行 `_defensive_action`，记录 `slot_count` 宏观动作，跳过底层 Q-Network 推理。
- [x] 1.5 在 `FineFT/RL/DiHFT/high_level/vae_routing_optuna.py` 的 `parser_all` 与 `prepare_base_args` 中增加 `--enable_non_main_contract_defense` 参数解析与透传。

## 2. 调度脚本集成与退化配置 (Script Integration)
- [x] 2.1 更新 `FineFT/script/test/DiHFT/high_level/vae_optuna_fu_10.sh`，默认开启 `--enable_non_main_contract_defense`，并通过 `ENABLE_NON_MAIN_DEFENSE` 环境变量支持一键退化消融运行。

## 3. 单元测试与验证 (Testing & Verification)
- [x] 3.1 编写 `FineFT/tests/rl/test_vae_routing_non_main_defense.py`：
  - 测试 `prepare_base_args` 正确透传 `--enable_non_main_contract_defense` 开关。
  - 测试在非主力状态（`role_tier = 0.0`）下触发规则平仓动作，记录槽位 9，且不调用 `agent_act`。
  - 测试在主力（`1.0`）与次主力（`0.5`）状态下正常进入 VAE 门控与 Agent 选择。
  - 测试开关关闭时（退化基准），即使处于非主力状态也正常调用 Agent。
  - 测试缺少 `prev_day_contract_role_tier` 特征时安全回退不抛异常。
  - 测试 `reconfigure_routing` 正确更新防御开关状态。
- [x] 3.2 运行相关既有测试套件，确认全部测试通过且无回归。
