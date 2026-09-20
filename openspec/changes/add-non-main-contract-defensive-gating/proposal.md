# Change: 增加非主力与次主力合约门控防御与退化消融开关 (add-non-main-contract-defensive-gating)

## Why

在多合约商品期货验证集上评估高层门控路由与超参数寻优（Optuna）时，部分合约可能处于远月冷门或交割前换月阶段，其流动性枯竭、盘口极浅、滑点剧烈。若门控网络无差别调度低层投机 Agent 频繁开仓，会因不真实的流动性冲击或异常走势产生严重回撤，从而扭曲多合约综合收益率（Portfolio Return）与胜率指标，误导超参选拔。因此，门控需要基于合约前一交易日角色特征实施防御策略（逐步平仓并保持空仓），并提供显式配置开关以支持严谨的退化消融测试。

## What Changes

1. **高层门控感知合约角色特征并实施防御策略**：
   - 门控决策层（`get_action`）在确定当前动作前，检查当前 Step 的状态向量中是否存在 `prev_day_contract_role_tier` 特征。
   - 当检测到当前合约特征不是主力（`1.0`）且不是次主力（`0.5`）——即档位值为 `0.0` 时，直接触发防御策略（调用逐步平仓 `rule_based_close` 回归空仓），并在宏观动作轨迹中记录防御槽位 `slot_count`。
   - 当合约处于次主力（`0.5`）或主力（`1.0`）时，保持原有 2D VAE 门控与低层 Agent 调度逻辑。
2. **VAE 滑动窗口时序队列连续追踪**：
   - 处于非主力防御期间，VAE 重构损失与分位数队列（`quantiles` deque）仍随环境步进正常计算与更新，避免时序滑动窗口出现断层，确保当合约切入次主力/主力时门控加权历史平滑过渡。
3. **新增显式消融/退化测试开关参数**：
   - 在高层路由核心工具模块及 Optuna 调参脚本中引入 `--enable_non_main_contract_defense` 参数（默认 `False`，保证向下基准与消融对比隔离）。
   - 在高层优化启动脚本（如 `vae_optuna_fu_10.sh`）中显式启用该参数，使 Optuna 在考虑非主力防御的真实实盘约束下选拔最佳门控超参。

## Capabilities

### New Capabilities
- `non-main-contract-defensive-gating`: 针对非主力/次主力合约状态的高层门控防御截断机制与参数化消融测试开关。

### Modified Capabilities
- `fineft-low-level-agent-selection`: 门控在多合约验证集上的回测仿真行为，在开启防御开关时受合约角色特征约束。

## Impact

- 涉及模块：
  - `FineFT/RL/DiHFT/high_level/vae_routing_util.py`
  - `FineFT/RL/DiHFT/high_level/vae_routing_optuna.py`
  - `FineFT/script/test/DiHFT/high_level/vae_optuna_fu_10.sh`
- 测试影响：
  - 增加对 `get_action` 在主力（1.0）、次主力（0.5）、非主力（0.0）状态下的行为验证单测。
  - 验证退化测试开关开启与关闭时宏观动作时序与外部动作的确定性差异。
