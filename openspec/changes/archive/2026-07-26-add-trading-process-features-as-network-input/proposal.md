# add-trading-process-features-as-network-input

## 背景与目标
原低层 Q 网络只接收市场技术指标 (state)、动作编码 (previous_action)、资金费率倒计时 (time) 和可用动作掩码 (avaliable_action)，缺乏对当前持仓暴露、收益及风险回撤的感知。
本变更将环境计算的 3 个无量纲 Trading Process Feature (`position_exposure`, `single_holding_return_rate`, `single_holding_max_drawdown`) 打包为 `trading_info` 数组，通过 `fc_trading` 独立层编码并接入 Q 网络输入，使 Agent 能进行风险感知的持仓/平仓决策。

## 关键决策
- **3 维无量纲特征**：`position_exposure` 归一化到 [-1, 1]，收益率与最大回撤使用比率，排除绝对盈亏值。
- **独立编码融合**：Qnet 中新增 `fc_trading` 映射到 `hidden_nodes` 维，与 `state_hidden` 等拼接。
- **强制全量迁移**：所有 `Qnet`/`ensemble_Qnet` 构造与调用点全量迁移显式传入 `trading_info`。
- **Replay Buffer 白名单**：定义 `NETWORK_INFO_KEYS` 常量，采样输出 shape 为 `(batch_size, 3)` 的 `trading_info`。

## 验收标准
- 环境在 reset/step 时 info 中正确返回 3 维 `trading_info`。
- Replay Buffer 采样时保留 `trading_info` shape 与数值。
- Qnet/ensemble_Qnet 输入包含 `trading_info`，缺失时调用失败。
