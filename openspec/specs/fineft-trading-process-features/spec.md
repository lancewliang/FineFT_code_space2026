# fineft-trading-process-features Specification

## Purpose
定义 Trading Process Features (持仓暴露、单次持仓收益率、单次持仓最大回撤率) 作为 Q 网络连续输入与 Replay Buffer 采样契约的规范。

## Requirements

### Requirement: Futures Trading Environment 暴露 Trading Process Feature
系统 SHALL 在 Futures Trading Environment 的 `reset()` 和 `step()` 返回的 `info` 字典中暴露 `trading_info` 数组。

#### Scenario: reset 和 step 的 info 包含 trading_info
- **WHEN** 环境执行 `reset()` 或 `step()`
- **THEN** 返回的 `info` 字典中 SHALL 包含 `trading_info` 键
- **AND** `trading_info` SHALL 包含无量纲特征：`position_exposure`、`single_holding_return_rate` 和 `single_holding_max_drawdown`
- **AND** `position_exposure` SHALL 为 `position / max_abs_position` 归一化持仓暴露 (范围 [-1, 1])
- **AND** 平仓或爆仓后，下一状态的 `trading_info` SHALL 为 `[0, 0, 0]`

#### Scenario: 同方向变仓与反向变仓的收益率/回撤语义
- **WHEN** 环境执行同方向加仓或减仓
- **THEN** 当前持仓不结束，`single_holding_return_rate` 与 `single_holding_max_drawdown` 延续计算
- **WHEN** 环境平仓至 0 或执行持仓反向变仓
- **THEN** 旧持仓结束，新持仓从 0 开始重新计算收益率与最大回撤

### Requirement: Low-level Q Networks 接收并编码 Trading Process Feature
系统 SHALL 在低层 Q 网络 (`Qnet` / `ensemble_Qnet`) 中增加独立线性层 `fc_trading` 编码 `trading_info`。

#### Scenario: Qnet 和 ensemble_Qnet 接收 trading_info 输入
- **WHEN** 调用 `Qnet.forward()` 或 `ensemble_Qnet.forward()`
- **THEN** 必须显式传入 `trading_info` (shape `(batch, TRADING_INFO_DIM)`)
- **AND** `fc_trading` 将 `trading_info` 映射为 `hidden_nodes` 维隐藏向量
- **AND** 该隐藏向量与 `state_hidden`、`previous_action_hidden` 及 `time` 隐藏向量拼接后再通过 `fc2` 节点计算 Q 值

### Requirement: Replay Buffer 保存并采样 Trading Process Feature
系统 SHALL 在 Replay Buffer 的存储与采样白名单中保留 `trading_info`。

#### Scenario: Replay Buffer sample 包含 trading_info
- **WHEN** 从 Replay Buffer 执行 `sample()` 或 `sample_evaluate()`
- **THEN** 返回的 `infos["trading_info"]` 与 `next_infos["trading_info"]` 的 shape 最后一维 SHALL 等于 `TRADING_INFO_DIM`
- **AND** dtype SHALL 为 float
