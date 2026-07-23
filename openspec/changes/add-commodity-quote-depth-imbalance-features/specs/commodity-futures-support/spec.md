## MODIFIED Requirements

### Requirement: 商品期货 quote 特征下采样
系统 SHALL 从秒频五档快照派生 quote 特征，并使用右闭右标窗口聚合到目标频率。

#### Scenario: 秒频 quote 快照
- **WHEN** 同一秒内存在多条原始快照
- **THEN** 该秒 quote 状态使用该秒最后一条快照
- **AND** 秒频标准层不 forward fill 缺失秒

#### Scenario: 右闭目标窗口
- **WHEN** 目标频率为 `5min`
- **THEN** `(09:00:00, 09:05:00]` 内的快照聚合到标记为 `09:05:00` 的 bar

#### Scenario: 同一交易 session 内空 quote 窗口 fail-fast
- **WHEN** 同一有效交易 session 内目标频率窗口没有任何秒频 quote 快照
- **THEN** 预处理报错，错误信息包含窗口标签和合约

#### Scenario: 跨交易 session 缺口不报 quote 缺失
- **WHEN** `TradingDay=20251103` 的夜盘事件时间包含 `2025-10-31 23:00:00`
- **AND** 下一条 quote bar 位于后续有效交易 session
- **THEN** 系统 SHALL NOT 因 `2025-10-31 23:05:00` 这类非交易时段窗口报 `Target window has no quote snapshots`
- **AND** 系统 SHALL 保留真实 `ActionDay + UpdateTime` timestamp，不按 `START_DATE` 过滤掉合法夜盘事件时间

#### Scenario: 整段 quote 输入为空 fail-fast
- **WHEN** quote 下采样输入没有任何秒频 quote 快照
- **THEN** 预处理报错，错误信息说明没有 quote snapshot

#### Scenario: quote 计数与状态特征
- **WHEN** 目标频率窗口存在秒频 quote 快照
- **THEN** 输出包含 Bid1/Ask1 价格和数量变化计数
- **AND** 输出包含 `spread`、`mid`、`imbalance_volume`、`bid`、`ask`、`bidsize` 和 `asksize` 的 OHLC/TWAP/AWAP 值
- **AND** 输出包含 `std_imbalance_volume`

#### Scenario: 多档盘口压力窗口统计
- **WHEN** quote 下采样输入包含 `BidVolume1` 到 `BidVolume5` 和 `AskVolume1` 到 `AskVolume5`
- **THEN** 系统 SHALL 在每条秒频 quote 快照上计算 `imbalance_1`、`imbalance_3` 和 `imbalance_5`
- **AND** `imbalance_1` SHALL 等于 `(BidVolume1 - AskVolume1) / (BidVolume1 + AskVolume1)`
- **AND** `imbalance_3` SHALL 等于 `(sum(BidVolume1..3) - sum(AskVolume1..3)) / (sum(BidVolume1..3) + sum(AskVolume1..3))`
- **AND** `imbalance_5` SHALL 等于 `(sum(BidVolume1..5) - sum(AskVolume1..5)) / (sum(BidVolume1..5) + sum(AskVolume1..5))`
- **AND** 目标频率窗口输出 SHALL 包含 `imbalance_1`、`imbalance_3` 和 `imbalance_5` 的 `open`、`high`、`low`、`close`、`awap`、`twap` 和 `std` 统计列
- **AND** `twap` 和 `awap` SHALL 与现有 quote 统计一致，使用窗口内简单均值
- **AND** `imbalance_1` 的窗口统计 SHALL 与旧 `imbalance_volume` 的同名统计数值一致

#### Scenario: 多档盘口压力零分母处理
- **WHEN** 某条 quote 快照在 `imbalance_1`、`imbalance_3` 或 `imbalance_5` 的 bid 与 ask volume 合计为 `0` 或为空
- **THEN** 对应逐快照压力值 SHALL 为 `0.0`
- **AND** 目标频率窗口内的多档压力统计 SHALL NOT 产生 `NaN`、`inf` 或 `-inf`

#### Scenario: 多档盘口压力输入非有限值 fail-fast
- **WHEN** quote 下采样输入的 `BidVolume1` 到 `BidVolume5` 或 `AskVolume1` 到 `AskVolume5` 任一列包含 `NaN`、`inf` 或 `-inf`
- **THEN** 系统 SHALL fail-fast
- **AND** 错误信息 SHALL 说明 quote volume 包含非有限值

#### Scenario: 多档盘口压力缺少深度列 fail-fast
- **WHEN** quote 下采样输入缺少 `BidVolume2` 到 `BidVolume5` 或 `AskVolume2` 到 `AskVolume5` 中任一必要列
- **THEN** 系统 SHALL fail-fast
- **AND** 系统 SHALL NOT 静默填充缺失深度或合成二到五档盘口数量
