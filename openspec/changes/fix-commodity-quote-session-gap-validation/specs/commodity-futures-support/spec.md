## MODIFIED Requirements

### Requirement: 商品期货配置
系统 SHALL 为燃料油提供商品期货配置，dataset 和输出 symbol 使用 `fu`，并声明商品合约交易单位和交易 session。

#### Scenario: 加载燃料油配置
- **WHEN** 商品期货预处理或环境初始化以 symbol `fu` 运行
- **THEN** 系统加载包含 `orderbook_depth=5`、`funding_enabled=false`、`buy_fee_rate=0.0001`、`sell_fee_rate=0.0003`、`contract_unit=10` 的商品配置
- **AND** 系统加载燃料油常规交易 session 配置
- **AND** 系统不要求 `download_operator` 输入

#### Scenario: 使用合约交易单位修正价格口径
- **WHEN** 系统从商品期货 `Volume` 和 `Turnover` 计算秒均价、OHLC 价格或 `vwap`
- **THEN** 系统 SHALL 使用商品配置中的 `contract_unit` 将价格口径修正为 `Turnover / Volume / contract_unit`
- **AND** 系统 SHALL 保持输出 `tradeval` 为原始成交额差分，不除以合约交易单位

#### Scenario: PnL、保证金和手续费不使用合约交易单位
- **WHEN** 系统计算商品期货 PnL、保证金或手续费
- **THEN** 计算过程不乘以或除以合约交易单位

#### Scenario: 交易 session 用于 quote gap 校验
- **WHEN** 商品 quote 下采样检查目标频率窗口连续性
- **THEN** 系统 SHALL 使用商品配置中的交易 session 判断相邻 quote bar 是否属于同一有效交易 session
- **AND** 系统 SHALL NOT 将跨 session、跨自然日、周末或休市时间的自然时间间隔视为缺失 quote snapshot

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
