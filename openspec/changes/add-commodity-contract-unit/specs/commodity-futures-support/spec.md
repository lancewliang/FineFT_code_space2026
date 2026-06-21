## MODIFIED Requirements

### Requirement: 商品期货配置
系统 SHALL 为燃料油提供商品期货配置，dataset 和输出 symbol 使用 `fu`，并声明商品合约交易单位。

#### Scenario: 加载燃料油配置
- **WHEN** 商品期货预处理或环境初始化以 symbol `fu` 运行
- **THEN** 系统加载包含 `orderbook_depth=5`、`funding_enabled=false`、`buy_fee_rate=0.0001`、`sell_fee_rate=0.0003`、`contract_unit=10` 的商品配置
- **AND** 系统不要求 `download_operator` 输入

#### Scenario: 使用合约交易单位修正价格口径
- **WHEN** 系统从商品期货 `Volume` 和 `Turnover` 计算秒均价、OHLC 价格或 `vwap`
- **THEN** 系统 SHALL 使用商品配置中的 `contract_unit` 将价格口径修正为 `Turnover / Volume / contract_unit`
- **AND** 系统 SHALL 保持输出 `tradeval` 为原始成交额差分，不除以合约交易单位

#### Scenario: PnL、保证金和手续费不使用合约交易单位
- **WHEN** 系统计算商品期货 PnL、保证金或手续费
- **THEN** 计算过程不乘以或除以合约交易单位

### Requirement: 商品期货基础特征下采样
系统 SHALL 从秒级 `Volume` 和 `Turnover` 差分计算商品期货 OHLCV 和估计成交方向特征。

#### Scenario: 秒级成交估计
- **WHEN** 同一秒内存在多条原始快照
- **THEN** 系统使用该秒最后一条快照的累计 `Volume` 和 `Turnover`
- **AND** 计算 `second_volume = Volume.diff()`、`second_tradeval = Turnover.diff()`、`second_avg_price = second_tradeval / second_volume / contract_unit`

#### Scenario: 聚合 vwap 使用价格口径
- **WHEN** 目标频率窗口内 `volume > 0`
- **THEN** 系统输出 `vwap = tradeval / volume / contract_unit`
- **AND** 输出 `tradeval` 仍为窗口内原始 `second_tradeval` 的合计值

#### Scenario: 无效成交额差分 fail-fast
- **WHEN** `second_volume > 0` 且 `second_tradeval` 为 0、缺失、负数或无效
- **THEN** 预处理报错，错误信息包含 timestamp、contract、`second_volume` 和 `second_tradeval`

#### Scenario: tick rule 估计方向
- **WHEN** 当前有效 `second_avg_price` 大于上一有效 `second_avg_price`
- **THEN** 该秒计为 `up` 并归入 `buy_estimated`
- **WHEN** 当前有效 `second_avg_price` 小于上一有效 `second_avg_price`
- **THEN** 该秒计为 `down` 并归入 `sell_estimated`
- **WHEN** 当前有效 `second_avg_price` 等于上一有效 `second_avg_price`
- **THEN** 该秒计为 `flat`，且不归入 buy 或 sell

#### Scenario: 有 quote 无成交时使用 LastPrice
- **WHEN** 目标频率窗口存在 quote 快照且 `volume=0`
- **THEN** 输出的 trade OHLC、`vwap`、`twap` 和 `awap` 使用上一笔有效 `LastPrice`
- **AND** 输出的 `volume` 和 `tradeval` 为 `0`
- **AND** 不增加估计 buy 或 sell 方向计数
