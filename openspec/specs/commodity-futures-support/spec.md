# commodity-futures-support Specification

## Purpose
定义上海期货交易所商品期货数据接入、主力连续化、五档行情下采样、特征生成和 FineFT 商品期货环境初始化的长期规格。

## Requirements
### Requirement: 商品期货配置
系统 SHALL 为燃料油提供商品期货配置，dataset 和输出 symbol 使用 `fu`。

#### Scenario: 加载燃料油配置
- **WHEN** 商品期货预处理或环境初始化以 symbol `fu` 运行
- **THEN** 系统加载包含 `orderbook_depth=5`、`funding_enabled=false`、`buy_fee_rate=0.0001`、`sell_fee_rate=0.0003` 的商品配置
- **AND** 系统不要求 `download_operator` 输入

#### Scenario: 不使用合约乘数
- **WHEN** 系统计算商品期货秒均价、PnL、保证金或手续费
- **THEN** 计算过程不乘以或除以合约乘数

### Requirement: 主力合约拼接
系统 SHALL 从本地五档 CSV 文件构建商品期货连续主力序列，日归属使用 `TradingDay`，事件
时间戳使用 `ActionDay + UpdateTime`。

#### Scenario: 扫描原始下载目录
- **WHEN** 用户将燃料油原始数据放置在 `data/原始下载/燃料油/2026`
- **THEN** 系统从 `data/原始下载/{品种中文名}/{YYYY}` 开始扫描数据
- **AND** 默认识别 `{MM}/{YYYYMMDD}/{合约}.csv` 层级下的合约 CSV 文件
- **AND** 系统按年份、月份和交易日顺序处理，以支持跨月和跨年的前一交易日主力识别

#### Scenario: 使用前一交易日成交量选择主力
- **WHEN** 参考 `TradingDay` 存在燃料油候选合约源文件
- **THEN** 系统基于前一交易日成交量最大的合格主力月份合约选择目标 `TradingDay` 主力
- **AND** 目标 `TradingDay` 的每行输出包含 `main_contract`、`source_contract` 和 `source_file` 元数据

#### Scenario: 前一日主力不可用时 fallback
- **WHEN** 前一日选出的主力合约在目标日缺失或目标日成交量为 0
- **THEN** 系统回退到目标日成交量最大的合格主力月份合约
- **AND** fallback 决策记录目标 `TradingDay` 和被选合约

#### Scenario: 夜盘时间戳归属
- **WHEN** 某行数据为 `TradingDay=20230104`、`ActionDay=20230103`、`UpdateTime=21:00:00.500`
- **THEN** 输出事件时间戳基于 `2023-01-03 21:00:00.500`
- **AND** 该行仍归属于日文件和主力选择键 `20230104`

#### Scenario: 换月不复权
- **WHEN** 相邻 `TradingDay` 的主力合约发生变化
- **THEN** 系统直接拼接被选合约，不做价格复权

### Requirement: 商品期货参考价下采样
系统 SHALL 从五档快照流派生商品期货环境参考价输出，并关闭 funding 行为。

#### Scenario: 参考价使用 LastPrice
- **WHEN** 商品快照包含有效 `LastPrice`、`BidPrice1` 和 `AskPrice1`
- **THEN** derivative 下采样输出的 `mark_price` 和 `index_price` 等于 `LastPrice`
- **AND** 输出包含 `timestamp`、`symbol`、`funding_timestamp`、`funding_rate`、`index_price` 和 `mark_price`

#### Scenario: LastPrice 回退到 midprice
- **WHEN** `LastPrice` 缺失、为 0 或超出有效涨跌停范围
- **THEN** derivative 下采样输出的 `mark_price` 和 `index_price` 为 `(BidPrice1 + AskPrice1) / 2`

#### Scenario: funding 关闭
- **WHEN** 生成商品期货环境数据
- **THEN** `funding_rate` 只作为兼容列输出且值为 `0`
- **AND** 下游商品环境不扣 funding，也不暴露 funding countdown 状态

### Requirement: 商品期货五档盘口下采样
系统 SHALL 使用真实 depth=5 下采样商品期货盘口快照，MUST NOT 合成第 6 到第 25 档。

#### Scenario: 输出五档列
- **WHEN** 燃料油 `fu` 运行 orderbook 下采样
- **THEN** 输出包含 `ask1_price` 到 `ask5_price`、`ask1_size` 到 `ask5_size`、`bid1_price` 到 `bid5_price`、`bid1_size` 到 `bid5_size`
- **AND** 输出不包含 `ask6_price`、`bid6_price`、`ask25_price` 或 `bid25_price`

#### Scenario: 异常最优报价 fail-fast
- **WHEN** 源数据行存在缺失 `BidPrice1`、缺失 `AskPrice1`、最优价为 0 或 `BidPrice1 >= AskPrice1`
- **THEN** 预处理报错，错误信息包含日期、合约和异常字段名

### Requirement: 商品期货基础特征下采样
系统 SHALL 从秒级 `Volume` 和 `Turnover` 差分计算商品期货 OHLCV 和估计成交方向特征。

#### Scenario: 秒级成交估计
- **WHEN** 同一秒内存在多条原始快照
- **THEN** 系统使用该秒最后一条快照的累计 `Volume` 和 `Turnover`
- **AND** 计算 `second_volume = Volume.diff()`、`second_tradeval = Turnover.diff()`、`second_avg_price = second_tradeval / second_volume`

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

### Requirement: 商品期货 quote 特征下采样
系统 SHALL 从秒频五档快照派生 quote 特征，并使用右闭右标窗口聚合到目标频率。

#### Scenario: 秒频 quote 快照
- **WHEN** 同一秒内存在多条原始快照
- **THEN** 该秒 quote 状态使用该秒最后一条快照
- **AND** 秒频标准层不 forward fill 缺失秒

#### Scenario: 右闭目标窗口
- **WHEN** 目标频率为 `5min`
- **THEN** `(09:00:00, 09:05:00]` 内的快照聚合到标记为 `09:05:00` 的 bar

#### Scenario: 空 quote 窗口 fail-fast
- **WHEN** 目标频率窗口没有任何秒频 quote 快照
- **THEN** 预处理报错，错误信息包含 `TradingDay`、窗口标签和合约

#### Scenario: quote 计数与状态特征
- **WHEN** 目标频率窗口存在秒频 quote 快照
- **THEN** 输出包含 Bid1/Ask1 价格和数量变化计数
- **AND** 输出包含 `spread`、`mid`、`imbalance_volume`、`bid`、`ask`、`bidsize` 和 `asksize` 的 OHLC/TWAP/AWAP 值

### Requirement: 商品期货 cross-section 与时间特征
系统 SHALL 使用可配置深度和显式 reward/execution manifest 生成商品期货特征。

#### Scenario: depth-aware snapshot 特征
- **WHEN** cross-section 以 `orderbook_depth=5` 运行
- **THEN** 系统生成 KLINE、QUOTE 和 depth=5 SNAPSHOT 特征
- **AND** 系统不生成依赖第 6 到第 25 档的特征

#### Scenario: 移除不可用特征
- **WHEN** 商品期货特征处理完成
- **THEN** 输出不包含 funding 特征、真实 index/mark basis 特征、真实逐笔成交数或未标记的真实 buy/sell 特征
- **AND** 估计方向特征以 `_estimated` 命名或在特征元数据中标记为 estimated

#### Scenario: feature selection target
- **WHEN** 商品期货 feature selection 为 `1`、`6`、`12` 等窗口计算 target
- **THEN** target 为 `mark_price.shift(-window) - mark_price`

#### Scenario: scale save 使用 manifest
- **WHEN** depth=5 商品数据运行 scale/save
- **THEN** reward/execution 列来自显式 manifest 或等价列列表
- **AND** 实现不假设前 106 列是 reward/execution 列

### Requirement: 商品期货环境初始化
系统 SHALL 使用 depth=5 商品数据初始化 FineFT 商品期货环境，并关闭 funding，按 symbol 配置买入/卖出费率。

#### Scenario: 使用商品数据 reset 环境
- **WHEN** `fu` 商品数据集包含 `df.feather`、`state_features.npy` 和商品环境配置
- **THEN** 商品环境使用 1-5 档 ask/bid 价格与数量数组初始化
- **AND** `reset()` 返回 observation 和 available action mask，且不包含 funding countdown 输入

#### Scenario: 商品交易手续费
- **WHEN** 商品环境 step 开仓或平仓
- **THEN** 买入方向成交金额使用 `buy_fee_rate=0.0001`
- **AND** 卖出方向成交金额使用 `sell_fee_rate=0.0003`

#### Scenario: 深度不足 fail-fast
- **WHEN** 请求的目标仓位无法在可用五档数量内完全成交
- **THEN** 商品环境 fail-fast，不部分成交、不静默拒单、不外推到第五档之外

### Requirement: 商品期货 smoke 验证
系统 SHALL 使用仓库样例数据和小型本地流程提供商品期货预处理与环境行为的聚焦验证。

#### Scenario: 样例数据测试
- **WHEN** 测试使用 `docs/上海商品交易所/fu2302.csv`
- **THEN** 测试验证时间戳解析、秒频标准化、右闭聚合、`Volume`/`Turnover` 差分、异常 quote 检查和 depth=5 输出列

#### Scenario: 端到端 smoke test
- **WHEN** 小型燃料油样例流程从主力拼接数据运行到 scale/save
- **THEN** 流程输出 FineFT 可读的商品数据集文件，并初始化可执行 `reset()` 和一次 `step()` 的商品环境
