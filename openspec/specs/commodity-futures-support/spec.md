# commodity-futures-support Specification

## Purpose
定义上海期货交易所商品期货数据接入、主力连续化、五档行情下采样、特征生成和 FineFT 商品期货环境初始化的长期规格。
## Requirements
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

### Requirement: 主力合约拼接
系统 SHALL 从本地五档 CSV 文件构建商品期货连续主力日文件，日归属使用 `TradingDay`，事件
时间戳使用 `ActionDay + UpdateTime`。

#### Scenario: 扫描原始下载目录
- **WHEN** 用户将燃料油原始数据放置在 `data/原始下载/燃料油/2026`
- **THEN** 系统从 `data/原始下载/{品种中文名}/{YYYY}` 开始扫描数据
- **AND** 默认识别 `{MM}/{YYYYMMDD}/{合约}.csv` 层级下的合约 CSV 文件
- **AND** 系统按年份、月份和交易日顺序处理，以支持跨月和跨年的前一交易日主力识别
- **AND** 当 `START_DATE` 和 `END_DATE` 跨越多个年份时，系统 SHALL 自动扫描该日期范围覆盖到的所有年份目录

#### Scenario: 使用前一交易日成交量选择主力
- **WHEN** 参考 `TradingDay` 存在燃料油候选合约源文件
- **THEN** 系统基于前一交易日成交量最大的合格主力月份合约选择目标 `TradingDay` 主力
- **AND** 目标 `TradingDay` 的每行输出包含 `main_contract`、`source_contract` 和 `source_file` 元数据
- **AND** 跨年连续运行时，系统 SHALL 保留前一交易日主力选择状态，不在年边界重置

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

#### Scenario: 按 TradingDay 写出连续主力日文件
- **WHEN** 用户运行 `stitch_main_contract.py` 并设置 `--output_dir PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/fu --start_date 2026-01-01 --end_date 2026-01-04 --symbol fu`
- **THEN** 系统 SHALL 为每个有源数据的 `TradingDay` 写出一个 CSV 文件
- **AND** 文件路径 SHALL 为 `PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/fu/{YYYY-MM-DD}.csv`
- **AND** 系统 MUST NOT 生成 `fu_2026-01-01_2026-01-04.csv` 或其他日期范围大 CSV

#### Scenario: stitch 跳过无源数据日期
- **WHEN** 日期范围包含 `2026-01-02` 但本地原始下载目录没有该交易日的候选合约 CSV
- **THEN** 系统不为 `2026-01-02` 生成 `CONTINUOUS_RAW/fu/2026-01-02.csv`
- **AND** 日志记录被跳过的日期
- **AND** 其他有源数据的日期继续生成日文件

#### Scenario: stitch 覆盖已有日文件
- **WHEN** `CONTINUOUS_RAW/fu/2026-01-05.csv` 已存在且用户重新运行覆盖该日期的 stitch 日期范围
- **THEN** 系统默认覆盖该日文件
- **AND** 日志记录被覆盖的输出路径

#### Scenario: stitch 坏数据 fail-fast
- **WHEN** 某个存在的合约源 CSV 缺少 `InstrumentID`、`TradingDay`、`ActionDay` 或 `UpdateTime`，或目标日没有可交易主力合约
- **THEN** 系统 SHALL 报错并停止本次 stitch
- **AND** 系统 MUST NOT 将该错误当作缺失日期静默跳过

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

### Requirement: 商品期货脚本入口支持日期范围
系统 SHALL 允许商品期货主流程通过 `START_DATE` / `END_DATE` 指定跨年的日期范围，并自动
生成该范围所需的连续主力日文件与后续处理文件。

#### Scenario: 日期范围驱动主流程
- **WHEN** 用户运行 `main.sh` 并设置 `START_DATE=2023-01-01`、`END_DATE=2026-03-01`
- **THEN** 系统自动覆盖 2023、2024、2025 和 2026 的原始目录扫描与主力拼接
- **AND** 系统输出按 `TradingDay` 拆分的连续主力日文件供后续下采样使用
- **AND** 系统 MUST NOT 构造或依赖单条跨年连续主力大 CSV

#### Scenario: 保持左闭右开语义
- **WHEN** 用户希望覆盖到 2026-02-28 的训练窗口
- **THEN** 系统继续使用左闭右开语义，要求 `END_DATE=2026-03-01`
- **AND** 脚本和日志文件名使用日期范围语义而不是单一年份语义

#### Scenario: YEAR 仅作兼容参数
- **WHEN** 用户继续传入 `YEAR`
- **THEN** 系统可以保留该参数作为兼容输入
- **AND** 主流程不再把单一年份作为唯一运行约束

#### Scenario: full process 传递日文件目录
- **WHEN** `fu_full_process.sh` 调用连续主力拼接和连续主力下采样
- **THEN** stitch 调用 SHALL 传递 `--output_dir PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/fu`
- **AND** downscale 调用 SHALL 传递 `--input_dir PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/fu --start_date <START_DATE> --end_date <END_DATE>`
- **AND** shell 脚本 MUST NOT 构造 `continuous_file="${symbol}_${start_date}_${end_date}.csv"` 作为 handoff

### Requirement: 商品期货 Polars 预处理兼容性
系统 SHALL 将 `data_preprocess/operator_futures/commodity` 商品期货核心预处理迁移到 Polars，并保持既有商品期货数据契约。

#### Scenario: 主力合约拼接输出兼容
- **WHEN** 商品期货主力合约拼接读取本地五档 CSV 文件
- **THEN** 系统使用 Polars 处理 CSV 读取、成交量计算、合格主力合约筛选、连续主力拼接和输出写入
- **AND** 输出继续包含 `main_contract`、`source_contract`、`source_file`、`main_contract_trading_day` 和 `main_contract_selection_reason` 元数据
- **AND** `TradingDay` 日归属和 `ActionDay + UpdateTime` 事件时间戳语义保持不变

#### Scenario: 商品 downscale 输出兼容
- **WHEN** 商品期货连续主力数据运行单日或按 `TradingDay` 下采样
- **THEN** 系统使用 Polars 生成 derivative reference、五档 orderbook、base features 和 quote features
- **AND** depth=5 输出不合成第 6 到第 25 档
- **AND** `LastPrice` 回退、funding 兼容列、Volume/Turnover 差分、tick rule 估计方向、右闭窗口聚合和 fail-fast 校验语义保持不变

#### Scenario: 商品 market_type 分支兼容
- **WHEN** `cross_section/create_feature.py`、`scale_describe_save/scale_save.py` 或 `feature_selection/ic_correlation.py` 以 `market_type=commodity_futures` 运行
- **THEN** 商品 reward/execution manifest、depth-aware feature generation、funding 关闭特征处理和 feature selection target 语义保持不变
- **AND** 输出列集合和列顺序继续满足商品期货现有 tests 和 downstream readers

### Requirement: 商品期货连续主力日文件下采样
系统 SHALL 从 `CONTINUOUS_RAW/{symbol}/{YYYY-MM-DD}.csv` 日文件目录按日期范围逐日生成商品期货下采样输出。

#### Scenario: downscale 逐日读取连续主力日文件
- **WHEN** 用户运行 `downscale_continuous_by_trading_day.py --input_dir PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/fu --start_date 2026-01-01 --end_date 2026-01-04 --output_root PREPROCESS_DATASET/commodity-futures --symbol fu --target_freq 5min`
- **THEN** 系统按左闭右开范围查找 `2026-01-01.csv`、`2026-01-02.csv` 和 `2026-01-03.csv`
- **AND** 对每个存在的日文件生成当天 `DOWNSCALE_DERTIC`、`DOWNSCALE_ORDERBOOK_25`、`BASE_FEATURE` 和 `COMMODITY_QUOTE_FEATURE` 输出
- **AND** 输出目录、Feather 文件名、列集合和商品期货特征语义保持不变

#### Scenario: downscale 缺失日文件 warning 后跳过
- **WHEN** 日期范围内 `CONTINUOUS_RAW/fu/2026-01-02.csv` 不存在
- **THEN** 系统记录 warning 并跳过该日期
- **AND** 系统继续处理其他存在的日文件
- **AND** 日志记录被跳过的日期汇总

#### Scenario: downscale 坏日文件 fail-fast
- **WHEN** 日期范围内存在的连续主力日文件缺少必需列、包含非法盘口、包含无效成交额差分或触发空 quote 窗口错误
- **THEN** 系统 SHALL 报错并停止本次 downscale
- **AND** 系统 MUST NOT 将该错误当作缺失日文件 warning 跳过

#### Scenario: downscale CLI 不再接受大文件输入
- **WHEN** 用户调用 `downscale_continuous_by_trading_day.py` 时只传递旧的 `--input PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/fu/fu_2026-01-01_2026-01-04.csv`
- **THEN** CLI 参数解析失败
- **AND** 用户必须改用 `--input_dir --start_date --end_date`

