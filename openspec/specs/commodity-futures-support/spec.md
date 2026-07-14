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
系统 SHALL 从本地五档 CSV 文件生成商品期货主力合约 summary，日归属使用 `TradingDay`，事件时间戳语义使用 `ActionDay + UpdateTime`。

#### Scenario: 扫描原始下载目录
- **WHEN** 用户将燃料油原始数据放置在 `data/原始下载/燃料油/2026`
- **THEN** 系统从 `data/原始下载/{品种中文名}/{YYYY}` 开始扫描数据
- **AND** 默认识别 `{MM}/{YYYYMMDD}/{合约}.csv` 层级下的合约 CSV 文件
- **AND** 当 `START_DATE` 和 `END_DATE` 跨越多个年份时，系统 SHALL 自动扫描该日期范围覆盖到的所有年份目录
- **AND** 系统 SHALL 使用左闭右开日期范围筛选 `TradingDay`

#### Scenario: 按自然月选择主力合约集合
- **WHEN** 日期范围内存在多个燃料油候选合约源文件
- **THEN** 系统 SHALL 按自然月统计每个合约的月成交量
- **AND** 单日成交量 SHALL 使用该合约该 `TradingDay` 源文件的 `Volume.max - Volume.min`
- **AND** 月成交量 SHALL 为该自然月内单日成交量之和
- **AND** 系统 SHALL 为每个自然月选择月成交量最高的前 2 个合约
- **AND** 系统 SHALL 额外选择同一自然月内至少 10 个实际交易日单日成交量大于配置阈值的合约
- **AND** 高成交量天数规则 SHALL 使用严格大于：`daily_volume > threshold`
- **AND** `fu` 的高成交量天数配置阈值 SHALL 为 `15000`
- **AND** 两条选择规则 SHALL 取并集；同一合约同一月份重复命中时只记录一次该月份
- **AND** 月成交量并列时 SHALL 按合约名升序稳定排序

#### Scenario: 配置化高成交量天数入选
- **WHEN** 燃料油合约 `fu2609` 在 `2026-03` 有 10 个实际交易日的 `daily_volume > 15000`
- **AND** `fu2609` 不是 `2026-03` 月成交量最高的前 2 个合约
- **THEN** 系统 SHALL 仍将 `fu2609` 加入 `2026-03` 主力合约集合
- **AND** summary 中 `fu2609.selected_months` SHALL 包含 `2026-03`

#### Scenario: 入选合约集合语义
- **WHEN** 某合约在任意自然月进入成交量前 2
- **THEN** 系统 SHALL 将该合约加入主力合约集合
- **AND** 系统 SHALL 以该合约首次入选月份的月初作为交易日窗口开始下限
- **AND** 系统 SHALL 以请求日期范围内该合约最后交易日前第 10 个合约交易日作为交易日窗口结束上限，且结束上限为包含式
- **AND** 系统 SHALL 为该合约只记录上述窗口内实际存在的 `TradingDay` 源文件
- **AND** 系统 SHALL NOT 要求该合约连续入选或连续交易

#### Scenario: 写出主力合约 summary JSON
- **WHEN** 用户运行 `stitch_main_contract.py` 并设置 `--output_dir PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/fu --start_date 2026-01-01 --end_date 2026-04-01 --symbol fu`
- **THEN** 系统 SHALL 写出 `PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/fu/main_contract_summary.json`
- **AND** summary SHALL 包含 `symbol`、`commodity_name`、`start_date`、`end_date`、`selection_rule` 和 `contracts`
- **AND** 系统 MUST NOT 写出 `PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/fu/{YYYY-MM-DD}.csv` 连续主力日文件
- **AND** 系统 MUST NOT 生成 `fu_2026-01-01_2026-04-01.csv` 或其他日期范围大 CSV

#### Scenario: summary contract 字段
- **WHEN** summary 中包含合约 `fu2601`
- **THEN** 该合约对象 SHALL 包含 `contract`、`start_trading_day`、`end_trading_day`、`trading_day_count`、`selected_months` 和 `trading_days`
- **AND** `start_trading_day` SHALL 等于该合约裁剪后 `trading_days` 中最小 `TradingDay`
- **AND** `end_trading_day` SHALL 等于该合约裁剪后 `trading_days` 中最大 `TradingDay`
- **AND** `trading_day_count` SHALL 等于 `len(trading_days)`
- **AND** 每个 `trading_days` 条目 SHALL 包含 `trading_day`、ISO `date`、`source_file` 和 `daily_volume`
- **AND** `daily_volume` SHALL 等于该合约该 `TradingDay` 源文件的 `Volume.max - Volume.min`

#### Scenario: 合约交易日窗口裁剪
- **WHEN** 合约 `fu2605` 首次入选月份为 `2026-03`，且该合约在请求日期范围内的最后 11 个 `TradingDay` 为 `20260318` 到 `20260401`
- **THEN** summary SHALL 只保留该合约 `TradingDay >= 20260301` 的实际交易日
- **AND** summary SHALL 排除该合约在请求日期范围内的最后 10 个交易日
- **AND** summary 中该合约 `end_trading_day` SHALL 等于请求日期范围内最后交易日前第 10 个交易日对应的 `TradingDay`
- **AND** 如果裁剪后该合约没有任何可保留交易日，系统 SHALL 报错并停止 summary 生成

#### Scenario: 夜盘时间戳语义保留
- **WHEN** 某行数据为 `TradingDay=20230104`、`ActionDay=20230103`、`UpdateTime=21:00:00.500`
- **THEN** summary 和后续读取 SHALL 继续将该源文件归属于 `TradingDay=20230104`
- **AND** 后续 downscale 仍基于 `2023-01-03 21:00:00.500` 生成事件时间戳

#### Scenario: summary 坏数据 fail-fast
- **WHEN** 某个存在的合约源 CSV 缺少 `InstrumentID`、`TradingDay`、`ActionDay`、`UpdateTime` 或 `Volume`
- **THEN** 系统 SHALL 报错并停止本次 summary 生成
- **AND** 系统 MUST NOT 将该错误当作缺失日期静默跳过

#### Scenario: summary 冲突数据 fail-fast
- **WHEN** 同一个 `TradingDay + contract` 命中多个源文件
- **THEN** 系统 SHALL 报错并停止本次 summary 生成
- **AND** 错误信息 SHALL 包含冲突的 `TradingDay`、contract 和源文件路径

#### Scenario: 无入选合约 fail-fast
- **WHEN** 日期范围内没有可交易候选合约或没有任何合约进入月度 top 2
- **THEN** 系统 SHALL 报错并停止本次 summary 生成

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
系统 SHALL 允许商品期货主流程通过 `START_DATE` / `END_DATE` 指定跨年的日期范围，并自动生成该范围所需的主力合约 summary 与后续按合约处理文件。

#### Scenario: 日期范围驱动主流程
- **WHEN** 用户运行商品期货主流程并设置 `START_DATE=2023-01-01`、`END_DATE=2026-03-01`
- **THEN** 系统自动覆盖 2023、2024、2025 和 2026 的原始目录扫描与 summary 生成
- **AND** 系统输出 `CONTINUOUS_RAW/{symbol}/main_contract_summary.json` 供后续下采样使用
- **AND** 系统 MUST NOT 构造或依赖单条跨年连续主力大 CSV
- **AND** 系统 MUST NOT 构造或依赖 `CONTINUOUS_RAW/{symbol}/{YYYY-MM-DD}.csv` 连续主力日文件

#### Scenario: 保持左闭右开语义
- **WHEN** 用户希望覆盖到 2026-02-28 的训练窗口
- **THEN** 系统继续使用左闭右开语义，要求 `END_DATE=2026-03-01`
- **AND** 脚本和日志文件名使用日期范围语义而不是单一年份语义

#### Scenario: YEAR 仅作兼容参数
- **WHEN** 用户继续传入 `YEAR`
- **THEN** 系统可以保留该参数作为兼容输入
- **AND** 主流程不再把单一年份作为唯一运行约束

#### Scenario: full process 传递 summary
- **WHEN** `fu_full_process.sh` 调用主力合约 summary 生成和下采样
- **THEN** stitch 调用 SHALL 传递 `--output_dir PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/{symbol}`
- **AND** downscale 调用 SHALL 传递 `--summary PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/{symbol}/main_contract_summary.json`
- **AND** shell 脚本 MUST NOT 构造 `continuous_file="${symbol}_${start_date}_${end_date}.csv"` 作为 handoff
- **AND** shell 脚本 MUST NOT 把 `CONTINUOUS_RAW/{symbol}` 当作日文件目录传给 downscale

#### Scenario: full process 按 summary 合约循环并在 union 后执行 scale save
- **WHEN** `main_contract_summary.json` 中包含合约 `fu2601` 和 `fu2605`
- **THEN** `fu_full_process.sh` SHALL 从 summary 读取合约列表
- **AND** `fu_full_process.sh` SHALL 分别为 `fu2601` 和 `fu2605` 调用 `cross_section`、`merge`、`concat`、`time_feature`、`merge_clean` 和 `ic_candidate`
- **AND** 每次合约级调用 SHALL 传递 `--symbols fu --contract <contract>`
- **AND** 所有合约 `ic_candidate` 完成后，`fu_full_process.sh` SHALL 调用品种级 `ic_union_finalize`
- **AND** `ic_union_finalize` 完成后，`fu_full_process.sh` SHALL 分别为每个合约调用 `scale_save`
- **AND** `fu_full_process.sh` SHALL NOT 保留独立后置的旧 `feature_union` 步骤

### Requirement: 商品期货 Polars 预处理兼容性
系统 SHALL 将 `data_preprocess/operator_futures/commodity` 商品期货核心预处理迁移到 Polars，并保持既有商品期货数据契约。

#### Scenario: 主力合约 summary 输出兼容
- **WHEN** 商品期货主力合约 summary 生成读取本地五档 CSV 文件
- **THEN** 系统使用 Polars 处理 CSV 读取、成交量计算、合格合约筛选、月度 top 2 选择和 summary 写入
- **AND** summary SHALL 保留 `TradingDay` 日归属和 `ActionDay + UpdateTime` 事件时间戳语义所需的源文件信息
- **AND** summary SHALL 提供后续 downscale 所需的 contract、date 和 source_file 明细

#### Scenario: 商品 downscale 输出兼容
- **WHEN** 商品期货 summary 源文件运行单日或按合约下采样
- **THEN** 系统使用 Polars 生成 derivative reference、五档 orderbook、base features 和 quote features
- **AND** depth=5 输出不合成第 6 到第 25 档
- **AND** `LastPrice` 回退、funding 兼容列、Volume/Turnover 差分、tick rule 估计方向、右闭窗口聚合和 fail-fast 校验语义保持不变

#### Scenario: 商品 market_type 分支兼容
- **WHEN** `cross_section/create_feature.py`、`scale_describe_save/scale_save.py` 或 `feature_selection/ic_correlation.py` 以 `market_type=commodity_futures` 运行
- **THEN** 商品 reward/execution manifest、depth-aware feature generation、funding 关闭特征处理和 feature selection target 语义保持不变
- **AND** 输出列集合和列顺序继续满足商品期货现有 tests 和 downstream readers

### Requirement: 商品期货连续主力日文件下采样
系统 SHALL 从 `main_contract_summary.json` 记录的源文件按合约和交易日生成商品期货下采样输出。

#### Scenario: downscale 读取 summary 处理全部合约
- **WHEN** 用户运行 `downscale_continuous_by_trading_day.py --summary PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/fu/main_contract_summary.json --output_root PREPROCESS_DATASET/commodity-futures --symbol fu --target_freq 5min`
- **THEN** 系统 SHALL 读取 summary 中全部 `contracts`
- **AND** 系统 SHALL 按每个合约的 `trading_days[].source_file` 读取原始 CSV
- **AND** 系统 SHALL 为每个合约实际存在的交易日生成 `DOWNSCALE_DERTIC`、`DOWNSCALE_ORDERBOOK_25`、`BASE_FEATURE` 和 `COMMODITY_QUOTE_FEATURE` 输出

#### Scenario: downscale 输出 contract-scoped 日文件
- **WHEN** summary 中合约 `fu2601` 包含 `date=2026-01-05`
- **THEN** downscale 输出路径 SHALL 使用 `{FEATURE_FOLDER}/fu/fu2601/5min/2026-01-05.feather`
- **AND** 输出列集合、窗口语义、depth=5 行为、商品价格口径和 fail-fast 校验 SHALL 保持现有商品期货特征语义

#### Scenario: downscale 单合约过滤
- **WHEN** 用户运行 `downscale_continuous_by_trading_day.py` 并传入 `--contract fu2601`
- **THEN** 系统 SHALL 只处理 summary 中 contract 为 `fu2601` 的源文件
- **AND** 如果 `fu2601` 不存在于 summary，系统 SHALL 报错

#### Scenario: downscale summary 校验 fail-fast
- **WHEN** summary 文件不存在、JSON 结构不合法、缺少必需字段或 `trading_day_count != len(trading_days)`
- **THEN** 系统 SHALL 报错并停止本次 downscale

#### Scenario: downscale source_file 缺失 fail-fast
- **WHEN** summary 中任一待处理 `source_file` 不存在
- **THEN** 系统 SHALL 报错并停止本次 downscale
- **AND** 系统 MUST NOT 将 summary 中列出的源文件缺失当作普通无交易日跳过

#### Scenario: downscale CLI 不再接受 input_dir 日文件目录
- **WHEN** 用户调用 `downscale_continuous_by_trading_day.py` 时只传递旧的 `--input_dir PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/fu --start_date 2026-01-01 --end_date 2026-01-04`
- **THEN** CLI 参数解析失败
- **AND** 用户必须改用 `--summary PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/fu/main_contract_summary.json`

### Requirement: 商品期货主流程步骤日志
系统 SHALL 为商品期货 preprocess 主流程的主要阶段生成独立步骤日志，并在总日志中记录阶段状态。

#### Scenario: 主流程生成步骤日志
- **WHEN** 用户运行 `data_preprocess/script_preprocess/future_upgraded/commodity/main.sh`，且 `SYMBOL=fu`、`TARGET_FREQ=5min`、`START_DATE=2025-11-03`、`END_DATE=2025-11-08`
- **THEN** 系统 SHALL 为 `stitch_main_contract`、`downscale_continuous_by_trading_day`、`cross_section`、`merge`、`concat`、`time_feature`、`merge_clean`、`ic_correlation` 和 `scale_save` 生成独立日志文件
- **AND** 每个步骤日志文件名 SHALL 包含 symbol、target_freq、start_date、end_date 和步骤名
- **AND** 每个步骤日志 SHALL 捕获该步骤的 stdout 和 stderr

#### Scenario: 总日志记录阶段状态
- **WHEN** 商品 preprocess 主流程执行任一主要步骤
- **THEN** 总日志 SHALL 记录该步骤的开始信息和步骤日志路径
- **AND** 当步骤成功完成时，总日志 SHALL 记录该步骤成功完成
- **AND** 当步骤失败时，总日志 SHALL 记录该步骤失败和对应日志路径

#### Scenario: 失败语义保持 fail-fast
- **WHEN** 任一主要步骤返回非 0 状态
- **THEN** 商品 preprocess 主流程 SHALL 以非 0 状态退出
- **AND** 系统 SHALL 保留失败步骤日志中的错误输出
- **AND** 系统 MUST NOT 因日志包装而继续执行后续主要步骤

#### Scenario: 现有子日志继续保留
- **WHEN** `cross_section` 或 `merge` 阶段继续按日期启动子任务日志
- **THEN** 系统 SHALL 保留现有按日期子日志目录和文件
- **AND** 新增步骤日志 MUST NOT 删除、重命名或替代这些子日志

### Requirement: 商品期货按合约生成因子文件
系统 SHALL 在商品期货多合约流程中按具体合约生成独立因子文件，并在未传 contract 时保留共享脚本旧路径行为。

#### Scenario: cross-section 按 contract 读写日文件
- **WHEN** `cross_section/create_feature.py` 以 `--symbols fu --contract fu2601 --target_freq 5min --date 2026-01-05` 运行
- **THEN** 系统 SHALL 从 `BASE_FEATURE/fu/fu2601/5min/2026-01-05.feather` 和 `DOWNSCALE_ORDERBOOK_25/fu/fu2601/5min/2026-01-05.feather` 读取输入
- **AND** 系统 SHALL 写出 `CROSS_SECTION/KLINE_FEATURE/fu/fu2601/5min/2026-01-05.feather`
- **AND** 系统 SHALL 写出 `CROSS_SECTION/QUOTES_FEATURE/fu/fu2601/5min/2026-01-05.feather`
- **AND** 系统 SHALL 写出 `CROSS_SECTION/SNAPSHOT_FEATURE/fu/fu2601/5min/2026-01-05.feather`

#### Scenario: merge 按 contract 读写日文件
- **WHEN** `merge_concat/merge.py` 以 `--symbols fu --contract fu2601 --target_freq 5min --date 2026-01-05` 运行
- **THEN** 系统 SHALL 从 downscale 和 cross-section 的 `fu/fu2601/5min` 日文件读取输入
- **AND** 系统 SHALL 写出 `MERGE_CONCAT/MERGED_FEATURE/fu/fu2601/5min/CONCURRENT_FEATURE/2026-01-05.feather`
- **AND** 系统 SHALL 写出 `MERGE_CONCAT/MERGED_FEATURE/fu/fu2601/5min/FUTURE_FEATURE/2026-01-05.feather`

#### Scenario: concat 按 contract 生成日期范围文件
- **WHEN** `merge_concat/concat.py` 以 `--symbols fu --contract fu2601 --target_freq 5min --start_date 2026-01-01 --end_date 2026-04-01` 运行
- **THEN** 系统 SHALL 从 `MERGE_CONCAT/MERGED_FEATURE/fu/fu2601/5min/...` 读取日文件
- **AND** 系统 SHALL 写出 `MERGE_CONCAT/CONCAT_FEATURE/fu/fu2601/5min/2026-01-01-2026-04-01.feather`

#### Scenario: time feature 按 contract 生成日期范围文件
- **WHEN** `time_operator/create_feature_multi_processing.py` 以 `--symbols fu --contract fu2601 --target_freq 5min --start_date 2026-01-01 --end_date 2026-04-01` 运行
- **THEN** 系统 SHALL 从 `MERGE_CONCAT/CONCAT_FEATURE/fu/fu2601/5min/2026-01-01-2026-04-01.feather` 读取输入
- **AND** 系统 SHALL 写出 `TIME_FEATURE/fu/fu2601/5min/2026-01-01-2026-04-01.feather`

#### Scenario: merge clean 按 contract 生成 all feature
- **WHEN** `merge_all/merge_clean.py` 以 `--symbols fu --contract fu2601 --target_freq 5min --start_date 2026-01-01 --end_date 2026-04-01` 运行
- **THEN** 系统 SHALL 从 `MERGE_CONCAT/CONCAT_FEATURE/fu/fu2601/5min/2026-01-01-2026-04-01.feather` 和 `TIME_FEATURE/fu/fu2601/5min/2026-01-01-2026-04-01.feather` 读取输入
- **AND** 系统 SHALL 写出 `ALL_FEATURE/fu/fu2601/5min/2026-01-01-2026-04-01.feather`

#### Scenario: feature selection 和 scale save 按 contract 生成日期范围目录
- **WHEN** `feature_selection/ic_correlation.py` 和 `scale_describe_save/scale_save.py` 以 `--symbols fu --contract fu2601 --target_freq 5min --start_date 2026-01-01 --end_date 2026-04-01` 运行
- **THEN** feature selection SHALL 读取 `ALL_FEATURE/fu/fu2601/5min/2026-01-01-2026-04-01.feather`
- **AND** feature selection SHALL 写出 `IC_RESULT/fu/fu2601/5min/2026-01-01-2026-04-01/`
- **AND** scale save SHALL 读取 `IC_RESULT/fu/fu2601/5min/2026-01-01-2026-04-01/`
- **AND** scale save SHALL 写出 `SCALE_SAVE/fu/fu2601/5min/2026-01-01-2026-04-01/`

#### Scenario: 未传 contract 时保留旧路径
- **WHEN** 共享 operator-futures 脚本未传入 `--contract`
- **THEN** 系统 SHALL 继续读写现有 `{symbol}/{target_freq}` 路径
- **AND** 系统 SHALL NOT 要求非商品期货或旧调用方提供 contract 参数

#### Scenario: 多合约日志和 skip 检查包含 contract
- **WHEN** 商品 full process 对多个合约运行后续阶段
- **THEN** 步骤日志文件名、skip 消息和输出存在性检查 SHALL 包含 `symbol` 和 `contract`
- **AND** 一个合约的日志或输出 SHALL NOT 覆盖另一个合约的日志或输出

### Requirement: 商品期货跨合约训练特征合集
系统 SHALL 在所有入选合约完成单合约特征选择和 scale save 后，生成品种级统一 state feature 合集，供后续同模型训练读取。

#### Scenario: 生成品种级 state feature union
- **WHEN** `main_contract_summary.json` 中包含合约 `fu2601` 和 `fu2605`
- **AND** `SCALE_SAVE/fu/fu2601/5min/2026-01-01-2026-04-01/state_features.npy` 包含 `["alpha", "beta"]`
- **AND** `SCALE_SAVE/fu/fu2605/5min/2026-01-01-2026-04-01/state_features.npy` 包含 `["beta", "gamma"]`
- **THEN** 系统 SHALL 写出 `FEATURE_UNION/fu/5min/2026-01-01-2026-04-01/state_features.npy`
- **AND** 该合集 SHALL 包含 `["alpha", "beta", "gamma"]`
- **AND** 系统 SHALL 写出同目录下的 `feature_union_manifest.json`
- **AND** manifest SHALL 包含 `symbol`、`target_freq`、`start_date`、`end_date`、`contracts`、`state_feature_count`、`state_features` 和每个合约的输入 `state_features.npy` 路径

#### Scenario: feature union 顺序稳定
- **WHEN** 系统生成跨合约 state feature union
- **THEN** 系统 SHALL 按 summary 中 `contracts` 的顺序读取每个合约
- **AND** 系统 SHALL 按每个合约 `state_features.npy` 内的原始顺序追加特征
- **AND** 重复 state feature SHALL 只保留第一次出现的位置
- **AND** 多次运行相同输入 SHALL 生成相同顺序的 union feature list

#### Scenario: feature union 缺失合约产物 fail-fast
- **WHEN** summary 中包含合约 `fu2605`
- **AND** `SCALE_SAVE/fu/fu2605/5min/2026-01-01-2026-04-01/state_features.npy` 不存在
- **THEN** 系统 SHALL 报错并停止 feature union 生成
- **AND** 错误信息 SHALL 包含缺失合约 `fu2605` 和缺失的 `state_features.npy` 路径

#### Scenario: full process 最后生成 feature union
- **WHEN** `fu_full_process.sh` 已对 summary 中所有合约完成 `scale_save`
- **THEN** `fu_full_process.sh` SHALL 调用品种级 feature union 生成步骤
- **AND** 该步骤 SHALL 使用同一次运行的 `summary_path`、`symbol`、`target_freq`、`start_date` 和 `end_date`
- **AND** feature union 日志、skip 检查和验证输出 SHALL 使用 symbol 级别，不绑定单个 contract

#### Scenario: validation 检查 feature union
- **WHEN** `validate_features.sh` 验证商品期货输出
- **THEN** 脚本 SHALL 检查 `FEATURE_UNION/{symbol}/{target_freq}/{start_date}-{end_date}/state_features.npy`
- **AND** 脚本 SHALL 检查 `FEATURE_UNION/{symbol}/{target_freq}/{start_date}-{end_date}/feature_union_manifest.json`
- **AND** 缺少任一 feature union 产物时验证 SHALL 失败

### Requirement: 商品期货多合约 feature selection union
系统 SHALL 将商品期货多合约 feature selection 拆分为 candidate 和 union finalize 两个阶段，确保所有合约使用同一份 union state feature 列表，并为每个合约生成按 union 过滤后的标准 `IC_RESULT` 数据文件。

#### Scenario: 单合约 candidate 阶段不写最终数据文件
- **WHEN** 商品期货合约 `fu2601` 运行 IC candidate 阶段，输入为 `PREPROCESS_DATASET/commodity-futures/ALL_FEATURE/fu/fu2601/5min/2026-01-01-2026-04-01.feather`
- **THEN** 系统 SHALL 写出 `PREPROCESS_DATASET/commodity-futures/IC_RESULT/fu/fu2601/5min/2026-01-01-2026-04-01/state_features_candidate.npy`
- **AND** 系统 SHALL 写出 `ic_window_<window>.json` 和 `correlation.csv`
- **AND** 系统 SHALL NOT 在 candidate 阶段写出标准 `df.feather`
- **AND** 系统 SHALL NOT 在 candidate 阶段写出标准 `state_features.npy`

#### Scenario: union finalize 生成品种级 state features
- **WHEN** `main_contract_summary.json` 包含合约 `fu2601` 和 `fu2605`
- **AND** 两个合约均已生成 `state_features_candidate.npy`
- **THEN** union finalize 阶段 SHALL 按 summary 合约列表读取所有 candidate feature 文件
- **AND** 系统 SHALL 去重合并候选特征并保持稳定顺序
- **AND** 系统 SHALL 写出 `PREPROCESS_DATASET/commodity-futures/FEATURE_UNION/fu/5min/2026-01-01-2026-04-01/state_features.npy`
- **AND** 系统 SHALL 写出 `feature_union_manifest.json`，记录每个合约 candidate 路径、candidate 特征数、union 特征数和最终合约输出路径

#### Scenario: union finalize 生成每个合约过滤后的 IC_RESULT
- **WHEN** union state features 为 `["f1", "f2", "f3"]`
- **AND** 合约 `fu2601` 和 `fu2605` 的 `ALL_FEATURE` 均包含 reward/execution 列和 `f1`、`f2`、`f3`
- **THEN** 系统 SHALL 为每个合约读取对应 `ALL_FEATURE/{symbol}/{contract}/{target_freq}/{start_date}-{end_date}.feather`
- **AND** 系统 SHALL 为每个合约写出 `IC_RESULT/{symbol}/{contract}/{target_freq}/{start_date}-{end_date}/df.feather`
- **AND** 每个合约的 `df.feather` SHALL 包含 `reward_features + union_state_features`
- **AND** 系统 SHALL 为每个合约写出标准 `state_features.npy`
- **AND** 每个合约标准 `state_features.npy` SHALL 与品种级 `FEATURE_UNION/state_features.npy` 内容一致

#### Scenario: union 特征缺列 fail-fast
- **WHEN** union state features 包含 `f3`
- **AND** 合约 `fu2605` 的 `ALL_FEATURE` 不包含 `f3`
- **THEN** union finalize SHALL 报错并停止
- **AND** 错误信息 SHALL 包含合约 `fu2605` 和缺失特征 `f3`
- **AND** 系统 SHALL NOT 静默丢弃 `f3`
- **AND** 系统 SHALL NOT 降级为使用 `fu2605` 自身 candidate 特征

#### Scenario: scale save 继续消费标准 IC_RESULT
- **WHEN** union finalize 已为合约 `fu2601` 写出标准 `IC_RESULT/fu/fu2601/5min/2026-01-01-2026-04-01/df.feather` 和 `state_features.npy`
- **THEN** `scale_save.py` SHALL 按现有接口读取该合约标准 `IC_RESULT` 输出
- **AND** `scale_save.py` SHALL 继续只负责缩放 state features 并保存 `SCALE_SAVE`
- **AND** `scale_save.py` SHALL NOT 负责生成 union、补齐缺列或降级选择合约自身 candidate 特征

### Requirement: 商品 FineFT 多合约数据集边界计算
系统 SHALL 从商品主力合约 summary 计算 FineFT 商品多合约数据集的全局 train、valid、test 日期边界，保证三个集合时序递进且日期不重叠。

#### Scenario: 按 5:3:2 计算全局边界
- **WHEN** `main_contract_summary.json` 包含多个合约及其有效 `trading_days[].date`
- **THEN** 系统 SHALL 对所有合约有效交易日取去重升序并集作为全局交易日轴
- **AND** 系统 SHALL 按 `train:valid:test = 5:3:2` 计算 `start`、`a`、`b`、`c`
- **AND** 系统 SHALL 生成左闭右开的集合范围：`train=[start,a)`、`valid=[a,b)`、`test=[b,c)`
- **AND** `start < a < b < c` SHALL 成立
- **AND** 三个集合的日期范围 SHALL 不重叠

#### Scenario: 合约集合归属由全局边界求交决定
- **WHEN** 合约 `fu2601` 的有效交易日跨越全局边界 `a` 和 `b`
- **THEN** 系统 SHALL 使用该合约有效交易日分别与 `[start,a)`、`[a,b)`、`[b,c)` 求交
- **AND** 合约同一个交易日 SHALL 最多归属于一个集合
- **AND** 系统 SHALL NOT 按行数或拼接后的连续行情重新计算该合约的集合归属

#### Scenario: 无法形成有效边界 fail-fast
- **WHEN** summary 中有效交易日不足以形成非空 train、valid 和 test 集合
- **THEN** 系统 SHALL 报错并停止数据集生成
- **AND** 错误信息 SHALL 说明无法满足 `start < a < b < c`

### Requirement: 商品 FineFT 数据集 manifest
系统 SHALL 为商品 FineFT 多合约数据集写出 `dataset_manifest.json`，描述边界、集合归属、输入路径、输出路径、输出行数和切片计划。

#### Scenario: 写出 manifest 边界和集合信息
- **WHEN** 商品多合约数据集工具完成边界计算
- **THEN** 系统 SHALL 写出 `dataset/{target_freq}/{symbol}/dataset_manifest.json`
- **AND** manifest SHALL 包含 `symbol`、`target_freq`、`split_ratio`、`boundaries`、`sets` 和 `state_features_path`
- **AND** `split_ratio` SHALL 记录 `{"train": 5, "valid": 3, "test": 2}`
- **AND** `boundaries` SHALL 记录 `start`、`a`、`b`、`c`

#### Scenario: manifest 记录合约级输入输出
- **WHEN** 合约 `fu2601` 在 train 集合命中至少一个交易日
- **THEN** manifest SHALL 在 `sets.train.contracts` 中记录 `contract=fu2601`
- **AND** 该记录 SHALL 包含命中的 `trading_days`
- **AND** 该记录 SHALL 包含输入 `SCALE_SAVE/{symbol}/{contract}/{target_freq}/{start_date}-{end_date}/df.feather` 路径
- **AND** 该记录 SHALL 包含阶段输出 `dataset/{target_freq}/{symbol}/train/df_<contract>.feather` 路径
- **AND** train 集合记录 SHALL 包含该合约贡献的 `train/slice/df_*.feather` 编号计划

#### Scenario: manifest 记录阶段输出行数
- **WHEN** 商品多合约数据集工具写出 `train/df_fu2601.feather`、`valid/df_fu2601.feather` 和 `test/df_fu2601.feather`
- **THEN** 每个包含 `output_path` 的合约记录 SHALL 包含 `output_row_count`
- **AND** `output_row_count` SHALL 等于该 `output_path` feather 文件的实际行数
- **AND** 每个集合 SHALL 包含 `contracts_total_count`
- **AND** `contracts_total_count` SHALL 等于该集合内所有合约 `output_row_count` 之和
- **AND** 系统 SHALL 在 `dataset_manifest.json` 中写出这些行数，使调用方无需读取 feather 文件即可知道单文件和集合总行数

#### Scenario: manifest 记录空命中或跳过原因
- **WHEN** 某合约在 valid 集合没有命中任何交易日
- **THEN** 系统 SHALL NOT 写出空的 `valid/df_<contract>.feather`
- **AND** manifest SHALL 记录该合约在 valid 集合为空命中或被跳过的原因

### Requirement: 商品 FineFT 阶段数据集生成
系统 SHALL 从合约级 `SCALE_SAVE` 输出生成 FineFT 商品阶段数据集，并停止生成旧的品种级 `train.feather`、`valid.feather` 和 `test.feather`。

#### Scenario: 生成合约级阶段数据文件
- **WHEN** manifest 中合约 `fu2601` 在 train、valid、test 集合均命中交易日
- **THEN** 系统 SHALL 读取 `SCALE_SAVE/fu/fu2601/5min/{start_date}-{end_date}/df.feather`
- **AND** 系统 SHALL 按 manifest 中列出的交易日过滤并按时间升序输出 `dataset/5min/fu/train/df_fu2601.feather`
- **AND** 系统 SHALL 输出 `dataset/5min/fu/valid/df_fu2601.feather`
- **AND** 系统 SHALL 输出 `dataset/5min/fu/test/df_fu2601.feather`
- **AND** 输出前 SHALL 重置 DataFrame index

#### Scenario: 不再生成旧集合大文件
- **WHEN** 商品多合约数据集工具生成阶段数据集
- **THEN** 系统 SHALL NOT 生成 `dataset/{target_freq}/{symbol}/train.feather`
- **AND** 系统 SHALL NOT 生成 `dataset/{target_freq}/{symbol}/valid.feather`
- **AND** 系统 SHALL NOT 生成 `dataset/{target_freq}/{symbol}/test.feather`

#### Scenario: 复制品种级 state features
- **WHEN** `FEATURE_UNION/{symbol}/{target_freq}/{start_date}-{end_date}/state_features.npy` 存在
- **THEN** 系统 SHALL 将该文件复制或等价写出到 `dataset/{target_freq}/{symbol}/state_features.npy`
- **AND** 商品训练 SHALL 使用该品种级 feature union 作为统一 state feature 列表

#### Scenario: 缺少必要输入 fail-fast
- **WHEN** manifest 中某个非空集合需要合约 `fu2601` 的 `df.feather`，但输入文件不存在
- **THEN** 系统 SHALL 报错并停止
- **AND** 错误信息 SHALL 包含缺失合约和缺失路径

### Requirement: 商品 FineFT 训练切片生成
系统 SHALL 从商品 train 阶段数据生成真正用于低层训练的 `train/slice/df_*.feather` 文件，切片连续编号且不跨合约、不跨 train 日期边界。

#### Scenario: train slice 连续编号
- **WHEN** `train/df_fu2601.feather` 和 `train/df_fu2605.feather` 均可切出训练片段
- **THEN** 系统 SHALL 在 `dataset/{target_freq}/{symbol}/train/slice/` 下写出 `df_0.feather`、`df_1.feather`、`df_2.feather` 等连续编号文件
- **AND** 编号 SHALL 从 0 开始且不跳号
- **AND** manifest SHALL 记录每个 slice 编号对应的 contract、源阶段文件和行范围
- **AND** manifest SHALL 记录每个 slice 输出文件的 `output_row_count`

#### Scenario: train short slice 不丢弃
- **WHEN** 合约 `fu2305` 的 train 阶段数据行数少于 `chunk_length`
- **THEN** 系统 SHALL 仍然写出一个 `train/slice/df_*.feather`
- **AND** 该 slice SHALL 只包含 `fu2305` 的 train 阶段数据
- **AND** manifest SHALL 记录该 slice 的 `output_row_count`
- **WHEN** 合约 train 阶段数据在完整 `chunk_length` 切片后仍有不足 `chunk_length` 的尾部行
- **THEN** 系统 SHALL 将该尾部行写出为短 slice
- **AND** 系统 SHALL NOT 为补齐短 slice 从其他合约、valid 或 test 阶段追加数据

#### Scenario: train slice 不跨合约
- **WHEN** 一个训练 slice 从 `train/df_fu2601.feather` 生成
- **THEN** 该 slice SHALL 只包含 `fu2601` 的行
- **AND** 该 slice SHALL NOT 包含任何其他合约的行

#### Scenario: early_stop 不跨 train 边界
- **WHEN** `chunk_length` 后追加 `early_stop` 行会越过同一合约的 train 阶段数据末尾
- **THEN** 系统 SHALL 将 slice 截断在同一合约 train 阶段数据内
- **AND** 系统 SHALL NOT 从 valid 或 test 阶段追加任何行
- **AND** 如果截断后 slice 不满足最小可用长度，系统 SHALL 跳过该 slice 并在 manifest 中记录原因

### Requirement: 商品 FineFT valid 动态切片生成
系统 SHALL 通过商品 data handler shell 对商品 valid 阶段数据逐合约执行市场动态切片，输出 `valid/<contract>/label_*/df_*.feather`，并保证动态片段不跨合约。

#### Scenario: 数据集工具不调用 slice model
- **WHEN** 商品多合约数据集工具生成 `dataset_manifest.json`、阶段数据和 train slice
- **THEN** `commodity_contract_dataset.py` SHALL NOT import or call `slice_model.py`
- **AND** `commodity_contract_dataset.py` SHALL NOT write valid dynamic slice files
- **AND** valid 动态切片 SHALL 留给商品 data handler shell 的后续独立阶段执行

#### Scenario: shell 逐合约调度 valid 动态切片
- **WHEN** `valid/df_fu2601.feather` 和 `valid/df_fu2605.feather` 均存在
- **THEN** 商品 data handler shell SHALL 分别对两个合约文件调用 `FineFT/datahandler/slice_model.py`
- **AND** 每次调用的 `--data_path` SHALL 指向 `dataset/{target_freq}/{symbol}/valid/df_<contract>.feather`
- **AND** 系统 SHALL NOT 在切片前把两个合约拼接成一个连续 valid DataFrame
- **AND** 输出的每个 `valid/<contract>/label_*/df_*.feather` SHALL 只包含单一合约的数据

#### Scenario: valid 动态切片保持 label 目录格式
- **WHEN** 动态标签数量为 5
- **THEN** 系统 SHALL 在 `dataset/{target_freq}/{symbol}/valid/<contract>/label_0` 到 `label_4` 下写出动态片段文件
- **AND** 文件编号 SHALL NOT 覆盖其他合约产生的片段
- **AND** manifest SHALL 记录每个 valid 动态片段对应的 contract、label 和输出路径

#### Scenario: valid processed 文件按合约隔离
- **WHEN** 商品 data handler shell 分别处理 `valid/df_fu2501.feather` 和 `valid/df_fu2505.feather`
- **THEN** `slice_model.py` SHALL 写出 `valid/processed/valid_processed_fu2501.feather`
- **AND** `slice_model.py` SHALL 写出 `valid/processed/valid_processed_fu2505.feather`
- **AND** 系统 SHALL NOT 对多个合约共用 `valid/valid_processed.feather`
- **AND** 系统 SHALL NOT 将不同合约的 label 片段写到同一个 `valid/label_*` 目录

#### Scenario: valid 数据不足时不跨合约补齐
- **WHEN** 某合约 valid 数据不足以执行动态切片
- **THEN** 系统 SHALL 跳过该合约的 valid 动态切片并在 manifest 中记录原因
- **AND** 系统 SHALL NOT 将其他合约数据拼接进该合约 valid 数据以满足最小长度

#### Scenario: valid slope 标签支持少量 segment
- **WHEN** 某合约 valid 动态切片在合并后只剩少量 segment
- **THEN** slope 标签阈值计算 SHALL NOT 因 segment 数量少而抛出 `IndexError`
- **AND** 系统 SHALL 使用该合约已有 segment 生成动态标签
- **AND** 系统 SHALL NOT 拼接其他合约数据来补齐 segment 数量

#### Scenario: valid 动态切片 manifest 记录合约和 label 行数
- **WHEN** `slice_model.py` 为合约 `fu2505` 写出 `valid/fu2505/label_0/df_0.feather`
- **THEN** 系统 SHALL 更新 `valid/slice_manifest.json`
- **AND** manifest SHALL 在合约视角记录 `fu2505` 每个非空 label 的文件路径、文件行数、文件数和总行数
- **AND** manifest SHALL 在 label 视角记录每个非空 label 跨合约的文件路径、合约、文件行数、文件数和总行数
- **AND** manifest SHALL NOT 记录没有生成文件的空 label
- **AND** 多个合约顺序调用 `slice_model.py` SHALL 累积更新 manifest，且同一合约重跑 SHALL 替换该合约旧记录

### Requirement: 商品 FineFT data handler 脚本入口
系统 SHALL 直接升级现有商品 data handler 脚本，让燃料油和铝的 FineFT 商品数据准备使用多合约 manifest 流程。

#### Scenario: 燃料油 data handler 调用新工具
- **WHEN** 用户运行 `FineFT/script/data/commodity_data_handler_fu.sh`
- **THEN** 脚本 SHALL 激活 `finetf` conda 环境
- **AND** 脚本 SHALL 调用商品多合约数据集工具
- **AND** 脚本 SHALL 传递 `--symbol fu`、summary 路径、`SCALE_SAVE` 根目录、`FEATURE_UNION/state_features.npy` 路径、输出根目录、`target_freq`、日期范围、`chunk_length` 和 `early_stop`
- **AND** 脚本 SHALL 在数据集工具完成后逐合约调用 `FineFT/datahandler/slice_model.py --data_path dataset/{target_freq}/fu/valid/df_<contract>.feather --timestamp timestamp`
- **AND** 脚本 SHALL NOT 调用旧的 `FineFT/datahandler/preprocess_data.py --trading_pair fu`
- **AND** 脚本 SHALL NOT 调用旧的 `FineFT/datahandler/slice_model.py --data_path dataset/fu/valid.feather`

#### Scenario: 铝 data handler 调用新工具
- **WHEN** 用户运行 `FineFT/script/data/commodity_data_handler_al.sh`
- **THEN** 脚本 SHALL 激活 `finetf` conda 环境
- **AND** 脚本 SHALL 调用商品多合约数据集工具
- **AND** 脚本 SHALL 传递 `--symbol al`、summary 路径、`SCALE_SAVE` 根目录、`FEATURE_UNION/state_features.npy` 路径、输出根目录、`target_freq`、日期范围、`chunk_length` 和 `early_stop`
- **AND** 脚本 SHALL 在数据集工具完成后逐合约调用 `FineFT/datahandler/slice_model.py --data_path dataset/{target_freq}/al/valid/df_<contract>.feather --timestamp timestamp`
- **AND** 脚本 SHALL NOT 调用旧的 `FineFT/datahandler/preprocess_data.py --trading_pair al`
- **AND** 脚本 SHALL NOT 调用旧的 `FineFT/datahandler/slice_model.py --data_path dataset/al/valid.feather`

#### Scenario: VAE 数据生成读取新 valid/test 结构
- **WHEN** 商品 data handler 完成多合约阶段数据和 valid 动态切片
- **THEN** 后续 VAE 数据生成 SHALL 从 `valid/<contract>/label_*/df_*.feather` 读取训练用动态片段
- **AND** 后续 VAE 数据生成 SHALL 写出 `VAE_data/<contract>/label_*.npy`
- **AND** 后续 VAE 数据生成 SHALL NOT 将不同合约的同一 label 聚合为单个 `VAE_data/label_*.npy`
- **AND** 后续 VAE 数据生成 SHALL 从 `test/df_<contract>.feather` 读取测试特征数组
- **AND** 后续 VAE 数据生成 SHALL 写出 `VAE_data/test/test_<contract>.npy`
- **AND** 后续 VAE 数据生成 SHALL NOT 将多合约 test 数据合并为单个 `VAE_data/test.npy`
- **AND** 后续 VAE 数据生成 SHALL NOT 要求 `test.feather` 存在

