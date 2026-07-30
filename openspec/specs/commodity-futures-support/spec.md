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
- **AND** summary SHALL 记录每个交易日或排名周期下各合约的主力身份：`main`、`sub` 或 `other`
- **AND** 系统 MUST NOT 写出 `PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/fu/{YYYY-MM-DD}.csv` 连续主力日文件
- **AND** 系统 MUST NOT 生成 `fu_2026-01-01_2026-04-01.csv` 或其他日期范围大 CSV

#### Scenario: summary contract 字段
- **WHEN** summary 中包含合约 `fu2601`
- **THEN** 该合约对象 SHALL 包含 `contract`、`start_trading_day`、`end_trading_day`、`trading_day_count`、`selected_months` 和 `trading_days`
- **AND** `start_trading_day` SHALL 等于该合约裁剪后 `trading_days` 中最小 `TradingDay`
- **AND** `end_trading_day` SHALL 等于该合约裁剪后 `trading_days` 中最大 `TradingDay`
- **AND** `trading_day_count` SHALL 等于 `len(trading_days)`
- **AND** 每个 `trading_days` 条目 SHALL 包含 `trading_day`、ISO `date`、`source_file` 和 `daily_volume`
- **AND** 每个 `trading_days` 条目 SHALL 包含该合约在当前交易日或排名周期的 `main_sub_role`
- **AND** `main_sub_role` SHALL 只能是 `main`、`sub` 或 `other`
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

#### Scenario: full process 在 dataset split 后执行特征选择和 scale save
- **WHEN** `main_contract_summary.json` 中包含合约 `fu2601` 和 `fu2605`
- **THEN** `fu_full_process.sh` SHALL 从 summary 读取合约列表
- **AND** `fu_full_process.sh` SHALL 分别为 `fu2601` 和 `fu2605` 调用 `cross_section`、`merge`、`concat`、`time_feature` 和 `merge_clean`
- **AND** 每次合约级调用 SHALL 传递 `--symbols fu --contract <contract>`
- **AND** 所有合约 `merge_clean` 完成后，`fu_full_process.sh` SHALL 只调用一次 `dataset_split`
- **AND** `dataset_split` 完成后，`fu_full_process.sh` SHALL 调用 `feature_selection_train`
- **AND** `feature_selection_train` 完成后，`fu_full_process.sh` SHALL 调用 `feature_selection_valid`
- **AND** `feature_selection_valid` 完成后，`fu_full_process.sh` SHALL 调用一次 `muti_contract_scale_save.py`
- **AND** `muti_contract_scale_save.py` SHALL 使用 `FEATURE_SELECTION/{target_freq}/{symbol}/train/state_features.npy`
- **AND** `muti_contract_scale_save.py` SHALL 扫描并读取 `SPLIT-TRAIN-VALID-TEST/{target_freq}/{symbol}/{train|valid|test}/*.feather` 中存在的 split 阶段合约文件
- **AND** `maintenance_margin_dict` SHALL 在 `muti_contract_scale_save.py` 完成后执行
- **AND** `fu_full_process.sh` SHALL NOT 在合约循环内的 `merge_clean` 后立即调用 `scale_save`
- **AND** `fu_full_process.sh` SHALL NOT 使用旧 `IC_RESULT` 作为本次商品特征评估输入源

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
- **WHEN** `cross_section/create_feature.py`、`scale_describe_save/scale_save.py`、`scale_describe_save/muti_contract_scale_save.py` 或 split 后 multi-contract feature selection 以 `market_type=commodity_futures` 运行
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
- **THEN** 系统 SHALL 为 `stitch_main_contract`、`downscale_continuous_by_trading_day`、`cross_section`、`merge`、`concat`、`time_feature`、`merge_clean`、`dataset_split`、`feature_selection_train`、`feature_selection_valid`、`scale_save` 和 `maintenance_margin_dict` 生成独立日志文件
- **AND** 每个步骤日志文件名 SHALL 包含 symbol、target_freq、start_date、end_date 和步骤名
- **AND** 每个步骤日志 SHALL 捕获该步骤的 stdout 和 stderr

#### Scenario: 总日志记录阶段状态
- **WHEN** 商品 preprocess 主流程执行任一主要步骤
- **THEN** 总日志 SHALL 记录该步骤的开始信息和步骤日志路径
- **AND** 当步骤成功完成时，总日志 SHALL 记录该步骤成功完成
- **AND** 当步骤失败时，总日志 SHALL 记录该步骤失败和对应日志路径

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

#### Scenario: feature selection 和 multi-contract scale save 按 stage/contract 生成文件
- **WHEN** 商品 full process 完成 split 后 feature selection
- **THEN** feature selection SHALL 读取 `SPLIT-TRAIN-VALID-TEST/{target_freq}/{symbol}/{train|valid}/{contract}.feather`
- **AND** feature selection SHALL 写出 `FEATURE_SELECTION/{target_freq}/{symbol}/{train|valid}/`
- **AND** train feature selection SHALL 写出 `FEATURE_SELECTION/{target_freq}/{symbol}/train/state_features.npy`
- **AND** valid feature selection SHALL 只写评估明细、汇总统计和 manifest/report
- **AND** `muti_contract_scale_save.py` SHALL 读取 `SPLIT-TRAIN-VALID-TEST/{target_freq}/{symbol}/{train|valid|test}/{contract}.feather`
- **AND** `muti_contract_scale_save.py` SHALL 使用 `FEATURE_SELECTION/{target_freq}/{symbol}/train/state_features.npy`
- **AND** `muti_contract_scale_save.py` SHALL 写出 `SCALE_SAVE/{symbol}/{target_freq}/{stage}/{contract}.feather`
- **AND** `muti_contract_scale_save.py` SHALL 同步写出 `SCALE_SAVE/{symbol}/{target_freq}/{stage}/{contract}.csv`

#### Scenario: 未传 contract 时保留旧路径
- **WHEN** 共享 operator-futures 脚本未传入 `--contract`
- **THEN** 系统 SHALL 继续读写现有 `{symbol}/{target_freq}` 路径
- **AND** 系统 SHALL NOT 要求非商品期货或旧调用方提供 contract 参数

#### Scenario: 多合约日志和 skip 检查包含 contract
- **WHEN** 商品 full process 对多个合约运行后续阶段
- **THEN** 步骤日志文件名、skip 消息和输出存在性检查 SHALL 包含 `symbol` 和 `contract`
- **AND** 一个合约的日志或输出 SHALL NOT 覆盖另一个合约的日志或输出

### Requirement: 商品期货跨合约训练特征合集
系统 SHALL 支持在所有入选合约完成单合约特征选择和 scale save 后，生成品种级统一 state feature 合集，供需要统一 state feature 列表的独立流程读取。

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

#### Scenario: full process 不再生成 feature union
- **WHEN** `fu_full_process.sh` 已对 summary 中所有合约完成 `scale_save`
- **THEN** `fu_full_process.sh` SHALL NOT 调用品种级 feature union 生成步骤
- **AND** `fu_full_process.sh` SHALL NOT 调用 `run_commodity_ic_union_finalize`
- **AND** 后续第 9 阶段 dataset split SHALL NOT 依赖 `FEATURE_UNION/{symbol}/{target_freq}/{start_date}-{end_date}/state_features.npy`

#### Scenario: validation 检查 feature union
- **WHEN** `validate_features.sh` 验证商品期货输出
- **THEN** 脚本 SHALL 检查 `FEATURE_UNION/{symbol}/{target_freq}/{start_date}-{end_date}/state_features.npy`
- **AND** 脚本 SHALL 检查 `FEATURE_UNION/{symbol}/{target_freq}/{start_date}-{end_date}/feature_union_manifest.json`

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

### Requirement: 商品 FineFT 数据集 manifest
系统 SHALL 为商品 FineFT 多合约数据集写出 `dataset_manifest.json`，描述 dataset split manifest 来源、集合归属、输入路径、输出路径、输出行数、state feature 清单和切片计划。

#### Scenario: 写出 manifest 来源和集合信息
- **WHEN** 商品 FineFT 数据集装配工具完成阶段合约文件复制
- **THEN** 系统 SHALL 写出 `dataset/{target_freq}/{symbol}/dataset_manifest.json`
- **AND** manifest SHALL 包含 `symbol`、`target_freq`、`dataset_split_manifest_path`、`sets` 和 `state_features_path`
- **AND** manifest SHALL NOT 要求 `split_ratio` 或 `boundaries` 字段驱动 FineFT 数据集生成
- **AND** manifest MAY 复制 dataset split manifest 中的 `range`、`trading_days`、`skipped_contracts`、`contracts_total_count` 或其他审计信息

#### Scenario: manifest 记录合约级输入输出
- **WHEN** dataset split manifest 在 `sets.train.contracts` 中声明合约 `fu2508`
- **THEN** FineFT manifest SHALL 在 `sets.train.contracts` 中记录 `contract=fu2508`
- **AND** 该记录 SHALL 包含输入 `SCALE_SAVE/{symbol}/{target_freq}/train/fu2508.feather` 路径
- **AND** 该记录 SHALL 包含阶段输出 `dataset/{target_freq}/{symbol}/train/fu2508.feather` 路径
- **AND** train 集合记录 SHALL 包含该合约贡献的 `train/slice/df_*.feather` 编号计划
- **AND** 如 dataset split manifest 提供 `trading_days` 或 `range`，FineFT manifest SHALL 可保留这些字段用于审计，但 SHALL NOT 使用它们过滤数据

#### Scenario: manifest 记录阶段输出行数
- **WHEN** 商品 FineFT 数据集装配工具写出 `train/fu2508.feather`、`valid/fu2508.feather` 或 `test/fu2508.feather`
- **THEN** 每个包含 `output_path` 的合约记录 SHALL 包含 `output_row_count`
- **AND** `output_row_count` SHALL 等于该 `output_path` feather 文件的实际行数
- **AND** 每个集合 SHALL 包含 `contracts_total_count`
- **AND** `contracts_total_count` SHALL 等于该集合内所有合约 `output_row_count` 之和
- **AND** 系统 SHALL 在 `dataset_manifest.json` 中写出这些行数，使调用方无需读取 feather 文件即可知道单文件和集合总行数

#### Scenario: manifest 记录空命中或跳过原因
- **WHEN** dataset split manifest 中某合约不属于 valid 集合或在 `sets.valid.skipped_contracts` 中记录跳过原因
- **THEN** 系统 SHALL NOT 要求该合约存在 `SCALE_SAVE/{symbol}/{target_freq}/valid/{contract}.feather`
- **AND** FineFT manifest SHALL 保留该合约在 valid 集合为空命中或被跳过的原因

### Requirement: 商品 FineFT 阶段数据集生成
系统 SHALL 从阶段化 `SCALE_SAVE` 输出装配商品 FineFT 阶段数据集，保留合约级 train、valid 和 test 文件。

#### Scenario: 复制合约级阶段数据文件
- **WHEN** dataset split manifest 声明合约 `fu2508` 在 train、valid 和 test 集合均存在
- **THEN** 系统 SHALL 读取 `SCALE_SAVE/fu/10min/train/fu2508.feather`
- **AND** 系统 SHALL 复制并写出 `dataset/10min/fu/train/fu2508.feather`
- **AND** 系统 SHALL 复制并写出 `dataset/10min/fu/valid/fu2508.feather`
- **AND** 系统 SHALL 复制并写出 `dataset/10min/fu/test/fu2508.feather`
- **AND** 输出 SHALL 保留输入 feather 的所有列和行
- **AND** 系统 SHALL NOT 额外写出 `dataset/{target_freq}/{symbol}/train.feather`、`valid.feather` 或 `test.feather`

#### Scenario: 复制训练阶段 state features
- **WHEN** `FEATURE_SELECTION/10min/fu/train/state_features.npy` 存在且非空
- **THEN** 系统 SHALL 将其复制为 `dataset/10min/fu/state_features.npy`
- **AND** 系统 SHALL NOT 要求旧 `FEATURE_UNION/{symbol}/{target_freq}/{start_date}-{end_date}/state_features.npy` 存在
- **AND** 系统 SHALL NOT 读取 `state_features.npy` 来筛选阶段 feather 输出列

#### Scenario: 缺少必要 SCALE_SAVE 输入 fail-fast
- **WHEN** dataset split manifest 声明 `train` 阶段需要合约 `fu2508`，但 `SCALE_SAVE/{symbol}/{target_freq}/train/fu2508.feather` 不存在
- **THEN** 系统 SHALL 报错并停止
- **AND** 错误信息 SHALL 包含缺失合约、stage 和缺失路径

#### Scenario: 复制后空数据 fail-fast
- **WHEN** 系统复制某个阶段合约 feather 后读取行数为 0
- **THEN** 系统 SHALL 报错并停止
- **AND** 错误信息 SHALL 包含合约、stage 和输出路径

### Requirement: 商品 FineFT 训练切片生成
系统 SHALL 从商品 train 阶段合约数据生成真正用于低层训练的 `train/slice/df_*.feather` 文件，切片连续编号且不跨合约、不跨 train 阶段文件。

#### Scenario: train slice 连续编号
- **WHEN** `train/fu2508.feather` 和 `train/fu2509.feather` 均可切出训练片段
- **THEN** 系统 SHALL 在 `dataset/{target_freq}/{symbol}/train/slice/` 下写出 `df_0.feather`、`df_1.feather`、`df_2.feather` 等连续编号文件
- **AND** 编号 SHALL 从 0 开始且不跳号
- **AND** manifest SHALL 记录每个 slice 编号对应的 contract、源阶段文件和行范围
- **AND** manifest SHALL 记录每个 slice 输出文件的 `output_row_count`

#### Scenario: train short slice 不丢弃
- **WHEN** 合约 `fu2508` 的 train 阶段数据行数少于 `chunk_length`
- **THEN** 系统 SHALL 仍然写出一个 `train/slice/df_*.feather`
- **AND** 该 slice SHALL 只包含 `fu2508` 的 train 阶段数据
- **AND** manifest SHALL 记录该 slice 的 `output_row_count`
- **WHEN** 合约 train 阶段数据在完整 `chunk_length` 切片后仍有不足 `chunk_length` 的尾部行
- **THEN** 系统 SHALL 将该尾部行写出为短 slice
- **AND** 系统 SHALL NOT 为补齐短 slice 从其他合约、valid 或 test 阶段追加数据

#### Scenario: train slice 不跨合约
- **WHEN** 一个训练 slice 从 `train/fu2508.feather` 生成
- **THEN** 该 slice SHALL 只包含 `fu2508` 的行
- **AND** 该 slice SHALL NOT 包含任何其他合约的行

#### Scenario: early_stop 不跨 train 阶段文件
- **WHEN** `chunk_length` 后追加 `early_stop` 行会越过同一合约的 train 阶段数据末尾
- **THEN** 系统 SHALL 将 slice 截断在同一合约 train 阶段数据内
- **AND** 系统 SHALL NOT 从 valid 或 test 阶段追加任何行
- **AND** 如果截断后 slice 为空，系统 SHALL 跳过该 slice

### Requirement: 商品 FineFT valid 动态切片生成
系统 SHALL 通过商品 data handler shell 对商品 valid 阶段数据逐合约执行市场动态切片，输出 `valid/<contract>/label_*/df_*.feather`，并保证动态片段不跨合约。

#### Scenario: 数据集工具不调用 slice model
- **WHEN** 商品 FineFT 数据集装配工具生成 `dataset_manifest.json`、阶段数据和 train slice
- **THEN** `commodity_contract_dataset.py` SHALL NOT import or call `slice_model.py`
- **AND** `commodity_contract_dataset.py` SHALL NOT write valid dynamic slice files
- **AND** valid 动态切片 SHALL 留给商品 data handler shell 的后续独立阶段执行

#### Scenario: shell 逐合约调度 valid 动态切片
- **WHEN** `valid/fu2508.feather` 和 `valid/fu2509.feather` 均存在
- **THEN** 商品 data handler shell SHALL 分别对两个合约文件调用 `FineFT/datahandler/slice_model.py`
- **AND** 每次调用的 `--data_path` SHALL 指向 `dataset/{target_freq}/{symbol}/valid/{contract}.feather`
- **AND** 系统 SHALL NOT 在切片前把两个合约拼接成一个连续 valid DataFrame
- **AND** 输出的每个 `valid/<contract>/label_*/df_*.feather` SHALL 只包含单一合约的数据

#### Scenario: valid 动态切片保持 label 目录格式
- **WHEN** 动态标签数量为 5
- **THEN** 系统 SHALL 在 `dataset/{target_freq}/{symbol}/valid/<contract>/label_0` 到 `label_4` 下写出动态片段文件
- **AND** 文件编号 SHALL NOT 覆盖其他合约产生的片段
- **AND** manifest SHALL 记录每个 valid 动态片段对应的 contract、label 和输出路径

#### Scenario: valid processed 文件按合约隔离
- **WHEN** 商品 data handler shell 分别处理 `valid/fu2508.feather` 和 `valid/fu2509.feather`
- **THEN** `slice_model.py` SHALL 写出 `valid/processed/valid_processed_fu2508.feather`
- **AND** `slice_model.py` SHALL 写出 `valid/processed/valid_processed_fu2509.feather`
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
- **WHEN** `slice_model.py` 为合约 `fu2508` 写出 `valid/fu2508/label_0/df_0.feather`
- **THEN** 系统 SHALL 更新 `valid/slice_manifest.json`
- **AND** manifest SHALL 在合约视角记录 `fu2508` 每个非空 label 的文件路径、文件行数、文件数和总行数
- **AND** manifest SHALL 在 label 视角记录每个非空 label 跨合约的文件路径、合约、文件行数、文件数和总行数
- **AND** manifest SHALL NOT 记录没有生成文件的空 label
- **AND** 多个合约顺序调用 `slice_model.py` SHALL 累积更新 manifest，且同一合约重跑 SHALL 替换该合约旧记录

### Requirement: 商品 FineFT data handler 脚本入口
系统 SHALL 直接升级现有商品 data handler 脚本，让燃料油和铝的 FineFT 商品数据准备使用 dataset split manifest 和阶段化 SCALE_SAVE 的多合约装配流程。

#### Scenario: 燃料油 data handler 调用新工具
- **WHEN** 用户运行 `FineFT/script/data/commodity_data_handler_fu.sh`
- **THEN** 脚本 SHALL 激活 `finetf` conda 环境
- **AND** 脚本 SHALL 调用商品多合约数据集工具
- **AND** 脚本 SHALL 传递 `--symbol fu`、`--dataset_split_manifest_path PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/{target_freq}/fu/dataset_split_manifest.json`、`--input_root PREPROCESS_DATASET/commodity-futures/SCALE_SAVE`、`--state_features_path PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/{target_freq}/fu/train/state_features.npy`、输出根目录、`target_freq`、`chunk_length` 和 `early_stop`
- **AND** 脚本 SHALL 在数据集工具完成后逐合约调用 `FineFT/datahandler/slice_model.py --data_path dataset/{target_freq}/fu/valid/{contract}.feather --timestamp timestamp`
- **AND** 脚本 SHALL NOT 调用旧的 `FineFT/datahandler/preprocess_data.py --trading_pair fu`
- **AND** 脚本 SHALL NOT 调用旧的 `FineFT/datahandler/slice_model.py --data_path dataset/fu/valid.feather`

#### Scenario: 铝 data handler 调用新工具
- **WHEN** 用户运行 `FineFT/script/data/commodity_data_handler_al.sh`
- **THEN** 脚本 SHALL 激活 `finetf` conda 环境
- **AND** 脚本 SHALL 调用商品多合约数据集工具
- **AND** 脚本 SHALL 传递 `--symbol al`、`--dataset_split_manifest_path PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/{target_freq}/al/dataset_split_manifest.json`、`--input_root PREPROCESS_DATASET/commodity-futures/SCALE_SAVE`、`--state_features_path PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/{target_freq}/al/train/state_features.npy`、输出根目录、`target_freq`、`chunk_length` 和 `early_stop`
- **AND** 脚本 SHALL 在数据集工具完成后逐合约调用 `FineFT/datahandler/slice_model.py --data_path dataset/{target_freq}/al/valid/{contract}.feather --timestamp timestamp`
- **AND** 脚本 SHALL NOT 调用旧的 `FineFT/datahandler/preprocess_data.py --trading_pair al`
- **AND** 脚本 SHALL NOT 调用旧的 `FineFT/datahandler/slice_model.py --data_path dataset/al/valid.feather`

#### Scenario: VAE 数据生成读取新 valid/test 结构
- **WHEN** 商品 data handler 完成多合约阶段数据和 valid 动态切片
- **THEN** 后续 VAE 数据生成 SHALL 从 `valid/<contract>/label_*/df_*.feather` 读取训练用动态片段
- **AND** 后续 VAE 数据生成 SHALL 写出 `VAE_data/<contract>/label_*.npy`
- **AND** 后续 VAE 数据生成 SHALL NOT 将不同合约的同一 label 聚合为单个 `VAE_data/label_*.npy`
- **AND** 后续 VAE 数据生成 SHALL 从 `test/{contract}.feather` 读取测试特征数组
- **AND** 后续 VAE 数据生成 SHALL 写出 `VAE_data/test/test_<contract>.npy`

### Requirement: 商品期货第 9 阶段 dataset split 入口
系统 SHALL 提供 `future_upgraded/9_dataset_split` 阶段入口，并在商品 full process 中于所有合约 `merge_clean` 完成后运行该阶段。

#### Scenario: shell stage 激活 finetf 环境
- **WHEN** `data_preprocess/script_preprocess/future_upgraded/9_dataset_split/dataset_split.sh` 运行
- **THEN** 脚本 SHALL 激活 `finetf` conda 环境
- **AND** 脚本 SHALL 调用 `operator_futures.dataset_split.dataset_split`
- **AND** 脚本 SHALL 传递 summary 路径、`ALL_FEATURE` 根目录、输出根目录、`symbol`、`target_freq`、`start_date`、`end_date` 和 split ratio 参数

#### Scenario: full process 只运行一次 dataset split
- **WHEN** `main_contract_summary.json` 包含多个合约
- **AND** `fu_full_process.sh` 已为每个合约完成 `merge_clean`
- **THEN** `fu_full_process.sh` SHALL 调用一次 `dataset_split`
- **AND** 该调用 SHALL 使用同一次运行的 `summary_path`、`symbol`、`target_freq`、`start_date` 和 `end_date`
- **AND** 该调用 SHALL NOT 绑定单个 `contract`
- **AND** 该调用 SHALL NOT 读取 `SCALE_SAVE` 作为输入根目录

### Requirement: 商品期货 dataset split manifest
系统 SHALL 为第 9 阶段商品 dataset split 写出 `dataset_split_manifest.json`，描述 split 边界、合约集合归属、输入输出路径、输出行数和跳过原因。

#### Scenario: 写出 split manifest 边界和集合信息
- **WHEN** `operator_futures.dataset_split.dataset_split` 完成边界计算
- **THEN** 系统 SHALL 写出 `dataset/{target_freq}/{symbol}/dataset_split_manifest.json`
- **AND** manifest SHALL 包含 `symbol`、`target_freq`、`split_ratio`、`boundaries` 和 `sets`
- **AND** `split_ratio` SHALL 记录 `{"train": 5, "valid": 3, "test": 2}`
- **AND** `boundaries` SHALL 记录 `start`、`a`、`b`、`c`

#### Scenario: split manifest 记录合约级输入输出
- **WHEN** 合约 `fu2601` 在 train 集合命中至少一个交易日
- **THEN** manifest SHALL 在 `sets.train.contracts` 中记录 `contract=fu2601`
- **AND** 该记录 SHALL 包含命中的 `trading_days`
- **AND** 该记录 SHALL 包含输入 `SCALE_SAVE/{symbol}/{contract}/{target_freq}/{start_date}-{end_date}/df.feather` 路径
- **AND** 该记录 SHALL 包含阶段输出 `dataset/{target_freq}/{symbol}/train/{contract}.feather` 路径
- **AND** 该记录 SHALL 包含 `output_row_count`

#### Scenario: split manifest 记录集合合并输出
- **WHEN** dataset split 写出 `train.feather`、`valid.feather` 和 `test.feather`
- **THEN** manifest SHALL 为每个集合记录顶层 merged output 路径
- **AND** manifest SHALL 为每个集合记录 `contracts_total_count`
- **AND** `contracts_total_count` SHALL 等于该集合内所有合约 `output_row_count` 之和

#### Scenario: split manifest 记录空命中或跳过原因
- **WHEN** 某合约在 valid 集合没有命中任何交易日
- **THEN** 系统 SHALL NOT 写出空的 `valid/{contract}.feather`
- **AND** manifest SHALL 记录该合约在 valid 集合为空命中或被跳过的原因

### Requirement: 商品期货涨跌停价 reward/execution 列
系统 SHALL 在商品期货当前行情 reward/execution 列中包含涨跌停价，并保持这些列不被 future shift。

#### Scenario: orderbook 下采样保留涨跌停价
- **WHEN** 商品期货 orderbook 下采样读取包含 `LowerLimitPrice` 和 `UpperLimitPrice` 的秒级快照
- **THEN** 下采样输出 SHALL 包含 `LowerLimitPrice` 和 `UpperLimitPrice`
- **AND** 两列 SHALL 使用与 orderbook 深度列相同的目标频率窗口取值语义
- **AND** 两列 SHALL 随 snapshot 输入进入 `CONCURRENT_FEATURE`

#### Scenario: reward manifest 包含涨跌停价
- **WHEN** 商品期货使用 `get_reward_execution_columns(depth=5)` 获取 reward/execution 列
- **THEN** 返回列 SHALL 包含 `LowerLimitPrice` 和 `UpperLimitPrice`
- **AND** 两列 SHALL 位于 depth-aware orderbook columns 之后、derivative reference columns 之前
- **AND** depth=5 商品 reward/execution 列总数 SHALL 为 29

#### Scenario: 涨跌停价不进入 state candidate
- **WHEN** 商品期货 feature selection 或 scale save 以 `market_type=commodity_futures` 和 `orderbook_depth=5` 运行
- **THEN** `LowerLimitPrice` 和 `UpperLimitPrice` SHALL 被识别为 reward/execution 列
- **AND** 两列 SHALL NOT 被作为 state candidate 特征参与选择或缩放

### Requirement: 商品期货单边盘口 snapshot 特征增强
系统 SHALL 为合法单边盘口生成有限且语义明确的 snapshot 截面特征。

#### Scenario: ask 侧空盘口生成有限特征
- **WHEN** snapshot 输入中 `ask1_size` 到当前 depth 的所有 ask size 总和为 0
- **AND** bid 侧 size 总和大于 0
- **AND** `ask1_price` 存在
- **THEN** snapshot 特征 SHALL 输出 `ask_side_empty = true`
- **AND** snapshot 特征 SHALL 输出 `bid_side_empty = false`
- **AND** 所有 `ask{i}_size_n` SHALL 等于 0
- **AND** `sell_wap` SHALL 等于 `ask1_price`
- **AND** `buy_sell_wap_spread` SHALL 使用增强后的 `buy_wap - sell_wap`
- **AND** 输出 SHALL NOT 包含 NaN 或 infinite 值

#### Scenario: bid 侧空盘口生成有限特征
- **WHEN** snapshot 输入中 `bid1_size` 到当前 depth 的所有 bid size 总和为 0
- **AND** ask 侧 size 总和大于 0
- **AND** `bid1_price` 存在
- **THEN** snapshot 特征 SHALL 输出 `bid_side_empty = true`
- **AND** snapshot 特征 SHALL 输出 `ask_side_empty = false`
- **AND** 所有 `bid{i}_size_n` SHALL 等于 0
- **AND** `buy_wap` SHALL 等于 `bid1_price`
- **AND** `buy_sell_wap_spread` SHALL 使用增强后的 `buy_wap - sell_wap`
- **AND** 输出 SHALL NOT 包含 NaN 或 infinite 值

#### Scenario: 正常双边盘口保持兼容
- **WHEN** snapshot 输入中 ask 侧和 bid 侧 size 总和均大于 0
- **THEN** `sell_wap`、`buy_wap`、`ask{i}_size_n` 和 `bid{i}_size_n` SHALL 使用原有加权与归一化公式
- **AND** `ask_side_empty` SHALL 为 false
- **AND** `bid_side_empty` SHALL 为 false
- **AND** 既有 snapshot 特征列的数值 SHALL 保持兼容

#### Scenario: 双侧空盘口 fail-fast
- **WHEN** snapshot 输入中 ask 侧和 bid 侧 size 总和均为 0
- **THEN** 系统 SHALL 将该输入视为非法盘口
- **AND** 系统 SHALL fail-fast，而不是静默填充为可训练特征

#### Scenario: time feature 输入不再因合法单边盘口失败
- **WHEN** 商品期货合法单边盘口已经生成 enhanced snapshot 特征并进入 `MERGE_CONCAT/CONCAT_FEATURE`
- **THEN** `time_feature_input` 非法值校验 SHALL NOT 因该合法单边盘口产生的 `sell_wap`、`buy_wap`、`buy_sell_wap_spread`、`ask{i}_size_n` 或 `bid{i}_size_n` 失败

### Requirement: 商品期货 split 后多合约特征选择流水线
系统 SHALL 在商品期货 dataset split 之后执行 train 多合约特征评估与筛选，并执行 valid 多合约评估与报告；后续 scale-save SHALL 只使用 train 产生的最终特征清单。

#### Scenario: train 阶段从 split train 文件生成最终特征清单
- **WHEN** `dataset_split` 已写出 `PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/5min/fu/train/fu2601.feather` 和 `fu2605.feather`
- **THEN** train feature selection SHALL 读取该 split train 目录下的合约级 feather 文件
- **AND** train feature selection SHALL 对所有 state 特征计算每合约 `Permutation Importance`、`CatBoost Importance`、`IC`、`RankIC` 和 `Sharpe`
- **AND** train feature selection SHALL 默认按窗口 `[1, 6, 12]` 计算指标
- **AND** train feature selection SHALL 将每合约明细写入 `PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/train/per_contract/`
- **AND** 每合约明细 SHALL 包含 `window` 字段
- **AND** train feature selection SHALL 汇总每个指标的 `Mean`、`Std` 和 `Median`
- **AND** train feature selection SHALL 依次执行 `Hard Filter`、`Stability Filter`、`Composite Score` 和 `Correlation Filter`
- **AND** train feature selection SHALL 写出 `PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/train/state_features.npy`
- **AND** train feature selection SHALL NOT 写出 `state_features_candidate.npy` 作为下游约定文件
- **AND** train feature selection SHALL 写出 `feature_selection_manifest.json`，记录输入 split 路径、合约、指标明细路径、汇总路径、筛选阶段结果、最终特征数、`windows_list` 和 `composite_drop_ratio`

#### Scenario: valid 阶段只使用 train 特征清单做评估报告
- **WHEN** train feature selection 已写出 `FEATURE_SELECTION/5min/fu/train/state_features.npy`
- **AND** `dataset_split` 已写出 `PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/5min/fu/valid/fu2601.feather`
- **THEN** valid feature selection SHALL 读取 split valid 目录下的合约级 feather 文件
- **AND** valid feature selection SHALL 仅对 train `state_features.npy` 中的 state 特征计算 `Permutation Importance`、`CatBoost Importance`、`IC`、`RankIC` 和 `Sharpe`
- **AND** valid feature selection SHALL 默认按窗口 `[1, 6, 12]` 计算指标
- **AND** valid feature selection SHALL 汇总每个指标的 `Mean`、`Std` 和 `Median`
- **AND** valid feature selection SHALL NOT 执行 `Hard Filter`、`Stability Filter`、`Composite Score` 或 `Correlation Filter`
- **AND** valid feature selection SHALL NOT 写出下游采用的 `PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/valid/state_features.npy`
- **AND** valid feature selection SHALL 写出 `feature_selection_manifest.json`，记录 train feature list 路径、输入 split 路径、指标明细路径、汇总路径、评估特征数和 `windows_list`

#### Scenario: 指标目标和多窗口口径
- **WHEN** feature selection 计算窗口 `window` 的任一指标
- **THEN** 系统 SHALL 使用 `mark_price.shift(-window) - mark_price` 作为未来收益 target
- **AND** 系统 SHALL 裁掉最后 `window` 行，使 feature values 与 target 长度一致
- **AND** 系统 SHALL 默认对 `windows_list=[1, 6, 12]` 中每个窗口分别生成每合约指标明细
- **AND** 系统 SHALL 在每合约指标明细中记录对应 `window`
- **AND** 系统 SHALL 按 feature 汇总所有合约和窗口上的指标 `Mean`、`Std` 和 `Median`

#### Scenario: IC 计算沿用原始 ic_correlation 口径
- **WHEN** feature selection 计算 state feature 的 `IC`
- **THEN** 系统 SHALL 对 feature 和 target 执行 pairwise NaN 过滤
- **AND** 当过滤后 feature 或 target 样本数小于 2 时，`IC` SHALL 为 `np.nan`
- **AND** 当过滤后 feature 或 target 标准差为 0 时，`IC` SHALL 为 `np.nan`
- **AND** 其他情况下 `IC` SHALL 为 feature 与 target 的 Pearson correlation

#### Scenario: RankIC 计算沿用原始 rank_ic_correlation 口径
- **WHEN** feature selection 计算 state feature 的 `RankIC`
- **THEN** 系统 SHALL 先检查原始 feature 和 target
- **AND** 当原始 feature 或 target 为空时，`RankIC` SHALL 为 `0.0`
- **AND** 当原始 feature 或 target 的 `np.nanstd` 为 0 时，`RankIC` SHALL 为 `0.0`
- **AND** 其他情况下系统 SHALL 使用 `np.argsort(np.argsort(values))` 生成 ranks
- **AND** 系统 SHALL 计算 feature ranks 与 target ranks 的 Pearson correlation
- **AND** 系统 SHALL 将 NaN、正无穷和负无穷结果转换为 `0.0`

#### Scenario: CatBoost Importance 沿用原始 catbooost 口径
- **WHEN** feature selection 计算 state feature 的 `CatBoost Importance`
- **THEN** 系统 SHALL 使用 `CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, loss_function="MAE", task_type="GPU", random_seed=42)`
- **AND** 系统 SHALL 使用同一窗口下的 feature matrix 和 target 构造 `train_pool` 与 `test_pool`
- **AND** 系统 SHALL 调用 `model.fit(train_pool, eval_set=test_pool, verbose=100)`
- **AND** 系统 SHALL 从 `model.get_feature_importance(train_pool)` 读取 feature importance
- **AND** 系统 SHALL NOT 在 CatBoost 不可用时降级为 `abs(IC)` 或其他替代指标

#### Scenario: Sharpe 使用单特征伪策略收益
- **WHEN** feature selection 计算 state feature `alpha` 的 Sharpe 指标
- **THEN** 系统 SHALL 在当前输入数据内对 `alpha` 执行列内 z-score
- **AND** 系统 SHALL 将 z-score 后的 `alpha` 与未来收益相乘得到伪收益序列
- **AND** 系统 SHALL 根据该伪收益序列计算 Sharpe
- **AND** 系统 SHALL NOT 使用跨 train 和 valid 的联合统计量计算该 Sharpe

#### Scenario: Permutation Importance 使用 IC 损失口径
- **WHEN** feature selection 计算 state feature 的 `Permutation Importance`
- **THEN** 系统 SHALL 以 `abs(IC(feature, target))` 作为 baseline
- **AND** 系统 SHALL 对 feature values 执行确定性 one-step roll 得到 shuffled feature
- **AND** 系统 SHALL 以 `max(baseline - abs(IC(shuffled_feature, target)), 0.0)` 作为 `Permutation Importance`
- **AND** 当任一 IC 结果为 NaN 时，系统 SHALL 在该差值计算中按 `0.0` 处理该 IC 分数

#### Scenario: Composite Score 按优先级删除低分因子
- **WHEN** feature selection 完成 `Hard Filter` 和 `Stability Filter`
- **THEN** `Hard Filter` SHALL 保留 `abs(RankIC_Mean) >= min_abs_ic` 的 features
- **AND** `Hard Filter` SHALL NOT 使用 `abs(IC_Mean)` 作为第一步硬过滤依据
- **AND** `Stability Filter` SHALL 保留 `IC_Std <= max_metric_std` 的 features
- **AND** `Composite Score` SHALL 先按 `abs(RankIC_Mean)` 降序排序
- **AND** 当 `abs(RankIC_Mean)` 相同时，`Composite Score` SHALL 按 `abs(Sharpe_Mean) + Permutation Importance_Mean` 降序排序
- **AND** 当存在 `SHAP Importance_Mean` 时，系统 SHALL 将其加入第二优先级分数
- **AND** 当前两级分数相同时，`Composite Score` SHALL 按 `CatBoost Importance_Mean` 降序排序
- **AND** 系统 SHALL 删除排序后底部 `composite_drop_ratio` 的 features，默认比例为 `0.1`
- **AND** 系统 SHALL 至少保留 1 个 feature
- **AND** 系统 SHALL 在 `feature_selection_manifest.json` 的 `filter_results` 中记录 `Composite Score` 保留列表和 `Composite Score Dropped` 删除列表
- **AND** `Correlation Filter` SHALL 在 Composite Score 删除后执行

#### Scenario: scale-save 使用训练集特征清单处理 split 阶段文件
- **WHEN** train feature selection 已得到最终 `FEATURE_SELECTION/5min/fu/train/state_features.npy`
- **AND** `SPLIT-TRAIN-VALID-TEST/5min/fu/train/fu2601.feather` 存在
- **AND** `SPLIT-TRAIN-VALID-TEST/5min/fu/valid/fu2601.feather` 不存在
- **THEN** `muti_contract_scale_save.py` SHALL 使用 train `state_features.npy` 作为 state feature 清单
- **AND** `muti_contract_scale_save.py` SHALL 只从 train split 全量拟合 robust scaler，一次生成整套 `center`、`scale`、`fallback` 和 `clip` 参数
- **AND** `muti_contract_scale_save.py` SHALL 将同一套 scaler 参数应用到 train、valid 和 test split 文件
- **AND** `muti_contract_scale_save.py` SHALL NOT 在 valid 或 test split 上重新拟合 scaler
- **AND** `muti_contract_scale_save.py` SHALL 处理存在的 `train/fu2601.feather`
- **AND** `muti_contract_scale_save.py` SHALL NOT 要求为缺失的 `valid/fu2601.feather` 生成输出
- **AND** `muti_contract_scale_save.py` SHALL 继续处理扫描到的其他存在阶段合约文件
- **AND** `muti_contract_scale_save.py` SHALL NOT 使用 valid 阶段产生的特征清单
- **AND** `muti_contract_scale_save.py` SHALL 写出 `SCALE_SAVE/fu/5min/scaler_manifest.json`
- **AND** `muti_contract_scale_save.py` SHALL 写出 `SCALE_SAVE/fu/5min/scale_diagnostics.csv`

#### Scenario: scale-save 输出只包含训练集选中特征
- **WHEN** `muti_contract_scale_save.py` 处理任一存在的 split 阶段合约文件
- **THEN** `muti_contract_scale_save.py` SHALL 写出 `SCALE_SAVE/fu/5min/{stage}/fu2601.feather`
- **AND** `muti_contract_scale_save.py` SHALL 同步写出 `SCALE_SAVE/fu/5min/{stage}/fu2601.csv`
- **AND** 输出 feather 和 csv SHALL 包含商品 reward/execution 列、train `state_features.npy` 中的 state features 和 `symbol`
- **AND** 系统 SHALL 将 state features 按 train-only robust scaler 进行标准化并默认裁剪到 `[-20, 20]`
- **AND** 系统 SHALL NOT 将未入选 state features 写入 scale-save 输出 feather 或 csv

#### Scenario: split-stage robust scaler fail-fast
- **WHEN** train split 输入目录不存在、没有合约 feather、train feature universe 为空、train 筛选结果为空、required feature column 缺失、clip bounds 无效或拟合统计量非有限
- **THEN** `muti_contract_scale_save.py` SHALL 报错并停止当前阶段
- **AND** 错误信息 SHALL 包含阶段名、缺失或为空的资源路径，以及相关合约或特征名
- **AND** 系统 SHALL NOT 静默跳过该合约
- **AND** 系统 SHALL NOT 写出下游可消费的 train `state_features.npy`

#### Scenario: 特征选择 fail-fast
- **WHEN** train 或 valid split 输入目录不存在、没有合约 feather、train feature universe 为空、train 筛选结果为空或 required feature column 缺失
- **THEN** feature selection SHALL 报错并停止当前阶段
- **AND** 错误信息 SHALL 包含阶段名、缺失或为空的资源路径，以及相关合约或特征名
- **AND** 系统 SHALL NOT 静默跳过该合约
- **AND** 系统 SHALL NOT 写出下游可消费的 train `state_features.npy`

### Requirement: 商品 FineFT 数据集装配输入契约
系统 SHALL 将 `FineFT/datahandler/commodity_contract_dataset.py` 收窄为 FineFT 商品数据集装配工具，从 `dataset_split_manifest.json` 获取阶段和合约元数据，并从阶段化 `SCALE_SAVE` 读取真实数据文件。

#### Scenario: 使用 dataset split manifest 代替 main contract summary
- **WHEN** 用户运行 `FineFT/datahandler/commodity_contract_dataset.py`
- **THEN** CLI SHALL 接收 `--dataset_split_manifest_path`
- **AND** 系统 SHALL 从该路径读取 `PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/{target_freq}/{symbol}/dataset_split_manifest.json`
- **AND** 系统 SHALL 使用该 manifest 中的 `sets.train.contracts`、`sets.valid.contracts` 和 `sets.test.contracts` 作为阶段合约列表
- **AND** 系统 SHALL NOT 读取 `main_contract_summary.json` 作为 FineFT 数据集生成输入

#### Scenario: 从阶段化 SCALE_SAVE 读取真实数据
- **WHEN** dataset split manifest 声明 `train` 阶段包含合约 `fu2508`
- **AND** CLI 传入 `--input_root PREPROCESS_DATASET/commodity-futures/SCALE_SAVE`、`--symbol fu` 和 `--target_freq 10min`
- **THEN** 系统 SHALL 读取 `PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/fu/10min/train/fu2508.feather`
- **AND** 系统 SHALL NOT 使用 manifest 中的旧 `input_path` 作为真实数据来源
- **AND** 系统 SHALL 对 `valid` 和 `test` 阶段使用同样的 `{input_root}/{symbol}/{target_freq}/{stage}/{contract}.feather` 路径规则

#### Scenario: 停止 FineFT 内部阶段切分
- **WHEN** `commodity_contract_dataset.py` 读取 dataset split manifest 和阶段化 SCALE_SAVE 文件
- **THEN** 系统 SHALL NOT 计算 train、valid 或 test split boundaries
- **AND** 系统 SHALL NOT 使用 `train_ratio`、`valid_ratio` 或 `test_ratio` 决定阶段归属
- **AND** 系统 SHALL NOT 按 `trading_days` 过滤 feather 行
- **AND** 系统 SHALL 直接复制已分阶段的合约文件作为 FineFT 阶段数据

#### Scenario: state features 路径使用训练阶段特征选择结果
- **WHEN** 用户运行 `commodity_contract_dataset.py`
- **THEN** CLI SHALL 接收 `--state_features_path`
- **AND** 推荐调用路径 SHALL 为 `PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/{target_freq}/{symbol}/train/state_features.npy`
- **AND** 系统 SHALL 复制该文件到 `dataset/{target_freq}/{symbol}/state_features.npy`
- **AND** CLI SHALL NOT 要求 `--feature_union_path`

#### Scenario: 输入契约不匹配 fail-fast
- **WHEN** `--dataset_split_manifest_path` 不存在、manifest 中 `symbol` 或 `target_freq` 与 CLI 不一致、`--state_features_path` 不存在、state feature 清单为空，或 manifest 声明的 SCALE_SAVE 阶段合约文件不存在
- **THEN** 系统 SHALL 报错并停止 FineFT 数据集生成
- **AND** 错误信息 SHALL 包含缺失或不匹配的路径、stage、contract、symbol 或 target_freq

### Requirement: 商品 VAE SHALL materialize cross-contract label training datasets
系统 SHALL 为商品期货 VAE 训练按 label 合并多合约样本，并将合并结果物化为训练输入文件。

#### Scenario: 合并存在的多合约 label 样本
- **WHEN** 用户为 `fu` 运行 VAE 训练并选择 `label_0`
- **THEN** 系统 SHALL 扫描 `dataset/10min/fu/VAE_data/<contract>/label_0.npy`
- **AND** 系统 SHALL 合并所有存在的二维 label 数组
- **AND** 系统 SHALL 写出 `dataset/10min/fu/VAE_data/train/label_0.npy`
- **AND** 合并后数组的列数 SHALL 等于每个源数组的 feature 维度
- **AND** 合并后数组的行数 SHALL 等于所有 included source arrays 的行数总和

#### Scenario: 写出训练集合并 manifest
- **WHEN** 系统写出 `VAE_data/train/label_0.npy`
- **THEN** 系统 SHALL 同步写出 `VAE_data/train/label_0_manifest.json`
- **AND** manifest SHALL 包含 `dataset_name`、`label`、`merged_path`、`total_samples` 和 `feature_dim`
- **AND** manifest SHALL 包含 `included_contracts`，其中每项记录 `contract`、`source_file` 和 `sample_count`
- **AND** manifest SHALL 包含 `missing_contracts`
- **AND** manifest SHALL 与合并训练集同一次生成并允许覆盖旧 manifest

#### Scenario: 缺失部分合约 label 时跳过并记录
- **WHEN** `VAE_data/fu2505/label_0.npy` 存在
- **AND** `VAE_data/fu2510/label_0.npy` 不存在
- **THEN** 系统 SHALL 使用 `fu2505` 的样本参与合并
- **AND** 系统 SHALL NOT 因 `fu2510` 缺少该 label 而停止训练
- **AND** 系统 SHALL 在 manifest 的 `missing_contracts` 中记录 `fu2510`

#### Scenario: 没有任何可用 label 样本时 fail-fast
- **WHEN** 用户为 `fu` 运行 VAE 训练并选择 `label_4`
- **AND** 没有任何 `VAE_data/<contract>/label_4.npy` 存在
- **THEN** 系统 SHALL 报错并停止训练
- **AND** 错误信息 SHALL 包含 `label_4` 和被扫描的 `VAE_data` 路径
- **AND** 系统 SHALL NOT 写出空的 `VAE_data/train/label_4.npy`

#### Scenario: 合并输入维度校验
- **WHEN** 系统合并 `label_0` 的合约数组
- **THEN** 每个源数组 SHALL 是二维数组
- **AND** 每个源数组 SHALL 至少包含一行样本
- **AND** 所有源数组的列数 SHALL 相同
- **AND** 任一数组不满足要求时系统 SHALL fail-fast
- **AND** 错误信息 SHALL 包含相关 `contract` 和 `source_file`

### Requirement: 商品 VAE SHALL train from the materialized cross-contract dataset
系统 SHALL 使用物化后的跨合约训练集训练每个 label 的通用 VAE 模型。

#### Scenario: VAE 训练读取物化训练集
- **WHEN** 用户为 `fu` 和 `label_0` 启动训练
- **THEN** VAE 训练 SHALL 读取 `dataset/10min/fu/VAE_data/train/label_0.npy`
- **AND** VAE 训练 SHALL NOT 读取 `dataset/10min/fu/VAE_data/label_0.npy`
- **AND** VAE 训练 SHALL NOT 要求旧的扁平 `VAE_data/label_0.npy` 存在

#### Scenario: 训练产物沿用 label 目录
- **WHEN** `label_0` 训练运行到保存 checkpoint 的 epoch
- **THEN** 系统 SHALL 在 `result/DiHFT/vae_results/fu/label_0/` 下保存 checkpoint
- **AND** `model_latest.pth` SHALL 表示最近一次保存的 checkpoint
- **AND** `model_latest.pth` MAY 被后续训练保存覆盖
- **AND** 系统 SHALL NOT 使用测试合约分析结果决定是否覆盖 `model_latest.pth`

#### Scenario: CLI 使用明确训练 flag
- **WHEN** 用户通过 `FineFT/RL/DiHFT/VAE/main.py` 启动商品 VAE 训练
- **THEN** CLI SHALL 支持明确的训练 flag，例如 `--train`
- **AND** 训练 flag SHALL 触发训练集合并、VAE 训练和训练后的分合约分析
- **AND** 系统 SHALL NOT 要求用户传入 `--if_train True` 才能训练

#### Scenario: 只重跑分析
- **WHEN** 用户希望使用已保存的 `model_latest.pth` 重跑测试分析
- **THEN** CLI SHALL 支持明确的 analyze-only 行为，例如 `--analyze-only`
- **AND** analyze-only SHALL 加载 `result/DiHFT/vae_results/fu/label_0/model_latest.pth`
- **AND** analyze-only SHALL NOT 重新合并训练集或重新训练模型

### Requirement: 商品 VAE SHALL analyze test contracts separately and produce aggregate outputs
系统 SHALL 对商品 VAE 测试合约逐合约分析，并输出分合约结果和总体汇总。

#### Scenario: 逐合约读取测试数组
- **WHEN** `dataset/10min/fu/VAE_data/test/test_fu2508.npy` 和 `test_fu2509.npy` 存在
- **THEN** 分析阶段 SHALL 分别读取每个 `test_<contract>.npy`
- **AND** 分析阶段 SHALL NOT 要求 `dataset/10min/fu/VAE_data/test.npy` 存在
- **AND** 分析阶段 SHALL NOT 在磁盘上写出合并后的 test 输入文件

#### Scenario: 输出分合约 logpx npy 和 csv
- **WHEN** 系统完成合约 `fu2508` 的 VAE 分析
- **THEN** 系统 SHALL 写出 `result/DiHFT/vae_results/fu/label_0/ood_logpx_fu2508.npy`
- **AND** 系统 SHALL 写出 `result/DiHFT/vae_results/fu/label_0/ood_logpx_fu2508.csv`
- **AND** CSV SHALL 包含列 `contract`、`source_file`、`row_index` 和 `logpx`
- **AND** CSV 的 `contract` 列 SHALL 等于 `fu2508`
- **AND** CSV 的 `source_file` 列 SHALL 等于该合约测试数组路径
- **AND** CSV 的 `row_index` SHALL 与 `ood_logpx_fu2508.npy` 中的 logpx 顺序一一对应

#### Scenario: 输出总体 logpx npy 和 csv
- **WHEN** 系统完成所有测试合约分析
- **THEN** 系统 SHALL 写出 `result/DiHFT/vae_results/fu/label_0/ood_logpx_all.npy`
- **AND** 系统 SHALL 写出 `result/DiHFT/vae_results/fu/label_0/ood_logpx_all.csv`
- **AND** `ood_logpx_all.npy` SHALL 按合约名稳定排序后的测试结果顺序拼接所有合约 logpx
- **AND** `ood_logpx_all.csv` SHALL 包含列 `contract`、`source_file`、`row_index` 和 `logpx`
- **AND** `ood_logpx_all.csv` SHALL 保留每一行对应的原始测试合约和源文件路径

#### Scenario: 输出 summary 统计
- **WHEN** 系统完成所有测试合约分析
- **THEN** 系统 SHALL 写出 `result/DiHFT/vae_results/fu/label_0/summary.json`
- **AND** summary SHALL 包含 `dataset_name` 和 `label`
- **AND** summary SHALL 为每个测试合约记录 `source_file`、`samples`、`logpx_mean`、`logpx_std`、`logpx_min` 和 `logpx_max`
- **AND** summary SHALL 记录总体 `samples`、`logpx_mean`、`logpx_std`、`logpx_min` 和 `logpx_max`
- **AND** summary SHALL NOT 输出 AUROC、AUPRC 或 FPR80，除非另有明确的 ID/OOD reference 定义

#### Scenario: 输出增强 summary 门控分析指标
- **WHEN** 系统完成 `label_0` 的训练后分析或 analyze-only 分析
- **THEN** `summary.json` SHALL 包含 `train_baseline`
- **AND** `train_baseline` SHALL 记录训练集 `source_file`、`input_samples`、`analyzed_samples`、`sample_mismatch`、`logpx_mean`、`logpx_std`、`logpx_min` 和 `logpx_max`
- **AND** `train_baseline` SHALL 包含 `quantiles`，其中至少包含 `q01`、`q05`、`q25`、`q50`、`q75`、`q95` 和 `q99`
- **AND** 每个测试合约 summary SHALL 记录 `input_samples`、`analyzed_samples` 和 `sample_mismatch`
- **AND** 每个测试合约 summary SHALL 包含相同 quantile keys 的 `quantiles`
- **AND** 每个测试合约 summary SHALL 包含 `acceptance`，其中至少包含 `ge_train_q01_pct`、`ge_train_q05_pct` 和 `ge_train_q50_pct`
- **AND** 总体 test summary SHALL 包含相同 quantile keys 的 `quantiles` 和相同 threshold keys 的 `acceptance`
- **AND** summary SHALL NOT 输出 accuracy、AUROC、AUPRC 或 FPR80，除非另有明确的真实 label 或 ID/OOD reference 定义

#### Scenario: 样本数完整性检查
- **WHEN** 分析阶段读取 `test_fu2508.npy`
- **THEN** 系统 SHALL 在 summary 中记录该文件的 `input_samples`
- **AND** 系统 SHALL 在 summary 中记录实际写入 logpx 的 `analyzed_samples`
- **AND** 当 `input_samples` 不等于 `analyzed_samples` 时 `sample_mismatch` SHALL 为 true
- **AND** 当 `input_samples` 等于 `analyzed_samples` 时 `sample_mismatch` SHALL 为 false
- **AND** `samples` SHALL 保持为 `analyzed_samples` 的兼容别名

#### Scenario: 输出跨 label routing summary
- **WHEN** `result/DiHFT/vae_results/fu/label_0` 到 `label_4` 的分合约 `ood_logpx_<contract>.npy` 均存在
- **THEN** 系统 SHALL 能生成 `result/DiHFT/vae_results/fu/routing_summary.json`
- **AND** routing summary SHALL 包含 `dataset_name`、`labels` 和 `score_type`
- **AND** routing summary SHALL 为每个测试合约记录 `samples`、`winner_counts` 和 `winner_pct`
- **AND** routing summary SHALL 记录总体 `samples`、`winner_counts` 和 `winner_pct`
- **AND** winner SHALL 表示同一 row 在所有 label logpx 中分数最高的 label
- **AND** routing summary SHALL 记录 `top1_top2_margin_mean`、`top1_top2_margin_q25` 和 `low_margin_pct`
- **AND** 若某合约在不同 label 下的 logpx 数量不一致，系统 SHALL 只比较共同长度并在 routing summary 中记录 sample mismatch 信息

#### Scenario: 测试输入维度校验
- **WHEN** 分析阶段读取 `test_fu2508.npy`
- **THEN** 测试数组 SHALL 是二维数组
- **AND** 测试数组的列数 SHALL 等于训练集 `feature_dim`
- **AND** 任一测试数组不满足要求时系统 SHALL fail-fast
- **AND** 错误信息 SHALL 包含相关 `contract` 和 `source_file`

### Requirement: 商品 VAE fu shell entry SHALL run the multi-contract workflow
系统 SHALL 提供适配当前商品多合约数据结构的 `fu` VAE shell 入口。

#### Scenario: fu VAE shell passes current data base path
- **WHEN** 用户运行 `FineFT/script/train/DiHFT/low_level/VAE_util_fu.sh`
- **THEN** 脚本 SHALL 激活 `finetf` conda 环境
- **AND** 脚本 SHALL 设置 `PYTHONPATH` 以包含 `FineFT`
- **AND** 脚本 SHALL 调用 `FineFT/RL/DiHFT/VAE/main.py`
- **AND** 每次调用 SHALL 传递 `--dataset_name fu`
- **AND** 每次调用 SHALL 传递 `--data_base_path dataset/10min`
- **AND** 每次调用 SHALL 使用明确训练 flag，例如 `--train`

#### Scenario: fu VAE shell launches all labels
- **WHEN** 用户运行 `VAE_util_fu.sh`
- **THEN** 脚本 SHALL 为 `label_0` 到 `label_4` 启动 VAE 训练
- **AND** 每个 label 的日志 SHALL 写入 `log/DiHFT/fu/VAE/`
- **AND** 每个 label 调用 SHALL 传递对应的 `--label_index`

#### Scenario: fu VAE shell limits concurrent training jobs
- **WHEN** 用户运行 `VAE_util_fu.sh`
- **THEN** 脚本 SHALL 默认最多同时运行 2 个 VAE 训练进程
- **AND** 脚本 SHALL 支持通过 `MAX_PARALLEL_JOBS` 环境变量覆盖并发上限
- **AND** `MAX_PARALLEL_JOBS` 不是正整数时脚本 SHALL fail-fast
- **AND** 任一 label 训练进程失败时脚本 SHALL 最终返回非 0

### Requirement: 商品期货主力合约 summary 构建 SHALL 使用 dataclass 对象内部状态
系统 SHALL 使用 dataclass 对象表达商品期货主力合约 summary 构建过程中的源文件发现、月度统计、合约入选和交易日裁剪状态，并在 `main_contract.py` 内部通过对象属性和对象方法传递这些状态，而不是通过裸 dict key 传递多层聚合数据。

#### Scenario: 源文件发现和月度统计以对象传递
- **WHEN** `load_contract_files_by_trading_day_for_years()` 扫描 `data/原始下载/{commodity_name}/{YYYY}` 目录
- **THEN** 函数 SHALL 返回 `TradingDayContractSources` 之类的 dataclass 对象集合
- **AND** 每个交易日条目 SHALL 使用对象记录 `trading_day` 以及其下的 contract/source_file 明细
- **AND** `build_main_contract_summary_model_for_date_range()` SHALL 使用对象表达按月成交量、按月高成交量天数、合约交易日历史和已选月份
- **AND** 主构建流程 SHALL NOT 把嵌套 dict 当作函数之间的中间状态边界

#### Scenario: 主力合约 summary 以对象返回
- **WHEN** `build_main_contract_summary_for_date_range()` 完成构建
- **THEN** 函数 SHALL 返回 `MainContractSummary` 对象
- **AND** 对象 SHALL 暴露 `symbol`、`commodity_name`、`start_date`、`end_date`、`selection_rule` 和 `contracts`
- **AND** 下游读取 summary 的调用方 SHALL 继续通过对象属性访问 contract、trading_days 和 source_file

### Requirement: 商品期货主力合约 summary JSON serialization SHALL preserve existing contract
系统 SHALL 只在 JSON 写入边界将商品期货主力合约 summary 对象序列化为 dict，并保持现有 `main_contract_summary.json` 结构兼容。

#### Scenario: JSON 结构保持兼容
- **WHEN** `write_main_contract_summary_for_date_range()` 写出 `main_contract_summary.json`
- **THEN** JSON SHALL 保持当前兼容结构，包含 `symbol`、`commodity_name`、`start_date`、`end_date`、`selection_rule` 和 `contracts`
- **AND** `contracts` 中每项 SHALL 保持 `contract`、`start_trading_day`、`end_trading_day`、`trading_day_count`、`selected_months` 和 `trading_days`
- **AND** 写出的 JSON payload SHALL 等于 `MainContractSummary.to_dict()` 的结果
- **AND** `load_main_contract_summary()` SHALL 继续读取同一 JSON 结构并返回 `MainContractSummary`

#### Scenario: 对象层不改变现有失败语义
- **WHEN** 现有原始 CSV 缺字段、重复 `TradingDay + contract`、无入选合约、空裁剪窗口或其他主力合约生成错误条件出现
- **THEN** 系统 SHALL 继续抛出当前相同类别的异常
- **AND** dataclass 对象层 SHALL NOT 吞掉底层异常或新增独立业务校验

### Requirement: 商品期货主力合约 summary object refactor SHALL be covered by focused tests
系统 SHALL 通过聚焦测试同时验证主力合约 summary 内部对象接口和外部 JSON 兼容性。

#### Scenario: focused tests assert object return types and attributes
- **WHEN** 执行 commodity 主力合约 summary 相关测试
- **THEN** 测试 SHALL 断言 `load_contract_files_by_trading_day_for_years()` 返回 dataclass 对象集合
- **AND** 测试 SHALL 断言 `build_main_contract_summary_for_date_range()` 返回 `MainContractSummary`
- **AND** 测试 SHALL 断言 `write_main_contract_summary_for_date_range()` 写出的内容与返回对象 `to_dict()` 一致
- **AND** 测试中针对返回值的业务断言 SHALL 使用对象属性访问

#### Scenario: focused tests assert JSON payload compatibility
- **WHEN** focused tests 读取 `main_contract_summary.json`
- **THEN** 测试 SHALL 断言 `json.loads(...)` 的结果等于对应 summary 对象 `to_dict()` 的结果
- **AND** 测试 SHALL 保留关键字段、层级、合约顺序和交易日窗口裁剪字段的兼容性断言
- **AND** focused verification SHALL 使用 `conda activate finetf && pytest data_preprocess/tests/test_commodity_main_contract.py`

### Requirement: 商品期货五档 OFI row-window 特征
系统 SHALL 从连续五档 quote 快照计算标准五档 OFI，并按固定输入行数聚合输出。

#### Scenario: 五档相邻快照 OFI 明细
- **WHEN** 输入 quote 快照包含 `timestamp`、`BidPrice1` 到 `BidPrice5`、`AskPrice1` 到 `AskPrice5`、`BidVolume1` 到 `BidVolume5` 和 `AskVolume1` 到 `AskVolume5`
- **THEN** 系统 SHALL 按 `timestamp` 全局排序后比较每条快照与上一条快照
- **AND** 第一条快照的每档 OFI SHALL 为 `0`
- **AND** 对每个 bid 档位，价格上移时使用 `+ 当前 BidVolume`，价格不变时使用 `当前 BidVolume - 上一条 BidVolume`，价格下移时使用 `- 上一条 BidVolume`
- **AND** 对每个 ask 档位，价格下移时使用 `- 当前 AskVolume`，价格不变时使用 `-(当前 AskVolume - 上一条 AskVolume)`，价格上移时使用 `+ 上一条 AskVolume`
- **AND** 输出 SHALL 包含 `ofi_bid1` 到 `ofi_bid5`、`ofi_ask1` 到 `ofi_ask5`、`ofi_bid`、`ofi_ask` 和 `ofi`
- **AND** `ofi_bid` SHALL 等于 `ofi_bid1` 到 `ofi_bid5` 的和，`ofi_ask` SHALL 等于 `ofi_ask1` 到 `ofi_ask5` 的和，`ofi` SHALL 等于 `ofi_bid + ofi_ask`

#### Scenario: 固定 12 行聚合
- **WHEN** 用户使用默认 `window_rows=12` 生成五档 OFI 特征
- **THEN** 系统 SHALL 每 12 条按时间排序后的输入快照输出一行 OFI bar
- **AND** 每行输出的 `timestamp` SHALL 为该组内最后一条快照的 `timestamp`
- **AND** 每行输出的 `nquote` SHALL 为该组输入快照数量
- **AND** 每行输出的所有 OFI 明细列和汇总列 SHALL 为该组内逐快照 OFI 的求和
- **AND** 最后不足 12 条输入快照的尾组 SHALL 保留输出

#### Scenario: OFI 归一化特征
- **WHEN** 系统输出五档 OFI bar
- **THEN** 输出 SHALL 包含 `ofi_norm`、`ofi_bid_norm` 和 `ofi_ask_norm`
- **AND** `ofi_norm` SHALL 等于 `ofi / sum(BidVolume1-5 + AskVolume1-5)`，分母为同一 OFI bar 内所有输入快照的五档 bid 与 ask volume 合计
- **AND** `ofi_bid_norm` SHALL 等于 `ofi_bid / sum(BidVolume1-5)`，分母为同一 OFI bar 内所有输入快照的五档 bid volume 合计
- **AND** `ofi_ask_norm` SHALL 等于 `ofi_ask / sum(AskVolume1-5)`，分母为同一 OFI bar 内所有输入快照的五档 ask volume 合计
- **AND** 当任一归一化分母为 `0` 时，对应归一化输出 SHALL 为 `0`

#### Scenario: OFI 比较跨行窗口连续
- **WHEN** 第 13 条快照开始新的 12 行 OFI bar
- **THEN** 第 13 条快照的 OFI SHALL 使用第 12 条快照作为上一条快照计算
- **AND** 系统 MUST NOT 在固定行数窗口边界重置相邻快照状态

#### Scenario: OFI 输入 fail-fast
- **WHEN** OFI 输入没有任何 quote 快照
- **THEN** 系统 SHALL 报错并说明没有 quote snapshot
- **WHEN** OFI 输入缺少任一五档价格或数量必需列
- **THEN** 系统 SHALL 报错并列出缺失列名
- **WHEN** OFI 输入的任一五档价格或数量必需列存在 null
- **THEN** 系统 SHALL 报错并列出存在 null 的列名
- **WHEN** OFI 输入的任一五档价格或数量必需列存在 NaN、`inf` 或 `-inf`
- **THEN** 系统 SHALL 报错并列出存在非有限值的列名
- **WHEN** `window_rows <= 0`
- **THEN** 系统 SHALL 报错并说明 `window_rows` 必须为正数

### Requirement: 商品期货 quote microstructure row-window 特征
系统 SHALL 从一档 quote 快照派生独立的 row-window microstructure 特征，不改变现有时间窗口 quote 下采样输出。

#### Scenario: 独立固定行窗口输出
- **WHEN** 系统调用 `downscale_quote_microstructure_features(second_df, window_rows=12)`
- **THEN** 系统 SHALL 按 `timestamp` 排序输入 quote 快照
- **AND** 系统 SHALL 每 12 条连续输入行输出一行 row-window 特征
- **AND** 系统 SHALL 保留不足 12 条的尾部窗口
- **AND** 输出 `timestamp` SHALL 使用窗口内最后一条 quote 快照时间
- **AND** 输出 SHALL 包含 `nquote`
- **AND** 系统 SHALL NOT 修改 `downscale_quote_features()` 的时间窗口输出语义
- **AND** 系统 SHALL NOT 将 microstructure 特征并入 OFI 输出

#### Scenario: microprice pressure 与 relative spread
- **WHEN** 输入 quote 快照包含 `BidPrice1`、`AskPrice1`、`BidVolume1` 和 `AskVolume1`
- **THEN** 系统 SHALL 对每条快照计算 `spread = AskPrice1 - BidPrice1`
- **AND** 系统 SHALL 对每条快照计算 `mid = (AskPrice1 + BidPrice1) / 2`
- **AND** 系统 SHALL 对每条快照计算 `microprice = (AskPrice1 * BidVolume1 + BidPrice1 * AskVolume1) / (BidVolume1 + AskVolume1)`
- **AND** 系统 SHALL 对每条快照计算 `microprice_pressure = (microprice - mid) / spread`
- **AND** 系统 SHALL 对每条快照计算 `relative_spread = spread / mid`
- **AND** row-window 输出 SHALL 包含 `mean_microprice_pressure`
- **AND** row-window 输出 SHALL 包含 `mean_relative_spread`
- **AND** row-window 输出 SHALL NOT 包含 `microprice_pressure` 或 `relative_spread` 的 OHLC 或 std 统计列

#### Scenario: spread 变化计数与比例
- **WHEN** row-window 内包含 quote 快照
- **THEN** 系统 SHALL 使用相邻快照的 `spread.diff()` 判断 spread 变化方向
- **AND** `spread.diff() > 0` SHALL 计入 `spread_widen_count`
- **AND** `spread.diff() < 0` SHALL 计入 `spread_narrow_count`
- **AND** `spread.diff() == 0` SHALL 计入 `spread_flat_count`
- **AND** 第一条快照没有前序 spread 时 SHALL 计入 `spread_flat_count`
- **AND** `spread_widen_count + spread_narrow_count + spread_flat_count` SHALL 等于 `nquote`
- **AND** `spread_widen_ratio` SHALL 等于 `spread_widen_count / nquote`

#### Scenario: 输入结构 fail-fast
- **WHEN** microstructure 特征输入为空
- **THEN** 系统 SHALL fail-fast
- **AND** 错误信息 SHALL 说明输入没有 quote snapshots
- **WHEN** `window_rows <= 0`
- **THEN** 系统 SHALL fail-fast
- **AND** 错误信息 SHALL 说明 `window_rows` 必须为正数
- **WHEN** 输入缺少 `timestamp`、`BidPrice1`、`AskPrice1`、`BidVolume1` 或 `AskVolume1` 中任一必要列
- **THEN** 系统 SHALL fail-fast
- **AND** 错误信息 SHALL 列出缺失列

#### Scenario: 输入非有限值 fail-fast
- **WHEN** `BidPrice1`、`AskPrice1`、`BidVolume1` 或 `AskVolume1` 任一列包含 `NaN`、`inf` 或 `-inf`
- **THEN** 系统 SHALL fail-fast
- **AND** 错误信息 SHALL 说明 microstructure 输入列包含非有限值
- **AND** 系统 SHALL NOT 生成包含 `NaN`、`inf` 或 `-inf` 的 microstructure 输出

#### Scenario: 派生零分母输出中性值
- **WHEN** `BidVolume1 + AskVolume1 == 0`
- **THEN** 对应快照的 `microprice_pressure` SHALL 为 `0.0`
- **WHEN** `spread == 0`
- **THEN** 对应快照的 `microprice_pressure` SHALL 为 `0.0`
- **WHEN** `mid == 0`
- **THEN** 对应快照的 `relative_spread` SHALL 为 `0.0`
- **AND** row-window 输出 SHALL NOT 包含 `NaN`、`inf` 或 `-inf`

### Requirement: 商品期货 quote queue pressure row-window 特征
系统 SHALL 在商品 quote microstructure row-window 输出中包含一档队列补充/撤单压力和单边盘口状态比例，并避免产生非法数值。

#### Scenario: 一档队列补充与撤单计数
- **WHEN** 系统调用 `downscale_quote_microstructure_features(second_df, window_rows=12)`
- **AND** 输入 quote 快照包含 `timestamp`、`BidPrice1`、`AskPrice1`、`BidVolume1` 和 `AskVolume1`
- **THEN** 系统 SHALL 按 `timestamp` 排序输入 quote 快照
- **AND** 每条快照 SHALL 与上一条快照比较一档价格和数量
- **AND** 第一条快照没有前序快照时 SHALL 不计入任何队列事件
- **AND** 当 `BidPrice1` 与上一条 `BidPrice1` 相同且 `BidVolume1` 上升时，该快照 SHALL 计入 `bid_refill_count`
- **AND** 当 `BidPrice1` 与上一条 `BidPrice1` 相同且 `BidVolume1` 下降时，该快照 SHALL 计入 `bid_deplete_count`
- **AND** 当 `AskPrice1` 与上一条 `AskPrice1` 相同且 `AskVolume1` 上升时，该快照 SHALL 计入 `ask_refill_count`
- **AND** 当 `AskPrice1` 与上一条 `AskPrice1` 相同且 `AskVolume1` 下降时，该快照 SHALL 计入 `ask_deplete_count`
- **AND** row-window 输出 SHALL 包含 `bid_refill_count`、`bid_deplete_count`、`ask_refill_count` 和 `ask_deplete_count`
- **AND** 每个 count 输出 SHALL 等于该 row-window 内逐快照事件命中次数之和

#### Scenario: 队列补充 imbalance
- **WHEN** row-window 内包含队列补充或撤单事件
- **THEN** 系统 SHALL 计算 `total_queue_events = bid_refill_count + bid_deplete_count + ask_refill_count + ask_deplete_count`
- **AND** 系统 SHALL 输出 `queue_refill_imbalance`
- **AND** `queue_refill_imbalance` SHALL 等于 `(bid_refill_count + ask_deplete_count - bid_deplete_count - ask_refill_count) / total_queue_events`
- **AND** 当 `total_queue_events == 0` 时，`queue_refill_imbalance` SHALL 为 `0.0`
- **AND** `queue_refill_imbalance` SHALL NOT 为 `NaN`、`inf` 或 `-inf`

#### Scenario: 空侧与涨跌停单边盘口比例
- **WHEN** 输入 quote 快照包含 `LastPrice`、`LowPrice`、`HighPrice`、`LowerLimitPrice` 和 `UpperLimitPrice`
- **THEN** row-window 输出 SHALL 包含 `bid_side_empty_ratio`
- **AND** row-window 输出 SHALL 包含 `ask_side_empty_ratio`
- **AND** row-window 输出 SHALL 包含 `limit_down_single_sided_ratio`
- **AND** row-window 输出 SHALL 包含 `limit_up_single_sided_ratio`
- **AND** `bid_side_empty_ratio` SHALL 等于该 row-window 内 bid 一档价格为空或 bid 一档 volume 为 `0` 的快照数除以 `nquote`
- **AND** `ask_side_empty_ratio` SHALL 等于该 row-window 内 ask 一档价格为空或 ask 一档 volume 为 `0` 的快照数除以 `nquote`
- **AND** `limit_down_single_sided_ratio` SHALL 等于该 row-window 内触及跌停、bid side empty 且 ask side 有效的快照数除以 `nquote`
- **AND** `limit_up_single_sided_ratio` SHALL 等于该 row-window 内触及涨停、ask side empty 且 bid side 有效的快照数除以 `nquote`
- **AND** 所有比例分母为 `0` 时 SHALL 输出 `0.0`
- **AND** 所有比例输出 SHALL NOT 为 `NaN`、`inf` 或 `-inf`

#### Scenario: queue pressure 输入 fail-fast
- **WHEN** microstructure queue pressure 输入缺少 `LastPrice`、`LowPrice`、`HighPrice`、`LowerLimitPrice` 或 `UpperLimitPrice` 中任一必要列
- **THEN** 系统 SHALL fail-fast
- **AND** 错误信息 SHALL 列出缺失列
- **WHEN** queue pressure 使用的任一数值列包含 `NaN`、`inf` 或 `-inf`
- **THEN** 系统 SHALL fail-fast
- **AND** 错误信息 SHALL 说明 microstructure 输入列包含非有限值

#### Scenario: 既有 microstructure 输出保持兼容
- **WHEN** 系统输出 queue pressure row-window 特征
- **THEN** 既有 `timestamp`、`nquote`、`mean_microprice_pressure`、`mean_relative_spread`、`spread_widen_count`、`spread_narrow_count`、`spread_flat_count` 和 `spread_widen_ratio` SHALL 保持输出
- **AND** 系统 SHALL NOT 修改 `downscale_quote_features()` 的时间窗口输出语义
- **AND** 系统 SHALL NOT 修改 `downscale_quote_ofi_features()` 的 OFI 输出语义


### Requirement: 商品期货 BASE_TIME_FEATURE 时间编码特征
系统 SHALL 生成 9 个非绝对 BASE_TIME_FEATURE 时间编码特征，强制作为 State Feature 保留并跳过 Robust Scaling。

#### Scenario: 9 个 BASE_TIME_FEATURE 列名与语义
- **WHEN** 生成商品期货 `BASE_TIME_FEATURE`
- **THEN** 输出包含 9 个列：`trading_minute_progress`、`morning_session`、`afternoon_session`、`night_session`、`is_opening_30m`、`is_closing_30m`、`contract_month_sin`、`contract_month_cos` 和 `contract_life_remaining_ratio`
- **AND** `trading_minute_progress` 为当前 timestamp 在所属 Trading Session 内的归一化进度
- **AND** `morning_session` / `afternoon_session` / `night_session` 为互斥 one-hot 标记
- **AND** `is_opening_30m` / `is_closing_30m` 为 session 独立首尾半小时标记
- **AND** `contract_month_sin` / `contract_month_cos` 为合约交割月份的 sin/cos 周期编码
- **AND** `contract_life_remaining_ratio` 为合约剩余生命周期比例

#### Scenario: Daily Merge join BASE_TIME_FEATURE
- **WHEN** 运行 daily merge
- **THEN** 按 `timestamp` 将 `BASE_TIME_FEATURE` join 到 `FUTURE_FEATURE`
- **AND** timestamp 不一致时 fail-fast

#### Scenario: Feature Selection 强制保留 BASE_TIME_FEATURE
- **WHEN** 运行 Feature Selection
- **THEN** 传入 `--mandatory_state_features` 保护 `BASE_TIME_FEATURE_COLUMNS` 9 个特征列
- **AND** 这些特征不参与 Hard Filter、Stability Filter、Composite Score、Correlation Filter 或 Blacklist 过滤，强制保留在 `state_features.npy` 中

#### Scenario: Scale Save 跳过 BASE_TIME_FEATURE 缩放
- **WHEN** 运行 Scale Save
- **THEN** 传入 `--passthrough_features` 包含 `BASE_TIME_FEATURE_COLUMNS` 9 个特征列
- **AND** 这些特征列直接保存原始编码值，不参与 robust scaler 的 fit、transform 或 clip

### Requirement: 商品期货风险与流动性 State Feature 滚动特征
系统 SHALL 从 5min 基础行情数据计算 6 个风险状态特征与 4 个流动性状态特征，按窗口配置输出带窗口后缀的 State Feature 列。

#### Scenario: 6 个风险状态特征列与公式
- **WHEN** 从 5min 行情数据 (`open`, `high`, `low`, `close`) 计算风险状态特征
- **THEN** 对每个配置窗口输出带 `{window}` 后缀的 6 个特征列：
- **AND** `atr_pct_{window}`: 真实波幅相对收盘价比例均值 (`mean(TR, N) / close * 100`)
- **AND** `historical_volatility_{window}`: 历史收益率标准差日化 (`std(r, N) * sqrt(bars_per_day)`)，`bars_per_day` 由品种 Trading Session 分钟数与 target_freq 推导
- **AND** `rolling_volatility_{window}`: 指数加权近期收益率波动率 (`ewm_std(r, N)`)
- **AND** `parkinson_volatility_{window}`: Parkinson 波动率 (`sqrt(mean(ln(high/low)^2, N) / (4*ln(2)))`)
- **AND** `garman_klass_volatility_{window}`: Garman-Klass 波动率 (`sqrt(max(mean(0.5*ln(high/low)^2 - (2*ln(2)-1)*ln(close/open)^2, N), 0))`)
- **AND** `realized_volatility_{window}`: 已实现波动率 (`sqrt(sum(r^2, N))`)

#### Scenario: 4 个流动性状态特征列与公式
- **WHEN** 从 5min 行情数据 (`volume`, `tradeval`, `open_interest`) 计算流动性状态特征
- **THEN** 对每个配置窗口输出带 `{window}` 后缀的 4 个特征列：
- **AND** `relative_volume_{window}`: 当前成交量相对窗口均量倍数 (`volume / mean(volume, N)`)
- **AND** `relative_amount_{window}`: 当前成交额相对窗口均额倍数 (`tradeval / mean(tradeval, N)`)
- **AND** `relative_open_interest_{window}`: 当前持仓量相对窗口均持仓量倍数 (`open_interest / mean(open_interest, N)`)
- **AND** `open_interest_change_ratio_{window}`: 持仓量相对 N 根 bar 前变化率 (`(open_interest_t - open_interest_{t-N}) / open_interest_{t-N}`)，分母 <= 0 时输出 `0.0`

#### Scenario: 下采样输出持仓量 open_interest
- **WHEN** `downscale.py` 从秒级快照生成 5min `BASE_FEATURE`
- **THEN** 输出 `open_interest` 列，其值为窗口内最后一条秒级快照的 `OpenInterest`
- **AND** 源数据缺少 `OpenInterest` 时 fail-fast

#### Scenario: 候选特征参与 Feature Selection 与 Scale Save
- **WHEN** 风险与流动性状态特征生成完成
- **THEN** 这些特征作为普通 candidate state feature 进入 Feature Selection 筛选
- **AND** 最终入选列进入 Scale Save 执行 train-only robust scaling 与 clip

### Requirement: 商品期货跨月合约结构 State Feature
系统 SHALL 生成跨月合约结构特征，覆盖主力/次主力动态配对与到期月份序列配对，并将其作为强制保留但参与缩放的 State Feature。

#### Scenario: 双重配对模式
- **WHEN** 生成跨月合约结构特征
- **THEN** 系统 SHALL 支持主力/次主力动态配对
- **AND** 主力合约 summary SHALL 为每个已完成交易日的每个合约记录事后动态身份：`main`、`sub` 或 `other`
- **AND** 生成交易日 T 的 CROSS_MONTH_FEATURE 时，主力合约 `main` 与次主力合约 `sub` SHALL 来自 T 之前最近一个可用交易日的 summary 动态身份
- **AND** 系统 SHALL NOT 使用交易日 T 当天完整成交量或持仓量计算出的事后动态身份生成 T 日特征
- **AND** 当交易日 T 不存在前一可用交易日动态身份时，系统 SHALL fail-fast 或使用明确记录的无未来信息 fallback
- **AND** CROSS_MONTH_FEATURE SHALL 输出当前合约角色 one-hot 特征，使下游数据能判断当前合约是主力合约、次主力合约，还是二者都不是
- **AND** 当当前合约身份为 `other` 时，系统 SHALL 计算当前合约分别相对主力合约与次主力合约的价格关系
- **AND** 综合排名规则 SHALL 是确定性的，并在成交量或持仓量并列时使用稳定排序
- **AND** 系统 SHALL 支持到期月份序列配对
- **AND** 到期月份序列配对 SHALL 按合约真实交割月份由近到远确定 `M_1`、`M_2` 和 `M_3`
- **AND** 系统 SHALL NOT 使用挂牌顺序或合约名自然排序替代真实交割月份排序

#### Scenario: No Absolute Price Rule
- **WHEN** 生成包含价格的跨月合约结构特征
- **THEN** 系统 SHALL NOT 输出原始价格水平列
- **AND** 系统 SHALL NOT 输出原始价格差列
- **AND** 系统 SHALL 仅输出无量纲、相对化或平稳化的价格表达
- **AND** 允许的价格表达 SHALL 包含对数收益差 `ln(P_1 / P_2)`
- **AND** 允许的价格表达 SHALL 包含相对百分比价差 `(P_1 - P_2) / P_1`
- **AND** 允许的价格表达 SHALL 包含蝶式结构相对比率 `(2 * P_M2 - P_M1 - P_M3) / P_M2`
- **AND** 允许的价格表达 SHALL 包含滚动窗口价差 Z-Score
- **AND** 成交量和持仓量跨月表达 SHALL 使用占比或相对比率，例如 `OI_2 / (OI_1 + OI_2)`

#### Scenario: 固定宽度 CROSS_MONTH_FEATURE 列契约
- **WHEN** 为任一当前合约生成跨月合约结构特征
- **THEN** 无论当前合约身份为 `main`、`sub` 或 `other`，系统 SHALL 输出相同数量、相同顺序的 CROSS_MONTH_FEATURE 列
- **AND** 输出 SHALL 包含角色 one-hot：`cm_contract_role_main`、`cm_contract_role_sub`、`cm_contract_role_other`
- **AND** 输出 SHALL 包含当前合约相对主力合约的 4 个特征：`cm_current_main_log_price_ratio`、`cm_current_main_relative_price_spread`、`cm_current_main_volume_share_current`、`cm_current_main_open_interest_share_current`
- **AND** 输出 SHALL 包含当前合约相对次主力合约的 4 个特征：`cm_current_sub_log_price_ratio`、`cm_current_sub_relative_price_spread`、`cm_current_sub_volume_share_current`、`cm_current_sub_open_interest_share_current`
- **AND** 输出 SHALL 包含市场主力/次主力结构的 4 个特征：`cm_main_sub_log_price_ratio`、`cm_main_sub_relative_price_spread`、`cm_main_sub_volume_share_sub`、`cm_main_sub_open_interest_share_sub`
- **AND** 输出 SHALL 包含到期月份序列结构的 7 个特征：`cm_m1_m2_log_price_ratio`、`cm_m2_m3_log_price_ratio`、`cm_m1_m2_relative_price_spread`、`cm_m2_m3_relative_price_spread`、`cm_m1_m2_m3_butterfly_ratio`、`cm_m1_m2_open_interest_share_m2`、`cm_m2_m3_open_interest_share_m3`
- **AND** 当当前合约与参照合约相同时，对应价格关系 SHALL 使用中性值 `0.0`
- **AND** 当当前合约与参照合约相同时，对应成交量或持仓量 share SHALL 使用同一分母公式计算；若分母小于等于 `0` 则输出 `0.0`

#### Scenario: 右闭右标窗口聚合与当前合约时间轴对齐
- **WHEN** 从多个合约生成目标频次跨月合约结构特征
- **THEN** 系统 SHALL 先对各合约独立执行右闭右标窗口聚合
- **AND** 系统 SHALL 以当前输出合约的时间轴作为基准执行 Left Join 对齐
- **AND** 次主力或远月合约在对齐后产生的无流动性窗口缺口 SHALL 填充为 `0.0`
- **AND** 系统 SHALL NOT 将缺失的整日跨月特征文件或缺失的必要输入文件当作流动性缺口
- **AND** 缺失整日跨月特征文件、缺失必要输入或无法解析交割月份时系统 SHALL fail-fast

#### Scenario: CROSS_MONTH_FEATURE 存储布局
- **WHEN** 跨月合约结构特征生成完成
- **THEN** 系统 SHALL 将结果独立写入 `PREPROCESS_DATASET/commodity-futures/CROSS_MONTH_FEATURE/{symbol}/{contract}/{target_freq}/{YYYY-MM-DD}.feather`
- **AND** 输出 SHALL 包含 `timestamp` 与跨月合约结构特征列
- **AND** 输出 SHALL NOT 包含 Reward/Execution 列
- **AND** 输出 SHALL NOT 包含原始价格水平列或原始价格差列

#### Scenario: Daily Merge join CROSS_MONTH_FEATURE
- **WHEN** 运行 daily merge
- **THEN** 系统 SHALL 按 `timestamp` 将 `CROSS_MONTH_FEATURE` join 到 `FUTURE_FEATURE`
- **AND** join 后的跨月合约结构特征 SHALL 属于 State Feature
- **AND** required 模式下缺失 `CROSS_MONTH_FEATURE` 文件 SHALL fail-fast
- **AND** 对齐后跨月合约结构特征中的 null 值 SHALL 填充为 `0.0`

#### Scenario: Feature Selection 强制保留 CROSS_MONTH_FEATURE
- **WHEN** 运行 Feature Selection
- **THEN** 系统 SHALL 将跨月合约结构特征列作为 `mandatory_state_features` 传入
- **AND** 这些特征 SHALL 不参与 Hard Filter、Stability Filter、Composite Score、Correlation Filter 或 Blacklist 过滤
- **AND** 这些特征 SHALL 强制保留在 `state_features.npy` 中
- **AND** 当 Feature Blacklist 包含任一跨月合约结构特征列时系统 SHALL fail-fast

#### Scenario: Scale Save 对 CROSS_MONTH_FEATURE 执行 Rolling Robust Scaling
- **WHEN** 运行 Scale Save
- **THEN** 跨月合约结构特征 SHALL 与其他连续 State Feature 一起参与 Rolling Robust Scaling 的 fit、transform 和 clip
- **AND** 跨月合约结构特征 SHALL NOT 被加入 `passthrough_features`
- **AND** Scale Manifest SHALL 将跨月合约结构特征记录为 scaled feature，而不是 passthrough state feature
