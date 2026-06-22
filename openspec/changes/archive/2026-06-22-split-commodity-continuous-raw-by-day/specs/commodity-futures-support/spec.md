## MODIFIED Requirements

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

## ADDED Requirements

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
