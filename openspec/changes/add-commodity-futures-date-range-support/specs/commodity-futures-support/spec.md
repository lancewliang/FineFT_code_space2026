## MODIFIED Requirements

### Requirement: 主力合约拼接
系统 SHALL 从本地五档 CSV 文件构建商品期货连续主力序列，日归属使用 `TradingDay`，事件
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

### Requirement: 商品期货脚本入口支持日期范围
系统 SHALL 允许商品期货主流程通过 `START_DATE` / `END_DATE` 指定跨年的日期范围，并自动
生成该范围所需的连续主力原始文件与后续处理文件。

#### Scenario: 日期范围驱动主流程
- **WHEN** 用户运行 `main.sh` 并设置 `START_DATE=2023-01-01`、`END_DATE=2026-03-01`
- **THEN** 系统自动覆盖 2023、2024、2025 和 2026 的原始目录扫描与主力拼接
- **AND** 系统输出单条跨年连续主力文件供后续下采样使用

#### Scenario: 保持左闭右开语义
- **WHEN** 用户希望覆盖到 2026-02-28 的训练窗口
- **THEN** 系统继续使用左闭右开语义，要求 `END_DATE=2026-03-01`
- **AND** 脚本和日志文件名使用日期范围语义而不是单一年份语义

#### Scenario: YEAR 仅作兼容参数
- **WHEN** 用户继续传入 `YEAR`
- **THEN** 系统可以保留该参数作为兼容输入
- **AND** 主流程不再把单一年份作为唯一运行约束
