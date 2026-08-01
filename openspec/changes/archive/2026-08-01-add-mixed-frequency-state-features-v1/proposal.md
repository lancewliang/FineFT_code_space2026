# add-mixed-frequency-state-features-v1

## 背景与目标
当前商品期货特征工程已有目标频率 bar 特征、`Base_Time_feature`、`Cross-Month Term Structure Feature`、`Rolling Window Feature`、`Risk State Feature` 和 `Liquidity State Feature`。这些特征覆盖了日内结构、合约生命周期、跨月结构和目标频率滚动状态，但每个日内 bar 还缺少上一完整日和上一完整自然周的低频市场背景。

本变更引入第一版 `Mixed-frequency State Feature`。v1 刻意收窄范围：日级仅使用上一 `TradingDay`，周级仅使用上一完整自然周；不实现日滑动窗口、周滑动窗口，也不实现当前日或当前周 period-to-date 特征。

低频基础数据先作为独立产物落盘：日基础数据一天一行，周基础数据一周一行。日基础数据与周基础数据由独立代码入口分别生成；日混频特征与周混频特征也由独立代码入口分别生成。最终 `MIXED_FREQUENCY_FEATURE` 仅合并已生成的日/周混频特征产物。

## 关键决策
- **上一周期语义**：任一目标频率 bar 只能看到上一完整 `TradingDay` 和上一完整自然周的统计。
- **自然周边界**：周级聚合按 `TradingDay` 所属自然周定义，而不是按 `Event Timestamp` 的自然日期或固定 5 个交易日窗口。
- **基础数据独立落盘**：先通过日基础数据入口生成 daily 表，通过周基础数据入口生成 weekly 表；daily 表一天一行，weekly 表一周一行。
- **特征独立生成**：日混频特征入口只消费 daily base，周混频特征入口只消费 weekly base；`MIXED_FREQUENCY_FEATURE` 只合并日/周混频特征文件。
- **v1 特征清单**：日级输出上一日收益、振幅、K 线形态、成交量、成交额、OpenInterest 变化和 turnover rate；周级输出上一周收益、振幅、K 线形态、成交量、成交额、OpenInterest 变化和 turnover rate。
- **State Feature 候选**：混频特征 join 到 future/state candidate feature frame，不进入 Reward/Execution 列。
- **Fail-fast 与有限值**：缺失必要输入列应 fail-fast；除早期无上一周期等明确 fallback 场景外，输出特征必须为有限数。

## 验收标准
- 对任一 `TradingDay = D` 的目标频率 bar，日级混频特征来自 `D` 之前最近一个可用 `TradingDay`。
- 对任一自然周 `W` 内的目标频率 bar，周级混频特征来自 `W` 之前最近一个完整自然周。
- 周一和周中 bar 都不能看到当前自然周聚合结果。
- 输出包含 v1 约定的上一日和上一周列，且列名清晰区分 `prev_day_*` 与 `prev_week_*`。
- `MIXED_FREQUENCY_BASE` 的 daily 输出每个 `TradingDay` 一行，weekly 输出每个自然周一行。
- Daily merge 将混频特征 join 到 future/state candidate feature frame，Reward/Execution frame 不包含这些列。
- 无上一日或无上一周的早期样本输出 deterministic finite fallback，不产生 NaN/Inf。
