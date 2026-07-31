# commodity-futures-feature-engineering Specification

## ADDED Requirements

### Requirement: 商品期货 Mixed-frequency State Feature v1
系统 SHALL 生成第一版 `Mixed-frequency State Feature`，将上一完整 `TradingDay` 和上一完整自然周的低频市场状态 join 到目标频率 bar 的 State Feature 候选列。

#### Scenario: 低频基础数据独立落盘
- **WHEN** 生成商品期货 `MIXED_FREQUENCY_BASE`
- **THEN** daily 基础数据每个 `TradingDay` 输出一行
- **AND** weekly 基础数据每个自然周输出一行
- **AND** daily 基础数据 SHALL 由独立日基础数据生成入口生成
- **AND** weekly 基础数据 SHALL 由独立周基础数据生成入口生成
- **AND** daily 与 weekly 基础数据包含 open、high、low、close、volume、tradeval、open_interest_first 和 open_interest_last
- **AND** weekly 基础数据的自然周归属由 `TradingDay` 决定
- **AND** 日混频特征 SHALL 由独立日混频特征入口消费 daily 基础数据生成
- **AND** 周混频特征 SHALL 由独立周混频特征入口消费 weekly 基础数据生成
- **AND** `MIXED_FREQUENCY_FEATURE` SHALL 合并已落盘的日混频特征与周混频特征

#### Scenario: 上一日特征列名与语义
- **WHEN** 生成商品期货 `Mixed-frequency State Feature`
- **THEN** 输出日级列：`prev_day_return`、`prev_day_range_pct`、`prev_day_body_pct`、`prev_day_upper_shadow_pct`、`prev_day_lower_shadow_pct`、`prev_day_volume`、`prev_day_tradeval`、`prev_day_open_interest_change` 和 `prev_day_turnover_rate`
- **AND** 所有 `prev_day_*` 列仅使用上一完整 `TradingDay` 的数据计算
- **AND** 不使用当前 `TradingDay` 的未完成或完整日终统计

#### Scenario: 上一自然周特征列名与语义
- **WHEN** 生成商品期货 `Mixed-frequency State Feature`
- **THEN** 输出周级列：`prev_week_return`、`prev_week_range_pct`、`prev_week_body_pct`、`prev_week_volume`、`prev_week_tradeval`、`prev_week_open_interest_change` 和 `prev_week_turnover_rate`
- **AND** 所有 `prev_week_*` 列仅使用上一完整自然周的数据计算
- **AND** 自然周归属由 `TradingDay` 决定
- **AND** 不使用当前自然周的未完成或完整周终统计

#### Scenario: Mixed-frequency Visibility Rule
- **WHEN** 目标频率 bar 属于 `TradingDay = D`
- **THEN** 日级混频特征来自 `D` 之前最近一个可用 `TradingDay`
- **AND** 周级混频特征来自 `D` 所属自然周之前最近一个完整自然周
- **AND** 周一和周中 bar 均不得看到当前自然周聚合结果

#### Scenario: v1 不生成低频滑动窗口和 period-to-date 特征
- **WHEN** 生成商品期货 `Mixed-frequency State Feature` v1
- **THEN** 系统 SHALL NOT 生成日滑动窗口特征
- **AND** 系统 SHALL NOT 生成周滑动窗口特征
- **AND** 系统 SHALL NOT 生成当前日 period-to-date 特征
- **AND** 系统 SHALL NOT 生成当前周 period-to-date 特征

#### Scenario: Daily Merge join Mixed-frequency State Feature
- **WHEN** daily merge 接收到 `Mixed-frequency State Feature`
- **THEN** 按 `timestamp` 将混频特征 join 到 future/state candidate feature frame
- **AND** Reward/Execution frame 不包含任何 `prev_day_*` 或 `prev_week_*` 混频特征列
- **AND** timestamp 不一致或缺失必要混频列时 fail-fast

#### Scenario: 有限值与早期样本 fallback
- **WHEN** 输入数据存在首个 `TradingDay`、首个自然周或非正 OpenInterest 分母等无法计算上一周期比率的场景
- **THEN** 输出 deterministic finite fallback
- **AND** 写出前不得包含 NaN、Inf 或非有限混频特征值
- **AND** 缺失必要 OHLCV/OpenInterest 输入列时 fail-fast
