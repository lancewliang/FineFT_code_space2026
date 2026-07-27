---
status: draft
source: user-request
---

# BASE_TIME_FEATURE State Features

## User Story

As a Commodity Preprocessing user
I want commodity futures BASE_TIME_FEATURE columns to be encoded as non-absolute State Feature inputs and always preserved through Feature Selection
So that the RL agent can learn intraday, session, contract-month, and expiry-context patterns without overfitting to absolute calendar or clock values

## Background

商品期货合约交易数据需要新增一组与 `BASE_FEATURE` 平级的 `BASE_TIME_FEATURE`：

| 特征语义 | 输入要求 | 说明 |
|---|---|---|
| 交易时间分钟 | 不能使用绝对分钟值 | 推荐使用周期编码，如 sin/cos 或相对 Trading Session 进度 |
| 交易时间段 | 不能使用绝对时段编号 | 使用编码表达早盘、下午盘、夜间盘，以及开盘半小时、收盘半小时等边界状态，而不是裸整数标签 |
| 合约交割月份 | 不能使用绝对月份值 | 从合约代码解析交割月份，并使用周期编码，避免模型记住裸月份编号 |
| 离合约结束还有多少天 | 不能使用绝对天数 | 在选择合约数据文件时从原始下载中该合约全部 `TradingDay` 找到真实最后交易日，后续归一化为合约生命周期剩余比例，或使用有界非绝对编码 |

这些 `BASE_TIME_FEATURE` 不是滚动窗口特征，属于必须保留的 `State Feature`。Feature Selection 的 Hard Filter、Stability Filter、Composite Score、Correlation Filter、Feature Blacklist 或其他后续过滤条件都不能删除它们；如果配置与必须保留规则冲突，应 fail-fast，而不是静默丢弃。Scale Save 必须保留这些列，但不对它们做 robust scaling。

`BASE_TIME_FEATURE` 产物路径使用 `PREPROCESS_DATASET/commodity-futures/BASE_TIME_FEATURE/{symbol}/{contract}/{target_freq}/{date}.feather`。daily merge 阶段将它 join 到 `FUTURE_FEATURE`，使后续 concat、Dataset Split、Feature Selection 和 Scale Save 自然消费这些列。上游 `Main Contract Summary` 每个 contract 记录提供 `last_trading_day` 和 `total_trading_day_count`，供 `BASE_TIME_FEATURE` 计算合约剩余生命周期比例。

`BASE_TIME_FEATURE` 输出列为：

| 列名 | 含义 |
|---|---|
| `trading_minute_progress` | 当前 timestamp 在所属 Trading Session 内按 session 持续分钟数归一化得到的进度 |
| `morning_session` | 早盘 one-hot 标记 |
| `afternoon_session` | 下午盘 one-hot 标记 |
| `night_session` | 夜间盘 one-hot 标记 |
| `is_opening_30m` | 当前 timestamp 是否位于所属 Trading Session 开盘半小时 |
| `is_closing_30m` | 当前 timestamp 是否位于所属 Trading Session 收盘半小时 |
| `contract_month_sin` | 合约交割月份的周期编码 sin 分量 |
| `contract_month_cos` | 合约交割月份的周期编码 cos 分量 |
| `contract_life_remaining_ratio` | 合约剩余生命周期比例 |

实现中使用唯一常量 `BASE_TIME_FEATURE_COLUMNS` 维护上述 9 个特征列名，生成器、Feature Selection、Scale Save 和测试共同引用，避免 mandatory 列、passthrough 列和实际输出列漂移。

## Scenarios

### 生成非绝对 BASE_TIME_FEATURE

Given:
- 商品期货合约数据包含 `timestamp`、`TradingDay` 和 `contract`

When:
- 运行商品期货特征工程流程

Then:
- `BASE_TIME_FEATURE` 以同日期同合约的 `BASE_FEATURE` timestamp 为主输入逐行生成
- 输出与 `BASE_FEATURE` 平级的 `BASE_TIME_FEATURE`，包含交易时间分钟、交易时间段、合约月份、合约剩余生命周期相关特征
- 输出路径为 `PREPROCESS_DATASET/commodity-futures/BASE_TIME_FEATURE/{symbol}/{contract}/{target_freq}/{date}.feather`
- 输出文件保留 `timestamp` 用于 daily merge 强校验和 join
- 输出文件不包含 `contract`、`TradingDay` 或其他非特征元数据列
- 这些特征不以原始绝对值形式进入模型输入

### Daily merge 并入 BASE_TIME_FEATURE

Given:
- 某个交易日存在 `BASE_FEATURE`
- 同日期的 `BASE_TIME_FEATURE` 文件可能存在或不存在

When:
- 运行 daily merge

Then:
- 若 `BASE_TIME_FEATURE` 文件存在，则按 `timestamp` join 到 `FUTURE_FEATURE`
- 若 `BASE_TIME_FEATURE` 文件不存在，则静默跳过并按普通流程生成 `FUTURE_FEATURE`
- 后续 concat、Dataset Split、Feature Selection 和 Scale Save 不需要额外入口即可消费这些列

### Daily merge 时间戳不一致时 fail-fast

Given:
- 某个交易日存在 `BASE_FEATURE`
- 同日期的 `BASE_TIME_FEATURE` 文件存在
- 两个文件的 timestamp 集合不一致

When:
- 运行 daily merge

Then:
- 流程 fail-fast，并说明 `BASE_TIME_FEATURE` 与 `BASE_FEATURE` timestamp 不一致
- 不生成 timestamp 错位的 `FUTURE_FEATURE`

### 交易时间分钟使用可泛化编码

Given:
- 两条记录分别处于不同交易分钟

When:
- 生成交易时间分钟特征

Then:
- 特征表达为 `trading_minute_progress`
- `trading_minute_progress` 按所属 Trading Session 的持续分钟数归一化，不按全天交易分钟数归一化
- 不输出裸的分钟数、HHMM、Unix timestamp 或其他绝对时间值作为 State Feature

### 交易时间段使用业务时间段编码

Given:
- 一条记录属于某个 Trading Session
- 交易时间段需要区分早盘、下午盘、夜间盘
- 交易时间段还需要标识开盘半小时和收盘半小时

When:
- 生成交易时间段特征

Then:
- 基础时段表达为互斥 one-hot：`morning_session`、`afternoon_session`、`night_session`
- 开盘半小时和收盘半小时表达为可叠加的 0/1 标记：`is_opening_30m`、`is_closing_30m`
- `is_opening_30m` 和 `is_closing_30m` 按每个 Trading Session 独立计算，不按整个交易日首尾计算
- 不输出裸 session id 或具有错误大小关系的整数标签作为 State Feature

### 合约交割月份使用非绝对编码

Given:
- 合约代码包含交割月份，如 `fu2605`

When:
- 生成合约交割月份特征

Then:
- 特征表达为 `contract_month_sin` 和 `contract_month_cos`
- 不输出裸月份编号或当前交易日自然月作为 State Feature

### Main Contract Summary 记录合约生命周期元数据

Given:
- 原始下载中包含该合约全部交易日期

When:
- 选择合约数据文件

Then:
- 在 `Main Contract Summary` 的每个 contract 记录中写入 `last_trading_day`
- `last_trading_day` 来自该合约全部 `TradingDay` 的最大日期
- 在 `Main Contract Summary` 的每个 contract 记录中写入 `total_trading_day_count`
- `total_trading_day_count` 来自该合约全部不同 `TradingDay` 的计数
- 该日期不是主力合约窗口结束日，也不是样本结束日

### 合约剩余时间使用非绝对编码

Given:
- `Main Contract Summary` 中该合约记录包含 `last_trading_day` 和 `total_trading_day_count`
- 当前记录属于某个合约的交易日

When:
- 生成离合约结束还有多少天的特征

Then:
- 特征基于 `last_trading_day` 计算剩余交易日数量，且包含当前交易日
- 特征输出为 `contract_life_remaining_ratio = remaining_trading_days / total_trading_day_count`
- 不输出未归一化的绝对剩余天数作为 State Feature

### 最后交易日仍有正剩余生命周期比例

Given:
- 当前记录的 `TradingDay` 等于该合约 `last_trading_day`
- `total_trading_day_count` 大于 0

When:
- 生成 `contract_life_remaining_ratio`

Then:
- `remaining_trading_days` 按包含当前交易日计算为 1
- `contract_life_remaining_ratio = 1 / total_trading_day_count`

### Feature Selection 通过 CLI 参数接收必须保留 State Feature

Given:
- 通过 CLI 参数 `--mandatory_state_features` 传入必须保留特征列表（如 `BASE_TIME_FEATURE_COLUMNS`）

When:
- Feature Selection 执行 Hard Filter、Stability Filter、Composite Score、Correlation Filter 和其他过滤步骤

Then:
- 传入的 mandatory 特征不参与 IC、RankIC、CatBoost、Permutation Importance、Sharpe 或相关性过滤等指标计算
- 最终 `state_features.npy` 必须包含全部 mandatory 特征
- 普通 Feature Selection 选出的特征排在前面，mandatory 特征按传入顺序追加在后
- 如果普通选择结果中已经包含某个 mandatory 特征，则最终列表去重，并只在 mandatory 追加位置保留一次
- Feature Selection Manifest 记录 `mandatory_state_features`，列出全部传入的 mandatory 特征列

### Feature Blacklist 不能删除必须保留特征

Given:
- 用户配置的 `feature_blacklist` 包含某个必须保留的 BASE_TIME_FEATURE

When:
- Feature Selection 运行

Then:
- 流程 fail-fast，并说明 blacklist 与必须保留特征冲突
- 不生成缺失必须保留特征的 `state_features.npy`

### 过滤后输出保留 BASE_TIME_FEATURE 列

Given:
- Feature Selection 已生成最终 state feature 列表

When:
- 写出每个合约的过滤后 `df.feather`

Then:
- 每个输出文件都包含全部必须保留的 BASE_TIME_FEATURE 列

### Scale Save 通过 CLI 参数接收 Passthrough State Feature

Given:
- 通过 CLI 参数 `--passthrough_features` 传入无需缩放的特征列表（如 `BASE_TIME_FEATURE_COLUMNS`）
- Scale Save 读取过滤后的合约数据

When:
- 执行 train-only robust scaler 和最终保存

Then:
- 指定的 passthrough 特征列出现在最终保存的 state feature 数据中
- Passthrough 特征列保持输入编码值，不参与 robust scaler 拟合、transform 或 clip
- Scale Manifest 记录 `passthrough_state_features`，列出全部 passthrough 特征列
- 其他非 passthrough 的 state feature 仍按现有规则缩放并裁剪
