---
status: draft
source: user-request
---

# Risk and Liquidity State Features

## User Story

As a Commodity Preprocessing user
I want commodity futures risk and liquidity State Feature columns generated from 5min market data
So that the RL agent can observe volatility regime, trading activity, and open-interest pressure when making trading decisions

## Background

商品期货需要新增一组风险与流动性 State Feature。这些特征不同于 `Base_Time_feature`：它们应作为普通 candidate state feature 进入后续 Feature Selection 和 Scale Save。

组件边界按是否需要历史窗口划分：

| 特征类型 | 归属组件 | 说明 |
|---|---|---|
| 滚动窗口特征 | `data_preprocess/operator_futures/time_operator/` | 需要跨 bar 历史窗口，基于已合并的 5min 特征计算 |
| 非滚动特征 | `data_preprocess/operator_futures/commodity/downscale.py` | 可在下采样窗口内直接由秒级快照或当前 bar 聚合得到 |

本故事列出的 10 个特征均需要历史窗口，因此归属 `Rolling Window Feature Generator`。`commodity/downscale.py` 只负责为这些滚动特征提供必要的基础列，例如 `open/high/low/close/volume/tradeval/open_interest`。

`open_interest` 由 `commodity/downscale.py` 在 `BASE_FEATURE` 中输出，语义为当前下采样窗口内最后一条秒级快照的持仓量。原始秒级输入字段要求为 `OpenInterest`，下采样输出列统一为 snake_case `open_interest`。缺少 `OpenInterest` 时在 downscale 阶段 fail-fast。`time_operator` 后续只消费 5min bar 级 `open_interest`，不回读原始秒级文件。

`historical_volatility_{window}` 的日化系数使用 `sqrt(bars_per_day)`。`bars_per_day` 根据品种 `CommodityConfig.trading_sessions` 的总交易分钟数和 `target_freq` 推导，不能硬编码为 24 小时市场的 bar 数。

`relative_amount_{window}` 的输入使用现有 `BASE_FEATURE.tradeval`。`amount` 只作为业务概念，不新增重复的 `amount` 基础列。

### 风险状态特征

这些特征复用 `time_operator --windows` 参数，输出列名格式为 `{feature_name}_{window}`。

| 基础列名 | 输出示例 | 概念 | 公式 |
|---|---|---|
| `atr_pct` | `atr_pct_12` | 平均真实波幅相对收盘价比例 | `TR_t = max(high_t - low_t, abs(high_t - close_{t-1}), abs(low_t - close_{t-1}))`; `atr_pct_t = mean(TR, N) / close_t * 100` |
| `historical_volatility` | `historical_volatility_12` | 历史收益率标准差，按 bar 数日化 | `r_t = ln(close_t / close_{t-1})`; `historical_volatility_t = std(r, N) * sqrt(bars_per_day)` |
| `rolling_volatility` | `rolling_volatility_12` | 指数加权近期收益率波动率 | `rolling_volatility_t = ewm_std(r, span=N)` |
| `parkinson_volatility` | `parkinson_volatility_12` | 基于 high/low 振幅的波动率估计 | `sqrt(mean(ln(high / low)^2, N) / (4 * ln(2)))` |
| `garman_klass_volatility` | `garman_klass_volatility_12` | 基于 open/high/low/close 的波动率估计 | `sqrt(max(mean(0.5 * ln(high / low)^2 - (2 * ln(2) - 1) * ln(close / open)^2, N), 0))` |
| `realized_volatility` | `realized_volatility_12` | 窗口内已实现波动率 | `sqrt(sum(r^2, N))` |

### 流动性状态特征

这些特征同样复用 `time_operator --windows` 参数，输出列名格式为 `{feature_name}_{window}`。

| 基础列名 | 输出示例 | 概念 | 公式 |
|---|---|---|
| `relative_volume` | `relative_volume_12` | 当前成交量相对当前及过去窗口均量倍数 | `volume_t / mean(volume_{t-N+1..t})` |
| `relative_amount` | `relative_amount_12` | 当前成交额相对当前及过去窗口均额倍数 | `tradeval_t / mean(tradeval_{t-N+1..t})` |
| `relative_open_interest` | `relative_open_interest_12` | 当前持仓量相对当前及过去窗口平均持仓量倍数 | `open_interest_t / mean(open_interest_{t-N+1..t})` |
| `open_interest_change_ratio` | `open_interest_change_ratio_12` | 持仓量相对 N 根 bar 前的变化率 | 若 `open_interest_{t-N} > 0`，则 `(open_interest_t - open_interest_{t-N}) / open_interest_{t-N}`；否则输出 `0` |

零分母、非法价格、非有限值或缺失必需输入必须 fail-fast 或输出明确的有限兜底值；不能让 `NaN`、`inf` 或 `-inf` 进入 State Feature。

## Scenarios

### 生成风险状态特征

Given:
- 商品期货 5min `BASE_FEATURE` 包含 `timestamp`、`open`、`high`、`low`、`close`
- 历史窗口内有足够 bar 可计算风险状态特征

When:
- 运行商品期货特征工程流程

Then:
- 对每个配置窗口输出 `atr_pct_{window}`
- 对每个配置窗口输出 `historical_volatility_{window}`
- 对每个配置窗口输出 `rolling_volatility_{window}`
- 对每个配置窗口输出 `parkinson_volatility_{window}`
- 对每个配置窗口输出 `garman_klass_volatility_{window}`
- 对每个配置窗口输出 `realized_volatility_{window}`
- 所有输出值为有限数值

### 生成流动性状态特征

Given:
- 商品期货 5min 特征数据包含 `timestamp`、`volume`、`tradeval`、`open_interest`
- 历史窗口内有足够 bar 可计算流动性状态特征

When:
- 运行商品期货特征工程流程

Then:
- 对每个配置窗口输出 `relative_volume_{window}`
- 对每个配置窗口输出 `relative_amount_{window}`
- 对每个配置窗口输出 `relative_open_interest_{window}`
- 对每个配置窗口输出 `open_interest_change_ratio_{window}`
- 所有输出值为有限数值

### 使用带窗口后缀的列名

Given:
- `time_operator --windows` 配置为 `12,20`

When:
- 生成风险与流动性状态特征

Then:
- 输出列名使用 `{feature_name}_{window}` 格式
- `atr_pct_12` 和 `atr_pct_20` 同时存在
- `relative_volume_12` 和 `relative_volume_20` 同时存在
- 不输出无窗口后缀的 `atr_pct` 或 `relative_volume`

### 沿用 time_operator warmup 裁剪行为

Given:
- `time_operator --windows` 包含多个窗口
- 最大窗口为 `max_window`

When:
- 生成风险与流动性状态特征

Then:
- 输出沿用现有 `time_operator` warmup 裁剪行为
- 前 `max_window + 1` 行不进入 `TIME_FEATURE`
- 不为 warmup 不足的风险或流动性状态特征强行填 0 后保留行

### 下采样输出持仓量基础列

Given:
- 商品期货秒级原始快照包含 `OpenInterest`
- 当前下采样窗口内存在至少一条快照

When:
- `commodity/downscale.py` 生成 `BASE_FEATURE`

Then:
- `BASE_FEATURE` 输出 `open_interest`
- `open_interest` 等于当前下采样窗口内最后一条秒级快照的持仓量
- `open_interest` 为有限数值

### 下采样缺少 OpenInterest 时 fail-fast

Given:
- 商品期货秒级原始快照缺少 `OpenInterest`

When:
- `commodity/downscale.py` 生成 `BASE_FEATURE`

Then:
- 流程 fail-fast，并说明缺少 `OpenInterest`
- 不生成缺少 `open_interest` 的 `BASE_FEATURE`

### 波动率特征使用过去窗口

Given:
- 当前 bar 的 timestamp 为 `t`
- 风险状态特征窗口大小为 `N`

When:
- 计算 timestamp=`t` 的风险状态特征

Then:
- 只使用 timestamp 小于等于 `t` 的 bar
- 不读取未来 bar
- 不跨合约混合滚动窗口
- 风险状态特征在 `data_preprocess/operator_futures/time_operator/` 中生成

### Historical Volatility 使用品种交易 Session 推导日化系数

Given:
- 商品期货品种配置包含 `CommodityConfig.trading_sessions`
- `target_freq` 为当前下采样频率

When:
- 计算 `historical_volatility_{window}`

Then:
- `bars_per_day` 根据该品种 Trading Session 总交易分钟数和 `target_freq` 推导
- `historical_volatility_{window}` 使用 `std(log_return, window) * sqrt(bars_per_day)`
- 不硬编码 24 小时市场的 bar 数

### 相对流动性特征使用过去窗口

Given:
- 当前 bar 的 timestamp 为 `t`
- 流动性状态特征窗口大小为 `N`

When:
- 计算 timestamp=`t` 的相对成交量、相对成交额和相对持仓量

Then:
- 分母来自同一合约当前及过去窗口内的均值
- 不读取未来 bar
- 不跨合约混合滚动窗口
- 流动性状态特征在 `data_preprocess/operator_futures/time_operator/` 中生成

### 缺少持仓量输入时 fail-fast

Given:
- 商品期货数据缺少 `open_interest`

When:
- 流程需要生成 `relative_open_interest` 或 `open_interest_change_ratio`

Then:
- 流程 fail-fast，并说明缺少 `open_interest`
- 不静默填 0
- 不跳过持仓量相关特征

### 非法价格或零分母不产生非有限值

Given:
- 输入数据中存在可能导致除零、对数非法或平方根非法的值

When:
- 计算风险与流动性状态特征

Then:
- 流程不能输出 `NaN`、`inf` 或 `-inf`
- 风险状态特征公式涉及的 `open`、`high`、`low`、`close` 如为非正数或非有限值，流程 fail-fast
- `garman_klass_volatility_{window}` 的平方根输入低于 0 时先裁剪到 0
- `open_interest_change_ratio_{window}` 遇到 `open_interest_{t-window} <= 0` 时输出 `0`
- 对无法合理兜底的输入 fail-fast，并指出异常列和 timestamp

### 特征进入后续 State Feature 流程

Given:
- 风险与流动性状态特征已经生成

When:
- 运行 concat、Dataset Split、Feature Selection 和 Scale Save

Then:
- 这些列作为普通 candidate state feature 被后续流程消费
- 这些列不加入 `mandatory_state_features`
- 这些列不作为 Scale Save passthrough feature
- Feature Selection 可以按现有指标筛选这些列
- Scale Save 对最终入选的这些列执行现有 train-only robust scaling 和裁剪
