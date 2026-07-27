# commodity-futures-support Specification

## Requirements

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
