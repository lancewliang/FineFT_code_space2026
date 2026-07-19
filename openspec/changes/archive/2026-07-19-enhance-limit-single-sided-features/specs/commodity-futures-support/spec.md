## ADDED Requirements

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
