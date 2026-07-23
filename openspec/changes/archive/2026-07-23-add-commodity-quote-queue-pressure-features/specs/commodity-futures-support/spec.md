## ADDED Requirements

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
