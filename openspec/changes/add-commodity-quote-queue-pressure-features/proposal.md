# add-commodity-quote-queue-pressure-features

## 背景与目标

当前商品期货 quote 的 row-window 特征已经覆盖微价压力和 spread 变化，但没有直接输出“价格不变时，一档 size 上升/下降”的队列行为，也没有把涨跌停单边盘口和空侧状态作为窗口比例特征输出。

本变更要补齐这两类信号：

- 挂单补充 / 撤单压力
- 单边 / 涨跌停盘口状态比例

特征继续走 `downscale_quote_microstructure_features()` 路径，按 `window_rows=12` 聚合，不使用 `target_freq` 时间窗口。

## 用户场景

- 研究者希望在秒级 quote 的 12 条窗口里，观察一档队列是补单还是撤单，而不是只看绝对 size。
- 训练流程需要把涨跌停单边盘口、bid/ask 空侧状态变成可直接消费的窗口特征。
- 下游希望特征输出稳定，不能出现 `NaN`、无穷大或静默修补后的非法值。

## 设计方向

在 `data_preprocess/operator_futures/commodity/downscale.py` 中扩展现有 `downscale_quote_microstructure_features(second_df, window_rows=12)`，保持函数入口不变，新增私有 helper 计算逐行事件和状态，再沿用现有 row-window 聚合。

逐行事件定义只看一档盘口：

- `bid_refill_count`: `BidPrice1` 与前一条相同且 `BidVolume1` 上升
- `bid_deplete_count`: `BidPrice1` 与前一条相同且 `BidVolume1` 下降
- `ask_refill_count`: `AskPrice1` 与前一条相同且 `AskVolume1` 上升
- `ask_deplete_count`: `AskPrice1` 与前一条相同且 `AskVolume1` 下降

窗口输出：

- 上述四个计数求和
- `queue_refill_imbalance = (bid_refill + ask_deplete - bid_deplete - ask_refill) / total_queue_events`
- `total_queue_events = bid_refill + bid_deplete + ask_refill + ask_deplete`

状态比例定义：

- `limit_up_single_sided_ratio`
- `limit_down_single_sided_ratio`
- `bid_side_empty_ratio`
- `ask_side_empty_ratio`

比例统一用窗口内 `nquote` 作为分母；分母为 0 时输出 `0.0`。第一条快照没有前值，不计入任何队列事件。

## 关键决策

- 不新增独立 pipeline，直接扩展 `downscale_quote_microstructure_features()`。
- 不使用 `target_freq`，只使用 `window_rows=12` 的 row-window。
- 队列行为只统计一档盘口，不扩展到 1-5 档汇总。
- 单边盘口状态复用现有涨跌停判定口径，不改修复逻辑。
- 所有新增归一化和比例都必须使用安全除法，禁止输出 `NaN` / `inf`。
- 输入若含非有限数值，继续 fail-fast。

## 范围边界

**包含：**

- 扩展 `downscale_quote_microstructure_features()` 输出队列 refill/deplete 计数。
- 新增 `queue_refill_imbalance`。
- 新增涨跌停单边和 bid/ask 空侧比例。
- 覆盖非法数值、零分母、窗口边界和单边盘口状态的测试。

**不包含（本次）：**

- 不修改 `downscale_quote_features()` 的时间窗口逻辑。
- 不合并现有的 quote depth imbalance 变更。
- 不把队列行为扩展成多档盘口汇总。
- 不修改现有 quote gap 校验和单边盘口修复逻辑。

## 验收标准

- [ ] `downscale_quote_ofi_features(second_df, window_rows=12)` 输出 `bid_refill_count`、`bid_deplete_count`、`ask_refill_count`、`ask_deplete_count`。
- [ ] 输出 `queue_refill_imbalance`，且分母为 0 时返回 `0.0`。
- [ ] 输出 `limit_up_single_sided_ratio`、`limit_down_single_sided_ratio`、`bid_side_empty_ratio`、`ask_side_empty_ratio`。
- [ ] 所有新增比例和归一化输出都不产生 `NaN` / `inf`。
- [ ] 输入缺少必要 quote 列或包含非有限值时，系统 fail-fast。
- [ ] 现有 `nquote` 和 OFI 相关输出保持不变。
