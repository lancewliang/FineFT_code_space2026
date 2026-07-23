# add-commodity-quote-depth-imbalance-features

## 背景与目标

当前商品期货 quote 下采样的盘口压力只使用一档数量：
`BidVolume1 -> bid_amount`、`AskVolume1 -> ask_amount`，并输出
`imbalance_volume = (bid_amount - ask_amount) / (bid_amount + ask_amount)`。
这没有覆盖 `BidVolume2-5` / `AskVolume2-5` 在 quote 时间窗口内的压力信息。

下游 `base_feature_util.py` 中的 `imblance_volume_oe` 是基于
`DOWNSCALE_ORDERBOOK_25` 每个 downscaled 快照计算的全深度单点特征，不是
`second_df` 窗口内的 `imbalance_1/3/5` 序列聚合，也没有对这些压力序列输出
OHLC、TWAP、AWAP 或 STD。

本变更扩展 `downscale_quote_features()`，在保留现有 `imbalance_volume` 兼容列的
前提下，新增多档盘口压力特征 `imbalance_1`、`imbalance_3` 和 `imbalance_5`，
并对每个压力序列输出窗口统计。

## 用户场景

- 研究或训练流程需要比较一档、三档、五档盘口压力，而不是只看一档压力。
- quote 特征需要在 `target_freq` 时间窗口内聚合原始秒级快照的多档压力序列。
- 下游仍依赖旧的 `imbalance_volume` 字段，新增特征不能破坏已有数据消费。

## 设计方向

采用原位扩展方案：新增私有 helper，但仍由 `downscale_quote_features()` 输出特征。

新增 `_depth_imbalance_expr(depth: int) -> pl.Expr`，负责生成某个盘口深度的
Polars 表达式：

`imbalance_k = (sum(BidVolume1..k) - sum(AskVolume1..k)) / (sum(BidVolume1..k) + sum(AskVolume1..k))`

新增 `_quote_window_stat_aggs(names: list[str], std_names: set[str] | None = None) -> list[pl.Expr]`，
负责生成窗口统计表达式。默认输出现有 quote 统计：
`open/high/low/close/awap/twap`；对 `std_names` 中的字段额外输出 `std_*`。

`downscale_quote_features()` 在 `_resample()` 前新增逐快照字段：

- `imbalance_1`
- `imbalance_3`
- `imbalance_5`

窗口输出新增：

- `open/high/low/close/awap/twap/std_imbalance_1`
- `open/high/low/close/awap/twap/std_imbalance_3`
- `open/high/low/close/awap/twap/std_imbalance_5`

旧字段 `imbalance_volume` 继续保留，并补充 `std_imbalance_volume`，使其与
`imbalance_1` 的统计集合保持一致。`twap` 和 `awap` 暂时沿用现有 quote 逻辑，
均为简单均值。

输入中的 `NaN` 或无穷大 volume 值应 fail-fast，避免坏数据被静默聚合。若某条
快照的压力分母为 0 或为空，则对应 `imbalance_k` 输出 `0.0`，表示中性盘口压力，
防止生成 `NaN` 或无穷大结果。

## 关键决策

- 直接扩展 `downscale_quote_features()`，不新增独立 pipeline 输出。
- 保留旧 `imbalance_volume` 兼容列；新增 `imbalance_1` 与其数值等价但命名纳入多档体系。
- `imbalance_1/3/5` 使用原始 quote 快照的前 1、3、5 档 bid/ask volume 求和。
- 新增 helper 只服务本函数，不引入跨模块抽象或配置。
- `twap` 和 `awap` 暂时保持现有简单均值语义。
- 对输入 `NaN` 或无穷大 fail-fast；对零分母输出 `0.0`，避免非法数值进入窗口聚合。
- 不复用 `time_operator` 中的 rolling 逻辑；该模块消费已生成特征，不负责 quote 下采样聚合。

## 范围边界

**包含：**

- 扩展 `downscale_quote_features()` 输出 `imbalance_1/3/5`。
- 为 `imbalance_1/3/5` 输出 `open/high/low/close/twap/awap/std`。
- 为旧 `imbalance_volume` 补充 `std_imbalance_volume`。
- 新增私有 helper 组织多档压力公式和 quote 窗口统计表达式。
- 覆盖多档压力公式、零分母、非法 volume、缺失深度列和兼容字段的测试。

**不包含（本次）：**

- 不修改 `time_operator` 的 rolling 特征逻辑。
- 不修改 `base_feature_util.py` 的单点 `imblance_volume_oe`。
- 不把多档压力改造成 OFI；OFI 属于独立 row-window 变更。
- 不改变现有 quote `target_freq` 时间窗口语义。
- 不合成缺失的二到五档盘口深度。

## 验收标准

- [ ] 给定包含 `BidVolume1-5` 和 `AskVolume1-5` 的 quote 快照，输出包含
      `imbalance_1`、`imbalance_3` 和 `imbalance_5` 的
      `open/high/low/close/awap/twap/std` 窗口统计列。
- [ ] `imbalance_1` 与旧 `imbalance_volume` 使用相同一档数量公式，窗口统计数值一致；
      旧 `imbalance_volume` 输出列继续存在，并新增 `std_imbalance_volume`。
- [ ] 当某条快照的某个深度 bid/ask volume 总和为 0 或为空时，对应 `imbalance_k`
      为 `0.0`，窗口输出不产生 `NaN` 或无穷大。
- [ ] 当输入 volume 含 `NaN` 或无穷大时，`downscale_quote_features()` fail-fast 并给出明确错误。
- [ ] 当缺少 `BidVolume2-5` 或 `AskVolume2-5` 中任一必要列时，系统 fail-fast，不静默填充深度。
- [ ] 现有 quote gap 校验、涨跌停单边一档规范逻辑和 `nquote` 统计保持不变。
