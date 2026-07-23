# add-commodity-quote-microstructure-features

## 背景与目标

当前商品期货 quote 特征已有时间窗口下采样和独立 row-window OFI 特征，但还缺少
一档队列压力和归一化价差类的独立 row-window 输出。

本变更新增独立的 quote microstructure 特征函数，用于输出 microprice 偏离、
relative spread 以及 spread 扩宽/收窄/持平计数。该函数不修改现有
`downscale_quote_features()` 的时间窗口语义，也不并入 OFI 输出。

## 用户场景

- 研究或训练流程需要使用一档买卖队列对下一跳方向的压力，而不是绝对价格。
- 跨品种、跨合约比较时，需要使用 `relative_spread = spread / mid` 这类归一化价差。
- 训练样本需要知道固定 quote 行窗口内 spread 是扩宽、收窄还是持平。

## 设计方向

新增独立函数 `downscale_quote_microstructure_features(second_df, window_rows=12)`。
输入 quote-like DataFrame 后，系统按 `timestamp` 排序，并使用一档价格和数量逐快照
计算：

- `spread = AskPrice1 - BidPrice1`
- `mid = (AskPrice1 + BidPrice1) / 2`
- `microprice = (AskPrice1 * BidVolume1 + BidPrice1 * AskVolume1) / (BidVolume1 + AskVolume1)`
- `microprice_pressure = (microprice - mid) / spread`
- `relative_spread = spread / mid`

逐快照 spread 变化方向通过相邻快照的 `spread.diff()` 判断：

- `spread_widen = spread.diff() > 0`
- `spread_narrow = spread.diff() < 0`
- `spread_flat = spread.diff() == 0`

第一条快照没有前序 spread，计入 `spread_flat`，保证每个窗口内
`spread_widen_count + spread_narrow_count + spread_flat_count == nquote`。

逐快照特征按固定 12 行窗口聚合。每 12 条连续输入行输出一行，`timestamp` 使用组内
最后一条快照时间，尾部不足 12 行的窗口保留。窗口输出精简列：

- `timestamp`
- `nquote`
- `mean_microprice_pressure`
- `mean_relative_spread`
- `spread_widen_count`
- `spread_narrow_count`
- `spread_flat_count`
- `spread_widen_ratio`

## 关键决策

- 新增独立 row-window 输出函数，不改动 `downscale_quote_features()`。
- 默认窗口大小为 `window_rows=12`，与现有 row-window OFI 默认值保持一致。
- `microprice_pressure` 和 `relative_spread` 只输出窗口均值，不输出 OHLC/std。
- 第一条 spread 变化计入 flat，保持计数闭合。
- 输入坏数据 fail-fast；合法零分母派生值输出 `0.0`，避免生成 `NaN` 或无穷大特征。

## 范围边界

**包含：**

- 新增独立 quote microstructure row-window 特征函数。
- 输出 microprice 偏离、归一化 spread 和 spread 变化计数/比例。
- 固定 12 行窗口聚合，保留不足 12 行的尾组。
- 对必要输入列的缺失、空输入、非法窗口大小、`NaN` 和无穷大 fail-fast。
- 对 `spread == 0`、`mid == 0` 或一档数量和为 0 的派生分母输出 `0.0`。

**不包含（本次）：**

- 不修改现有 `downscale_quote_features()` 的时间窗口输出语义。
- 不把 microstructure 特征并入 OFI 输出。
- 不输出 `microprice_pressure` 或 `relative_spread` 的 OHLC/std。
- 不使用 `target_freq` 时间窗口聚合。
- 不合成缺失的一档盘口价格或数量。

## 验收标准

- [ ] 给定覆盖不同一档价格和数量的 quote 快照，系统输出正确的
      `mean_microprice_pressure` 和 `mean_relative_spread`。
- [ ] 给定覆盖 spread 扩宽、收窄和持平的 quote 快照，系统输出正确的
      `spread_widen_count`、`spread_narrow_count`、`spread_flat_count` 和
      `spread_widen_ratio`，且三类计数之和等于 `nquote`。
- [ ] 给定 13 条 quote 输入和默认 `window_rows=12`，系统输出两行，第一行
      `nquote=12`，第二行 `nquote=1`，且 `timestamp` 均为各组最后一条快照时间。
- [ ] 输入为空、缺少任一必要列、或 `window_rows <= 0` 时，系统 fail-fast 并给出明确错误。
- [ ] 任一必要一档价格或数量列包含 `NaN`、`inf` 或 `-inf` 时，系统 fail-fast 并给出明确错误。
- [ ] 当派生计算遇到 `spread == 0`、`mid == 0` 或一档数量和为 0 时，对应归一化特征输出
      `0.0`，窗口输出不产生 `NaN` 或无穷大。
