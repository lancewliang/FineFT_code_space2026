# fix-commodity-quote-session-gap-validation

## 背景与目标

商品期货主流程按 `TradingDay` 过滤日期范围，但事件时间戳使用真实的
`ActionDay + UpdateTime`。因此 `START_DATE=2025-11-01` 时，属于
`TradingDay=20251103` 的夜盘数据可以包含 `2025-10-31 21:00-23:00`
的事件时间。

当前 `downscale_quote_features()` 使用连续自然时间检查 quote bar 缺口。
当夜盘最后一个 bar 是 `2025-10-31 23:00:00`，下一段行情进入后续交易日
日盘时，代码会误报 `Target window has no quote snapshots:
2025-10-31 23:05:00`。这个时间处于非交易时段，不应被当成缺失 quote
snapshot。

目标是让商品 quote gap 校验理解交易 session：保留有效交易 session 内的
缺口 fail-fast，但不要把跨 session、跨自然日、周末或休市时间当成必须有
quote snapshot 的窗口。

## 用户场景

- 用户运行商品期货完整预处理，设置 `START_DATE=2025-11-01`、
  `END_DATE=2026-03-01`、`TARGET_FREQ=5min`、`SYMBOL=fu`，期望按
  `TradingDay` 左闭右开处理范围内交易日。
- 夜盘事件时间早于 `START_DATE` 但归属于范围内 `TradingDay` 时，系统应
  正常处理。
- 夜盘结束后到下一交易 session 开始前没有 quote snapshot 时，系统不应报
  自然时间缺口错误。
- 如果同一个有效交易 session 内缺少目标频率窗口的 quote snapshot，系统仍
  应报错以暴露数据质量问题。

## 设计方向

采用 session-aware gap validation。

在商品配置中为 `fu` 增加交易 session 描述，覆盖燃料油常规交易时段。quote
下采样继续使用现有 right-closed、right-labeled bar 语义；完成聚合后，
`downscale_quote_features()` 根据目标频率和交易 session 判断相邻输出 bar
是否位于同一 session：

- 同一 session 内相邻 bar 间隔超过 `target_freq` 时，继续抛出明确异常。
- 跨 session、跨自然日、跨周末或休市间隔，不做连续自然时间缺口校验。
- 输入 `second_df` 为空时继续 fail-fast，避免整段无 quote 数据静默通过。

不把非交易时间窗口补齐为合成 quote bar，也不简单删除全部 gap 校验。

## 关键决策

- `START_DATE` / `END_DATE` 继续基于 `TradingDay`，不改为事件自然日过滤。
- 夜盘事件时间早于 `START_DATE` 是合法行为，继续保留真实
  `ActionDay + UpdateTime` timestamp。
- quote gap 校验从连续自然时间改为同一交易 session 内连续性校验。
- 有效交易 session 内缺口仍是数据质量错误，应 fail-fast。
- 跨 session 的自然时间缺口不是 quote 缺失，不应报错。
- 本次不引入合成 quote snapshot，不做前向填充非交易时段。

## 范围边界

**包含：**

- 为商品配置增加燃料油交易 session 信息。
- 调整 `downscale_quote_features()` 的 quote gap 校验逻辑。
- 更新商品 downscale 测试，覆盖夜盘结束后非交易时段不误报。
- 保留 session 内 quote 缺口 fail-fast 的回归测试。

**不包含（本次）：**

- 改变主力合约选择、`TradingDay` 日期范围或 `ActionDay + UpdateTime`
  时间戳语义。
- 改变 base feature、orderbook、derivative reference 的下采样输出。
- 接入交易所节假日完整日历。
- 将 `COMMODITY_QUOTE_FEATURE` 接入主训练合并链路。
- 为非交易时间生成填充 bar 或前向填充 quote feature。

## 验收标准

- [ ] `START_DATE=2025-11-01` 且范围内 `TradingDay` 包含
      `2025-10-31` 夜盘事件时，quote 下采样不因 `2025-10-31 23:05:00`
      这类非交易时段窗口报错。
- [ ] 同一有效交易 session 内缺少 `target_freq` 对应 quote window 时，
      仍抛出 `Target window has no quote snapshots` 类错误。
- [ ] 输入 quote snapshot 为空时仍 fail-fast。
- [ ] 现有商品 downscale 测试通过。
- [ ] 使用 `conda activate finetf` 后运行相关 Python 测试命令通过。
