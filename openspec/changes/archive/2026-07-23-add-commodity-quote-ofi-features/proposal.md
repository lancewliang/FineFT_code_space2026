# add-commodity-quote-ofi-features

## 背景与目标

当前商品期货 quote 下采样只包含报价变化次数、静态一档量不平衡，以及 spread/mid/price/size 的窗口统计。它没有实现标准 OFI，也没有基于相邻盘口快照按 bid/ask 价格上移、下移或不变来加减 size。

本变更新增五档 row-window OFI 特征，用于描述连续盘口快照之间的五档 order flow imbalance。OFI 不混入现有按时间窗口聚合的 quote 特征，避免改变 `downscale_quote_features()` 的 `target_freq` 语义。

## 用户场景

- 研究或训练流程需要使用基于盘口相邻快照变化的标准 OFI，而不是静态 volume imbalance。
- 输入 quote 行可能代表 1 分钟、5 分钟或 10 分钟等不同时间含义，需要按固定行数聚合 OFI。
- 调试 OFI 方向时，需要保留 bid/ask 分解项和五档明细列。

## 设计方向

新增独立的五档 OFI 计算路径。输入 quote-like DataFrame 后，系统按 `timestamp` 全局排序，使用 `BidPrice1-5`、`AskPrice1-5`、`BidVolume1-5`、`AskVolume1-5` 逐快照计算每档 bid/ask OFI。

每档 bid 口径：

- `bid_price` 上移：`+ 当前 bid_size`
- `bid_price` 不变：`当前 bid_size - 前一笔 bid_size`
- `bid_price` 下移：`- 前一笔 bid_size`

每档 ask 口径：

- `ask_price` 下移：`- 当前 ask_size`
- `ask_price` 不变：`-(当前 ask_size - 前一笔 ask_size)`
- `ask_price` 上移：`+ 前一笔 ask_size`

系统输出 `ofi_bid1` 到 `ofi_bid5`、`ofi_ask1` 到 `ofi_ask5`，并汇总得到 `ofi_bid`、`ofi_ask` 和 `ofi`。逐快照 OFI 按固定 `window_rows=12` 聚合，每 12 条连续输入行输出一行，`timestamp` 使用组内最后一条快照时间，OFI 列使用组内求和，`nquote` 记录组内行数。最后不足 12 行的尾组保留。

## 关键决策

- OFI 作为独立 row-window quote feature 输出，不改变现有时间窗口 quote 下采样语义。
- 相邻快照比较跨 12 行窗口边界连续进行，窗口边界不重置 OFI。
- 使用五档盘口明细并保留 bid/ask 分解项，便于排查方向来源。
- 第一条快照没有上一条可比对象，各档 OFI 置为 `0`。
- 五档缺列、空输入和非法 `window_rows` 直接 fail-fast，不静默填充深度缺失。

## 范围边界

**包含：**

- 新增独立五档 OFI 特征函数。
- 支持固定 12 行窗口聚合，保留不足 12 行的尾组。
- 输出每档 `ofi_bid*`、`ofi_ask*` 和汇总 `ofi_bid`、`ofi_ask`、`ofi`。
- 覆盖标准方向、固定行数聚合、跨窗口连续比较、缺列和空输入测试。

**不包含（本次）：**

- 不修改现有 `downscale_quote_features()` 的时间窗口输出语义。
- 不把 OFI 自动并入现有时间窗口 quote feature frame。
- 不实现按 `target_freq` 时间窗口聚合 OFI。
- 不合成缺失的二到五档盘口深度。

## 验收标准

- [ ] 给定覆盖 bid 上移、不变、下移和 ask 上移、不变、下移的五档快照，系统输出正确的 `ofi_bid1..5`、`ofi_ask1..5`、`ofi_bid`、`ofi_ask` 和 `ofi`。
- [ ] 给定 13 条 quote 输入和 `window_rows=12`，系统输出两行，第一行 `nquote=12`，第二行 `nquote=1`，且 `timestamp` 均为各组最后一条快照时间。
- [ ] 第 13 条快照的 OFI 使用第 12 条快照作为前序状态，不因固定行窗口边界重置。
- [ ] 输入为空、缺少任一五档价格或数量列、或 `window_rows <= 0` 时，系统 fail-fast 并给出明确错误。

## Amendments

### 2026-07-23: 新增 OFI 归一化特征

原因：raw OFI 直接携带盘口挂单量绝对尺度，训练或研究流程还需要相对量纲的 OFI 特征。

摘要：

- 新增 `ofi_norm = ofi / sum(BidVolume1-5 + AskVolume1-5)`。
- 新增 `ofi_bid_norm = ofi_bid / sum(BidVolume1-5)`。
- 新增 `ofi_ask_norm = ofi_ask / sum(AskVolume1-5)`。
- 分母使用同一个 12 行 row-window 内所有输入快照的对应五档 volume 合计。
- 分母为 `0` 时归一化特征输出 `0`，避免生成 inf/null。

### 2026-07-23: 拦截 OFI 输入 NaN 和无穷大

原因：OFI 和归一化 OFI 直接使用五档 price/volume，输入中的 NaN 或正负无穷会污染 raw OFI、分母和归一化输出。

摘要：

- OFI 输入校验新增非有限值检查。
- 任一五档价格或数量列包含 NaN、`inf` 或 `-inf` 时 fail-fast。
- 错误信息列出包含非有限值的列名。
