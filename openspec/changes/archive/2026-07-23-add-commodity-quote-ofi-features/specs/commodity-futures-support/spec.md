## ADDED Requirements

### Requirement: 商品期货五档 OFI row-window 特征
系统 SHALL 从连续五档 quote 快照计算标准五档 OFI，并按固定输入行数聚合输出。

#### Scenario: 五档相邻快照 OFI 明细
- **WHEN** 输入 quote 快照包含 `timestamp`、`BidPrice1` 到 `BidPrice5`、`AskPrice1` 到 `AskPrice5`、`BidVolume1` 到 `BidVolume5` 和 `AskVolume1` 到 `AskVolume5`
- **THEN** 系统 SHALL 按 `timestamp` 全局排序后比较每条快照与上一条快照
- **AND** 第一条快照的每档 OFI SHALL 为 `0`
- **AND** 对每个 bid 档位，价格上移时使用 `+ 当前 BidVolume`，价格不变时使用 `当前 BidVolume - 上一条 BidVolume`，价格下移时使用 `- 上一条 BidVolume`
- **AND** 对每个 ask 档位，价格下移时使用 `- 当前 AskVolume`，价格不变时使用 `-(当前 AskVolume - 上一条 AskVolume)`，价格上移时使用 `+ 上一条 AskVolume`
- **AND** 输出 SHALL 包含 `ofi_bid1` 到 `ofi_bid5`、`ofi_ask1` 到 `ofi_ask5`、`ofi_bid`、`ofi_ask` 和 `ofi`
- **AND** `ofi_bid` SHALL 等于 `ofi_bid1` 到 `ofi_bid5` 的和，`ofi_ask` SHALL 等于 `ofi_ask1` 到 `ofi_ask5` 的和，`ofi` SHALL 等于 `ofi_bid + ofi_ask`

#### Scenario: 固定 12 行聚合
- **WHEN** 用户使用默认 `window_rows=12` 生成五档 OFI 特征
- **THEN** 系统 SHALL 每 12 条按时间排序后的输入快照输出一行 OFI bar
- **AND** 每行输出的 `timestamp` SHALL 为该组内最后一条快照的 `timestamp`
- **AND** 每行输出的 `nquote` SHALL 为该组输入快照数量
- **AND** 每行输出的所有 OFI 明细列和汇总列 SHALL 为该组内逐快照 OFI 的求和
- **AND** 最后不足 12 条输入快照的尾组 SHALL 保留输出

#### Scenario: OFI 归一化特征
- **WHEN** 系统输出五档 OFI bar
- **THEN** 输出 SHALL 包含 `ofi_norm`、`ofi_bid_norm` 和 `ofi_ask_norm`
- **AND** `ofi_norm` SHALL 等于 `ofi / sum(BidVolume1-5 + AskVolume1-5)`，分母为同一 OFI bar 内所有输入快照的五档 bid 与 ask volume 合计
- **AND** `ofi_bid_norm` SHALL 等于 `ofi_bid / sum(BidVolume1-5)`，分母为同一 OFI bar 内所有输入快照的五档 bid volume 合计
- **AND** `ofi_ask_norm` SHALL 等于 `ofi_ask / sum(AskVolume1-5)`，分母为同一 OFI bar 内所有输入快照的五档 ask volume 合计
- **AND** 当任一归一化分母为 `0` 时，对应归一化输出 SHALL 为 `0`

#### Scenario: OFI 比较跨行窗口连续
- **WHEN** 第 13 条快照开始新的 12 行 OFI bar
- **THEN** 第 13 条快照的 OFI SHALL 使用第 12 条快照作为上一条快照计算
- **AND** 系统 MUST NOT 在固定行数窗口边界重置相邻快照状态

#### Scenario: OFI 输入 fail-fast
- **WHEN** OFI 输入没有任何 quote 快照
- **THEN** 系统 SHALL 报错并说明没有 quote snapshot
- **WHEN** OFI 输入缺少任一五档价格或数量必需列
- **THEN** 系统 SHALL 报错并列出缺失列名
- **WHEN** OFI 输入的任一五档价格或数量必需列存在 null
- **THEN** 系统 SHALL 报错并列出存在 null 的列名
- **WHEN** OFI 输入的任一五档价格或数量必需列存在 NaN、`inf` 或 `-inf`
- **THEN** 系统 SHALL 报错并列出存在非有限值的列名
- **WHEN** `window_rows <= 0`
- **THEN** 系统 SHALL 报错并说明 `window_rows` 必须为正数
