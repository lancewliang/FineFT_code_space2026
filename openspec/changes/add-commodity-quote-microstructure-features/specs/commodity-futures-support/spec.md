## ADDED Requirements

### Requirement: 商品期货 quote microstructure row-window 特征
系统 SHALL 从一档 quote 快照派生独立的 row-window microstructure 特征，不改变现有时间窗口 quote 下采样输出。

#### Scenario: 独立固定行窗口输出
- **WHEN** 系统调用 `downscale_quote_microstructure_features(second_df, window_rows=12)`
- **THEN** 系统 SHALL 按 `timestamp` 排序输入 quote 快照
- **AND** 系统 SHALL 每 12 条连续输入行输出一行 row-window 特征
- **AND** 系统 SHALL 保留不足 12 条的尾部窗口
- **AND** 输出 `timestamp` SHALL 使用窗口内最后一条 quote 快照时间
- **AND** 输出 SHALL 包含 `nquote`
- **AND** 系统 SHALL NOT 修改 `downscale_quote_features()` 的时间窗口输出语义
- **AND** 系统 SHALL NOT 将 microstructure 特征并入 OFI 输出

#### Scenario: microprice pressure 与 relative spread
- **WHEN** 输入 quote 快照包含 `BidPrice1`、`AskPrice1`、`BidVolume1` 和 `AskVolume1`
- **THEN** 系统 SHALL 对每条快照计算 `spread = AskPrice1 - BidPrice1`
- **AND** 系统 SHALL 对每条快照计算 `mid = (AskPrice1 + BidPrice1) / 2`
- **AND** 系统 SHALL 对每条快照计算 `microprice = (AskPrice1 * BidVolume1 + BidPrice1 * AskVolume1) / (BidVolume1 + AskVolume1)`
- **AND** 系统 SHALL 对每条快照计算 `microprice_pressure = (microprice - mid) / spread`
- **AND** 系统 SHALL 对每条快照计算 `relative_spread = spread / mid`
- **AND** row-window 输出 SHALL 包含 `mean_microprice_pressure`
- **AND** row-window 输出 SHALL 包含 `mean_relative_spread`
- **AND** row-window 输出 SHALL NOT 包含 `microprice_pressure` 或 `relative_spread` 的 OHLC 或 std 统计列

#### Scenario: spread 变化计数与比例
- **WHEN** row-window 内包含 quote 快照
- **THEN** 系统 SHALL 使用相邻快照的 `spread.diff()` 判断 spread 变化方向
- **AND** `spread.diff() > 0` SHALL 计入 `spread_widen_count`
- **AND** `spread.diff() < 0` SHALL 计入 `spread_narrow_count`
- **AND** `spread.diff() == 0` SHALL 计入 `spread_flat_count`
- **AND** 第一条快照没有前序 spread 时 SHALL 计入 `spread_flat_count`
- **AND** `spread_widen_count + spread_narrow_count + spread_flat_count` SHALL 等于 `nquote`
- **AND** `spread_widen_ratio` SHALL 等于 `spread_widen_count / nquote`

#### Scenario: 输入结构 fail-fast
- **WHEN** microstructure 特征输入为空
- **THEN** 系统 SHALL fail-fast
- **AND** 错误信息 SHALL 说明输入没有 quote snapshots
- **WHEN** `window_rows <= 0`
- **THEN** 系统 SHALL fail-fast
- **AND** 错误信息 SHALL 说明 `window_rows` 必须为正数
- **WHEN** 输入缺少 `timestamp`、`BidPrice1`、`AskPrice1`、`BidVolume1` 或 `AskVolume1` 中任一必要列
- **THEN** 系统 SHALL fail-fast
- **AND** 错误信息 SHALL 列出缺失列

#### Scenario: 输入非有限值 fail-fast
- **WHEN** `BidPrice1`、`AskPrice1`、`BidVolume1` 或 `AskVolume1` 任一列包含 `NaN`、`inf` 或 `-inf`
- **THEN** 系统 SHALL fail-fast
- **AND** 错误信息 SHALL 说明 microstructure 输入列包含非有限值
- **AND** 系统 SHALL NOT 生成包含 `NaN`、`inf` 或 `-inf` 的 microstructure 输出

#### Scenario: 派生零分母输出中性值
- **WHEN** `BidVolume1 + AskVolume1 == 0`
- **THEN** 对应快照的 `microprice_pressure` SHALL 为 `0.0`
- **WHEN** `spread == 0`
- **THEN** 对应快照的 `microprice_pressure` SHALL 为 `0.0`
- **WHEN** `mid == 0`
- **THEN** 对应快照的 `relative_spread` SHALL 为 `0.0`
- **AND** row-window 输出 SHALL NOT 包含 `NaN`、`inf` 或 `-inf`
