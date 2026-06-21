## ADDED Requirements

### Requirement: 商品期货 Polars 预处理兼容性
系统 SHALL 将 `data_preprocess/operator_futures/commodity` 商品期货核心预处理迁移到 Polars，并保持既有商品期货数据契约。

#### Scenario: 主力合约拼接输出兼容
- **WHEN** 商品期货主力合约拼接读取本地五档 CSV 文件
- **THEN** 系统使用 Polars 处理 CSV 读取、成交量计算、合格主力合约筛选、连续主力拼接和输出写入
- **AND** 输出继续包含 `main_contract`、`source_contract`、`source_file`、`main_contract_trading_day` 和 `main_contract_selection_reason` 元数据
- **AND** `TradingDay` 日归属和 `ActionDay + UpdateTime` 事件时间戳语义保持不变

#### Scenario: 商品 downscale 输出兼容
- **WHEN** 商品期货连续主力数据运行单日或按 `TradingDay` 下采样
- **THEN** 系统使用 Polars 生成 derivative reference、五档 orderbook、base features 和 quote features
- **AND** depth=5 输出不合成第 6 到第 25 档
- **AND** `LastPrice` 回退、funding 兼容列、Volume/Turnover 差分、tick rule 估计方向、右闭窗口聚合和 fail-fast 校验语义保持不变

#### Scenario: 商品 market_type 分支兼容
- **WHEN** `cross_section/create_feature.py`、`scale_describe_save/scale_save.py` 或 `feature_selection/ic_correlation.py` 以 `market_type=commodity_futures` 运行
- **THEN** 商品 reward/execution manifest、depth-aware feature generation、funding 关闭特征处理和 feature selection target 语义保持不变
- **AND** 输出列集合和列顺序继续满足商品期货现有 tests 和 downstream readers
