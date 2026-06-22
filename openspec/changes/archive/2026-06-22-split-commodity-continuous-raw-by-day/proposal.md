# split-commodity-continuous-raw-by-day

## 背景与目标

商品期货当前 `CONTINUOUS_RAW` 会按日期范围生成单个连续主力大 CSV，例如 `fu_2023-01-01_2024-01-01.csv`。随着日期范围变长，该文件会变得过大，导致后续 downscale 入口必须一次性读取大文件后再按 `TradingDay` 分组，增加 IO 和内存峰值。

目标是将商品期货连续主力原始行情改为按 `TradingDay` 生成日文件，避免大区间单文件，并让后续 downscale 入口直接按日期范围逐日读取日文件。

## 用户场景

- 用户运行燃料油 `fu` 商品期货完整预处理，希望 `CONTINUOUS_RAW` 不再生成巨大区间文件，而是生成可逐日消费的连续主力行情文件。
- 用户重跑某个日期范围时，希望已有日文件可被覆盖，符合批处理重跑习惯。
- 用户允许日期范围内缺少部分非交易日或缺数据日，但希望日志明确记录哪些日期被跳过。

## 设计方向

采用按 `TradingDay` 日文件输出的方案，破坏旧的大连续文件 CLI 契约。

`stitch_main_contract.py` 从“写一个大 CSV 文件”改为“写一个日文件目录”。它接收 `--output_dir --start_date --end_date --symbol --commodity_name`，扫描原始多合约 CSV，按主力选择规则生成连续主力数据，并为每个有数据的交易日写出：

```text
PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/{symbol}/{YYYY-MM-DD}.csv
```

`downscale_continuous_by_trading_day.py` 从读取 `--input <big.csv>` 改为读取 `--input_dir --start_date --end_date`。它按左闭右开日期范围逐日查找 `CONTINUOUS_RAW/{symbol}/{YYYY-MM-DD}.csv`，存在则生成当天 downscale outputs，缺失则 warning 后跳过。

后续 cross-section、merge、concat、time feature、feature selection 仍按日期范围逐日处理，不改变下游 Feather 输出目录和数据契约。

## 关键决策

- `CONTINUOUS_RAW/{symbol}/{YYYY-MM-DD}.csv` 是唯一连续主力原始产物。
- 不再生成 `fu_start_end.csv` 这类大区间连续文件。
- `stitch_main_contract.py` 改用 `--output_dir`，移除旧 `--output <big.csv>` 语义。
- `downscale_continuous_by_trading_day.py` 改用 `--input_dir --start_date --end_date`，移除旧 `--input <big.csv>` 语义。
- 日期范围语义保持左闭右开：包含 `start_date`，不包含 `end_date`。
- stitch 阶段某日无原始合约数据时不生成文件，并记录 skipped date。
- downscale 阶段某日日文件缺失时 warning 后跳过，并记录 skipped date 汇总。
- 文件存在但内容坏时继续 fail-fast，不静默跳过。
- 目标日文件已存在时默认覆盖，并记录覆盖路径。

## 范围边界

**包含：**

- 调整商品期货连续主力生成逻辑，按 `TradingDay` 输出日文件。
- 调整 `stitch_main_contract.py` CLI 参数和写出行为。
- 调整 `downscale_continuous_by_trading_day.py` CLI 参数和读取行为。
- 更新 `fu_full_process.sh`，传递 `CONTINUOUS_RAW/{symbol}` 目录和日期范围，不再拼接大文件路径。
- 增加或更新测试，覆盖按日输出、缺失日跳过、已存在文件覆盖、旧大文件不再生成、shell 参数传递。

**不包含（本次）：**

- 不改变主力合约选择规则。
- 不改变 `TradingDay` 日归属和 `ActionDay + UpdateTime` 事件时间戳语义。
- 不改变 downscale 后的 Feather 输出目录、文件名或列契约。
- 不新增交易日历配置；日期范围内缺失文件按 warning 跳过处理。
- 不保留旧大文件输入/输出 CLI 兼容路径。

## 验收标准

- [ ] `stitch_main_contract.py` 按日期范围输出 `CONTINUOUS_RAW/{symbol}/{YYYY-MM-DD}.csv` 日文件。
- [ ] `stitch_main_contract.py` 不再生成 `fu_start_end.csv` 大区间连续文件。
- [ ] `downscale_continuous_by_trading_day.py` 支持 `--input_dir --start_date --end_date` 并逐日读取日文件。
- [ ] 日期范围内缺失日文件时 downscale 记录 warning 并跳过，不中断整个流程。
- [ ] 文件存在但内容缺列、非法盘口、无可交易主力等数据错误继续 fail-fast。
- [ ] 已存在的 `CONTINUOUS_RAW` 日文件默认覆盖，并记录日志。
- [ ] `fu_full_process.sh` 不再构造 `continuous_file` 大文件路径，改为传递日文件目录和日期范围。
- [ ] 现有下游 cross-section、merge、concat、time feature、feature selection 的输入日期范围和输出契约保持不变。
