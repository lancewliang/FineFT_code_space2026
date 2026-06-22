## 1. 主力连续化日文件输出

- [x] 1.0 主力连续化日文件输出完成（与 plan-ready.md Task 1 和 superpowers plan Task 1 同步） <!-- 已实现: stitch 按 TradingDay 输出日文件并移除旧 output 大文件 CLI -->
- [x] 1.1 修改 `data_preprocess/tests/test_commodity_main_contract.py`，增加日期范围按 `TradingDay` 返回日帧、无源日期跳过、已有日文件覆盖日志的测试。
- [x] 1.2 修改 `data_preprocess/operator_futures/commodity/main_contract.py`，提取日期范围每日主力选择结果，并新增写出 `CONTINUOUS_RAW/{symbol}/{YYYY-MM-DD}.csv` 的目录输出函数。
- [x] 1.3 修改 `data_preprocess/operator_futures/commodity/stitch_main_contract.py` 和 `data_preprocess/tests/test_commodity_main_contract_cli.py`，将 CLI 从 `--output` 改为 `--output_dir --start_date --end_date`，并验证不再生成日期范围大 CSV。

## 2. 连续主力日文件下采样

- [x] 2.0 连续主力日文件下采样完成（与 plan-ready.md Task 2 和 superpowers plan Task 2 同步） <!-- 已实现: downscale 改为按 input_dir 和日期范围逐日读取并跳过缺失日 -->
- [x] 2.1 修改 `data_preprocess/tests/test_commodity_main_contract_cli.py`，增加 `downscale_continuous_by_trading_day.py` 的 `--input_dir --start_date --end_date` CLI 测试，覆盖存在日文件输出和缺失日 warning skip。
- [x] 2.2 修改 `data_preprocess/operator_futures/commodity/downscale_continuous_by_trading_day.py`，按左闭右开日期范围逐日读取 `YYYY-MM-DD.csv`，缺失文件 warning 后跳过，存在坏文件保持 fail-fast。
- [x] 2.3 运行商品 downscale 聚焦测试，确认 `DOWNSCALE_DERTIC`、`DOWNSCALE_ORDERBOOK_25`、`BASE_FEATURE` 和 `COMMODITY_QUOTE_FEATURE` 输出契约不变。

## 3. 主流程脚本、文档与验证

- [x] 3.0 主流程脚本、文档与验证完成（与 plan-ready.md Task 3 和 superpowers plan Task 3 同步） <!-- 已实现: full process 改为传 CONTINUOUS_RAW 日文件目录并完成文档与验证 -->
- [x] 3.1 修改 `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`，传递 `CONTINUOUS_RAW/{symbol}` 目录和日期范围，不再构造 `continuous_file` 大 CSV 路径。
- [x] 3.2 修改 `data_preprocess/tests/test_commodity_main_contract_cli.py` 和 `docs/上海商品交易所/commodity_futures_preprocess.md`，验证并记录新的日文件命名、CLI 参数和 full-process 调用方式。
- [x] 3.3 运行 `conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py data_preprocess/tests/test_commodity_downscale.py -q`、`bash -n data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`、`openspec validate split-commodity-continuous-raw-by-day --strict` 和 `git diff --check`。
