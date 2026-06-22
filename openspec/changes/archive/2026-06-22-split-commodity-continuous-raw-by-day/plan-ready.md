# 实现计划：split-commodity-continuous-raw-by-day

## 来源
- 提案：openspec/changes/split-commodity-continuous-raw-by-day/proposal.md
- 设计：openspec/changes/split-commodity-continuous-raw-by-day/design.md
- 规格：openspec/changes/split-commodity-continuous-raw-by-day/specs/
- 任务：openspec/changes/split-commodity-continuous-raw-by-day/tasks.md

## 实现步骤

### Task 1: 主力连续化日文件输出
- [x] **任务完成**（与 superpowers plan `Task 1`、`tasks.md` 对应条目同步勾选）
- 目标：让商品主力拼接按 `TradingDay` 输出 `CONTINUOUS_RAW/{symbol}/{YYYY-MM-DD}.csv`，缺失源日期跳过并记录日志，已有日文件默认覆盖并记录路径，旧日期范围大 CSV 不再生成。
- 改动文件：`data_preprocess/operator_futures/commodity/main_contract.py`、`data_preprocess/operator_futures/commodity/stitch_main_contract.py`、`data_preprocess/tests/test_commodity_main_contract.py`、`data_preprocess/tests/test_commodity_main_contract_cli.py`。
- 验证方式：运行 `conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py -q`，确认日文件生成、跳过、覆盖和 CLI breaking change 测试通过。

### Task 2: 连续主力日文件下采样
- [x] **任务完成**（与 superpowers plan `Task 2`、`tasks.md` 对应条目同步勾选）
- 目标：让 `downscale_continuous_by_trading_day.py` 改为 `--input_dir --start_date --end_date`，逐日读取 `YYYY-MM-DD.csv`，缺失日 warning skip，坏日文件保持 fail-fast，并保持所有下采样输出目录、文件名和列契约不变。
- 改动文件：`data_preprocess/operator_futures/commodity/downscale_continuous_by_trading_day.py`、`data_preprocess/tests/test_commodity_main_contract_cli.py`、`data_preprocess/tests/test_commodity_downscale.py`。
- 验证方式：运行 `conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract_cli.py data_preprocess/tests/test_commodity_downscale.py -q`，确认目录输入、缺失日 warning、输出 Feather 路径和既有 bad-data fail-fast 测试通过。

### Task 3: 主流程脚本、文档与验证
- [x] **任务完成**（与 superpowers plan `Task 3`、`tasks.md` 对应条目同步勾选）
- 目标：同步 `fu_full_process.sh` 和商品预处理文档，移除 `continuous_file` 大 CSV handoff，保留下游 cross-section、merge、concat、time feature、feature selection 和 scale/save 的日期范围契约。
- 改动文件：`data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`、`data_preprocess/tests/test_commodity_main_contract_cli.py`、`docs/上海商品交易所/commodity_futures_preprocess.md`、`openspec/changes/split-commodity-continuous-raw-by-day/tasks.md`、`openspec/changes/split-commodity-continuous-raw-by-day/plan-ready.md`、`docs/superpowers/plans/2026-06-22-split-commodity-continuous-raw-by-day.md`。
- 验证方式：运行 `bash -n data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`、`conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py data_preprocess/tests/test_commodity_downscale.py -q`、`openspec validate split-commodity-continuous-raw-by-day --strict` 和 `git diff --check`。
