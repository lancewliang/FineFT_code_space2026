# 实现计划：add-commodity-futures-date-range-support

## 来源
- 提案：openspec/changes/add-commodity-futures-date-range-support/proposal.md
- 设计：openspec/changes/add-commodity-futures-date-range-support/design.md
- 规格：openspec/changes/add-commodity-futures-date-range-support/specs/
- 任务：openspec/changes/add-commodity-futures-date-range-support/tasks.md

## 实现步骤

### Task 1: 主力连续化实现
- [x] **任务完成**（与 superpowers plan `Task 1`、`tasks.md` 对应条目同步勾选）
- 目标：实现从 `START_DATE` / `END_DATE` 推导年份集合、跨年扫描原始目录、跨年保持 `previous_frames`，并按 `TradingDay` 左闭右开过滤连续主力输出。
- 改动文件：`data_preprocess/operator_futures/commodity/main_contract.py`、`data_preprocess/tests/test_commodity_main_contract.py`。
- 验证方式：运行 `conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py -q`，确认年份推导、跨年主力选择和输出元数据测试通过。

### Task 2: 商品主流程脚本适配
- [x] **任务完成**（与 superpowers plan `Task 2`、`tasks.md` 对应条目同步勾选）
- 目标：让 `stitch_main_contract.py`、`main.sh` 和 `fu_full_process.sh` 接收日期范围并使用该范围驱动日志与输出命名，同时保留 `--year` / `YEAR` 的兼容语义。
- 改动文件：`data_preprocess/operator_futures/commodity/stitch_main_contract.py`、`data_preprocess/script_preprocess/future_upgraded/commodity/main.sh`、`data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`、`data_preprocess/tests/test_commodity_main_contract_cli.py`。
- 验证方式：运行 `conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract_cli.py -q` 和 `bash -n data_preprocess/script_preprocess/future_upgraded/commodity/main.sh data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`，确认脚本不再依赖单一年份输入。

### Task 3: 验证与回归
- [x] **任务完成**（与 superpowers plan `Task 3`、`tasks.md` 对应条目同步勾选）
- 目标：运行 OpenSpec 和商品期货回归测试，确认跨年支持不破坏现有商品能力。
- 改动文件：`openspec/changes/add-commodity-futures-date-range-support/tasks.md`、`openspec/changes/add-commodity-futures-date-range-support/plan-ready.md`、`docs/superpowers/plans/2026-06-21-add-commodity-futures-date-range-support.md`。
- 验证方式：运行 `openspec validate add-commodity-futures-date-range-support --strict`、`conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py -q`、`git diff --check`，并记录结果。
