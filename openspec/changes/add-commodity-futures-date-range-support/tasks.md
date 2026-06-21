## 1. 主力连续化实现

- [x] 1.0 主力连续化实现完成（与 `plan-ready.md` Task 1 和 superpowers plan Task 1 同步） <!-- 已实现: 日期范围年份推导、跨年扫描和主力状态连续 -->
- [x] 1.1 修改 `data_preprocess/operator_futures/commodity/main_contract.py`，支持从日期范围推导年份集合并跨年扫描原始目录。
- [x] 1.2 修改主力拼接逻辑，保证 `previous_frames` 在跨年日期范围内不重置，并按 `TradingDay` 左闭右开过滤输出。
- [x] 1.3 新增跨年主力拼接测试，覆盖 `END_DATE` 右开年份推导、2023 年末到 2024 年初的主力连续选择和输出元数据。

## 2. 商品主流程脚本适配

- [x] 2.0 商品主流程脚本适配完成（与 `plan-ready.md` Task 2 和 superpowers plan Task 2 同步） <!-- 已实现: CLI 和 shell 主流程改为日期范围驱动 -->
- [x] 2.1 修改 `data_preprocess/operator_futures/commodity/stitch_main_contract.py`，新增 `--start_date` / `--end_date` 参数并保留 `--year` 兼容路径。
- [x] 2.2 修改 `data_preprocess/script_preprocess/future_upgraded/commodity/main.sh`，去除单年强依赖并改为日期范围驱动日志/输出命名。
- [x] 2.3 修改 `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`，按日期范围传递主流程参数并保持现有左闭右开语义。
- [x] 2.4 更新 shell 级 smoke / CLI 测试，覆盖 `START_DATE=2023-01-01` 到 `END_DATE=2026-03-01` 的调用语义。

## 3. 验证与回归

- [x] 3.0 验证与回归完成（与 `plan-ready.md` Task 3 和 superpowers plan Task 3 同步） <!-- 已实现: 回归测试、OpenSpec 校验和格式检查通过 -->
- [x] 3.1 运行跨年主力拼接测试、CLI 测试与 shell 语法检查。
- [x] 3.2 运行 `openspec validate add-commodity-futures-date-range-support --strict` 并记录跳过项或环境限制。
