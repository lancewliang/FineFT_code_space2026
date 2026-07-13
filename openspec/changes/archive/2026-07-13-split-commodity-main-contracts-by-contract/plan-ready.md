# 实现计划：split-commodity-main-contracts-by-contract

## 来源
- 提案：openspec/changes/split-commodity-main-contracts-by-contract/proposal.md
- 设计：openspec/changes/split-commodity-main-contracts-by-contract/design.md
- 规格：openspec/changes/split-commodity-main-contracts-by-contract/specs/
- 任务：openspec/changes/split-commodity-main-contracts-by-contract/tasks.md

## Amendments

### 2026-07-12: Summary model bean construction
- 原因：summary JSON 的嵌套结构需要明确字段归属，避免在选择逻辑里手工拼装多层 dict/list。
- 影响规格：外部 JSON schema 不变；无需新增行为规格。
- 影响任务：新增 `tasks.md` Task 5，原最终验证顺延为 Task 6。

### 2026-07-12: Summary model bean deserialization
- 原因：读取 `main_contract_summary.json` 的代码也应通过 `MainContractSummary` model 使用字段，避免消费侧继续直接操作多层 dict/list。
- 影响规格：外部 JSON schema 不变；无需新增行为规格。
- 影响任务：新增 `tasks.md` Task 7 和 Task 8。

### 2026-07-12: Daily volume in summary trading days
- 原因：summary 日级交易日明细需要直接包含该合约当天总成交量，避免消费方为检查成交量重新读取原始 CSV。
- 影响规格：`commodity-futures-support/spec.md` 的 summary contract 字段新增 `trading_days[].daily_volume`。
- 影响任务：新增 `tasks.md` Task 9 和 Task 10。

### 2026-07-13: Contract trading-window clipping
- 原因：入选合约不应输出完整日期范围内所有交易日；有效窗口需要从首次入选月份开始，并排除请求日期范围内该合约最后交易日前 10 个交易日。
- 影响规格：`commodity-futures-support/spec.md` 的入选合约集合语义和 summary contract 字段。
- 影响任务：新增 `tasks.md` Task 11 和 Task 12。

### 2026-07-13: Date-range-relative final-10-day cutoff
- 来源：close 阶段代码审查发现实现使用请求日期范围内的合约交易日序列计算最后 10 个交易日 cutoff；用户要求将 spec 修改为当前实现语义。
- 影响：规格和计划文档中的 cutoff 描述改为“请求日期范围内该合约最后交易日前第 10 个交易日”；无需新增代码实现任务。

### 2026-07-13: High-volume-day main contract rule
- 原因：除月成交量 top 2 外，一个月内有足够多高成交量交易日的合约也应被视为主力合约。
- 影响规格：`commodity-futures-support/spec.md` 的自然月主力合约选择规则。
- 影响任务：新增 `tasks.md` Task 13 和 Task 14。

### 2026-07-13: Cross-contract training feature union
- 原因：各合约独立 feature selection / scale save 后最终 state feature 集合可能不同；这些合约用于同一个模型训练时，需要一个品种级统一特征合集。
- 影响规格：`commodity-futures-support/spec.md` 新增商品期货跨合约训练特征合集需求。
- 影响任务：新增 `tasks.md` Task 15 和 Task 16。

## 实现步骤

### Task 1: Main-contract summary generation
- [x] **任务完成**（与 superpowers plan `Task 1`、`tasks.md` 对应条目同步勾选）
- 目标：把商品主力阶段从写连续主力日 CSV 改为写 `CONTINUOUS_RAW/{symbol}/main_contract_summary.json`。summary 按自然月统计合约成交量，月成交量为日 `Volume.max - Volume.min` 求和，每月取 top 2；合约只要任意月入选，就记录该合约在日期范围内实际存在的全部交易日源文件。
- 改动文件：`data_preprocess/operator_futures/commodity/main_contract.py`、`data_preprocess/operator_futures/commodity/stitch_main_contract.py`、`data_preprocess/tests/test_commodity_main_contract.py`、`data_preprocess/tests/test_commodity_main_contract_cli.py`。
- 验证方式：运行 `conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py -q`，确认月度 top-2、summary 字段、`trading_day_count`、错误处理、CLI 写 JSON 和不再写日 CSV 的测试通过。

### Task 2: Summary-driven downscale
- [x] **任务完成**（与 superpowers plan `Task 2`、`tasks.md` 对应条目同步勾选）
- 目标：让 `downscale_continuous_by_trading_day.py` 从 summary 读取 `source_file`，默认处理所有合约，可选 `--contract` 过滤单合约，并把四类 downscale 输出写到 `{FEATURE_FOLDER}/{symbol}/{contract}/{target_freq}/{date}.feather`。
- 改动文件：`data_preprocess/operator_futures/commodity/downscale_continuous_by_trading_day.py`、`data_preprocess/tests/test_commodity_main_contract_cli.py`、`data_preprocess/tests/test_commodity_downscale.py`。
- 验证方式：运行 `conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract_cli.py data_preprocess/tests/test_commodity_downscale.py -q`，确认 `--summary`、全合约处理、`--contract` 过滤、summary 校验、缺失 source fail-fast 和 contract-scoped 输出路径通过。

### Task 3: Contract-scoped downstream Python paths
- [x] **任务完成**（与 superpowers plan `Task 3`、`tasks.md` 对应条目同步勾选）
- 目标：为 downscale 后的共享 Python 入口增加可选 `--contract` 参数；传入时所有输入输出路径使用 `{symbol}/{contract}/{target_freq}`，未传时保持旧 `{symbol}/{target_freq}` 行为。
- 改动文件：`data_preprocess/operator_futures/cross_section/create_feature.py`、`data_preprocess/operator_futures/merge_concat/merge.py`、`data_preprocess/operator_futures/merge_concat/concat.py`、`data_preprocess/operator_futures/time_operator/create_feature_multi_processing.py`、`data_preprocess/operator_futures/merge_all/merge_clean.py`、`data_preprocess/operator_futures/feature_selection/ic_correlation.py`、`data_preprocess/operator_futures/scale_describe_save/scale_save.py`、相关聚焦测试文件。
- 验证方式：运行相关路径契约测试和 `conda run -n finetf pytest data_preprocess/tests/test_commodity_feature_pipeline.py -q`，确认 contract-scoped 和 legacy no-contract 两套路径都可用，且日文件/区间文件粒度不变。

### Task 4: Commodity shell scripts, validation, and docs
- [x] **任务完成**（与 superpowers plan `Task 4`、`tasks.md` 对应条目同步勾选）
- 目标：更新 commodity shell 编排：主流程先生成 summary，再解析合约列表，按合约运行 downscale 后续阶段并传 `--contract`；日志、skip 检查和验证脚本都包含 contract 维度。同步商品预处理文档。
- 改动文件：`data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`、`data_preprocess/script_preprocess/future_upgraded/commodity/main_fu.sh`、`data_preprocess/script_preprocess/future_upgraded/commodity/main_al.sh`、`data_preprocess/script_preprocess/future_upgraded/commodity/commodity_process.sh`、`data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh`、`data_preprocess/script_preprocess/future_upgraded/commodity/flatten_aluminum_raw_csv.sh`、`data_preprocess/tests/test_commodity_main_contract_cli.py`、`docs/上海商品交易所/commodity_futures_preprocess.md`。
- 验证方式：运行 shell 聚焦测试、`bash -n` 检查所有 commodity shell 脚本，确认 summary 解析、contract 循环、日志命名、skip 检查、验证脚本和文档示例都匹配新契约。

### Task 5: Summary model bean refactor
- [x] **任务完成**（与 superpowers plan `Task 5`、`tasks.md` 对应条目同步勾选）
- 目标：把 summary 构建从手写嵌套 dict/list 改为类型化 model bean：顶层 `MainContractSummary` 包含 `MainContractSummaryContract` 列表，合约对象包含交易日明细对象；最终统一通过 model 转 dict/JSON，保持 `main_contract_summary.json` 外部字段和值不变。
- 改动文件：`data_preprocess/operator_futures/commodity/main_contract.py`、`data_preprocess/tests/test_commodity_main_contract.py`。
- 验证方式：运行 `conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py::test_stitch_main_contract_cli_outputs_summary_json -q`，确认 model serialization、summary builder 和 CLI JSON 输出仍匹配既有契约。

### Task 6: Verification
- [x] **任务完成**（与 superpowers plan `Task 6`、`tasks.md` 对应条目同步勾选）
- 目标：执行本变更的最终验证，确保 OpenSpec、测试、shell 语法和 diff hygiene 都通过。
- 改动文件：`openspec/changes/split-commodity-main-contracts-by-contract/tasks.md`、`openspec/changes/split-commodity-main-contracts-by-contract/plan-ready.md`、`docs/superpowers/plans/2026-07-12-split-commodity-main-contracts-by-contract.md`。
- 验证方式：运行 `conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py data_preprocess/tests/test_commodity_downscale.py data_preprocess/tests/test_commodity_feature_pipeline.py -q`、所有 commodity shell 脚本的 `bash -n`、`openspec validate split-commodity-main-contracts-by-contract --strict` 和 `git diff --check`。

### Task 7: Summary model bean deserialization
- [x] **任务完成**（与 superpowers plan `Task 7`、`tasks.md` 对应条目同步勾选）
- 目标：读取 `main_contract_summary.json` 时先反序列化为 `MainContractSummary`，消费方通过 `MainContractSummaryContract` 和 `MainContractSummaryTradingDay` 对象访问字段；保留必要的 `to_dict()` JSON 兼容出口。
- 改动文件：`data_preprocess/operator_futures/commodity/main_contract.py`、`data_preprocess/operator_futures/commodity/downscale_continuous_by_trading_day.py`、`data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`、`data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh`、`data_preprocess/tests/test_commodity_main_contract.py`、`data_preprocess/tests/test_commodity_main_contract_cli.py`、`data_preprocess/tests/test_commodity_downscale.py`。
- 验证方式：运行 `conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py data_preprocess/tests/test_commodity_downscale.py -q`，确认 model 反序列化、summary 校验、downscale、shell 合约解析和 validation helper 都使用 typed model 后仍匹配既有契约。

### Task 8: Post-deserialization verification
- [x] **任务完成**（与 superpowers plan `Task 8`、`tasks.md` 对应条目同步勾选）
- 目标：执行本次读取侧 model refactor 的最终验证，确保测试、shell 语法、OpenSpec 和 diff hygiene 都通过。
- 改动文件：`openspec/changes/split-commodity-main-contracts-by-contract/tasks.md`、`openspec/changes/split-commodity-main-contracts-by-contract/plan-ready.md`、`docs/superpowers/plans/2026-07-12-split-commodity-main-contracts-by-contract.md`。
- 验证方式：运行 `conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py data_preprocess/tests/test_commodity_downscale.py -q`、所有 commodity shell 脚本的 `bash -n`、`openspec validate split-commodity-main-contracts-by-contract --strict` 和 `git diff --check`。

### Task 9: Daily volume in summary trading days
- [x] **任务完成**（与 superpowers plan `Task 9`、`tasks.md` 对应条目同步勾选）
- 目标：让 `main_contract_summary.json` 的每个 `contracts[].trading_days[]` 条目包含 `daily_volume`，字段值等于该合约该 `TradingDay` 源文件的 `Volume.max - Volume.min`；构建和读取都继续通过 `MainContractSummary` typed model。
- 改动文件：`data_preprocess/operator_futures/commodity/main_contract.py`、`data_preprocess/tests/test_commodity_main_contract.py`、`data_preprocess/tests/test_commodity_main_contract_cli.py`、`data_preprocess/tests/test_commodity_downscale.py`。
- 验证方式：运行 `conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py data_preprocess/tests/test_commodity_downscale.py -q`，确认 serialization、deserialization、summary builder 和 summary fixture 消费均包含并校验 `daily_volume`。

### Task 10: Daily-volume verification
- [x] **任务完成**（与 superpowers plan `Task 10`、`tasks.md` 对应条目同步勾选）
- 目标：执行本次 summary schema 增量的最终验证，确保测试、shell 语法、OpenSpec 和 diff hygiene 都通过。
- 改动文件：`openspec/changes/split-commodity-main-contracts-by-contract/tasks.md`、`openspec/changes/split-commodity-main-contracts-by-contract/plan-ready.md`、`docs/superpowers/plans/2026-07-12-split-commodity-main-contracts-by-contract.md`。
- 验证方式：运行 `conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py data_preprocess/tests/test_commodity_downscale.py -q`、所有 commodity shell 脚本的 `bash -n`、`openspec validate split-commodity-main-contracts-by-contract --strict` 和 `git diff --check`。

### Task 11: Contract trading-window clipping
- [x] **任务完成**（与 superpowers plan `Task 11`、`tasks.md` 对应条目同步勾选）
- 目标：修改 summary 生成规则：合约入选集合仍由自然月 top 2 决定；每个入选合约的 `trading_days` 只保留从首次入选月份月初开始，到请求日期范围内该合约最后交易日前第 10 个交易日为止的实际交易日。
- 改动文件：`data_preprocess/operator_futures/commodity/main_contract.py`、`data_preprocess/tests/test_commodity_main_contract.py`、必要时更新 CLI/downscale summary fixture。
- 验证方式：运行 `conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py data_preprocess/tests/test_commodity_downscale.py -q`，确认起止边界、`start_trading_day`、`end_trading_day`、`trading_day_count`、`daily_volume` 和空窗口 fail-fast 均符合新规则。

### Task 12: Contract trading-window verification
- [x] **任务完成**（与 superpowers plan `Task 12`、`tasks.md` 对应条目同步勾选）
- 目标：执行本次合约交易日窗口裁剪的最终验证，确保测试、shell 语法、OpenSpec 和 diff hygiene 都通过。
- 改动文件：`openspec/changes/split-commodity-main-contracts-by-contract/tasks.md`、`openspec/changes/split-commodity-main-contracts-by-contract/plan-ready.md`、`docs/superpowers/plans/2026-07-12-split-commodity-main-contracts-by-contract.md`。
- 验证方式：运行 `conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py data_preprocess/tests/test_commodity_downscale.py -q`、所有 commodity shell 脚本的 `bash -n`、`openspec validate split-commodity-main-contracts-by-contract --strict` 和 `git diff --check`。

### Task 13: High-volume-day main contract rule
- [x] **任务完成**（与 superpowers plan `Task 13`、`tasks.md` 对应条目同步勾选）
- 目标：扩展主力合约选择规则：自然月 top 2 仍保留；同时，同一自然月内至少 10 个实际交易日 `daily_volume > 配置阈值` 的合约也入选该月。阈值从 commodity config 读取，`fu=15000`。
- 改动文件：`data_preprocess/operator_futures/commodity/config.py`、`data_preprocess/operator_futures/commodity/main_contract.py`、`data_preprocess/tests/test_commodity_main_contract.py`、必要时更新 CLI summary 断言。
- 验证方式：运行 `conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py data_preprocess/tests/test_commodity_downscale.py -q`，确认非 top2 但满足高成交量天数规则的合约被选中，严格大于阈值、配置阈值和既有 top2 行为都正确。

### Task 14: High-volume-day verification
- [x] **任务完成**（与 superpowers plan `Task 14`、`tasks.md` 对应条目同步勾选）
- 目标：执行高成交量天数规则的最终验证，确保测试、shell 语法、OpenSpec 和 diff hygiene 都通过。
- 改动文件：`openspec/changes/split-commodity-main-contracts-by-contract/tasks.md`、`openspec/changes/split-commodity-main-contracts-by-contract/plan-ready.md`、`docs/superpowers/plans/2026-07-12-split-commodity-main-contracts-by-contract.md`。
- 验证方式：运行 `conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py data_preprocess/tests/test_commodity_downscale.py -q`、所有 commodity shell 脚本的 `bash -n`、`openspec validate split-commodity-main-contracts-by-contract --strict` 和 `git diff --check`。

### Task 15: Cross-contract training feature union
- [x] **任务完成**（与 superpowers plan `Task 15`、`tasks.md` 对应条目同步勾选）
- 目标：新增最终 feature union 步骤：从 `main_contract_summary.json` 中读取所有合约，加载每个合约最终 `SCALE_SAVE/{symbol}/{contract}/{target_freq}/{start_date}-{end_date}/state_features.npy`，按 summary 合约顺序和合约内特征顺序稳定去重，写出 `FEATURE_UNION/{symbol}/{target_freq}/{start_date}-{end_date}/state_features.npy` 和 `feature_union_manifest.json`。
- 改动文件：`data_preprocess/operator_futures/feature_selection/contract_feature_union.py`、`data_preprocess/operator_futures/util.py`、`data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`、`data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh`、`data_preprocess/tests/test_commodity_feature_pipeline.py`、`data_preprocess/tests/test_commodity_main_contract_cli.py`、`docs/上海商品交易所/commodity_futures_preprocess.md`。
- 验证方式：运行 `conda run -n finetf pytest data_preprocess/tests/test_commodity_feature_pipeline.py data_preprocess/tests/test_commodity_main_contract_cli.py -q`，确认 union 顺序、重复去除、缺失合约 fail-fast、full process 调用和 validation 检查都正确。

### Task 16: Feature-union verification
- [x] **任务完成**（与 superpowers plan `Task 16`、`tasks.md` 对应条目同步勾选）
- 目标：执行 feature union 需求的最终验证，确保测试、shell 语法、OpenSpec 和 diff hygiene 都通过。
- 改动文件：`openspec/changes/split-commodity-main-contracts-by-contract/tasks.md`、`openspec/changes/split-commodity-main-contracts-by-contract/plan-ready.md`、`docs/superpowers/plans/2026-07-12-split-commodity-main-contracts-by-contract.md`。
- 验证方式：运行 `conda run -n finetf pytest data_preprocess/tests/test_commodity_feature_pipeline.py data_preprocess/tests/test_commodity_main_contract_cli.py -q`、所有 commodity shell 脚本的 `bash -n`、`openspec validate split-commodity-main-contracts-by-contract --strict` 和 `git diff --check`。
