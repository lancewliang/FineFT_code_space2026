# 实现计划：enhance-limit-single-sided-features

## 来源
- 提案：openspec/changes/enhance-limit-single-sided-features/proposal.md
- 设计：openspec/changes/enhance-limit-single-sided-features/design.md
- 规格：openspec/changes/enhance-limit-single-sided-features/specs/
- 任务：openspec/changes/enhance-limit-single-sided-features/tasks.md

## 实现步骤

### Task 1: 扩展商品涨跌停价 reward/execution 合同
- [x] **任务完成**（与 superpowers plan `Task 1`、`tasks.md` 对应条目同步勾选）
- 目标：让商品 orderbook 下采样输出 `LowerLimitPrice`、`UpperLimitPrice`，并让商品 reward/execution manifest 将两列识别为当前行情列。
- 改动文件：`data_preprocess/operator_futures/commodity/downscale.py`、`data_preprocess/operator_futures/commodity/schema.py`、`data_preprocess/tests/test_commodity_downscale.py`、`data_preprocess/tests/test_commodity_config_schema.py`、`data_preprocess/tests/test_commodity_feature_pipeline.py`
- 验证方式：运行 `pytest data_preprocess/tests/test_commodity_downscale.py data_preprocess/tests/test_commodity_config_schema.py data_preprocess/tests/test_commodity_feature_pipeline.py -q`，并确认新增列顺序和 reward/state 分离断言通过。

### Task 2: 增强单边盘口 snapshot 特征计算
- [x] **任务完成**（与 superpowers plan `Task 2`、`tasks.md` 对应条目同步勾选）
- 目标：在 `process_snapshot_features` 中处理 ask 空、bid 空、正常双边和双侧空输入，输出有限特征和 `ask_side_empty`、`bid_side_empty`。
- 改动文件：`data_preprocess/operator_futures/cross_section/base_feature_util.py`、`data_preprocess/tests/test_commodity_feature_pipeline.py`、`data_preprocess/tests/test_polars_feature_generation.py`
- 验证方式：运行 `pytest data_preprocess/tests/test_commodity_feature_pipeline.py data_preprocess/tests/test_polars_feature_generation.py -q`，确认单边盘口无 NaN/Inf、正常盘口兼容、双侧空 fail-fast。

### Task 3: 更新列合同文档和 expected columns
- [x] **任务完成**（与 superpowers plan `Task 3`、`tasks.md` 对应条目同步勾选）
- 目标：同步 snapshot 特征列数量、商品 reward/environment 文档、expected columns 和相关验证说明。
- 改动文件：`data_preprocess/operator_futures/feature_validation/expected_columns.py`、`docs/datapreprocess/5.SNAPSHOT_FEATURE_82_COLUMNS.md`、`docs/datapreprocess/6.TIME_FEATURE_3375_COLUMNS.md`、`docs/datapreprocess/1.DATA_PREPROCESS_REWARD_ENVIRONMENT_106_COLUMNS(挂单).md`
- 验证方式：运行 expected-column 相关测试和 `rg` 检查旧 82/106/3375 文案是否仍误导新增商品列合同。

### Task 4: 端到端聚焦验证与 OpenSpec 校验
- [x] **任务完成**（与 superpowers plan `Task 4`、`tasks.md` 对应条目同步勾选）
- 目标：验证合法单边盘口可以通过 time feature 输入质量校验，并完成 OpenSpec strict validation。
- 改动文件：`data_preprocess/tests/test_time_operator_polars.py`、`data_preprocess/tests/test_time_operator_pandas_create_feature.py`、`openspec/changes/enhance-limit-single-sided-features/tasks.md`
- 验证方式：运行 `pytest data_preprocess/tests/test_time_operator_polars.py data_preprocess/tests/test_time_operator_pandas_create_feature.py -q` 和 `openspec validate enhance-limit-single-sided-features --strict`。
