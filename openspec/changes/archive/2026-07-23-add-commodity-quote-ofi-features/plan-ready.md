# 实现计划：add-commodity-quote-ofi-features

## 来源
- 提案：openspec/changes/add-commodity-quote-ofi-features/proposal.md
- 设计：无（OpenSpec 判定无需）
- 规格：openspec/changes/add-commodity-quote-ofi-features/specs/
- 任务：openspec/changes/add-commodity-quote-ofi-features/tasks.md

## 实现步骤

### Task 1: Add five-depth OFI direction tests
- [x] **任务完成**（与 superpowers plan `Task 1`、`tasks.md` 对应条目同步勾选）
- 目标：用精确样例锁定 `ofi_bid1..5`、`ofi_ask1..5`、`ofi_bid`、`ofi_ask`、`ofi` 和第一行置 0 语义。
- 改动文件：`data_preprocess/tests/test_commodity_downscale.py`
- 验证方式：`source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_computes_five_depth_direction_math -q`，实现前预期因缺少 `downscale_quote_ofi_features` 失败，实现后通过。

### Task 2: Add fixed-row aggregation and boundary tests
- [x] **任务完成**（与 superpowers plan `Task 2`、`tasks.md` 对应条目同步勾选）
- 目标：验证默认 `window_rows=12` 每 12 条输出一行、尾组保留、timestamp 取组内最后一条，并验证第 13 条仍使用第 12 条作为上一条快照。
- 改动文件：`data_preprocess/tests/test_commodity_downscale.py`
- 验证方式：`source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_aggregates_every_twelve_rows_and_keeps_tail data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_compares_across_row_window_boundary -q`，实现前预期失败，实现后通过。

### Task 3: Add OFI input validation tests
- [x] **任务完成**（与 superpowers plan `Task 3`、`tasks.md` 对应条目同步勾选）
- 目标：锁定空输入、缺少五档必需列、五档必需列含 null、`window_rows <= 0` 的 fail-fast 行为。
- 改动文件：`data_preprocess/tests/test_commodity_downscale.py`
- 验证方式：`source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_rejects_empty_input data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_rejects_missing_depth_columns data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_rejects_null_depth_values data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_rejects_invalid_window_rows -q`，实现前预期失败，实现后通过。

### Task 4: Implement row-window five-depth OFI
- [x] **任务完成**（与 superpowers plan `Task 4`、`tasks.md` 对应条目同步勾选）
- 目标：在 `downscale.py` 中新增独立 `downscale_quote_ofi_features(second_df, window_rows=12, depth=5)` 和小型 helper，不改变现有 `downscale_quote_features()`。
- 改动文件：`data_preprocess/operator_futures/commodity/downscale.py`
- 验证方式：`source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_computes_five_depth_direction_math data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_aggregates_every_twelve_rows_and_keeps_tail data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_compares_across_row_window_boundary data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_rejects_empty_input data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_rejects_missing_depth_columns data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_rejects_null_depth_values data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_rejects_invalid_window_rows -q`，预期全部通过。

### Task 5: Verify OFI change
- [x] **任务完成**（与 superpowers plan `Task 5`、`tasks.md` 对应条目同步勾选）
- 目标：运行 OpenSpec strict 校验、完整 commodity downscale 测试模块和 Python 编译检查，确认新增 OFI 未破坏现有下采样行为。
- 改动文件：无新增代码改动；验证 `openspec/changes/add-commodity-quote-ofi-features/**`、`data_preprocess/operator_futures/commodity/downscale.py`、`data_preprocess/tests/test_commodity_downscale.py`
- 验证方式：`openspec validate add-commodity-quote-ofi-features --strict`；`source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_downscale.py -q`；`source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && python -m py_compile data_preprocess/operator_futures/commodity/downscale.py data_preprocess/tests/test_commodity_downscale.py`，预期全部返回 0。

## Amendments

### 2026-07-23: 新增 OFI 归一化特征
- 原因：raw OFI 直接携带盘口挂单量绝对尺度，需要补充相对量纲的 `ofi_norm`、`ofi_bid_norm` 和 `ofi_ask_norm`。
- 影响规格：openspec/changes/add-commodity-quote-ofi-features/specs/commodity-futures-support/spec.md
- 影响任务：tasks.md `1.6`、`1.7`、`1.8`

## 追加实现步骤

### Task 6: Add normalized OFI tests
- [x] **任务完成**（与 superpowers plan `Task 6`、`tasks.md` 对应条目同步勾选）
- 目标：验证 `ofi_norm`、`ofi_bid_norm`、`ofi_ask_norm` 使用同一 12 行窗口内五档 volume 合计作为分母，并验证分母为 0 时输出 0。
- 改动文件：`data_preprocess/tests/test_commodity_downscale.py`
- 验证方式：`source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_outputs_normalized_ofi data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_zeroes_normalized_ofi_when_denominator_is_zero -q`，实现前预期失败，实现后通过。

### Task 7: Implement normalized OFI outputs
- [x] **任务完成**（与 superpowers plan `Task 7`、`tasks.md` 对应条目同步勾选）
- 目标：在 `downscale_quote_ofi_features()` 的 row-window 聚合中加入 bid/ask/total volume 分母，并输出 `ofi_norm`、`ofi_bid_norm`、`ofi_ask_norm`。
- 改动文件：`data_preprocess/operator_futures/commodity/downscale.py`
- 验证方式：`source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_outputs_normalized_ofi data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_zeroes_normalized_ofi_when_denominator_is_zero -q`，预期全部通过。

### Task 8: Verify normalized OFI change
- [x] **任务完成**（与 superpowers plan `Task 8`、`tasks.md` 对应条目同步勾选）
- 目标：运行 OpenSpec strict 校验、完整 commodity downscale 测试模块和 Python 编译检查，确认归一化 OFI 变更完整。
- 改动文件：无新增代码改动；验证 `openspec/changes/add-commodity-quote-ofi-features/**`、`data_preprocess/operator_futures/commodity/downscale.py`、`data_preprocess/tests/test_commodity_downscale.py`
- 验证方式：`openspec validate add-commodity-quote-ofi-features --strict`；`source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_downscale.py -q`；`source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && python -m py_compile data_preprocess/operator_futures/commodity/downscale.py data_preprocess/tests/test_commodity_downscale.py`，预期全部返回 0。

### 2026-07-23: 拦截 OFI 输入 NaN 和无穷大
- 原因：五档 price/volume 中的 NaN、`inf` 或 `-inf` 会污染 raw OFI、分母和归一化输出。
- 影响规格：openspec/changes/add-commodity-quote-ofi-features/specs/commodity-futures-support/spec.md
- 影响任务：tasks.md `1.9`、`1.10`、`1.11`

### Task 9: Add non-finite OFI input test
- [x] **任务完成**（与 superpowers plan `Task 9`、`tasks.md` 对应条目同步勾选）
- 目标：验证五档价格或数量列出现 NaN、`inf` 或 `-inf` 时 fail-fast，并列出坏列名。
- 改动文件：`data_preprocess/tests/test_commodity_downscale.py`
- 验证方式：`source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_rejects_non_finite_depth_values -q`，实现前预期失败，实现后通过。

### Task 10: Implement non-finite OFI input validation
- [x] **任务完成**（与 superpowers plan `Task 10`、`tasks.md` 对应条目同步勾选）
- 目标：在 `_validate_ofi_input()` 中检查五档价格/数量列是否包含 NaN、`inf` 或 `-inf`，发现后抛出包含列名的 `ValueError`。
- 改动文件：`data_preprocess/operator_futures/commodity/downscale.py`
- 验证方式：`source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_downscale.py::test_downscale_quote_ofi_features_rejects_non_finite_depth_values -q`，预期通过。

### Task 11: Verify non-finite OFI validation
- [x] **任务完成**（与 superpowers plan `Task 11`、`tasks.md` 对应条目同步勾选）
- 目标：运行 OpenSpec strict 校验、完整 commodity downscale 测试模块和 Python 编译检查，确认坏数据防御完整。
- 改动文件：无新增代码改动；验证 `openspec/changes/add-commodity-quote-ofi-features/**`、`data_preprocess/operator_futures/commodity/downscale.py`、`data_preprocess/tests/test_commodity_downscale.py`
- 验证方式：`openspec validate add-commodity-quote-ofi-features --strict`；`source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_downscale.py -q`；`source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && python -m py_compile data_preprocess/operator_futures/commodity/downscale.py data_preprocess/tests/test_commodity_downscale.py`，预期全部返回 0。
