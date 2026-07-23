# 实现计划：add-commodity-quote-depth-imbalance-features

## 来源
- 提案：openspec/changes/add-commodity-quote-depth-imbalance-features/proposal.md
- 设计：无（OpenSpec 判定无需）
- 规格：openspec/changes/add-commodity-quote-depth-imbalance-features/specs/
- 任务：openspec/changes/add-commodity-quote-depth-imbalance-features/tasks.md

## 实现步骤

### Task 1: Add quote depth imbalance regression tests
- [x] **任务完成**（与 superpowers plan `Task 1`、`tasks.md` 对应条目同步勾选）
- 目标：先写出能覆盖新增多档盘口压力契约的 focused tests，包括 `imbalance_1/3/5` 窗口统计、旧字段兼容、零分母中性值、非有限 volume fail-fast 和缺深度列 fail-fast。
- 改动文件：`data_preprocess/tests/test_commodity_downscale.py`
- 验证方式：先运行新增测试确认旧实现失败；实现后运行 `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest data_preprocess/tests/test_commodity_downscale.py -k "depth_imbalance or quote_depth" -q` 确认通过。

### Task 2: Implement quote depth imbalance helpers and aggregation
- [x] **任务完成**（与 superpowers plan `Task 2`、`tasks.md` 对应条目同步勾选）
- 目标：在 `downscale.py` 中新增私有 helper，扩展 `downscale_quote_features()` 以输出 `imbalance_1/3/5` 的 `open/high/low/close/awap/twap/std`，并为旧 `imbalance_volume` 补充 `std_imbalance_volume`。
- 改动文件：`data_preprocess/operator_futures/commodity/downscale.py`
- 验证方式：运行 Task 1 的 focused tests，确认新增特征数值、非法值处理和现有 quote gap / 涨跌停单边行为都通过。

### Task 3: Validate quote depth imbalance artifacts
- [x] **任务完成**（与 superpowers plan `Task 3`、`tasks.md` 对应条目同步勾选）
- 目标：对实现和规格做最终校验，确保 Python 语法、focused tests 和 OpenSpec strict validate 都通过。
- 改动文件：`data_preprocess/operator_futures/commodity/downscale.py`、`data_preprocess/tests/test_commodity_downscale.py`、`openspec/changes/add-commodity-quote-depth-imbalance-features/{proposal.md,specs/,tasks.md,plan-ready.md}`
- 验证方式：运行 `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && python -m py_compile data_preprocess/operator_futures/commodity/downscale.py data_preprocess/tests/test_commodity_downscale.py && pytest data_preprocess/tests/test_commodity_downscale.py -k "depth_imbalance or quote_depth or limit_single_sided_quote or quote_gap" -q && openspec validate add-commodity-quote-depth-imbalance-features --strict`。
