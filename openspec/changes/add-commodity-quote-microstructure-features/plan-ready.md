# 实现计划：add-commodity-quote-microstructure-features

## 来源
- 提案：openspec/changes/add-commodity-quote-microstructure-features/proposal.md
- 设计：无（OpenSpec 判定无需）
- 规格：openspec/changes/add-commodity-quote-microstructure-features/specs/
- 任务：openspec/changes/add-commodity-quote-microstructure-features/tasks.md

## 实现步骤

### Task 1: Add quote microstructure regression tests
- [ ] **任务完成**（与 superpowers plan `Task 1`、`tasks.md` 对应条目同步勾选）
- 目标：先写 focused regression tests，覆盖 microprice pressure 与 relative spread 均值、spread widen/narrow/flat 计数和比例、默认 12 行窗口与尾组保留、输入校验、非有限值拒绝和零分母中性输出。
- 改动文件：`data_preprocess/tests/test_commodity_downscale.py`
- 验证方式：实现前运行 `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_downscale.py -k "quote_microstructure" -q`，预期因缺少 `downscale_quote_microstructure_features` 导入或实现失败；实现后预期通过。

### Task 2: Implement quote microstructure row-window function
- [ ] **任务完成**（与 superpowers plan `Task 2`、`tasks.md` 对应条目同步勾选）
- 目标：在 `downscale.py` 中新增 `downscale_quote_microstructure_features(second_df, window_rows=12)` 和小型私有 helper，逐快照计算 `microprice_pressure`、`relative_spread` 和 spread 变化方向，再按固定行窗口输出精简特征列。
- 改动文件：`data_preprocess/operator_futures/commodity/downscale.py`
- 验证方式：运行 Task 1 的 focused tests，确认公式、计数、固定窗口、fail-fast 和零分母保护全部通过，同时不改动现有 quote 与 OFI 函数行为。

### Task 3: Validate quote microstructure artifacts
- [ ] **任务完成**（与 superpowers plan `Task 3`、`tasks.md` 对应条目同步勾选）
- 目标：对实现和规格做最终校验，确保 Python 语法、focused tests、相关 quote/OFI 回归测试和 OpenSpec strict validate 都通过。
- 改动文件：`data_preprocess/operator_futures/commodity/downscale.py`、`data_preprocess/tests/test_commodity_downscale.py`、`openspec/changes/add-commodity-quote-microstructure-features/{proposal.md,specs/,tasks.md,plan-ready.md}`、`docs/superpowers/plans/2026-07-24-add-commodity-quote-microstructure-features.md`
- 验证方式：运行 `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && python -m py_compile data_preprocess/operator_futures/commodity/downscale.py data_preprocess/tests/test_commodity_downscale.py && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_downscale.py -k "quote_microstructure or quote_ofi" -q && openspec validate add-commodity-quote-microstructure-features --strict`。
