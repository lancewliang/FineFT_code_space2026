# 实现计划：add-commodity-quote-queue-pressure-features

## 来源
- 提案：openspec/changes/add-commodity-quote-queue-pressure-features/proposal.md
- 设计：无（OpenSpec 判定无需）
- 规格：openspec/changes/add-commodity-quote-queue-pressure-features/specs/
- 任务：openspec/changes/add-commodity-quote-queue-pressure-features/tasks.md

## 实现步骤

### Task 1: Add quote queue pressure regression tests
- [x] **任务完成**（与 superpowers plan `Task 1`、`tasks.md` 对应条目同步勾选）
- 目标：先写 focused regression tests，覆盖一档 refill/deplete 计数、`queue_refill_imbalance`、零事件中性输出、涨跌停单边和空侧比例、缺失 limit 列 fail-fast、非有限值拒绝，并确认既有 microstructure 输出仍存在。
- 改动文件：`data_preprocess/tests/test_commodity_downscale.py`
- 验证方式：实现前运行 `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_downscale.py -k "quote_queue_pressure or quote_microstructure" -q`，预期新增 queue pressure 断言失败；实现后预期通过。

### Task 2: Implement quote queue pressure row-window outputs
- [x] **任务完成**（与 superpowers plan `Task 2`、`tasks.md` 对应条目同步勾选）
- 目标：在 `downscale.py` 中扩展 microstructure 输入校验和 row-window 聚合，逐快照计算一档 queue refill/deplete 事件、bid/ask 空侧状态和涨跌停单边状态，再输出计数、比例和安全归一化的 `queue_refill_imbalance`。
- 改动文件：`data_preprocess/operator_futures/commodity/downscale.py`
- 验证方式：运行 Task 1 的 focused tests，确认新增特征数值、零分母保护、fail-fast 和既有 microstructure 行为全部通过，同时不改动 quote 时间窗口与 OFI 函数行为。

### Task 3: Validate quote queue pressure artifacts
- [x] **任务完成**（与 superpowers plan `Task 3`、`tasks.md` 对应条目同步勾选）
- 目标：对实现和规格做最终校验，确保 Python 语法、focused tests、相关 quote/OFI 回归测试和 OpenSpec strict validate 都通过。
- 改动文件：`data_preprocess/operator_futures/commodity/downscale.py`、`data_preprocess/tests/test_commodity_downscale.py`、`openspec/changes/add-commodity-quote-queue-pressure-features/{proposal.md,specs/,tasks.md,plan-ready.md}`、`docs/superpowers/plans/2026-07-24-add-commodity-quote-queue-pressure-features.md`
- 验证方式：运行 `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && python -m py_compile data_preprocess/operator_futures/commodity/downscale.py data_preprocess/tests/test_commodity_downscale.py && PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_downscale.py -k "quote_queue_pressure or quote_microstructure or quote_ofi" -q && openspec validate add-commodity-quote-queue-pressure-features --strict`。
