# 实现计划：add-scale-save-nan-validation

## 来源
- 提案：openspec/changes/add-scale-save-nan-validation/proposal.md
- 设计：无（OpenSpec 判定无需）
- 规格：openspec/changes/add-scale-save-nan-validation/specs/
- 任务：openspec/changes/add-scale-save-nan-validation/tasks.md

## 实现步骤

### Task 1: Add focused scale-save CLI tests
- [x] **任务完成**（与 superpowers plan `Task 1`、`tasks.md` 对应条目同步勾选）
- 目标：在 `data_preprocess/tests/test_feature_selection_polars.py` 增加输入 NaN、输出 NaN、正常成功路径的 scale-save CLI 覆盖。
- 改动文件：`data_preprocess/tests/test_feature_selection_polars.py`
- 验证方式：运行新增的失败路径测试，确认在实现前失败；实现后运行三个 scale-save 测试并通过。

### Task 2: Add scale-save NaN validation helper
- [x] **任务完成**（与 superpowers plan `Task 2`、`tasks.md` 对应条目同步勾选）
- 目标：在 `data_preprocess/operator_futures/scale_describe_save/scale_save.py` 新增小型 Polars DataFrame 校验函数，返回含 NaN 的列并抛出包含阶段和路径的 `ValueError`。
- 改动文件：`data_preprocess/operator_futures/scale_describe_save/scale_save.py`
- 验证方式：运行针对 helper 的 import/syntax 检查，并运行新增测试确认错误信息含 `input`、`output`、路径和列名。

### Task 3: Wire validation into scale-save flow
- [x] **任务完成**（与 superpowers plan `Task 3`、`tasks.md` 对应条目同步勾选）
- 目标：在读取主输入 feather 后立即调用校验函数，并在最终 `out` 写出前调用校验函数，确保失败时没有输出文件落盘。
- 改动文件：`data_preprocess/operator_futures/scale_describe_save/scale_save.py`
- 验证方式：运行输入 NaN、输出 NaN、正常成功路径测试；失败路径断言输出目录内四类输出文件不存在。

### Task 4: Run focused verification
- [x] **任务完成**（与 superpowers plan `Task 4`、`tasks.md` 对应条目同步勾选）
- 目标：在 `finetf` conda 环境下运行聚焦测试、OpenSpec 严格校验和语法检查。
- 改动文件：无代码改动；更新实现计划 checkbox 时同步 `tasks.md`、`plan-ready.md`、superpowers plan。
- 验证方式：运行 `conda activate finetf` 后的 `pytest ...`、`python -m py_compile ...`，以及 `openspec validate add-scale-save-nan-validation --strict`。
