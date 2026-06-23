# 实现计划：add-commodity-preprocess-step-logs

## 来源
- 提案：openspec/changes/add-commodity-preprocess-step-logs/proposal.md
- 设计：openspec/changes/add-commodity-preprocess-step-logs/design.md
- 规格：openspec/changes/add-commodity-preprocess-step-logs/specs/
- 任务：openspec/changes/add-commodity-preprocess-step-logs/tasks.md

## 实现步骤

### Task 1: Step-level logging implementation
- [x] **任务完成**（与 superpowers plan `Task 1`、`tasks.md` 对应条目同步勾选）
- 目标：为商品 preprocess 主流程 9 个主要阶段添加独立步骤日志，并保持总日志、失败停止语义和既有子日志不变。
- 改动文件：`data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`、`data_preprocess/tests/test_commodity_scripts_docs.py` 或新增聚焦脚本测试文件。
- 验证方式：运行聚焦 pytest，检查步骤日志文件名、总日志阶段状态、stderr 捕获、失败返回码和 cross-section/merge 子日志路径未变。

### Task 2: Verification
- [x] **任务完成**（与 superpowers plan `Task 2`、`tasks.md` 对应条目同步勾选）
- 目标：完成 shell 语法、OpenSpec strict 校验和相关测试，确认变更可进入 build。
- 改动文件：`openspec/changes/add-commodity-preprocess-step-logs/tasks.md`、`openspec/changes/add-commodity-preprocess-step-logs/plan-ready.md`、`docs/superpowers/plans/2026-06-22-add-commodity-preprocess-step-logs.md`。
- 验证方式：运行 `bash -n data_preprocess/script_preprocess/future_upgraded/commodity/main.sh data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`、`conda run -n finetf pytest data_preprocess/tests/test_commodity_scripts_docs.py -q`、`openspec validate add-commodity-preprocess-step-logs --strict`。
