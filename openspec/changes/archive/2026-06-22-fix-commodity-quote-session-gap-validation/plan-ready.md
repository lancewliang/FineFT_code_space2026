# 实现计划：fix-commodity-quote-session-gap-validation

## 来源
- 提案：openspec/changes/fix-commodity-quote-session-gap-validation/proposal.md
- 设计：无（OpenSpec 判定无需）
- 规格：openspec/changes/fix-commodity-quote-session-gap-validation/specs/
- 任务：openspec/changes/fix-commodity-quote-session-gap-validation/tasks.md

## 实现步骤

### Task 1: 商品交易 session 配置
- [x] **任务完成**（与 superpowers plan `Task 1`、`tasks.md` 对应条目同步勾选）
- 目标：在商品配置中增加可校验的交易 session 结构，并为 `fu` 配置燃料油常规交易时段，供 quote gap 校验使用。
- 改动文件：`data_preprocess/operator_futures/commodity/config.py`、`data_preprocess/tests/test_commodity_config_schema.py`。
- 验证方式：运行 `conda run -n finetf pytest data_preprocess/tests/test_commodity_config_schema.py -q`，确认 session 字段、`fu` session 值和非法 session 校验通过。

### Task 2: Quote gap session-aware 校验
- [x] **任务完成**（与 superpowers plan `Task 2`、`tasks.md` 对应条目同步勾选）
- 目标：让 `downscale_quote_features()` 只在同一有效交易 session 内检查 target window 连续性，跨 session、跨自然日、周末或休市间隔不报 quote 缺失。
- 改动文件：`data_preprocess/operator_futures/commodity/downscale.py`、`data_preprocess/tests/test_commodity_downscale.py`。
- 验证方式：运行 `conda run -n finetf pytest data_preprocess/tests/test_commodity_downscale.py -q`，确认 `2025-10-31 23:05:00` 非交易时段不误报、同一 session 内缺口仍 fail-fast、空输入仍 fail-fast。

### Task 3: 规格与回归验证
- [x] **任务完成**（与 superpowers plan `Task 3`、`tasks.md` 对应条目同步勾选）
- 目标：运行 OpenSpec、商品配置/downscale 回归测试和 diff 检查，确认规格、实现计划和代码变更保持一致。
- 改动文件：`openspec/changes/fix-commodity-quote-session-gap-validation/tasks.md`、`openspec/changes/fix-commodity-quote-session-gap-validation/plan-ready.md`、`docs/superpowers/plans/2026-06-22-fix-commodity-quote-session-gap-validation.md`。
- 验证方式：运行 `openspec validate fix-commodity-quote-session-gap-validation --strict`、`conda run -n finetf pytest data_preprocess/tests/test_commodity_config_schema.py data_preprocess/tests/test_commodity_downscale.py -q`、`git diff --check`。
