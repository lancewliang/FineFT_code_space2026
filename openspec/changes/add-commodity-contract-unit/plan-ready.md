# 实现计划：add-commodity-contract-unit

## 来源
- 提案：openspec/changes/add-commodity-contract-unit/proposal.md
- 设计：无（OpenSpec 判定无需）
- 规格：openspec/changes/add-commodity-contract-unit/specs/
- 任务：openspec/changes/add-commodity-contract-unit/tasks.md

## 实现步骤

### Task 1: 商品配置合约交易单位
- [x] **任务完成**（与 superpowers plan `Task 1`、`tasks.md` 对应条目同步勾选）
- 目标：在商品配置中新增必填正数 `contract_unit`，将 `fu` 配置为 `10`，并用测试固定该配置契约。
- 改动文件：`data_preprocess/operator_futures/commodity/config.py`、`data_preprocess/tests/test_commodity_config_schema.py`。
- 验证方式：运行 `conda run -n finetf pytest data_preprocess/tests/test_commodity_config_schema.py -q`，确认配置字段和值校验通过。

### Task 2: 商品成交价格口径修正
- [x] **任务完成**（与 superpowers plan `Task 2`、`tasks.md` 对应条目同步勾选）
- 目标：让商品成交估算读取 `contract_unit`，将 `second_avg_price` 和聚合 `vwap` 修正为价格口径，同时保持 `tradeval` 为原始成交额。
- 改动文件：`data_preprocess/operator_futures/commodity/downscale.py`、`data_preprocess/operator_futures/commodity/downscale_single_day.py`、`data_preprocess/operator_futures/commodity/downscale_continuous_by_trading_day.py`、`data_preprocess/tests/test_commodity_downscale.py`。
- 验证方式：运行 `conda run -n finetf pytest data_preprocess/tests/test_commodity_downscale.py -q`，确认 `fu` 的 `contract_unit=10` 时价格为 `Turnover / Volume / 10` 且 `tradeval` 未归一化。

### Task 3: 规格与回归验证
- [ ] **任务完成**（与 superpowers plan `Task 3`、`tasks.md` 对应条目同步勾选）
- 目标：同步规格/文档中旧的“不引入 contract_multiplier”描述，并运行 OpenSpec、商品期货测试和 diff 检查。
- 改动文件：`docs/上海商品交易所/commodity_futures_preprocess.md`、`openspec/changes/add-commodity-contract-unit/tasks.md`、`openspec/changes/add-commodity-contract-unit/plan-ready.md`、`docs/superpowers/plans/2026-06-21-add-commodity-contract-unit.md`。
- 验证方式：运行 `openspec validate add-commodity-contract-unit --strict`、`conda run -n finetf pytest data_preprocess/tests/test_commodity_config_schema.py data_preprocess/tests/test_commodity_downscale.py -q`、`git diff --check`，确认规格、测试和格式检查通过。
