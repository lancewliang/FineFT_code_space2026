# 实现计划：refactor-commodity-main-contract-objects

## 来源
- 提案：openspec/changes/refactor-commodity-main-contract-objects/proposal.md
- 设计：无（OpenSpec 判定无需）
- 规格：openspec/changes/refactor-commodity-main-contract-objects/specs/
- 任务：openspec/changes/refactor-commodity-main-contract-objects/tasks.md

## 实现步骤

### Task 1: Add focused commodity main-contract tests
- [x] **任务完成**（与 superpowers plan `Task 1`、`tasks.md` 对应条目同步勾选）
- 目标：在 `data_preprocess/tests/test_commodity_main_contract.py` 把源文件发现、summary 构建和 JSON 写出的断言改成对象断言，固定 `to_dict()` 兼容性。
- 改动文件：`data_preprocess/tests/test_commodity_main_contract.py`
- 验证方式：`conda activate finetf && pytest data_preprocess/tests/test_commodity_main_contract.py -q`

### Task 2: Add main-contract build state dataclasses
- [x] **任务完成**（与 superpowers plan `Task 2`、`tasks.md` 对应条目同步勾选）
- 目标：在 `data_preprocess/operator_futures/commodity/main_contract.py` 增加源文件发现和构建状态的 dataclass，替换嵌套 dict 的中间传递。
- 改动文件：`data_preprocess/operator_futures/commodity/main_contract.py`
- 验证方式：`conda activate finetf && pytest data_preprocess/tests/test_commodity_main_contract.py::test_load_contract_files_by_trading_day_for_years_returns_paths -q`

### Task 3: Refactor summary build and write boundaries
- [x] **任务完成**（与 superpowers plan `Task 3`、`tasks.md` 对应条目同步勾选）
- 目标：让 `build_main_contract_summary_for_date_range()` 返回 `MainContractSummary`，并让 `write_main_contract_summary_for_date_range()` 只在 JSON 边界调用 `to_dict()`。
- 改动文件：`data_preprocess/operator_futures/commodity/main_contract.py`、`data_preprocess/tests/test_commodity_main_contract.py`
- 验证方式：`conda activate finetf && pytest data_preprocess/tests/test_commodity_main_contract.py::test_build_main_contract_summary_selects_monthly_top_two_contracts data_preprocess/tests/test_commodity_main_contract.py::test_write_main_contract_summary_for_date_range_writes_json -q`

### Task 4: Run focused verification
- [x] **任务完成**（与 superpowers plan `Task 4`、`tasks.md` 对应条目同步勾选）
- 目标：运行 commodity focused tests、Python 语法检查和 OpenSpec strict 校验，确认对象化重构与规格一致。
- 改动文件：无代码改动；根据验证结果只修正前面任务引入的问题
- 验证方式：`conda activate finetf && pytest data_preprocess/tests/test_commodity_main_contract.py`、`conda activate finetf && python -m py_compile data_preprocess/operator_futures/commodity/main_contract.py`、`openspec validate refactor-commodity-main-contract-objects --strict`
