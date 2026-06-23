# 实现计划：add-feature-validation-reference

## 来源
- 提案：openspec/changes/add-feature-validation-reference/proposal.md
- 设计：openspec/changes/add-feature-validation-reference/design.md
- 规格：openspec/changes/add-feature-validation-reference/specs/
- 任务：openspec/changes/add-feature-validation-reference/tasks.md

## 实现步骤

### Task 1: Reference layout and entrypoint
- [x] **任务完成**（与 superpowers plan `Task 1`、`tasks.md` 对应条目同步勾选）
- 目标：新增独立商品特征验证 shell 入口、Python package/CLI 骨架，并复制旧 pandas reference 模块到 validation-only namespace。
- 改动文件：`data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh`、`data_preprocess/operator_futures/feature_validation/__init__.py`、`data_preprocess/operator_futures/feature_validation/validate_features.py`、`data_preprocess/operator_futures/feature_validation/pandas_reference/**`。
- 验证方式：运行 shell 语法检查、CLI help/import smoke、reference 模块 import 测试，并确认 `main.sh` 未被修改。

### Task 2: Validation rules and comparison engine
- [x] **任务完成**（与 superpowers plan `Task 2`、`tasks.md` 对应条目同步勾选）
- 目标：实现固定 expected columns、stage validators、timestamp 抽样逐列比较、Markdown/JSON 报告和退出码。
- 改动文件：`data_preprocess/operator_futures/feature_validation/expected_columns.py`、`data_preprocess/operator_futures/feature_validation/validators.py`、`data_preprocess/operator_futures/feature_validation/compare.py`、`data_preprocess/operator_futures/feature_validation/report.py`、`data_preprocess/operator_futures/feature_validation/validate_features.py`。
- 验证方式：运行 comparator/report/validators 单元测试，确认 `abs_diff <= 1e-9`、缺失列、多余列、未验证列、mismatch 和退出码行为符合规格。

### Task 3: Verification
- [x] **任务完成**（与 superpowers plan `Task 3`、`tasks.md` 对应条目同步勾选）
- 目标：补齐单元测试、reference 适配测试和五日商品样例 smoke test，并运行聚焦测试与 OpenSpec strict 校验。
- 改动文件：`data_preprocess/tests/test_feature_validation_*.py`、`data_preprocess/tests/test_commodity_main_contract_cli.py` 或新增专用 shell smoke 测试文件。
- 验证方式：运行 `conda run -n finetf pytest data_preprocess/tests/test_feature_validation_*.py -q`、`bash -n data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh`、`openspec validate add-feature-validation-reference --strict`。
