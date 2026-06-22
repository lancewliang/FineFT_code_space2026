# 实现计划：migrate-commodity-preprocess-to-polars

## 来源
- 提案：openspec/changes/migrate-commodity-preprocess-to-polars/proposal.md
- 设计：openspec/changes/migrate-commodity-preprocess-to-polars/design.md
- 规格：openspec/changes/migrate-commodity-preprocess-to-polars/specs/
- 任务：openspec/changes/migrate-commodity-preprocess-to-polars/tasks.md

## 实现步骤

### Task 1: Time feature Polars migration
- [x] **任务完成**（与 superpowers plan `Task 1`、`tasks.md` 对应条目同步勾选）
- 目标：将 time feature 入口和 helper 从 pandas + multiprocessing 改为 Polars 原生实现，保持 `TIME_FEATURE` 输出路径、列名、timestamp 对齐、商品 depth=5 和通用 depth=25 兼容。
- 改动文件：`data_preprocess/operator_futures/time_operator/create_feature_multi_processing.py`、`data_preprocess/operator_futures/time_operator/multi_processing_util.py`、`data_preprocess/tests/test_commodity_main_contract_cli.py` 或新增聚焦测试文件。
- 验证方式：运行 `conda run -n finetf pytest <time feature 聚焦测试> -q`，并运行真实或 fixture CLI 检查 `TIME_FEATURE/<symbol>/<freq>/<start>-<end>.feather` 生成。

### Task 2: Core feature selection Polars migration
- [x] **任务完成**（与 superpowers plan `Task 2`、`tasks.md` 对应条目同步勾选）
- 目标：迁移 `ic_correlation.py`、`rank_ic_correlation.py` 和 `cor_util.py` 到 Polars/NumPy，保持 reward/state 拆分、target 计算、相关性输出、selected feature 顺序和文件契约。
- 改动文件：`data_preprocess/operator_futures/feature_selection/ic_correlation.py`、`data_preprocess/operator_futures/feature_selection/rank_ic_correlation.py`、`data_preprocess/operator_futures/feature_selection/cor_util.py`、相关测试文件。
- 验证方式：运行聚焦 feature selection 测试，确认 `df.feather`、`df_rank.feather`、`state_features*.npy`、`correlation.csv`、`ic_window_<window>.json` 输出兼容。

### Task 3: ML feature selection Polars migration
- [x] **任务完成**（与 superpowers plan `Task 3`、`tasks.md` 对应条目同步勾选）
- 目标：迁移 `catbooost.py` 和 `lasso_linear.py` 的数据准备、筛列和输出写入到 Polars，模型调用边界使用 NumPy 或库原生输入，不引入 pandas。
- 改动文件：`data_preprocess/operator_futures/feature_selection/catbooost.py`、`data_preprocess/operator_futures/feature_selection/lasso_linear.py`、相关测试文件。
- 验证方式：运行聚焦 ML feature selection 测试，确认脚本不 import pandas，输出 feature metadata 和 Feather 文件路径兼容。

### Task 4: Remove-duplicates and scale/save Polars migration
- [x] **任务完成**（与 superpowers plan `Task 4`、`tasks.md` 对应条目同步勾选）
- 目标：迁移 `remove_duplicates_feature.py` 和 `scale_save.py` 到 Polars，保持去重/相关性过滤、reward 列保留、state 缩放、`df_describe.csv` 和 `SCALE_SAVE` 输出契约。
- 改动文件：`data_preprocess/operator_futures/feature_selection/remove_duplicates_feature.py`、`data_preprocess/operator_futures/scale_describe_save/scale_save.py`、相关测试文件。
- 验证方式：运行聚焦 remove-duplicates/scale-save 测试，确认商品 depth=5 reward manifest 和通用 reward 列兼容，`df.feather`、`state_features.npy`、`df_describe.csv` 生成。

### Task 5: Validation and smoke
- [x] **任务完成**（与 superpowers plan `Task 5`、`tasks.md` 对应条目同步勾选）
- 目标：完成 pandas import scan、商品 5 天主流程 smoke、相关 pytest、OpenSpec strict 校验和 diff 检查，确认变更可进入 build/close 后续阶段。
- 改动文件：目标测试文件、`openspec/changes/migrate-commodity-preprocess-to-polars/tasks.md`、`openspec/changes/migrate-commodity-preprocess-to-polars/plan-ready.md`、`docs/superpowers/plans/2026-06-22-migrate-commodity-preprocess-to-polars.md`。
- 验证方式：运行 `conda run -n finetf pytest data_preprocess/tests -q` 的相关子集、`bash data_preprocess/script_preprocess/future_upgraded/commodity/main.sh`、日志错误扫描、`openspec validate migrate-commodity-preprocess-to-polars --strict`、`git diff --check`。
