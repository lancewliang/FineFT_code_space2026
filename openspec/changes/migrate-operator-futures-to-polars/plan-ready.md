# 实现计划：migrate-operator-futures-to-polars

## 来源
- 提案：openspec/changes/migrate-operator-futures-to-polars/proposal.md
- 设计：openspec/changes/migrate-operator-futures-to-polars/design.md
- 规格：openspec/changes/migrate-operator-futures-to-polars/specs/
- 任务：openspec/changes/migrate-operator-futures-to-polars/tasks.md

## 实现步骤

### Task 1: Dependency and compatibility harness
- [x] **任务完成**（与 superpowers plan `Task 1`、`tasks.md` 对应条目同步勾选）
- 目标：先让环境明确依赖 Polars，并建立迁移期间可复用的输出兼容断言，避免每个模块各自写一套列顺序、timestamp、dtype/schema 和浮点比较逻辑。
- 改动文件：`data_preprocess/requirements.txt`、`data_preprocess/tests/conftest.py` 或 `data_preprocess/tests/test_polars_compat.py`、`openspec/changes/migrate-operator-futures-to-polars/compatibility-notes.md`。
- 验证方式：运行 `conda run -n finetf python -c "import polars"`、`conda run -n finetf pytest data_preprocess/tests/test_polars_compat.py -q`（若 helper 放入 `conftest.py`，则运行引用 helper 的最小测试文件），并确认 `compatibility-notes.md` 保持无未决项或记录完整未决项。

### Task 2: Binance downscale and IO paths
- [ ] **任务完成**（与 superpowers plan `Task 2`、`tasks.md` 对应条目同步勾选）
- 目标：迁移 Binance orderbook 与 derivative ticker 的 CSV/Feather 读取、时间列转换、first-in-window 降采样、列重命名、前向填充和 Feather 写出，同时保持 CLI 和输出路径不变。
- 改动文件：`data_preprocess/operator_futures/orderbook_25/down_scale_single_shot.py`、`data_preprocess/operator_futures/orderbook_25/down_scale_single_shot_base_other.py`、`data_preprocess/operator_futures/derivative_ticker/down_scale_single_shot.py`、`data_preprocess/operator_futures/derivative_ticker/down_scale_single_shot_base_other.py`、相关 `data_preprocess/tests` 测试文件。
- 验证方式：运行针对 downscale helper 的聚焦 pytest；如无现成 raw Binance 小样例，则运行 Python 级小 DataFrame 单元测试验证 timestamp、列名、first/ffill 语义，并记录缺少真实 raw 数据导致的 CLI smoke 限制。

### Task 3: Binance feature generation paths
- [ ] **任务完成**（与 superpowers plan `Task 3`、`tasks.md` 对应条目同步勾选）
- 目标：迁移 Binance quotes/trades 基础特征、cross-section 特征和 time_operator 滚动时间特征，保持 quote counts、OHLC、side grouped trade features、depth-aware cross-section、窗口语义、列顺序和 no-future-leakage 行为。
- 改动文件：`data_preprocess/operator_futures/features_related/base_feature.py`、`data_preprocess/operator_futures/features_related/feature_util.py`、`data_preprocess/operator_futures/cross_section/base_feature_util.py`、`data_preprocess/operator_futures/cross_section/create_feature.py`、`data_preprocess/operator_futures/time_operator/create_feature.py`、`data_preprocess/operator_futures/time_operator/create_feature_multi_processing.py`、`data_preprocess/operator_futures/time_operator/multi_processing_util.py`、`data_preprocess/operator_futures/time_operator/time_operator_util.py`、相关 `data_preprocess/tests` 测试文件。
- 验证方式：运行特征 helper 的聚焦 pytest，断言列名、列顺序、timestamp 对齐和 `rtol=1e-12, atol=1e-12` 数值兼容；运行 `conda run -n finetf pytest data_preprocess/tests -q` 的相关子集，确保商品已有测试未被共享分支破坏。

### Task 4: Merge, scale, and feature-selection paths
- [ ] **任务完成**（与 superpowers plan `Task 4`、`tasks.md` 对应条目同步勾选）
- 目标：迁移 merge/concat/merge_clean、scale/save 和核心 feature_selection 路径，保持 timestamp inner join、重复 timestamp 取 first、future feature shift、ffill、reward/execution/state feature 列选择和输出文件契约。
- 改动文件：`data_preprocess/operator_futures/merge_concat/merge.py`、`data_preprocess/operator_futures/merge_concat/concat.py`、`data_preprocess/operator_futures/merge_all/merge_clean.py`、`data_preprocess/operator_futures/scale_describe_save/scale_save.py`、`data_preprocess/operator_futures/feature_selection/cor_util.py`、`data_preprocess/operator_futures/feature_selection/ic_correlation.py`、`data_preprocess/operator_futures/feature_selection/lasso_linear.py`、`data_preprocess/operator_futures/feature_selection/rank_ic_correlation.py`、`data_preprocess/operator_futures/feature_selection/remove_duplicates_feature.py`、`data_preprocess/operator_futures/feature_selection/catbooost.py`、相关 `data_preprocess/tests` 测试文件。
- 验证方式：运行 merge/scale/feature-selection 聚焦 pytest，验证输出路径、列顺序、join/shift/ffill 语义和 `market_type` 分支；对无法在当前 workspace 跑通的大文件 CLI，记录缺少输入数据而非跳过兼容断言。

### Task 5: Commodity futures Polars migration
- [ ] **任务完成**（与 superpowers plan `Task 5`、`tasks.md` 对应条目同步勾选）
- 目标：迁移 `operator_futures/commodity` 专用链路，保持主力合约选择、跨年日期范围行为、元数据列、timestamp 归一化、depth=5、LastPrice fallback、funding 兼容列、Volume/Turnover 差分、tick rule、右闭聚合、fail-fast 校验和商品 manifest 语义。
- 改动文件：`data_preprocess/operator_futures/commodity/main_contract.py`、`data_preprocess/operator_futures/commodity/stitch_main_contract.py`、`data_preprocess/operator_futures/commodity/downscale.py`、`data_preprocess/operator_futures/commodity/downscale_single_day.py`、`data_preprocess/operator_futures/commodity/downscale_continuous_by_trading_day.py`、`data_preprocess/operator_futures/commodity/schema.py`、`data_preprocess/operator_futures/commodity/config.py`、`data_preprocess/tests/test_commodity_config_schema.py`、`data_preprocess/tests/test_commodity_downscale.py`、`data_preprocess/tests/test_commodity_feature_pipeline.py`、`data_preprocess/tests/test_commodity_main_contract.py`、`data_preprocess/tests/test_commodity_main_contract_cli.py`。
- 验证方式：运行 `conda run -n finetf pytest data_preprocess/tests/test_commodity_config_schema.py data_preprocess/tests/test_commodity_downscale.py data_preprocess/tests/test_commodity_feature_pipeline.py data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py -q`，并确认 `docs/上海商品交易所/fu2302.csv` 样例覆盖的商品行为仍然通过。

### Task 6: Validation and migration closure
- [ ] **任务完成**（与 superpowers plan `Task 6`、`tasks.md` 对应条目同步勾选）
- 目标：完成规格校验、测试回归、代表性 smoke 记录和手工性能记录，确保迁移完成后可以进入 close 阶段。
- 改动文件：`openspec/changes/migrate-operator-futures-to-polars/tasks.md`、`openspec/changes/migrate-operator-futures-to-polars/plan-ready.md`、`docs/superpowers/plans/2026-06-21-migrate-operator-futures-to-polars.md`、`openspec/changes/migrate-operator-futures-to-polars/compatibility-notes.md`。
- 验证方式：运行 `openspec validate migrate-operator-futures-to-polars --strict`、`conda run -n finetf pytest data_preprocess/tests -q`、代表性 Binance futures 与 commodity futures smoke 命令、`git diff --check`，并在验证记录或计划勾选备注中写明性能记录和无法运行项的具体原因。
