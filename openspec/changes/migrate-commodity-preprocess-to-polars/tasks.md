## 1. Time feature Polars migration

- [x] 1.0 Time feature Polars migration complete（与 `plan-ready.md` Task 1 和 superpowers plan Task 1 同步）<!-- 已实现: time_operator 入口与 helper 已切换到 Polars，CLI depth 兼容测试通过 -->
- [x] 1.1 Add focused tests for `create_feature_multi_processing.py` covering commodity depth=5, generic depth=25, output column names, row trimming, timestamp alignment, and no future leakage.
- [x] 1.2 Replace pandas and multiprocessing usage in `data_preprocess/operator_futures/time_operator/create_feature_multi_processing.py` and `data_preprocess/operator_futures/time_operator/multi_processing_util.py` with Polars-native helpers.
- [x] 1.3 Verify time feature CLI output path and file name compatibility for `TIME_FEATURE/<symbol>/<freq>/<start>-<end>.feather`.

## 2. Core feature selection Polars migration

- [x] 2.0 Core feature selection Polars migration complete（与 `plan-ready.md` Task 2 和 superpowers plan Task 2 同步）<!-- 已实现: IC/rank IC 流程已切到 Polars，输出文件与聚焦测试通过 -->
- [x] 2.1 Add focused tests for `ic_correlation.py`, `rank_ic_correlation.py`, and `cor_util.py` covering reward/state splitting, target calculation, correlation output, selected feature names, and artifact paths.
- [x] 2.2 Replace pandas usage in `ic_correlation.py`, `rank_ic_correlation.py`, and `cor_util.py` with Polars and NumPy equivalents.
- [x] 2.3 Verify `df.feather`, `df_rank.feather`, `state_features.npy`, `state_features_rank.npy`, `correlation.csv`, and `ic_window_<window>.json` compatibility.

## 3. ML feature selection Polars migration

- [x] 3.0 ML feature selection Polars migration complete（与 `plan-ready.md` Task 3 和 superpowers plan Task 3 同步）<!-- 已实现: lasso 与 catboost 流程已切到 Polars，测试通过 -->
- [x] 3.1 Add focused tests for `catbooost.py` and `lasso_linear.py` covering pandas-free input preparation, output artifact paths, and selected feature metadata.
- [x] 3.2 Replace pandas usage in `catbooost.py` and `lasso_linear.py` with Polars data preparation and NumPy or library-native model boundaries.
- [x] 3.3 Verify the scripts do not import pandas and still write their expected selected-feature outputs.

## 4. Remove-duplicates and scale/save Polars migration

- [x] 4.0 Remove-duplicates and scale/save Polars migration complete（与 `plan-ready.md` Task 4 和 superpowers plan Task 4 同步）<!-- 已实现: scale_save 和 remove_duplicates_feature 已迁移到 Polars，测试通过 -->
- [x] 4.1 Add focused tests for `remove_duplicates_feature.py` and `scale_save.py` covering duplicate/correlation filtering, reward column preservation, state scaling, `df_describe.csv`, and depth-aware commodity reward columns.
- [x] 4.2 Replace pandas usage in `remove_duplicates_feature.py` with Polars and NumPy equivalents while preserving output paths.
- [x] 4.3 Replace pandas usage in `scale_describe_save/scale_save.py` with Polars while preserving `df.feather`, `state_features.npy`, and `df_describe.csv`.

## 5. Validation and smoke

- [x] 5.0 Validation and smoke complete（与 `plan-ready.md` Task 5 和 superpowers plan Task 5 同步）<!-- 已实现: pandas import scan、OpenSpec strict、diff check、pytest 回归均完成 -->
- [x] 5.1 Add or update an import-scan test proving target files do not import pandas.
- [x] 5.2 Run focused post-merge Polars tests and commodity regression tests.
- [x] 5.3 Run commodity five-day `main.sh` and scan logs for `Traceback`, `FileNotFound`, and `KeyError`.
- [x] 5.4 Run `openspec validate migrate-commodity-preprocess-to-polars --strict` and `git diff --check`.
