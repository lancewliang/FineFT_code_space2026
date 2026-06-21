## 1. Dependency and compatibility harness

- [x] 1.0 Dependency and compatibility harness complete（与 `plan-ready.md` Task 1 和 superpowers plan Task 1 同步） <!-- 已实现: 添加 Polars 必需依赖和兼容性测试 helper -->
- [ ] 1.1 Update `data_preprocess/requirements.txt` to add required `polars` dependency while preserving existing dependency pins.
- [ ] 1.2 Add focused compatibility helpers in `data_preprocess/tests` for comparing Polars-produced outputs against existing expected contracts, including column order, timestamps, row order, dtype/schema checks where practical, and float tolerance `rtol=1e-12, atol=1e-12`.
- [ ] 1.3 Keep `openspec/changes/migrate-operator-futures-to-polars/compatibility-notes.md` current if implementation finds a suspected pandas bug or unavoidable schema/dtype difference.

## 2. Binance downscale and IO paths

- [x] 2.0 Binance downscale and IO paths complete（与 `plan-ready.md` Task 2 和 superpowers plan Task 2 同步） <!-- 已实现: orderbook 和 derivative ticker downscale 改为 Polars IO 与窗口聚合 -->
- [ ] 2.1 Migrate `data_preprocess/operator_futures/orderbook_25/*.py` to Polars while preserving timestamp conversion, first-in-window downsampling, forward fill, renamed depth columns, CLI behavior, and Feather output paths.
- [ ] 2.2 Migrate `data_preprocess/operator_futures/derivative_ticker/*.py` to Polars while preserving timestamp/funding timestamp conversion, selected columns, first-in-window downsampling, forward fill, CLI behavior, and Feather output paths.
- [ ] 2.3 Add or update focused tests/smoke checks for the migrated Binance downscale paths using existing small fixtures or minimal existing test patterns.

## 3. Binance feature generation paths

- [ ] 3.0 Binance feature generation paths complete（与 `plan-ready.md` Task 3 和 superpowers plan Task 3 同步）
- [ ] 3.1 Migrate `data_preprocess/operator_futures/features_related/base_feature.py` and `feature_util.py` to Polars while preserving quote counts, quote OHLC, trade OHLCV, side-grouped features, column order, `exchange`, `symbol`, and ffill behavior.
- [ ] 3.2 Migrate `data_preprocess/operator_futures/cross_section/base_feature_util.py` and `create_feature.py` to Polars while preserving KLINE, QUOTES, SNAPSHOT, normalization, depth-aware behavior, and `market_type` branches.
- [ ] 3.3 Migrate `data_preprocess/operator_futures/time_operator/*.py` to Polars while preserving rolling feature names, window semantics, index/timestamp alignment, and no-future-leakage behavior.
- [ ] 3.4 Add or update focused tests/smoke checks for feature-generation compatibility using existing small data patterns.

## 4. Merge, scale, and feature-selection paths

- [ ] 4.0 Merge, scale, and feature-selection paths complete（与 `plan-ready.md` Task 4 和 superpowers plan Task 4 同步）
- [ ] 4.1 Migrate `data_preprocess/operator_futures/merge_concat/merge.py` and `concat.py` to Polars while preserving same-day concat, cross-day concat, duplicate timestamp `first` semantics, target-frequency gaps, future feature shift, inner join, ffill, and output paths.
- [ ] 4.2 Migrate `data_preprocess/operator_futures/merge_all/merge_clean.py` to Polars while preserving timestamp inner join of cross-section and time features.
- [ ] 4.3 Migrate `data_preprocess/operator_futures/scale_describe_save/scale_save.py` and core `feature_selection` scripts to Polars where they participate in preprocessing outputs, while preserving reward/execution columns, state features, selected feature files, and `market_type` behavior.
- [ ] 4.4 Add or update focused tests/smoke checks for merge, scale, and feature-selection output compatibility.

## 5. Commodity futures Polars migration

- [ ] 5.0 Commodity futures Polars migration complete（与 `plan-ready.md` Task 5 和 superpowers plan Task 5 同步）
- [ ] 5.1 Migrate `data_preprocess/operator_futures/commodity/main_contract.py` and `stitch_main_contract.py` to Polars while preserving main-contract selection, date-range behavior, metadata columns, timestamp normalization, duplicate data checks, and fail-fast errors.
- [ ] 5.2 Migrate `data_preprocess/operator_futures/commodity/downscale.py`, `downscale_single_day.py`, and `downscale_continuous_by_trading_day.py` to Polars while preserving derivative reference, depth=5 orderbook, base features, quote features, right-closed windows, ffill behavior, and validation errors.
- [ ] 5.3 Preserve commodity `schema.py`, `config.py`, and `market_type=commodity_futures` branch behavior for reward/execution manifest, depth-aware features, funding-disabled data, and feature selection targets.
- [ ] 5.4 Update existing `data_preprocess/tests/test_commodity_*` tests only as needed to validate the same commodity contracts under the Polars implementation.

## 6. Validation and migration closure

- [ ] 6.0 Validation and migration closure complete（与 `plan-ready.md` Task 6 和 superpowers plan Task 6 同步）
- [ ] 6.1 Run `openspec validate migrate-operator-futures-to-polars --strict` and fix all validation errors.
- [ ] 6.2 Run focused data preprocessing tests with `conda run -n finetf pytest data_preprocess/tests -q`.
- [ ] 6.3 Run representative Binance futures and commodity futures smoke commands using existing small samples or document why a smoke command cannot run in the current workspace.
- [ ] 6.4 Record manual before/after timing notes for a representative end-to-end preprocessing path and confirm expected total runtime improvement is at least 30%, without adding a benchmark script.
