# Tasks

## 1. Implementation

- [x] 1.1 Add focused quote downscale regression tests in `data_preprocess/tests/test_commodity_downscale.py`, covering `imbalance_1/3/5` OHLC/TWAP/AWAP/STD outputs, `imbalance_1` compatibility with `imbalance_volume`, zero-denominator neutral output, non-finite volume fail-fast, and missing depth-column fail-fast. <!-- 已实现: 新增多档 quote imbalance RED 回归测试并确认旧实现失败 -->
- [x] 1.2 Implement private quote-depth imbalance helpers and extend `downscale_quote_features()` in `data_preprocess/operator_futures/commodity/downscale.py`, preserving existing quote gap and limit single-sided behavior while adding `imbalance_1/3/5` and `std_imbalance_volume`. <!-- 已实现: 新增 quote depth imbalance helper 与窗口统计聚合，保留右闭右标 quote 语义 -->
- [x] 1.3 Run focused validation for the quote-depth imbalance change, including targeted pytest, Python compile checks for changed Python files, and `openspec validate add-commodity-quote-depth-imbalance-features --strict`. <!-- 已实现: py_compile、focused pytest、完整 commodity downscale pytest 与 OpenSpec strict validate 均通过 -->
