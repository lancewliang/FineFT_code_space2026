# Tasks

## 1. Implementation

- [x] 1.1 Add focused quote microstructure regression tests in `data_preprocess/tests/test_commodity_downscale.py`, covering formula means, spread widen/narrow/flat counts and ratio, default 12-row aggregation with retained tail window, fail-fast input validation, non-finite input rejection, and derived zero-denominator neutral outputs. <!-- 已实现: 新增 microstructure 回归测试并已通过 focused pytest -->
- [x] 1.2 Implement `downscale_quote_microstructure_features()` and its small private helpers in `data_preprocess/operator_futures/commodity/downscale.py`, preserving existing `downscale_quote_features()` and `downscale_quote_ofi_features()` behavior. <!-- 已实现: 新增独立 row-window microstructure 特征函数 -->
- [x] 1.3 Run focused validation for the quote microstructure change, including Python compile checks for changed Python files, targeted pytest, and `openspec validate add-commodity-quote-microstructure-features --strict`. <!-- 已实现: py_compile / pytest / openspec validate 已通过 -->
