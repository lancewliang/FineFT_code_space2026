## 1. Implementation

- [x] 1.1 Add focused tests in `data_preprocess/tests/test_commodity_downscale.py` for five-depth OFI direction math, summary columns, and first-row zero behavior. <!-- 已实现: 添加五档 OFI 方向数学测试并确认 RED -->
- [x] 1.2 Add focused tests in `data_preprocess/tests/test_commodity_downscale.py` for fixed 12-row aggregation, tail groups, and cross-window continuous comparison. <!-- 已实现: 添加 12 行聚合、尾组保留和跨窗口连续比较 RED 测试 -->
- [x] 1.3 Add focused tests in `data_preprocess/tests/test_commodity_downscale.py` for empty input, missing required depth columns, null depth values, and invalid `window_rows`. <!-- 已实现: 添加 OFI 输入 fail-fast RED 测试 -->
- [x] 1.4 Implement `downscale_quote_ofi_features()` and its OFI expression helpers in `data_preprocess/operator_futures/commodity/downscale.py`. <!-- 已实现: 新增五档 row-window OFI helper 和公开函数，聚焦测试通过 -->
- [x] 1.5 Verify the new OFI tests and run the commodity downscale test module under the `finetf` conda environment. <!-- 已实现: OpenSpec 校验、完整 commodity downscale 测试和 py_compile 均通过 -->
- [x] 1.6 Add focused tests in `data_preprocess/tests/test_commodity_downscale.py` for `ofi_norm`, `ofi_bid_norm`, `ofi_ask_norm`, and zero-denominator behavior. <!-- 已实现: 添加归一化 OFI RED 测试和分母为 0 行为测试 -->
- [x] 1.7 Implement row-window OFI normalization outputs in `data_preprocess/operator_futures/commodity/downscale.py`. <!-- 已实现: 输出 ofi_norm、ofi_bid_norm、ofi_ask_norm 并处理零分母 -->
- [x] 1.8 Verify the normalized OFI tests, OpenSpec strict validation, and the commodity downscale test module under the `finetf` conda environment. <!-- 已实现: OpenSpec 校验、完整 commodity downscale 测试和 py_compile 均通过 -->
- [x] 1.9 Add a focused test in `data_preprocess/tests/test_commodity_downscale.py` for rejecting NaN and infinite OFI depth values. <!-- 已实现: 添加 NaN/inf 输入 RED 测试 -->
- [x] 1.10 Implement non-finite OFI input validation in `data_preprocess/operator_futures/commodity/downscale.py`. <!-- 已实现: 拦截五档价格/数量列中的 NaN、inf 和 -inf -->
- [x] 1.11 Verify non-finite validation, OpenSpec strict validation, and the commodity downscale test module under the `finetf` conda environment. <!-- 已实现: 坏数据测试通过，最终验证通过 -->
