# Tasks

## 1. Implementation

- [ ] 1.1 Add focused quote downscale regression tests in `data_preprocess/tests/test_commodity_downscale.py`, covering `imbalance_1/3/5` OHLC/TWAP/AWAP/STD outputs, `imbalance_1` compatibility with `imbalance_volume`, zero-denominator neutral output, non-finite volume fail-fast, and missing depth-column fail-fast.
- [ ] 1.2 Implement private quote-depth imbalance helpers and extend `downscale_quote_features()` in `data_preprocess/operator_futures/commodity/downscale.py`, preserving existing quote gap and limit single-sided behavior while adding `imbalance_1/3/5` and `std_imbalance_volume`.
- [ ] 1.3 Run focused validation for the quote-depth imbalance change, including targeted pytest, Python compile checks for changed Python files, and `openspec validate add-commodity-quote-depth-imbalance-features --strict`.
