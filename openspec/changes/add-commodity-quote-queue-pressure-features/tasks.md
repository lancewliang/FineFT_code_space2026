# Tasks

## 1. Implementation

- [ ] 1.1 Add focused quote queue pressure regression tests in `data_preprocess/tests/test_commodity_downscale.py`, covering refill/deplete counts, `queue_refill_imbalance`, zero-event neutral output, single-sided state ratios, missing limit-column fail-fast, non-finite input rejection, and preservation of existing microstructure outputs.
- [ ] 1.2 Extend `downscale_quote_microstructure_features()` and its private helpers in `data_preprocess/operator_futures/commodity/downscale.py`, preserving existing `downscale_quote_features()` and `downscale_quote_ofi_features()` behavior while adding queue pressure and single-sided ratio outputs.
- [ ] 1.3 Run focused validation for the quote queue pressure change, including Python compile checks for changed Python files, targeted pytest, and `openspec validate add-commodity-quote-queue-pressure-features --strict`.
