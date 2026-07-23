# Close Issues

## Resolved

- `data_preprocess/operator_futures/commodity/downscale.py:789` and `data_preprocess/operator_futures/commodity/downscale.py:852`
- `bid_side_empty_ratio` / `ask_side_empty_ratio` do not fully satisfy the spec wording for "price empty" states. The implementation rejects null best prices up front and only treats `size <= 0` as empty, so a true empty-price snapshot cannot be represented and zero-price sentinels are not counted either.
- This matters because the new ratio columns can silently become volume-only and miss valid single-sided states under the written requirement.
- Fix by aligning validation and empty-side detection with the intended representation of empty price, then add a regression test for a price-empty side state separate from the volume-zero case.
- Resolution: treat non-positive best prices as the empty-price sentinel while keeping null validation strict. Added a regression test for zero-price bid/ask sides with positive size and updated `_quote_side_empty_expr()` to count `price <= 0`.
