# 03 — Generate Delivery-Month Sequence Features

**What to build:** Extend the same cross-month feature generation code file used for Main-Sub Dynamic Pairing to generate Delivery Month Sequence Pairing features for `M_1`, `M_2`, and `M_3`. The slice should sort active contracts by real delivery month, produce relative price and liquidity structure features, and apply the same alignment, gap-fill, and fail-fast semantics.

**Blocked by:** 02 — Generate Main-Sub Cross-Month Features.

**Status:** done

- [x] Delivery Month Sequence Pairing sorts by real delivery month, not listing order or contract-name natural order.
- [x] The same cross-month feature generation file owns both Main-Sub Dynamic Pairing and Delivery Month Sequence Pairing generation.
- [x] The output includes allowed `M_1/M_2/M_3` relative price, butterfly, volume-ratio, and open-interest-ratio features.
- [x] Fewer-than-required active contracts and unparseable delivery months follow explicit fail-fast or documented fallback behavior.
- [x] Valid post-alignment liquidity gaps are filled with `0.0`.
- [x] Tests verify delivery-month ordering, feature formulas, shared generator behavior, and illegal absolute-price rejection.
