# 05 — Verify Mixed-frequency State Feature End-to-end Contracts

**What to build:** Verify the OpenSpec v1 contracts end to end for Mixed-frequency State Feature support and keep v1 out-of-scope behavior unchanged.

**Blocked by:** 04 — Wire Mixed-frequency State Feature Through Commodity Feature Pipeline.

**Status:** complete

- [x] OpenSpec validation passes for the mixed-frequency change.
- [x] Relevant commodity futures feature engineering tests pass.
- [x] Outputs contain no NaN or Inf mixed-frequency feature values.
- [x] v1 does not introduce daily sliding-window, weekly sliding-window, current-day period-to-date, or current-week period-to-date features.
