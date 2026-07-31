# 02 — Generate Previous Calendar-week Mixed-frequency State Feature

**What to build:** Generate `prev_week_*` Mixed-frequency State Feature columns for each target-frequency bar using only the previous complete natural week, with week membership derived from TradingDay.

**Blocked by:** 01 — Generate Previous TradingDay Mixed-frequency State Feature.

**Status:** complete

- [x] Bars in calendar week `W` receive weekly features from the immediately previous complete natural week before `W`.
- [x] Week membership is derived from TradingDay.
- [x] Monday and mid-week bars do not see current-week aggregation.
- [x] First-week and invalid-denominator cases produce deterministic finite values.
