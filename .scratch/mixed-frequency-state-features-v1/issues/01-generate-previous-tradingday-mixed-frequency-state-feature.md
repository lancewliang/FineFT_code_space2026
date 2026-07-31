# 01 — Generate Previous TradingDay Mixed-frequency State Feature

**What to build:** Generate `prev_day_*` Mixed-frequency State Feature columns for each target-frequency bar using only the previous complete TradingDay. The output should cover prior-day return, range, candlestick shape, volume, tradeval, OpenInterest change, and turnover rate, with deterministic finite fallback for early rows.

**Blocked by:** None — can start immediately.

**Status:** complete

- [x] Bars in TradingDay `D` receive daily features from the immediately previous available TradingDay before `D`.
- [x] No `prev_day_*` feature uses current TradingDay statistics.
- [x] First-day and invalid-denominator cases produce deterministic finite values.
- [x] Missing required OHLCV/OpenInterest input columns fail fast.
