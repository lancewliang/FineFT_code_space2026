# 03 — Join Mixed-frequency State Feature Into Daily Feature Assembly

**What to build:** Allow daily feature assembly to accept Mixed-frequency State Feature input and join it by timestamp into the future/state candidate feature frame only.

**Blocked by:** 01 — Generate Previous TradingDay Mixed-frequency State Feature; 02 — Generate Previous Calendar-week Mixed-frequency State Feature.

**Status:** complete

- [x] Mixed-frequency columns are present in the future/state candidate feature frame after merge.
- [x] Reward/Execution frame does not contain `prev_day_*` or `prev_week_*` columns.
- [x] Timestamp mismatch or missing required mixed-frequency columns fails fast.
