# 02: Full Seven-Horizon Semantics, Session Transitions, and Label Invalidation

**What to build:** Extend the independent feature audit pipeline across all seven required prediction horizons: 1h, 2h, 4h active trading minutes (60, 120, 240 active minutes skipping breaks and closures), and TradingDay-aligned endpoints for Today (D close), NextTD (D+1 close), Next5TD (D+5 close), and Next15TD (D+15 close). Provide rigorous handling for session crossings, overnight gaps, holidays, zero-length Today horizons, and contract expiry. Deliver fine-grained label invalidation tracking so that any missing target price, stale quote, expired contract, or non-positive price emits an explicit invalid reason code and validity mask rather than falling back to zeros, shorter horizons, or neighbouring contracts.

**Blocked by:** 01: Minimal Tracer Bullet: Audit Validation & End-to-End Single-Horizon Run

**Status:** completed

- [x] Hourly horizons accumulate exactly 60, 120, and 240 active trading minutes based on exchange session calendars, correctly crossing intra-day breaks, night sessions, and multiple trading days without pausing the clock for missing bars.
- [x] Multi-day horizons (Today, NextTD, Next5TD, Next15TD) resolve endpoints strictly to the day-session close of TradingDays D, D+1, D+5, and D+15 from the current decision moment without off-by-one counting errors.
- [x] Decision moments outside trading sessions (e.g. overnight or weekend bar timestamps) advance conservatively to the opening of the next active trading session.
- [x] Today labels with no remaining future interval before day-session close are marked with reason code `ZERO_LENGTH_HORIZON` while allowing multi-day and hourly horizons to evaluate independently.
- [x] Both directional signed terminal returns and absolute terminal returns are computed for all seven horizons for the same contract throughout the horizon span.
- [x] Unobservable endpoints due to contract expiry, missing tail observations, stale quotes (>60s), or ordering conflicts produce reasoned invalid labels with reason codes (`CONTRACT_EXPIRED`, `STALE_PRICE`, `MISSING_TARGET_PRICE`, `UNVERIFIED_TIMING`).
- [x] Output artifacts include row-level invalid reason records and feature-level valid coverage ratios across all seven horizons.
- [x] Automated tests verify active trading time accumulation across edge cases: lunch breaks, night-to-day transitions, weekends, holidays, exact session close endpoints, and partial bars.
