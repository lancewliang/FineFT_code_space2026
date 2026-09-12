# 01: Core 2D Regime Calibration Engine

**What to build:** A standalone, mathematical core calibration engine that accepts multi-contract time-series data, extracts macro market turning-point segments, simultaneously calculates segment slope and log-return volatility, pools segment scores across contracts to fit tercile thresholds, strictly enforces the negative lower slope invariant ($T_{slope}[0] < 0 < T_{slope}[1]$), applies limit-state overrides, and maps every time step to a 3×3 orthogonal grid ID (`regime_grid_id \in [0, 8]`).

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] Turning-point segment-and-merge correctly extracts macro market segments and scores both percentage slope and return volatility for multi-contract datasets
- [ ] Pooled quantile calibration calculates tercile thresholds for slope and volatility across participating contracts
- [ ] Lower slope threshold is verified to be strictly negative ($T_{slope}[0] < 0.0$) and upper slope threshold strictly positive ($T_{slope}[1] > 0.0$), failing fast with an informative error if invariants are violated
- [ ] Every time step is assigned consistent discrete labels (`slope_label \in {0, 1, 2}`, `volatility_label \in {0, 1, 2}`, and synthesized `regime_grid_id = volatility_label * 3 + slope_label \in [0, 8]`)
- [ ] Limit-up and limit-down market states correctly override row labels to designated boundary extremes per project conventions
- [ ] Pure functional execution with zero disk I/O side effects, fully covered by automated unit tests
