# 06: Within-Contract Augmented Dickey-Fuller Stationarity Gate

**What to build:** Create `operator_futures.feature_selection.muti_contract.stationarity_audit` to run Augmented Dickey-Fuller (ADF) unit-root stationarity tests on continuous contract series ($p_{	ext{ADF}} < 0.05$ on $\ge 70\%$ of contracts), enforce an automated fallback safeguard ($\ge 25$ survivors), and verify persistence half-life ($	au_{1/2} \ge 2.0$ bars) and sign alternation rate ($	ext{SAR} \le 0.40$).

**Blocked by:** 04-pre-selection-winsorization-and-data-hygiene.md

**Status:** completed

- [x] Implement `execute_stationarity_audit` in `stationarity_audit.py` with multi-contract ADF testing.
- [x] Implement automated fallback relaxation to $p < 0.10$ and minimum 25 survivors floor if too few features pass.
- [x] Implement lag-1 autocorrelation half-life and sign alternation rate turnover filtering across all features.
- [x] Unit tests in `test_stationarity_audit.py` verify rejection of random walks, passage of mean-reverting series, and fallback trigger behavior.
