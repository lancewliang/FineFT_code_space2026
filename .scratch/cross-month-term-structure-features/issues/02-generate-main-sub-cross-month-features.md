# 02 — Generate Main-Sub Cross-Month Features

**What to build:** Extend the main contract summary so it records each contract's observed dynamic role for every completed trading day (`main`, `sub`, or `other`), then generate Main-Sub Dynamic Pairing features through the shared cross-month feature generation code path. For feature trading day T, cross-month feature generation must consume the previous available trading day's summary roles, never T's same-day roles, so the feature does not use future information from the current trading day. The generator should aggregate each contract independently to the target bar frequency, align to the current contract's feature output, and fill valid liquidity gaps with `0.0`.

**Blocked by:** 01 — Cross-Month Feature Contract And Guards.

**Status:** done

- [x] `main_contract.py` records each contract's ranking-period role as `main`, `sub`, or `other` in the main contract summary.
- [x] Main-Sub Dynamic Pairing records deterministic liquidity ranking and stable tie-break behavior.
- [x] For feature trading day T, Main-Sub Dynamic Pairing resolves `main`, `sub`, and `other` from the previous available trading day's summary roles.
- [x] Feature generation fails fast or follows an explicit documented fallback when no previous trading-day roles exist.
- [x] The generated cross-month feature output includes one-hot role features for whether the current contract is `main`, `sub`, or `other`.
- [x] When the current contract is `other`, the generated features describe its relative relationship to both the main contract and the sub contract.
- [x] The generated output contains `timestamp` and allowed cross-month relative feature columns only.
- [x] Cross-month generation consumes independently aggregated target-frequency bars before cross-month alignment.
- [x] Valid post-alignment liquidity gaps are filled with `0.0`.
- [x] Missing required source data fails fast rather than producing all-zero features.
- [x] Tests verify that feature day T uses previous available trading-day roles rather than same-day roles.
- [x] Tests verify the main/sub feature formulas, alignment behavior, gap fill, and No Absolute Price Rule enforcement.
