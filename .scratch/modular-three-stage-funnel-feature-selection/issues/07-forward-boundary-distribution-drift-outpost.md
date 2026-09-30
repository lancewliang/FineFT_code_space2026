# 07: Forward Boundary Distribution Drift Outpost Gate

**What to build:** Upgrade `operator_futures.feature_selection.muti_contract.distribution_audit` to support zero-isolated adaptive quantile binning and a Forward Boundary Drift Gate evaluating $	ext{PSI}_{	ext{forward}}(f) \le 0.15$ between the aggregated training distribution and the earliest chronological validation contract outpost ($C_{	ext{valid\_1}}$), reading strictly feature columns without reference prices or returns.

**Blocked by:** 04-pre-selection-winsorization-and-data-hygiene.md

**Status:** completed

- [x] Add zero-isolated adaptive quantile binning to `distribution_audit.py` to prevent bin collapse on zero-inflated distributions.
- [x] Implement forward boundary drift calculation against the earliest validation contract outpost.
- [x] Verify that validation frame reading extracts strictly state feature columns and excludes target prices and returns.
- [x] Unit tests in `test_distribution_audit.py` verify that zero-inflated features retain 10 bins and that forward drifting features are rejected.
