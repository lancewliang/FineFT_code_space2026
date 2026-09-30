# 02: Resequence Three-Stage Funnel and Implement Predictive Audit Module

**What to build:** Create `operator_futures.feature_selection.muti_contract.predictive_audit` encapsulating vectorized forward returns, IC, RankIC, cross-contract Sign Consistency ($\ge 0.75$), RankIC IR ($\ge 0.40$), and anti-causality anomaly screening. Resequence the pipeline so these fast $O(N)$ linear gates execute ahead of heavy machine learning model fitting.

**Blocked by:** 01-types-and-io-manager-extraction.md

**Status:** completed

- [x] Implement `execute_predictive_audit` in `predictive_audit.py` to evaluate forward returns across contracts and apply Hard RankIC, Sign Consistency, and Stability IR gates.
- [x] Include anti-causality sanity check flagging features with suspicious $|IC| > 0.30$.
- [x] Implement Benjamini-Hochberg False Discovery Rate (FDR $\le 0.05$) correction.
- [x] Unit tests in `test_predictive_audit.py` verify vectorized gate behavior, anti-causality detection, and FDR thresholding on synthetic multi-contract frames.
