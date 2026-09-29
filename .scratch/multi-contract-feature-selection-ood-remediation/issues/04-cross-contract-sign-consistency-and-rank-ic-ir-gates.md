# 04: Cross-Contract Sign Consistency and RankIC Information Ratio Gates

**What to build:** Replace the defunct `RankIC_Std <= 1.0` stability check in `muti_contract/pipeline.py` with an active cross-contract directional sign consistency gate ($\text{SignConsistency} \ge 0.75$) and RankIC Information Ratio gate ($IR_{\text{RankIC}} \ge 0.40$), while elevating `min_abs_ic` to 0.02.

**Blocked by:** 02: Front-Loaded Feature Blacklist Execution in Multi-Contract Selection Pipeline, 03: Multi-Contract Distribution Drift Gate

**Status:** closed

- [x] Calculate `SignConsistency` as $\max(\sum \mathbb{I}(r_c > 0), \sum \mathbb{I}(r_c < 0)) / N_{\text{contracts}}$ across training contracts for each evaluated window.
- [x] Filter out any candidate feature failing $\text{SignConsistency} \ge 0.75$ on the target decision window.
- [x] Calculate $IR_{\text{RankIC}} = |\overline{RankIC}| / (s_{RankIC} + 10^{-6})$ and enforce $IR_{\text{RankIC}} \ge 0.40$ in `_ordered_filter_features`.
- [x] Elevate default `min_abs_ic` from 0.01 to 0.02 in parser and CLI pipelines.
- [x] Manifest and unit tests updated to track and verify sign consistency and IR filtering.
