# 05: Contract-Normalized Decentralized Correlation Matrix and OOD-Aware Priority Scoring

**What to build:** Eliminate Simpson's paradox in `pipeline.py` by computing intra-contract correlation matrices $R_c$ within each contract and aggregating them via sample-weighted average $\bar{R} = \sum w_c R_c$, and re-weight Composite Score priority to favor stationary features ($0.40 \cdot \text{Rank}(1/\text{PSI}) + 0.35 \cdot \text{Rank}(|\overline{RankIC}|) + 0.25 \cdot \text{Rank}(\text{CatBoost\_Imp})$).

**Blocked by:** 03: Multi-Contract Distribution Drift Gate, 04: Cross-Contract Sign Consistency and RankIC Information Ratio Gates

**Status:** closed

- [x] Replace naive vertical concatenation of unscaled DataFrames with contract-wise correlation calculation and sample-size weighted averaging in `cor_util.py` and `pipeline.py`.
- [x] Composite Score priority ranking incorporates inverse mean PSI rank, prioritizing low-drift features during greedy correlation pruning.
- [x] Greedy correlation deduplication at threshold 0.70 runs on $\bar{R}$, preserving genuine stationary signals.
- [x] Unit tests verify that differences in contract baseline price levels (e.g. 2,000 vs 4,000) do not produce spurious correlations or distort feature selection.
