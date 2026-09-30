# 05: Spearman Rank Correlation Matrix and Ward Hierarchical Clustering Deduplication

**What to build:** Create `operator_futures.feature_selection.muti_contract.orthogonal_dedup` to compute contract-normalized Spearman rank correlation matrices and perform Ward minimum-variance hierarchical clustering deduplication with dynamic capacity constraint $K \in [50, 70]$ and Variance Inflation Factor ($	ext{VIF} \le 10.0$) checks, while supporting a `--dedup_method {cluster, greedy}` compatibility flag.

**Blocked by:** 03-target-horizon-catboost-with-purged-embargo.md

**Status:** completed

- [x] Implement contract-normalized Spearman rank correlation matrix calculation in `orthogonal_dedup.py`.
- [x] Implement Ward hierarchical clustering dendrogram cutting with dynamic capacity calibration ($50 \le K \le 70$).
- [x] Verify that selected cluster representatives have $	ext{VIF} \le 10.0$.
- [x] Maintain `--dedup_method greedy` backward-compatible execution mode.
- [x] Unit tests in `test_orthogonal_dedup.py` verify rank invariance to nonlinear shifts, dynamic cluster capacity bounds, and VIF pruning.
