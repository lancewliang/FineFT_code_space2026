# 04: In-Memory Pre-Selection Winsorization and Data Hygiene Module

**What to build:** Create `operator_futures.feature_selection.muti_contract.data_hygiene` to execute near-zero variance filtering ($\sigma^2 \le 10^{-6}$), mode-frequency quasi-constant pruning ($\ge 0.98$), front-loaded blacklist/ablation, and double-sided in-memory Winsorization ($5	imes	ext{IQR}$) in Stage 1, while preserving the strict immutability of raw split files on disk.

**Blocked by:** 01-types-and-io-manager-extraction.md

**Status:** completed

- [x] Implement `execute_data_hygiene` in `data_hygiene.py` rejecting zero-variance and quasi-constant columns.
- [x] Apply in-memory $5	imes	ext{IQR}$ Winsorization on candidate state feature columns without altering source feather files on disk.
- [x] Integrate front-loaded blacklist and regex ablation rules.
- [x] Unit tests in `test_data_hygiene.py` verify that constant and pulse columns are dropped, extreme spikes are clamped in memory, and source feather files remain unchanged.
