# 05: Pairwise Matched-Horizon and Matched-Frequency Comparison Engine

**What to build:** Deliver the independent `compare` CLI subcommand for matched-sample comparisons across horizons and data frequencies. Execute strictly matched pairwise evaluations on common valid observations (matching exact contract, decision time, and evaluation date), keeping native valid-sample metrics distinct from matched-sample metrics. Compute paired difference statistics and paired bootstrap difference intervals. Output a versioned comparison manifest (`audit_comparison_manifest.json`) and a comparative Markdown report with tabular diffs and scatter summaries, providing explicit diagnostics when inputs lack overlapping samples or carry incompatible protocol versions.

**Blocked by:** 04: Joint Moving-Block Bootstrap Uncertainty Protocol and Audit Report Generation

**Status:** completed

- [x] CLI subcommand `compare` loads two or more completed audit run manifests and verifies compatibility of schema version, protocol version, and target definitions.
- [x] Matched-horizon comparison aligns observations across different horizons within a run to their common valid intersection, preserving native sample metrics alongside matched metrics.
- [x] Matched-frequency comparison aligns 10min and 30min runs on exact contract, decision timestamp, and evaluation date keys without interpolation or nearest-neighbour approximations.
- [x] Paired difference statistics and paired bootstrap difference intervals are computed for correlation and absolute return metrics.
- [x] Comparisons with empty or insufficient common observation sets output explicit `NO_COMMON_SAMPLES` status without fabricating zero differences or imputed values.
- [x] Versioned `audit_comparison_manifest.json` is published with input run identifiers, common sample counts, and execution metadata.
- [x] Comparative Markdown report is rendered, detailing paired correlation differences, denominator contrasts, and explanatory notes regarding native lookback window differences.
- [x] Automated tests assert proper alignment on synthetic dual-frequency datasets, rejection of mismatched protocol versions, and mathematical correctness of paired statistics.
