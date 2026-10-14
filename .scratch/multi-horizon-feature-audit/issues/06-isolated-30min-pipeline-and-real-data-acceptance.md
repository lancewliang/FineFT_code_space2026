# 06: Isolated 30min Input Pipeline & Dual-Frequency Real Data Acceptance on fu

**What to build:** Provide an isolated preprocessing utility to generate missing 30min Stage 1–6 inputs for the `fu` commodity symbol without executing Stage 7/8 or altering shared summaries. Execute the complete multi-horizon feature audit pipeline in the project's `finetf` Conda environment on real `fu` data for both 10min and 30min frequencies across all seven horizons. Run the comparison engine to produce real matched-frequency and matched-horizon audit comparisons. Verify zero side effects on the existing codebase by asserting SHA256 checksums of legacy Stage 6/7/8 inputs and configurations before and after execution. Record measured runtime and peak memory usage to complete and sign off the real-data acceptance matrix.

**Blocked by:** 04: Joint Moving-Block Bootstrap Uncertainty Protocol and Audit Report Generation, 05: Pairwise Matched-Horizon and Matched-Frequency Comparison Engine

**Status:** completed

- [x] Dedicated preprocessing tool generates 30min Stage 1–6 split inputs for `fu` (`train.feather`, `valid.feather`, `dataset_split_manifest.json`) without executing Stage 7/8 selection or modifying existing shared summary files.
- [x] Full audit pipeline runs to completion on real `fu` 10min data across all seven prediction horizons, producing valid run manifests, row labels, metrics, heatmaps, and audit reports.
- [x] Full audit pipeline runs to completion on real `fu` 30min data across all seven prediction horizons, producing valid run manifests, row labels, metrics, heatmaps, and audit reports.
- [x] Comparison CLI executes successfully on the completed 10min and 30min runs, publishing `audit_comparison_manifest.json` and a paired comparison report with real coverage metrics.
- [x] Real-data observations demonstrate expected invalid label behaviors, including `ZERO_LENGTH_HORIZON` on late-session Today targets and `CONTRACT_EXPIRED` on maturing contract tails.
- [x] Legacy isolation check confirms SHA256 fingerprints of existing 10min Stage 6/7/8 files, triple-stream blacklists, and pipeline configurations remain 100% identical before and after all audit executions.
- [x] Measured execution wall-clock time and peak RSS memory are documented in the final run manifest and acceptance report on the declared execution hardware.
