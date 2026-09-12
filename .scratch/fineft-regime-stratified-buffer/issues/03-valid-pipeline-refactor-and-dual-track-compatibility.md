# 03: Validation Pipeline Refactor and Dual-Track Directory Compatibility

**What to build:** A refactored validation calibration pipeline that delegates core segment extraction and scoring to the shared 2D calibration engine while independently fitting validation tercile thresholds. The pipeline exports dual-track physical slice hierarchies (`slope` and `volatility`) to ensure complete backward compatibility with the downstream 2D agent selector, while embedding synthesized 2D regime grid IDs in all generated validation slices.

**Blocked by:** 01: Core 2D Regime Calibration Engine

**Status:** ready-for-agent

- [ ] Validation calibration delegates wave extraction and 2D scoring to the shared calibration engine
- [ ] Validation thresholds are fitted independently across all participating validation contracts with scope recorded as `valid_all_contracts`
- [ ] Dual-track directory structure (`valid/slope` and `valid/volatility`) is generated with valid slice files and slice manifests
- [ ] All exported validation slice files contain row-level `slope_label`, `volatility_label`, and `regime_grid_id` columns
- [ ] Downstream 2D agent selector can read generated validation directories and timestamps without any schema or structural incompatibility
- [ ] Automated validation pipeline tests verify independent calibration, directory structure, and slice column presence
