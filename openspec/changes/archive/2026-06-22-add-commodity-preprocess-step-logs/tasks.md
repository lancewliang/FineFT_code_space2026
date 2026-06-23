## 1. Step-level logging implementation

- [x] 1.1 Add focused shell/script tests for commodity step log naming, stage status output, stderr capture, and fail-fast behavior. <!-- 已实现: added main.sh-backed step log tests plus failure propagation coverage -->
- [x] 1.2 Add a reusable step logging wrapper in `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`. <!-- 已实现: added run_commodity_logged_step wrapper -->
- [x] 1.3 Route `stitch_main_contract`, `downscale_continuous_by_trading_day`, `cross_section`, `merge`, `concat`, `time_feature`, `merge_clean`, `ic_correlation`, and `scale_save` through the wrapper. <!-- 已实现: routed all nine commodity stages through step logging -->
- [x] 1.4 Preserve existing `main.sh` total log behavior and existing cross-section/merge per-date child logs. <!-- 已实现: kept main.sh unchanged and preserved child log paths -->

## 2. Verification

- [x] 2.1 Run focused commodity script tests and confirm all new log assertions pass. <!-- 已实现: focused commodity script tests pass -->
- [x] 2.2 Run `bash -n` on commodity shell scripts and `openspec validate add-commodity-preprocess-step-logs --strict`. <!-- 已实现: shell syntax and OpenSpec strict validation pass -->
