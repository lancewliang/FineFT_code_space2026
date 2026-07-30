# 04 — Merge CROSS_MONTH_FEATURE Into Daily State Frame

**What to build:** Make daily merge consume generated `CROSS_MONTH_FEATURE` data as State Feature input. The merged future-state frame should include cross-month columns, preserve reward/execution separation, fail fast when required cross-month files are missing, and fill null values created by timestamp alignment with `0.0`.

**Blocked by:** 04a — 写出 CROSS_MONTH_FEATURE 文件.

**Status:** done

**说明:** 04a 已补齐上游生成入口和 shell 触发，daily merge required 模式现在有标准 `CROSS_MONTH_FEATURE` 产物可消费。

- [x] Daily merge joins `CROSS_MONTH_FEATURE` by `timestamp` into the future-state frame.
- [x] Cross-month columns do not appear in reward/execution outputs.
- [x] Required mode fails fast when the whole daily cross-month feature file is absent.
- [x] Null values in cross-month feature columns after merge are filled with `0.0`.
- [x] Tests verify successful merge, missing-file fail-fast behavior, and reward/state column separation.
