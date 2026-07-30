# 05 — Preserve Cross-Month Features Through Feature Selection

**What to build:** Configure the commodity futures full-process shell pipeline so Cross-Month Term Structure Feature columns flow through Feature Selection as `mandatory_state_features`. The Python Feature Selection interface already supports mandatory state features; this ticket wires the fixed cross-month column list into the shell invocation alongside Base_Time_feature columns.

**Blocked by:** 04 — Merge CROSS_MONTH_FEATURE Into Daily State Frame.

**注意:** 04 依赖 04a 先实际写出 `CROSS_MONTH_FEATURE` 文件，否则 required daily merge 没有上游产物可消费。

**Status:** done

- [x] The commodity futures shell pipeline defines the fixed cross-month feature column list.
- [x] The commodity futures shell pipeline passes Base_Time_feature columns and Cross-Month Term Structure Feature columns together through `--mandatory_state_features`.
- [x] The Python Feature Selection mandatory-feature behavior remains unchanged.
- [x] Shell tests verify that the full-process feature selection invocation preserves cross-month features.
