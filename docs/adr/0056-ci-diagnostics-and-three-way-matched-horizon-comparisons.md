---
status: accepted
---

# 补齐特征审计 CI 诊断证据链与三维共同样本期限比对

为了解决多预测期限特征审计中置信区间（CI）缺失原因未记录（`MISSING_CI_REASON_UNRECORDED`）以及跨期限比较缺失（`NOT_AVAILABLE`）的结构性缺口，对审计引擎及报告体系进行双向增强。用户于 2026-10-16 访谈确认此决定。

### 1. 物理根因与样本容量现实
在 10 分钟燃料油（fu）基准数据中，有效交易日总数为 370 天。研究协议 `evidence_protocol` 要求移动区块长度为 20 天、有效区块数 $\ge 20$（需要至少 400 个交易日）。当前实际仅产生 $\lfloor 370 / 20 \rfloor = 18$ 个有效区块（$18 < 20$），双倍敏感性区块为 $\lfloor 370 / 40 \rfloor = 9$ 个（$9 < 10$）。因此，Bootstrap 判定置信区间不满足渐近一致性要求属于正常统计拒绝。

### 2. CI 诊断证据链（不降低门槛凑数）
1. 严禁为了“有数字好看”而盲目下调区块长度或降低有效区块最低门槛（坚决捍卫 Protocol Q14 的统计严谨性）。
2. 在核心指标宽表 `feature_metrics.csv` 中如实记录结构化状态字段：`ci_status`（如 `INSUFFICIENT_EFFECTIVE_BLOCKS`）、`ci_status_2x`、`ci_valid_trading_days`、`ci_block_length`、`ci_effective_blocks`、`ci_min_effective_blocks`、`ci_null_reason`。
3. 在审计根目录输出期限级诊断总表 `ci_diagnostics.csv`，按 `horizon × target_kind` 汇总完整的块划分与门槛对比参数。

### 3. 三维共同样本期限比对（单行宽表呈现）
跨期限原生样本由于窗口边界 Purge 机制，样本覆盖率从 1h 的 99.30% 单调递减至 5day（Next5TD）的 86.13%，直接比较原生 IC 会引入严重的幸存者偏差。
1. 确立 `1h ↔ 1day (NextTD) ↔ 5day (Next5TD)` 三维共同交集样本基准。在 fu/10min 中，三维交集决策样本达到 39,661 条（占全量 86.01%），即 99.86% 的 5day 有效点均同时满足 1h 和 1day 标签有效性。
2. 产物 `matched_horizon_comparisons.csv` 采用单行宽表结构，将单个因子的全周期轨迹整合在同一行：
   - 基础信息：`feature`, `selected_streams`, `matched_samples`, `sample_retention_ratio`
   - 共同样本纯净 IC：`ic_common_1h`, `ic_common_1day`, `ic_common_5day`
   - 跨尺度配对增量：`delta_1h_to_1day`, `delta_1day_to_5day`, `delta_1h_to_5day`
   - 信号动力学轨迹：`trajectory_category`（如 `CLEAN_DECAY`, `CLEAN_EMERGENCE`, `DAY_RESONANCE`, `ALL_PERSISTENT`）
3. 彻底排除样本流失偏差，为多尺度门控强化学习 Agent 提供纯净的物理跨周期有效性输入。

### 4. 双向报告体系增强
1. **上游 `audit_report.md`**：新增 `Bootstrap Uncertainty & Horizon Block Diagnostics` 与 `Matched Common-Sample Horizon Comparisons` 章节，直接呈现真实区块数（$18 < 20$）与三维共同样本比对表。
2. **下游 `executive_summary.md`**：状态由 `NOT_AVAILABLE` 提升为 `COMPLETED_MATCHED_EVALUATION`，并基于单行宽表数据输出真实衰减/涌现归因，提示防伪归因边界。
