# target_freq 拆分为 decision_freq + aggregation_window — 需求讨论稿

> **状态：讨论稿（未决定实施）**
> 本文档仅为 grill-with-docs 讨论会话的结论归档，**尚未决定是否实施**。
> 不应被视为已批准的需求或已生效的领域模型。`CONTEXT.md` 与 `docs/adr/` 未被修改。
>
> 创建时间：2026-08-01
> 讨论方式：grill-with-docs（结合 /domain-modeling 技能）

---

## 1. 背景

当前 `data_preprocess` 中的 `target_freq` 参数同时承担两个职责：

1. **决策行间隔**：输出 base feature bar 之间的时间间隔（即 polars `group_by_dynamic` 的 `every`）。
2. **聚合窗口宽度**：每个 bar 回看聚合的时间窗口宽度（即 `group_by_dynamic` 的 `period`，省略时默认等于 `every`）。

在 [downscale.py:78-89](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/data_preprocess/operator_futures/commodity/downscale.py#L78-L89) 的 `_resample` 中：

```python
def _resample(frame, target_freq, aggs):
    return frame.sort("timestamp").group_by_dynamic(
        "timestamp", every=_polars_freq(target_freq), closed="right", label="right"
    ).agg(*aggs).sort("timestamp")
```

由于 `period=` 被省略，`every == period == target_freq`，两个职责被绑定为同一个值，无法表达「每 1min 决策一次、但每次回看 5min」这类**重叠回看窗口**语义。

## 2. 目标

将 `target_freq` 拆分为两个独立参数：

- **`decision_freq`**：决策频率，输出行间隔时间（对应 `group_by_dynamic` 的 `every`）。
- **`aggregation_window`**：聚合窗口，每行回看聚合的时间窗口宽度（对应 `group_by_dynamic` 的 `period`）。
- **`timestamp`**：表示决策时间，即回看窗口的右端点。

**等价性约束**：当 `decision_freq == aggregation_window` 时，与当前 `target_freq` 的行为位级一致（`period` 省略即默认等于 `every`）。

## 3. 讨论结论汇总（Q&A 决策表）

| 编号 | 问题 | 结论 |
|------|------|------|
| Q1 | 是否支持重叠回看窗口（`aggregation_window > decision_freq`）？ | **是**。重叠回看窗口是本次拆分真正想要的新能力。 |
| Q1b | `window_rows_list` 的含义？ | **纠正，不重新定义**（见 Q5）。 |
| Q2 | 是否允许 `aggregation_window < decision_freq`？ | **不允许**；允许 `aggregation_window == decision_freq`（便于测试）。 |
| Q3 | 聚合窗口跨交易日边界时如何处理？ | **(b) 在当前连续交易日起点截断**，窗口不跨连续交易日。 |
| Q4 | 空窗口丢弃规则是否改变？ | **保持不变**：`(t - aggregation_window, t]` 内 `nquote == 0` 仍丢弃。 |
| Q5 | `window_rows_list` 是否随拆分重新锚定？ | **(a) 纠正而非重新定义**：`window_rows` 仍为秒级行数，OFI/微观结构特征保持秒级分辨率，**不在本次拆分范围内**。 |
| Q6 | `BASE_FEATURE` 路径布局？ | **(b) 嵌套**：`BASE_FEATURE/{symbol}/{contract}/{decision_freq}/{aggregation_window}/{date}.feather`。 |
| Q7(i) | 截断的「交易日起点」是否包含同一交易日内更早的连续交易日（如夜盘）？ | **仅当前连续交易日**，不跨连续交易日。 |
| Q7(ii) | 交易日首个（被截断的）bar 保留还是丢弃？ | **保留**，并增加一个「部分窗口」标志列。 |
| Q9 | 迁移策略？ | **(a) 硬切换**：移除 `target_freq`，所有 base feature 须以新参数对重新生成，旧产物失效。 |
| Q10 | 命名？ | **确认非对称命名** `decision_freq` / `aggregation_window`。 |
| Q11 | 部分窗口标志列的细节？ | **见下文「待确认项」**——本轮未明确确认，按建议默认值记录。 |
| Q12 | 跨月/混频特征是否有独立的 `aggregation_window`？ | **无**：它们消费已重采样的 base feature bar，从路径继承两个参数，仅用于路径解析。 |
| Q13 | 间隔检测 delta 如何重锚定？ | **确认**：间隔*告警*使用 `decision_freq`；`nquote==0` 丢弃规则使用 `aggregation_window` 覆盖范围。 |

## 4. 范围

### 4.1 在范围内（受拆分影响）

凡是通过 `_resample` / `group_by_dynamic` 基于 `target_freq` 重采样的特征产物：

- OHLC / quote 统计 bar（[downscale.py:600-627](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/data_preprocess/operator_futures/commodity/downscale.py#L600-L627)、[downscale.py:1321](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/data_preprocess/operator_futures/commodity/downscale.py#L1321)）
- 跨月结构特征（[cross_month_feature.py:232](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/data_preprocess/operator_futures/commodity/cross_month_feature.py#L232)）——从 base feature 路径继承两参数
- 日级/周级混频状态特征（[mixed_frequency_feature.py:115](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/data_preprocess/operator_futures/commodity/mixed_frequency_feature.py#L115)、[daily_mixed_frequency_feature.py:307](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/data_preprocess/operator_futures/commodity/daily_mixed_frequency_feature.py#L307)、[weekly_mixed_frequency_feature.py:172](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/data_preprocess/operator_futures/commodity/weekly_mixed_frequency_feature.py#L172)）——从路径继承两参数
- `BASE_FEATURE` 路径布局与 `日内 Bar 数` 推导

### 4.2 不在范围内（保持现状）

- **OFI / 微观结构特征**：`window_rows` 仍为秒级快照行数，与 `target_freq`/`decision_freq`/`aggregation_window` 解耦。
  - 依据：[downscale.py:1050-1066](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/data_preprocess/operator_futures/commodity/downscale.py#L1050-L1066) 中 `_ofi_row_index // window_rows` 作用于 `second_df`（秒级），不经过 `_resample(target_freq)`。
  - `_quote_window_stat_aggs`（[downscale.py:940-958](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/data_preprocess/operator_futures/commodity/downscale.py#L940-L958)）在 `target_freq` 重采样 bin 内仅做 first/max/min/last/mean/std，不使用 `window_rows`。
- 日级/周级混频特征内部使用的「上一完整交易日 / 上一完整自然周」滚动窗口（与 `target_freq` 无关，保持现有可见性约束）。

## 5. 待确认项（Q11）

部分窗口标志列的细节本轮**未被明确确认**，以下为建议默认值，实施前需复核：

- **列名**：`_partial_window`（bool），沿用仓库内部列下划线前缀惯例（如 `_source_line_number`、`_ofi_row_index`）。
- **置 True 条件**：当回看窗口的有效时长短于 `aggregation_window` 时——无论是因为交易日起点截断（Q3/Q7），还是因为数据集/合约的最开始处数据起点不足一个完整窗口。两种情况统一处理，避免特殊分支。
- **作用域**：在 base feature bar（downscale 输出）上设置；跨月/混频等下游消费者从 bar 继承该标志。
- **State Feature 归属**：作为元数据列排除在 State Feature 候选之外（归类同 **Reward/Execution 列**），不进入 RL agent 观测，仅用于诊断与下游门控。

## 6. 影响面概览

- **接口变更**：`downscale_*`、`write_*_feature_for_day` 等函数签名中 `target_freq: str` 替换为 `decision_freq: str, aggregation_window: str`。
- **路径变更**：`BASE_FEATURE` 目录多一层，旧路径全部失效（Q9 硬切换）。
- **行为新增**：重叠回看窗口（`aggregation_window > decision_freq`）——相邻决策行共享数据，产生高度相关的连续状态，下游训练与特征选择需知晓。
- **校验新增**：`aggregation_window >= decision_freq`（Q2），违反时 fail-fast。
- **交易日截断**：需在 `group_by_dynamic` 之外按连续交易日对窗口下界做 floor 处理，并写入 `_partial_window` 标志。
- **间隔检测**：[downscale.py:1323-1333](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/data_preprocess/operator_futures/commodity/downscale.py#L1323-L1333) 的 gap 告警 delta 改用 `decision_freq`。
- **下游推导**：`日内 Bar 数` 改为按 `decision_freq` 推导（= 交易日时长 / decision_freq）。

## 7. 本目录文件清单

| 文件 | 说明 |
|------|------|
| `README.md` | 本文件——需求讨论稿总览 |
| `CONTEXT-变更提案.md` | 拟对 `CONTEXT.md` 术语表的增改条目（草稿，未写入正式 CONTEXT.md） |
| `ADR-0005-draft-重叠回看窗口与路径布局.md` | 拟新增 ADR 草稿（status: proposed，未写入 `docs/adr/`） |

## 8. 决策门槛提醒

本次拆分仅在「需要重叠回看窗口」时才有意义（Q1=是）。若最终判定重叠窗口并非真实需求，则拆分退化为纯重命名，应直接保留 `target_freq`，本讨论稿可归档不实施。
