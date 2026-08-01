---
status: proposed
---

# 重叠回看窗口与 BASE_FEATURE 路径布局（草稿）

> **状态：proposed（讨论稿，未决定实施）**
> 本 ADR 草稿**未**写入 `docs/adr/`，仅为讨论结论归档。
> 编号 0005 为占位，正式采纳时按 `docs/adr/` 现有最大编号递增。

## 上下文与决策

当前 `data_preprocess` 的 `target_freq` 同时承担「决策行间隔」与「聚合窗口宽度」两个职责，因 polars `group_by_dynamic` 省略 `period=` 时默认等于 `every`，二者被绑定为同一值，无法表达「每 1min 决策一次、每次回看 5min」的重叠回看窗口。

决定将 `target_freq` 拆分为 `decision_freq`（`every`，行间隔）与 `aggregation_window`（`period`，回看窗口宽度），并在 `group_by_dynamic` 中显式分别传参。当两者相等时行为与历史 `target_freq` 位级一致；大于时启用重叠回看窗口（本次拆分的核心目的）。`timestamp` 表示决策时间，即回看窗口右端点。

## 解耦机制

在 [downscale.py:78-89](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/data_preprocess/operator_futures/commodity/downscale.py#L78-L89) 的 `_resample` 中，由

```python
group_by_dynamic("timestamp", every=_polars_freq(target_freq), closed="right", label="right")
```

改为

```python
group_by_dynamic(
    "timestamp",
    every=_polars_freq(decision_freq),
    period=_polars_freq(aggregation_window),
    closed="right",
    label="right",
)
```

- `every=decision_freq`：决策行栅格。
- `period=aggregation_window`：每条决策行回看窗口宽度。
- `closed="right", label="right"`：窗口 `(t - aggregation_window, t]`，标记为 `t`，保持 **右闭右标窗口** 语义不变。

校验：`aggregation_window >= decision_freq`，违反时 fail-fast（Q2）。相等时 `period` 显式等于 `every`，与省略 `period` 的历史行为位级一致。

## 范围：仅 _resample 路径，OFI 不在内（Q5=a）

拆分仅作用于通过 `_resample` / `group_by_dynamic` 基于 `target_freq` 重采样的产物（OHLC、quote 统计、跨月 bar、混频 bar）。

OFI 与微观结构特征的 `window_rows` 仍为秒级快照行数（[downscale.py:1050-1066](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/data_preprocess/operator_futures/commodity/downscale.py#L1050-L1066) 的 `_ofi_row_index // window_rows` 作用于 `second_df`，不经 `_resample`），与拆分解耦，保持秒级分辨率。`_quote_window_stat_aggs` 在重采样 bin 内仅做 first/max/min/last/mean/std，不使用 `window_rows`。

## BASE_FEATURE 路径布局（Q6=b）

重采样输出同时依赖 `decision_freq` 与 `aggregation_window`，路径须编码两者。采用嵌套布局：

```
BASE_FEATURE/{symbol}/{contract}/{decision_freq}/{aggregation_window}/{date}.feather
```

跨月与混频特征从该路径读取已重采样 bar，并继承两个参数（Q12）——它们**没有**独立的 `aggregation_window`，`target_freq` 参数仅用于路径解析。原 `BASE_FEATURE/{symbol}/{contract}/{target_freq}/{date}.feather` 路径废弃。

## 交易日截断与部分窗口标志（Q3=b, Q7）

聚合窗口在连续交易日起点处截断，不跨连续交易日（Q7-i：仅当前连续交易日，不延伸至同一交易日内更早的连续交易日如夜盘）。交易日首个被截断的 bar **保留**（Q7-ii），并新增 bool 列 `_partial_window` 置 `True`；该列作为元数据排除在 State Feature 候选之外（同 Reward/Execution 列），不进入 RL agent 观测。

> ⚠️ Q11 细节（列名、置位条件、作用域、归属）本轮**未被明确确认**，以上为建议默认值，实施前需复核。

实现上 `group_by_dynamic` 不原生支持按连续交易日截断，需在窗口下界按当前连续交易日起点做 floor，或将重采样按连续交易日分组后分别执行再拼接。

## 间隔检测重锚定（Q13）

[downscale.py:1323-1333](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/data_preprocess/operator_futures/commodity/downscale.py#L1323-L1333) 的间隔*告警*使用 `_target_freq_delta(target_freq)` 检测 `current - previous > target_delta`，是行间隔检查，拆分后改用 `decision_freq`。`nquote == 0` 的空窗口丢弃规则（Q4）保持不变，仍按 `aggregation_window` 覆盖范围判定。

注意：重叠模式下相邻决策行共享数据，空窗口（`(t - aggregation_window, t]` 内无快照）出现概率降低，但丢弃规则语义不变。

## 迁移：硬切换（Q9=a）

移除 `target_freq`，所有 base feature 须以 `(decision_freq, aggregation_window)` 重新生成；旧 `BASE_FEATURE` 路径产物失效。因相等情形已证位级一致，等频配置可平滑映射为 `decision_freq = aggregation_window = 原 target_freq`，但路径布局变更使旧产物仍需重新生成到新路径。

## 下游推导修订

`日内 Bar 数` 改为按 `decision_freq` 推导（= 交易日时长 / decision_freq），与 `aggregation_window` 无关。

## 后果

- **正向**：获得重叠回看窗口能力；两个职责解耦，语义清晰；`timestamp` 决策时间语义明确。
- **代价**：相邻决策行数据共享导致连续状态高度相关（自相关），下游特征选择 IC 评估与 RL 训练需知晓该特性；`BASE_FEATURE` 全量重新生成；接口与路径破坏性变更。
- **可逆性**：低——路径布局与下游消费者变更使回退成本高，故记录本 ADR。

## 决策门槛

本拆分仅在「需要重叠回看窗口」时才有意义。若最终判定重叠窗口并非真实需求，拆分退化为纯重命名，应保留 `target_freq`，本 ADR 不采纳。
