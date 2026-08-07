# Agent 形态分类：K 线形态与策略形态双独立分类器

## 背景

为给每个 `(label, epoch, bin_index)` agent triple 打上可解释的"类型"标签，以便回答"哪个 agent 在哪个 label 的哪个 K 线形态下用哪种策略盈利"，需要确定分类架构。前期讨论中混淆了"K 线形状"与"agent 动作曲线形状"，且在 segment 是否作为分类维度上反复。

## 决策

采用**两个正交的分类器**，输出一张明细表：

1. **K 线形态分类器**（行情侧）：读 `mark_price`/`volume` 序列，在 label_1~label_5 segment 内按 N=20 步不重叠窗口识别，输出 7 类——KT1 突破 / KT2 回调 / KT3 加速 / KM1 V 反转 / KM2 箱体 / KM3 背离 / KX1 涨跌停。KX1 专用于 label_0/label_6（trajectory 极短，median=3 步，不进窗口识别）。单窗口**单选**，命中顺序 `KX1→KM1→KT2→KT1→KT3→KM2→KM3→未分类`（KT2 优先于 KT1 以保留回调形态）。
2. **策略形态分类器**（agent 侧）：读行为轨迹整体（`position_after` + `mark_price` + `volume` + `cumulative_realized_pnl`）识别 agent 动作与行情的关系模式，6 类——ST1 突破即时 / ST2 回调加仓 / ST3 金字塔 / SM1 硬边界 / SM2 网格 / SM3 背离增强。单窗口**多选**，一个窗口可命中多类（如 ST1+SM3：阶跃动作 + 背离过滤不互斥）。
3. **明细表 schema**：`(label, epoch, bin_index, K 线形态, 策略形态, 盈亏值)`。构成 7×6 组合空间（K 线含 KX1）；一个 triple 可命中多行，天然支持"一个 agent 属于多种类型"。
4. **分类键** = `(label, epoch, bin_index)`，对齐 `selection_manifest.json` 的部署产物；`initial_action` 不进键（部署产物不带它，且 3 条起步强相关会重复计数），降级为诊断切片轴；`segment`（df_0..n）是 label 内文件切片，不构成分类维度，仅作为 K 线数据的来源文件。
5. **盈亏归因**：每行盈亏 = 窗口内已实现 PnL 变化 + 浮动 PnL 变化。K 线形态单选不重复计入；策略形态多选重复计入各命中行，聚合时按形态 distinct 计算总盈亏（不跨形态求和，避免放大）。
6. **阈值标定**：6 类判别阈值先用提议值实现，跑真实 Detail CSV 看特征分布再调（迭代式），不预先标定。

## 否决的替代

- **单列合一（K 线形态 = 策略，6 类）**：信息量低，无法表达"同一突破行情下不同 agent 用不同策略"。否决。
- **per-triple 直接分类（路线 B）**：不涌现多类型。否决。
- **segment 作分类维度**：segment 只表示一阶趋势同质性，不表示二阶 K 线形态；且 segment 内形态会演变。否决为分类维度，保留为数据来源文件。
- **整个 segment 作为 K 线形态识别窗口**：segment 内形态可能从突破演变到加速，整体识别会掩盖内部形态切换。采用 segment 内滑窗。
- **`initial_action` 进分类键**：与部署产物（`selection_manifest.json` 不带 initial_action）接缝错位，且 3 条起步强相关投票重复计数。否决。

## 后果

- 需新建两个分类器模块（行情侧 + agent 侧），各自标定阈值。
- 明细表行数 = triples × 窗口数 × 命中形态数，规模较大；需按需聚合视图。
- 6×6 组合空间稀疏（如 label_4 上涨段不会有 KM2 箱体形态），实际有效组合远小于 36。
- "agent 属于多种类型"通过 triple 命中多行涌现，无需额外 rollup 机制。
