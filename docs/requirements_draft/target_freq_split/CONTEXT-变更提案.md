# CONTEXT.md 变更提案（草稿）

> **状态：讨论稿（未决定实施）**
> 本文件**不是**正式 `CONTEXT.md` 的一部分，仅为拟增改条目的预览。
> 实施前不应合并进仓库根目录的 `CONTEXT.md`。
>
> 依据：grill-with-docs 讨论结论（见同目录 `README.md` 决策表）。

---

## 拟新增条目

以下条目拟加入 `CONTEXT.md` 的 `### Data And Preprocessing` 小节（紧随 **右闭右标窗口** 之后）。

### **决策频率 (decision_freq)**

下采样输出 base feature bar 之间的行间隔时间，对应 polars `group_by_dynamic` 的 `every`，决定决策行的时间栅格密度。
_Avoid_: 决策周期、行频率、step_freq

### **聚合窗口 (aggregation_window)**

每条决策行回看聚合的时间窗口宽度，对应 polars `group_by_dynamic` 的 `period`；窗口区间为 `(t - aggregation_window, t]`，`t` 为决策时间（右端点）。当 `aggregation_window > decision_freq` 时相邻决策行共享数据，形成重叠回看窗口。
_Avoid_: lookback、回看期、window（不明确时）

### **聚合窗口约束 (Aggregation Window Constraint)**

`aggregation_window` 必须 `>= decision_freq`；相等时与历史 `target_freq` 行为位级一致。小于时无物理意义（会丢弃决策间数据），由 fail-fast 校验拒绝。该约束保证回看窗口始终覆盖到决策点的连续数据。
_Avoid_: 窗口大小限制

### **重叠回看窗口 (Overlapping Lookback Window)**

`aggregation_window > decision_freq` 时产生的相邻决策行数据共享现象；同一条秒级快照会出现在 `aggregation_window / decision_freq` 条输出行中，导致连续状态高度相关。这是本次拆分的核心新能力，下游特征选择与 RL 训练需知晓该自相关。
_Avoid_: 滑动窗口（不明确时）、rolling window

### **部分窗口标志 (_partial_window)**  ⚠️ 待确认（Q11）

base feature bar 上的 bool 元数据列，当回看窗口有效时长短于 `aggregation_window` 时置 `True`——包括连续交易日起点截断与数据起点不足两种情形。作为元数据排除在 State Feature 候选之外（同 Reward/Execution 列），不进入 RL agent 观测，仅用于诊断与下游门控。
_Avoid_: partial_flag、is_truncated

> 注：该条目细节（列名、置位条件、作用域、State Feature 归属）本轮讨论未被明确确认，以上为建议默认值，实施前需复核。

### **目标频率 (target_freq)**  [deprecated]

历史参数，同时表示决策行间隔与聚合窗口宽度（两者相等）。已被 `decision_freq` 与 `aggregation_window` 取代，采用硬切换移除（见 ADR-0005）。仅作为遗留术语保留在文档中，不再出现在代码接口与路径中。
_Avoid_: freq、frequency

---

## 拟修订条目

### **右闭右标窗口 (Right-closed Right-labeled Window)** —— 修订

区间 `(t - k, t]` 内的秒级快照聚合到标记为 `t` 的 bar，是 FineFT 下采样的标准窗口语义。修订明确：`k = aggregation_window`（聚合窗口宽度），决策行步长 `= decision_freq`（决策频率）；当 `k == decision_freq` 时窗口不重叠，等价于历史 `target_freq` 语义。窗口下界在连续交易日起点处被截断（不跨连续交易日），被截断的 bar 置 `_partial_window = True` 并保留。
_Avoid_: 右开窗口、左闭窗口

### **日内 Bar 数 (Bars Per Day)** —— 修订

根据商品期货品种 `Trading Session` 总交易时长和**决策频率** `decision_freq` 推导的每日 bar 数（= 交易日时长 / decision_freq），用于 Historical Volatility 等日化计算。修订明确：推导仅依赖 `decision_freq`（行间隔），与 `aggregation_window` 无关。
_Avoid_: 24 小时固定 bar 数、自然日 bar 数

---

## 拟新增条目（OFI 范围澄清，建议加入 `### Feature Engineering`）

### **OFI 窗口行数 (window_rows)**

OFI 与微观结构特征中按秒级快照行数分块聚合的参数（`_ofi_row_index // window_rows` 作用于 `second_df`），与 `decision_freq`/`aggregation_window` 解耦；`window_rows=12` 表示 12 个连续秒级快照行（标称 12 秒，存在缺秒间隔）。不在 target_freq 拆分范围内，保持秒级分辨率。
_Avoid_: ofi_window、聚合行数
