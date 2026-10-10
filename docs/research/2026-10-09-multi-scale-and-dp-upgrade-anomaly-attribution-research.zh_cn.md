# 多尺度跨频状态与离线 DP 专家因果硬锁升级异常深度归因研究报告

- **报告日期**：2026-10-09
- **研究员**：AI Quantitative Research Team
- **实验标识**：`fu/10min_parallel`（升级后 Trial 67 对比历史基线 Trial 200）
- **对应数据与产物目录**：
  - 最新实验诊断数据：`analysis_result/DiHFT/high_level_heurstic/fu/10min_parallel/diagnostics/`
  - 最新选优结果：`analysis_result/DiHFT/high_level_heurstic/fu/10min_parallel/best_result.csv`
  - 历史基线诊断报告：`docs/research/2026-10-08-dihft-latest-experiment-diagnostics-report.zh_cn.md`
  - 特征工程与选择清单：`PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/10min/fu/train/feature_selection_manifest.json`
  - 分布漂移审计指标：`PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/10min/fu/train/distribution_audit_metrics.csv`
  - 核心源码定位：
    - `FineFT/RL/DiHFT/high_level/vae_routing_util.py:1122`（`reset_routing_state`）与 `1460`（`holding_return`）
    - `FineFT/env/env_class/futures_util.py:1506-1518`（`create_optimal_q_table` 因果硬锁）
    - `data_preprocess/operator_futures/feature_selection/muti_contract/distribution_audit.py:257-270`
- **执行方法论依据**：`docs/research/dihft_strategy_diagnostics_methodology.zh_cn.md`
- **关联架构决策**：ADR 0051（因果体制开仓硬锁与移动止盈）、ADR 0054（多尺度跨频状态与三频预测漏斗）

---

## 1. 执行摘要与评测实证对比 (Executive Summary & Empirical Contrast)

### 1.1 核心结论：负向异常确证
根据 ADR 0051 与 ADR 0054 实施的“多尺度跨频状态表征”与“离线 DP 专家因果硬锁”升级后，最新完整评测实验（Trial 67）不仅未实现预期的性能提升，反而出现了**系统性业绩暴跌与多合约大面积失真**。
总体判定结论为：**升级严重不及预期，策略出现多重软硬件级深层短路（CRITICAL FAILURE / REGRESSION）**。

### 1.2 组合核心评价指标对比矩阵

| 评价维度 | 核心量化指标 | 2026-10-08 基线 (Trial 200) | 2026-10-09 最新升级 (Trial 67) | 变动幅度 | 状态判定 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **净盈利能力** | 组合净回报率 (`portfolio_return_pct`) | **`+6.47%`** (+4,655.71 元) | **`+1.53%`** (+1,104.73 元) | **-76.3%** | **严重倒退** |
| **盘面择时能力** | 盘面毛回报率 (`portfolio_gross_return_pct`) | **`+7.42%`** (+5,345.00 元) | **`+2.56%`** (+1,840.00 元) | **-65.6%** | **Alpha 严重衰退** |
| **横截面稳健性** | 多合约胜率 (`win_rate_pct`) | **`100.0%`** (12/12 全面盈利) | **`66.67%`** (仅 8/12 盈利，4 支亏损) | **-33.33 pct** | **不合格** |
| **风险调整收益** | 平均年化夏普比率 (`mean_annual_sr`) | **`1.22`** | **`0.33`** | **-73.0%** | **不合格** |
| **摩擦吞噬控制** | 摩擦吞噬率 (`friction_consumption_ratio_pct`) | **`12.90%`** (689.29 元) | **`39.96%`** (735.27 元) | **+27.06 pct** | **显著恶化** |
| **单点尾部风险** | 最差单合约亏损占比 (`worst_loss / capital`) | **`0.0%`** (无亏损合约) | **`-4.43%`** (`fu2508` 亏损 265.98 元) | **尾部风险重现** | **预警** |
| **调仓交易频次** | 全周期交易次数 (`total_trades`) | `385` 次 | `320` 次 | -16.9% | 变相更低频 |
| **资金换手倍数** | 组合本金换手倍数 (`portfolio_turnover_multiple`) | `16.36x` | `13.57x` | -17.1% | 换手略降 |

### 1.3 逐合约全维度收益与行为对比明细

| 合约代码 | 评估区间 | 步数 | 标的 B&H 1x | 基线净利 (10-08) | 本次净利 (10-09) | 净利变动额 | 本次毛利 | 本次摩擦 | 本次交易数 | 本次年化SR | 异常归因速览 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **fu2409** | 07-23 ~ 08-16 | 727 | -2.47% | +2.08 | **+82.12** | +80.04 | +111.0 | 28.88 | 8 | 1.20 | 表现正常，做空捕捉下跌 |
| **fu2411** | 07-23 ~ 10-17 | 2,143 | -0.77% | +429.18 | **+566.98** | +137.80 | +588.0 | 21.02 | 12 | 2.64 | 唯一触发追踪止盈，遗留幽灵价格 |
| **fu2412** | 09-06 ~ 11-15 | 1,605 | +16.88% | +395.36 | **+39.73** | **-355.63** | +23.0 | -16.73 | 10 | 0.79 | 被幽灵价格拦截 31 次开多 |
| **fu2501** | 07-23 ~ 12-17 | 3,819 | +7.64% | +93.96 | **+121.35** | +27.39 | +191.0 | 69.65 | 24 | 0.67 | 被幽灵价格拦截 207 次开多 |
| **fu2503** | 09-06 ~ 02-14 | 3,869 | +36.83% | +1,004.14 | **-173.22** | **-1,177.36** | -143.0 | 30.22 | 28 | -1.01 | **灾难性崩盘**：被幽灵价格拦截 131 次 |
| **fu2505** | 10-15 ~ 04-16 | 4,764 | +2.37% | +339.17 | **+37.54** | **-301.63** | +222.0 | 184.46 | 66 | 0.13 | 利润严重缩水 89% |
| **fu2507** | 01-09 ~ 06-16 | 3,903 | +3.23% | +508.30 | **-181.61** | **-689.91** | -74.0 | 107.61 | 52 | -0.62 | 由盈转亏，空头占比过高逆势亏损 |
| **fu2508** | 01-09 ~ 04-16 | 2,880 | -1.39% | +512.00 | **-265.98** | **-777.98** | -184.0 | 81.98 | 38 | -1.52 | 由盈转亏，微观判断严重失灵 |
| **fu2509** | 01-09 ~ 06-16 | 3,578 | -8.66% | +194.20 | **+203.96** | +9.76 | +336.0 | 132.04 | 50 | 0.72 | 表现基本持平 |
| **fu2510** | 02-14 ~ 06-16 | 1,263 | +6.50% | +525.68 | **+543.28** | +17.60 | +583.0 | 39.72 | 16 | 3.18 | 表现稳定 |
| **fu2511** | 06-10 ~ 06-27 | 535 | +2.36% | +0.00 | **-27.79** | -27.79 | +3.0 | 30.79 | 2 | -3.41 | 滑点手续费吞噬导致微亏 |
| **fu2601** | 05-13 ~ 06-27 | 1,265 | +6.28% | +337.89 | **+158.36** | -179.53 | +184.0 | 25.64 | 14 | 1.24 | 利润缩水 53% |
| **全组合** | **合计 / 均值** | **30,351** | - | **+4,655.71** | **+1,104.73** | **-3,550.98** | **+1,840.0** | **735.27** | **320** | **0.33** | **毛利塌缩 65.6%，净利塌缩 76.3%** |


---

## 2. 根因一：特征选择分布漂移门误杀导致宏观连续动量特征全军覆没 (Macro Continuous Vision Zeroed by PSI Drift Gate)

### 2.1 理论预期与实际产物的严重背离
在 ADR 0054 中，架构明确要求为系统引入跨越 720 与 1440 步（覆盖 20~40 交易日）的连续宏观算子，包括：
- 长期价格趋势斜率与波动比：`trend_beta_720`, `trend_beta_1440`, `trend_to_noise_720`, `trend_to_noise_1440`
- 标度不变价格偏离与动量变化率：`mark_price_ema_deviation_720`, `mark_price_ema_deviation_1440`, `mark_price_roc_720`, `mark_price_roc_1440`
- 跨月期限结构长周期滚动 Z-score：`cm_current_main_spread_rolling_zscore_720/1440`, `cm_main_sub_spread_rolling_zscore_720/1440`

其理论预期是：底层 Q-Net 获得连续因果宏观视界，在面临大牛市中途 2~3 小时的微观急跌时，长期偏离度与动量特征将死死锚定多头认知，根除逆势做空摸顶。

然而，审查最新的特征选择清单 `feature_selection_manifest.json` 与状态特征集 `rl_state_features.npy` 发现：
- **宏观连续价格动量算子入选数为：0！**
- 底层强化学习决策流 `rl_stream` 中，最终入选的 Macro 特征仅有 4 个：
  1. `prev_5_day_trade_imbalance_quantile_rank`
  2. `prev_10_day_trade_imbalance_quantile_rank`
  3. `prev_15_day_trade_imbalance_quantile_rank`
  4. `prev_4_week_trade_imbalance_quantile_rank`
- 这 4 个特征全为日频成交量失衡的分位数排名，**完全不包含任何价格方向、均线偏离度或宏观趋势斜率信息**！低层智能体在价格维度依然是彻头彻尾的“宏观盲人”。

### 2.2 误杀发生的具体机制：分布漂移审计门控逻辑短路
审查 `data_preprocess/operator_futures/feature_selection/muti_contract/distribution_audit.py` 与 `distribution_audit_metrics.csv`：
特征选择流水线在将特征输入“三尺度分层前瞻漏斗”（Predictive Funnel）之前，首先执行跨合约分布漂移审计（Distribution Drift Audit）：
```python
# distribution_audit.py:257-270
def _feature_passes_drift(f_name: str, eff_base_mean: float, eff_base_pair: float) -> bool:
    f_tier = classify_feature_scale(f_name)
    tier_cfg = SCALE_TIER_CONFIGS.get(f_tier)
    eff_mean = max(eff_base_mean, tier_cfg.max_mean_psi if tier_cfg else 0.0)
    eff_pair = max(eff_base_pair, tier_cfg.max_pair_psi if tier_cfg else 0.0)
    return (
        mean_psi_dict[f_name] <= eff_mean
        and max_pair_psi_dict[f_name] <= eff_pair
        and (
            forward_outpost_frame is None
            or forward_psi_dict[f_name] <= forward_outpost_max_psi
        )
    )
```
根据 `types.py:SCALE_TIER_CONFIGS`，Macro 层的准入阈值为：`max_mean_psi = 0.35`，`max_pair_psi = 1.00`。

### 2.3 实测 PSI 崩塌数据证据
下表提取自 `PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/10min/fu/train/distribution_audit_metrics.csv` 中全部 16 个宏观连续长周期特征的实测审计结果：

| 连续宏观特征算子 | 跨合约平均 PSI (`mean_psi`) | 最大配对 PSI (`max_pair_psi`) | 前瞻验证 PSI (`forward_psi`) | 阈值要求 (`mean<=0.35, pair<=1.00`) | 漂移门判定 (`drift_passed`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `trend_beta_720` | **3.515** | **16.396** | **9.504** | 严重超标 | **False (Dropped)** |
| `trend_to_noise_720` | **1.001** | **7.178** | **0.917** | 严重超标 | **False (Dropped)** |
| `trend_beta_1440` | **6.896** | **24.721** | **5.287** | 严重超标 | **False (Dropped)** |
| `trend_to_noise_1440` | **3.041** | **17.516** | **3.875** | 严重超标 | **False (Dropped)** |
| `mark_price_ema_deviation_720` | **1.224** | **10.315** | **1.656** | 严重超标 | **False (Dropped)** |
| `mark_price_roc_720` | **2.967** | **17.817** | **8.371** | 严重超标 | **False (Dropped)** |
| `mark_price_ema_deviation_1440`| **3.197** | **25.799** | **3.200** | 严重超标 | **False (Dropped)** |
| `mark_price_roc_1440` | **5.743** | **25.066** | **6.292** | 严重超标 | **False (Dropped)** |
| `cm_current_main_spread_rolling_zscore_720` | **3.988** | **21.978** | **8.005** | 严重超标 | **False (Dropped)** |
| `cm_spread_rolling_zscore_720` | **3.988** | **21.978** | **8.005** | 严重超标 | **False (Dropped)** |
| `cm_current_main_spread_rolling_zscore_1440`| **4.375** | **23.323** | **10.304**| 严重超标 | **False (Dropped)** |
| `cm_spread_rolling_zscore_1440` | **4.375** | **23.323** | **10.304**| 严重超标 | **False (Dropped)** |
| `cm_current_sub_spread_rolling_zscore_720` | **4.473** | **25.726** | **2.637** | 严重超标 | **False (Dropped)** |
| `cm_current_sub_spread_rolling_zscore_1440`| **4.901** | **25.945** | **4.365** | 严重超标 | **False (Dropped)** |
| `cm_main_sub_spread_rolling_zscore_720` | **0.609** | **2.835** | **1.908** | 严重超标 | **False (Dropped)** |
| `cm_main_sub_spread_rolling_zscore_1440` | **1.331** | **6.887** | **4.333** | 严重超标 | **False (Dropped)** |

### 2.4 数学本质剖析：将“周期体制差异”错误当成“特征不稳定漂移”
跨合约 PSI（Population Stability Index）公式为：
$$\text{PSI} = \sum_{k=1}^K (P_k - Q_k) \ln\left(\frac{P_k + \epsilon}{Q_k + \epsilon}\right)$$
- 期货多合约横截面数据横跨数年：例如 `fu2503` 是持续上涨超 36% 的大牛市合约，而 `fu2509` 则是持续下跌超 8% 的大熊市合约。
- 在大牛市合约中，长周期价格动量 `mark_price_roc_720` 的取值集中在正区间（$[+0.05, +0.25]$）；在大熊市合约中，其取值集中在负区间（$[-0.20, -0.05]$）。
- 当直方图分箱计算两者的相对经验概率分布时，牛市合约在负值分箱上的概率接近 0，熊市合约在正值分箱上的概率接近 0，导致对数比率项 $\ln(P_k / Q_k)$ 剧烈爆炸，单对合约的 PSI 轻松突破 10~25！
- 相比之下，唯一幸存的 4 个特征是因为采用了**合约内部横截面分位数排序（Quantile Rank）**：分位数转换强行将所有分布映射为 $[0, 1]$ 上的均匀分布 $U(0, 1)$，因而其跨合约 PSI 恒等于 0。
- **致命缺陷**：分布漂移审计门缺乏对长周期宏观连续因子的自适应认知，将金融市场必然存在的“宏观大势周期波动”简单粗暴地判定为“特征不可用漂移”，在第一道门禁直接灭活了 100% 的连续宏观动量因子。ADR 0054 的设计初衷在数据预处理阶段便彻底落空。


---

## 3. 根因二：DP 专家因果锁注入极端惩罚导致底层 Q-Net 状态混叠崩溃 (DP Oracle Bellman Lock & POMDP State Aliasing)

### 3.1 离线 DP 专家表的因果硬锁逻辑
在 `FineFT/env/env_class/futures_util.py:1506-1518` 中，系统为离线动态规划（DP）最优 Q 表求解注入了趋势开仓硬锁：
```python
# futures_util.py:1506-1518 (create_optimal_q_table)
elif (
    enable_trend_entry_lock
    and regime_grid_ids_array is not None
    and (
        (int(regime_grid_ids_array[current_timestamp_index]) % 3 == 2 and future_position < 0)
        or (int(regime_grid_ids_array[current_timestamp_index]) % 3 == 0 and future_position > 0)
    )
):
    q_table[current_timestamp_index, current_action, future_action] = (
        -max_punishment  # 赋以 -1e10 的绝对死亡惩罚
    )
```
在训练数据加载模块 `FineFT/RL/DiHFT/low_level/qtable_config.py:34` 中，`enable_trend_entry_lock: bool = True` 默认被强制开启。

### 3.2 离线 DP 求解的信息特权与 Bellman 递推
离线 DP 专家在求解时，依赖分段线性回归（PLR）拟合后标注的高层网格标签 `regime_grid_id`。
- 在标注为牛市的切片（`regime_grid_id % 3 == 2`），只要底层智能体尝试产生空头持仓（`future_position < 0`），转移回报直接被强制置为 $-10^{10}$。
- 在标注为熊市的切片（`regime_grid_id % 3 == 0`），只要底层智能体尝试产生多头持仓（`future_position > 0`），转移回报同样被置为 $-10^{10}$。
- 在向后逆序递推（Backward Bellman Optimality）过程中，最优动作序列 $a^*_t = \arg\max_a Q^*(s_t, a)$ 彻底杜绝了任何“逆势开仓”动作。

### 3.3 POMDP 状态混叠（State Aliasing）的数理矛盾
然而，在强化学习预训练阶段（`FineFT/RL/DiHFT/low_level/parallel_pretrain.py:150-165`，`run_exhaustive_warmup`），系统将 DP 专家的示范序列收集进回放经验池，并让底层 Q-Net 进行拟合学习：
1. **输入信息的物理隔离**：根据 ADR 0037/0041/0051 的三流特征解耦原则，低层 Q-Net **绝不允许输入 `regime_grid_id`**，以防信息泄漏和过拟合。
2. **宏观视野的物理缺失**：正如根因一所揭示，由于分布漂移门误杀，底层 Q-Net 的 139 维状态向量中完全不包含任何连续宏观价格动量特征。
3. **部分可观测马尔可夫决策过程（POMDP）状态混叠**：
   - 考虑某一局部微观形态 $s_{\text{local}}$（例如：10分钟内盘口卖单积压、短周期 RSI 超买、局部微跌）：
   - 在历史牛市切片中，由于 `regime_grid_id` 施加的硬锁，DP 专家在此刻强制执行买入持有或观望，严禁做空（做空对应 $-10^{10}$）；
   - 在历史熊市切片中，面对高度相似的 $s_{\text{local}}$，DP 专家在此刻要求积极做空并严禁买入（买入对应 $-10^{10}$）。
   - 对于输入仅为 $s_{\text{local}}$ 的底层学生网络而言，**同一微观状态在经验池中被映射到了截然相反的最优策略分布与相差 $10^{10}$ 数量级的 Q 值回归目标！**
4. **梯度爆炸与策略瘫痪**：
   - 传统的 Q-learning 与 MSE TD 损失函数根本无法收敛于这种条件依赖于不可见变量的离散二值极端跳变分布；
   - 强行反向传播导致学生网络的隐藏层权重遭受严重梯度扰动与价值扭曲，子网络不仅未能学会“识别大趋势顺势而为”，反而丧失了微观层面的正常择时敏感度，导致子模型整体质量严重退化。


---

## 4. 根因三：高层评测状态泄漏导致跨合约幽灵突破价冻结交易 (Cross-Contract State Leakage & Phantom Hurdle Lock)

### 4.1 核心实证现象：牛市合约被幽灵冷却锁死
在逐合约分析中，`fu2503` 的业绩坍塌最为触目惊心：
- 该合约标的现货价格在评估期内暴涨 **`+36.83%`**（对应 5x 杠杆标的涨幅 `+184.14%`），为过去两年最强劲的四个月单边大牛市。
- 在 2026-10-08 基线（Trial 200）中，策略顺应大势斩获 **`+1,004.14` 元**（+16.74% 净回报），年化夏普比率高达 **2.07**。
- 但在本次升级（Trial 67）中，`fu2503` 的净利润不仅归零，更是反向亏损 **`-173.22` 元**（-2.89%），单合约回吐高达 **1,177.36 元**！

通过提取并解析 `result/DiHFT/high_level/fu/10min_parallel/..._trial_67/contracts/` 下的动作决策原因序列 `action_decision_reason_history.npy`，发现了震撼的数据实证：

| 合约代码 | 评估总步数 | 实际触发移动止盈 (`Reason 9`) | 触发追踪止盈冷却拦截 (`Reason 10`) | 趋势硬锁开仓拦截 (`Reason 8`) | 核心异常现象与机制剖析 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `fu2409` | 727 | 0 | 0 | 1 | 正常 |
| `fu2411` | 2,143 | **2** | 6 | 1 | **污染源**：在合约末期高点（~3,300 元）触发 2 次止盈 |
| `fu2412` | 1,605 | 0 | **31** | 4 | **被幽灵锁拦截 31 次开多**（本身 0 次止盈） |
| `fu2501` | 3,819 | 0 | **207** | 6 | **被幽灵锁拦截 207 次开多**（本身 0 次止盈） |
| `fu2503` | 3,869 | 0 | **131** | 5 | **被幽灵锁拦截 131 次开多**（本身 0 次止盈，错失大牛市） |
| `fu2505` | 4,764 | 0 | **7** | 12 | **被幽灵锁拦截 7 次开多** |
| `fu2507` ~ `fu2601` | 13,423 | 0 | 0 | 25 | 价格突破 3,300 元后幽灵门槛自动失效 |

在 `fu2411` 之后的 4 支合约中，移动追踪止盈本身（Reason 9）**从未触发过一次**，但后续合约却被“移动止盈冷却与突破拦截”（Reason 10）强行掐断开仓多达 **376 次**！

### 4.2 源码审查：`reset_routing_state` 致命漏清变量
审查多合约回测评估循环中的状态重置入口：
在 `FineFT/RL/DiHFT/high_level/vae_routing_util.py:1122-1144` 中，定义了合约切换时的重置函数：
```python
# vae_routing_util.py:1122-1144
def reset_routing_state(self):
    self.quantiles = {
        axis: [
            deque(maxlen=self.axis_window_lengths[axis])
            for _ in range(self.num_labels)
        ]
        for axis in ("slope", "volatility")
    }
    self.action = self.zero_position_action
    self.macro_action_history = []
    self.flat_action = self.zero_position_action
    self.remaining_persist = 0
    self.current_action = self.flat_action
    self.action_decision_reason_history = []
    self.cooldown_remaining_steps = 0
    self.last_stopped_position = 0.0
    self.consecutive_stop_loss_count = 0
    self.circuit_breaker_remaining_steps = 0
    self.hard_stop_loss_count = 0
    self.cooldown_intercept_count = 0
    self.circuit_breaker_suspension_count = 0
    self.previous_step_position = 0.0
    self.active_trade_stopped = False
    # 致命缺陷：完全没有重置以下移动追踪止盈的核心风控状态变量！
    # self.trailing_stop_active = False
    # self.trailing_stop_remaining_steps = 0
    # self.trailing_stop_last_position = 0.0
    # self.trailing_stop_hurdle_price = None
    # self.trailing_stop_hurdle_slope = None
    # self.trailing_stop_count = 0
    # self.trailing_stop_cooldown_intercept_count = 0
```
该函数负责清空硬止损（Hard Stop Loss）和断路器（Circuit Breaker）的相关状态，但**完全遗漏了 ADR 0051 新增的全部移动追踪止盈状态属性**！

### 4.3 跨合约状态泄漏与幽灵突破价阻断逻辑
1. **泄漏发生**：在评测第二支合约 `fu2411` 时，标的价格运行至高位 3,300 元左右，触发了 2 次移动追踪止盈（Reason 9）。此时路由对象的实例变量被赋值：
   - `self.trailing_stop_last_position = 1.0`（记录上次止盈平多）
   - `self.trailing_stop_hurdle_price = 3300.0+`（记录峰值基准价）
   - `self.trailing_stop_hurdle_slope = 2`（记录止盈时的宏观牛市体制）
2. **幽灵继承**：评测切换至 `fu2412`、`fu2501`、`fu2503` 时，由于未调用重置，这些变量被原封不动地继承下来。
3. **时空门禁永久阻断**：
   审查 `FineFT/RL/DiHFT/high_level/vae_routing_util.py:1575-1615`（Tier 4 时空门禁逻辑）：
   ```python
   # vae_routing_util.py:1575-1615
   if self.enable_trailing_stop and self.trailing_stop_last_position != 0.0:
       target_pos, _ = map_action_to_position_leverage(...)
       if target_pos * self.trailing_stop_last_position > 0:  # 智能体想开多
           if self.trailing_stop_hurdle_slope is not None and slope_index != self.trailing_stop_hurdle_slope:
               # 体制未反转，不满足重置条件
               ...
           else:
               price_not_broken = False
               if self.trailing_stop_require_peak_breakout and self.trailing_stop_hurdle_price is not None:
                   if self.trailing_stop_last_position > 0:
                       price_not_broken = (current_markprice < self.trailing_stop_hurdle_price)
               if is_in_cooldown or price_not_broken:
                   # 拦截开多动作，强行置平！
                   action = self._defensive_action(...)
                   self.action_decision_reason_history.append(ActionDecisionReasons.TRAILING_STOP_COOLDOWN)
                   return action
   ```
4. **致命恶果**：
   - `fu2503` 合约开盘点位在 2,684 ~ 2,737 元。
   - 当高层 VAE 准确识别出牛市大势（`slope_index == 2`），底层智能体产生顺势做多信号（`target_pos > 0`）时，拦截器执行检查：
     `current_markprice (2737) < trailing_stop_hurdle_price (3300)`！
   - 价格突破条件判定为：**未突破前序合约的幽灵峰值价格**！
   - 于是，在长达数周的单边大牛市主升浪中，智能体每次试图顺势建多，均被该幽灵价格判定为“处于止盈冷却未突破期”，强制剥夺开仓权（累计拦截 131 次）！
   - 智能体只能在大牛市中眼睁睁看着价格飙升而被迫空仓观望，甚至偶有做空微亏，导致最大盈利来源彻底枯竭。


---

## 5. 根因四：Tier 4 移动追踪止盈名义价值分母错配导致风控完全死锁钝化 (Notional vs Capital Denominator Mismatch)

### 5.1 收益率分母设计错配的源码实证
在 `FineFT/RL/DiHFT/high_level/vae_routing_util.py:1458-1465` 中，移动追踪止盈引擎计算持仓收益率的代码如下：
```python
# vae_routing_util.py:1458-1465
if (
    self.enable_trailing_stop
    and current_pos_float != 0.0
    and current_markprice > 0.0
):
    notional = abs(current_pos_float) * current_markprice
    holding_return = current_unrealized_pnl / notional
```
注意此处分母 `notional` 的计算：它等于**当前持仓手数的绝对值乘以合约当前标记价格（即名义合约总市值）**，而**非该笔持仓实际占用的保证金本金（Initial Margin / Allocated Capital）**！

### 5.2 金融杠杆下的量纲级放大与门槛不可达性
1. **名义价值 vs 保证金资本**：
   - 在高频期货交易中，燃料油合约每手 10 吨，当标记价为 3,000 元/吨时，1 手多头的名义合约价值为：
     $$\text{Notional} = 10 \times 3,000 = 30,000 \text{ 元}$$
   - 在系统设定的 5 倍杠杆（20% 保证金率）下，该笔持仓实际占用的资金仅为：
     $$\text{Required Capital} = 30,000 / 5 = 6,000 \text{ 元}$$
2. **激活阈值 `--trailing_stop_activation_threshold 0.08`（8%）的极端苛刻性**：
   - 若收益率分母为名义总市值（30,000 元），激活移动止盈所需的未实现浮盈为：
     $$\text{Unrealized PnL} \ge 30,000 \times 0.08 = 2,400 \text{ 元}$$
   - 对应标的现货价格单次持仓内的绝对涨幅为：
     $$\Delta P = 2,400 / 10 = 240 \text{ 点（即现货无杠杆单波暴涨 +8.0%）}$$
   - 若将其折算回策略账户本金（6,000 元），这相当于要求单笔波段实现 **`+40.0%` 的惊人暴利**才能激活追踪止盈！
3. **高频实盘中的物理不可达**：
   - 在 10 分钟 K 线级别的高频/日内回测中，单次持仓的平均周期仅为几十至百余步，燃料油现货在单次持仓周期内暴涨 240 点（8%）属于极低概率的罕见极端事件。
   - 这直接导致全测试集 12 支合约、累计 30,351 步的漫长交易历史中，**除了 `fu2411` 在极端跳空中偶然触及了 2 次阈值之外，其余 11 支合约的触发次数全为 0**！
   - 设计用于“回撤锁利、截断尾部风险”的 Tier 4 移动追踪止盈引擎，在 99.99% 的时间里由于量纲分母错配而完全“脑死亡”，根本未能发挥预期的防守作用。


---

## 6. 系统化架构修复与工程演进路线图 (Systematic Remediation Roadmap)

为彻底解决上述四个互锁的致命缺陷，恢复并超越基线性能，制定以下四项精准的工程与算法修复方案：

### 6.1 修复一：修正多合约特征漂移审计逻辑（Macro-Safe Distribution Audit）
- **核心文件**：`data_preprocess/operator_futures/feature_selection/muti_contract/distribution_audit.py` 与 `types.py`
- **修复方案**：
  1. **宏观特征周期自适应豁免或放宽**：在 `_feature_passes_drift` 中，针对被归类为 `macro` 的连续算子（`trend_beta_720/1440`, `mark_price_ema_deviation_720/1440`, `mark_price_roc_720/1440`, `trend_to_noise_720/1440`），停止直接施加静态跨合约直方图 PSI 门控，或将其最大允许 PSI 大幅放宽至周期安全界限（例如 `max_mean_psi=5.0`, `max_pair_psi=25.0`）；
  2. **算子标度自归一化（Self-Normalized Operators）**：对于跨合约长周期动量因子，在特征提取算子内部采用全局滚动或扩张窗口 Z-Score 归一化（例如 $\frac{\text{mark\_price} - \text{EMA}_{720}}{\text{ATR}_{720}}$），使其在统计上具备跨合约分布稳定性；
  3. **重新执行特征预处理与选择**：确保 10~15 个连续因果宏观动量与偏离度特征真正入选 `rl_state_features.npy`，彻底赋予低层智能体宏观视野。

### 6.2 修复二：彻底消除多合约评测状态泄漏（Leak-Proof Routing Reset）
- **核心文件**：`FineFT/RL/DiHFT/high_level/vae_routing_util.py`
- **修复方案**：
  在 `reset_routing_state(self)` 中，完整补齐移动追踪止盈状态变量的重置：
  ```python
  def reset_routing_state(self):
      # ... 既有重置逻辑保持不变 ...
      # 补齐全量移动追踪止盈状态重置：
      self.trailing_stop_active = False
      self.trailing_stop_remaining_steps = 0
      self.trailing_stop_last_position = 0.0
      self.trailing_stop_hurdle_price = None
      self.trailing_stop_hurdle_slope = None
      self.trailing_stop_count = 0
      self.trailing_stop_cooldown_intercept_count = 0
  ```
  在合约切换时坚决阻断跨合约幽灵突破价格的跨期污染，彻底释放 `fu2503`、`fu2501`、`fu2412` 等牛市合约的顺势开仓能力。

### 6.3 修复三：校准移动追踪止盈收益率基准与激活阈值（Capital-Aligned Trailing Stop）
- **核心文件**：`FineFT/RL/DiHFT/high_level/vae_routing_util.py:1463`
- **修复方案**：
  将持仓收益率的分母由名义总市值修正为保证金/占用本金基准：
  ```python
  # 方案 A：修正为实际占用保证金分母
  effective_capital = (abs(current_pos_float) * current_markprice) / max(current_leverage, 1.0)
  holding_return = current_unrealized_pnl / effective_capital
  ```
  或者在保持名义市值分母的前提下，将 `--trailing_stop_activation_threshold` 从 `0.08`（8% 名义）校准至符合日内波动的真实水平（如 `0.015` ~ `0.025`，对应名义 1.5%~2.5%，折算 5x 杠杆收益率为 7.5%~12.5%）。使追踪止盈在高频波段中能够真正被有效激活与精准止盈。

### 6.4 修复四：消除 DP Oracle 状态混叠与平滑强化学习目标（POMDP-Aligned Soft DP）
- **核心文件**：`FineFT/env/env_class/futures_util.py:1506` 与 `FineFT/RL/DiHFT/low_level/qtable_config.py`
- **修复方案**：
  1. **严禁在学生网络部分可观测（POMDP）条件下施加 $-10^{10}$ 极端断崖惩罚**；
  2. 将 DP 专家的顺势引导改为**柔性换手逆势惩罚（Soft Adverse Turnover Penalty）**（依托 ADR 0050 的 `turnover_adverse_ratio`，给逆势换手施加 2~5 倍的手续费/滑点惩罚，而非 $-10^{10}$ 梯度炸弹）；
  3. 或者在低层 Q-Net 真正获得宏观连续算子观测特征、消除状态混叠之后，再审慎进行特权学习蒸馏（Privileged Learning Distillation）。

---

## 7. 总结与行动建议 (Action Items & Verification)

| 阶段 | 任务目标 | 核心操作 | 验证标准 |
| :--- | :--- | :--- | :--- |
| **P0 紧急修复** | 修复回测引擎状态泄漏 | 在 `vae_routing_util.py:reset_routing_state` 中补齐 7 个追踪止盈重置变量 | 单元测试通过，`fu2503` 不再出现 Reason 10 幽灵拦截 |
| **P0 紧急修复** | 校准追踪止盈分母与阈值 | 修正 `holding_return` 计算分母为保证金本金，重调激活阈值 | 追踪止盈在盈利回撤时能够正常触发截断 |
| **P1 核心修复** | 修复宏观特征分布漂移门误杀 | 调整 `distribution_audit.py` 宏观算子 PSI 放宽策略，重跑特征选择 | `rl_state_features.npy` 包含 10+ 价格动量与偏离度连续算子 |
| **P1 核心修复** | 消除离线 DP 专家状态混叠 | 将 `futures_util.py` 中 `-max_punishment` 改为柔性逆势换手惩罚 | Warmup 阶段 TD loss 平稳下降，子网络无梯度爆炸 |
| **P2 全流程验证** | 重跑低层重训与高层验证 | 重训低层 Q-Net，并在 12 支验证合约上重跑回测评测 | 净回报恢复并超越基线（目标 $>+8.0\%$），胜率达标 $\ge 90\%$ |


---

## 8. Trial 336 评测复现与最新训练“结果依然很差”终极归因 (Trial 336 Post-Mortem & Timeline Disconnect)

### 8.1 评测现象：Trial 336 依然严重不及基线
在修复了高层 `reset_routing_state` 状态泄漏及保证金分母量纲后，系统运行产出了最新的高层评测结果 **Trial 336**（`analysis_result/DiHFT/high_level_heurstic/fu/10min_parallel/best_result.csv`）：
- 组合净回报率：**`+1.59%`**（+1,142.61 元，年化 SR = 1.31）；
- 对比历史基线 Trial 200（**`+6.47%`**，+4,655.71 元）：**下降 -75.4%**；
- 胜率：仅 **75.0%**（12 支合约中 3 支亏损：`fu2507` 亏 -344.3 元，`fu2508` 亏 -116.5 元，`fu2509` 亏 -60.0 元）；
- 核心疑问：代码已计划去除 $-10^{10}$ 悬崖惩罚与 `enable_trend_entry_lock`，为何最新训练与分析结果依然只有 +1.59%？

### 8.2 终极根因：时序错位与“假性新模型”陷阱 (The Timeline Disconnect)
通过审查系统文件时间戳，发现了确凿的**时序断层事实**：

| 流水线阶段 / 关键文件 | 文件路径 | 修改/生成时间戳 (CST) | 运行时代码状态 |
| :--- | :--- | :--- | :--- |
| **Step 3 (低层训练结束)** | `result/.../weights_advantage_pretrain/epoch_62/` | **2026-10-09 13:36:13** | **仍为旧代码** (`enable_trend_entry_lock=True`, 写入 $-10^{10}$) |
| **Step 5 (模型组装完成)** | `analysis_result/.../two_dimensional_selection/model.pth` | **2026-10-09 14:01:41** | 组装自 13:36 产出的旧 checkpoint (Epoch 31~50) |
| **Step 7 (Optuna 优化结束)** | `log/.../optuna/10min_parallel/optuna.log` | **2026-10-09 14:15:34** | 基于旧 `model.pth` 运行 400 轮试验，选出 Trial 336 |
| **Step 8 (高层分析落盘)** | `analysis_result/.../best_result.csv` | **2026-10-09 14:15:40** | 落盘 Trial 336 结果 (+1.59%) |
| **代码修复 1** | `FineFT/RL/DiHFT/low_level/qtable_config.py` | **2026-10-09 16:19:00** | 将 `enable_trend_entry_lock` 默认值改为 `False` |
| **代码修复 2** | `FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py` | **2026-10-09 16:20:04** | 暴露 `--enable_trend_entry_lock` 参数 |
| **训练启动脚本** | `FineFT/script/train/train_commodity_fu_10.sh` | **此前未修改 (00:58)** | 未包含 `--enable_trend_entry_lock False` |

**铁证结论**：
**所谓“最新跑完的 Trial 336”，其底层的全部 9 个 Q-Net 模型是在 10:13 至 13:36 CST 期间训练完成的，完全基于旧的 $-10^{10}$ 悬崖惩罚逻辑！针对低层参数的代码修复是在 16:19 CST 才修改的，距离训练结束已过去整整 2 小时 43 分钟！在此之后，Step 3（低层重训）从未被重新执行过！**
因此，Trial 336 本质上依然是在严重损坏的低层模型权重上做出的无效挣扎。

### 8.3 低层模型全面瘫痪（Flat Collapse）的量化实测
对目前处于激活状态的 `model.pth` 进行前向推断实测，揭示了低层模型遭受 $-10^{10}$ 梯度重创后的瘫痪状态：
1. **Q 值病态失衡**：
   - 9 个子网络从 Flat 开仓推断，平仓 Q 值均处于 **`[-240, -270]`**，而做多 Q 值处于 **`[-307, -333]`**，做空 Q 值处于 **`[-330, -360]`**；
   - 平仓动作相比于做多/做空拥有 **`+60 ~ +90` 的绝对 Q 值优势**！
2. **牛市专家拒不顺势做多**：
   - 在大牛市主力合约 `fu2503`（现货暴涨 +36.8%）中，Slot 8（高波动牛市专家）给出的动作为：
     **Flat = 88.9%，Long = 11.1%，Short = 0.0%**！
   - 即便在最确定的大牛行情中，牛市专家依然有近 90% 的时间坚决躺平。
3. **全测试集动作分布崩溃**：
   - 在 Trial 336 的全测试集 30,351 步中，实际执行微观动作为：
     - **Flat (空仓/观望)**: **`81.70%`** (24,797 步)
     - **Short (做空)**: **`11.66%`** (3,539 步)
     - **Long (做多)**: **`6.64%`** (2,015 步)
   - 在整体向上的燃料油行情中，策略做空次数竟然接近做多的 **2 倍**，做多意愿被彻底瓦解。

### 8.4 数理本质：POMDP 局部可观测下的纳什避险崩溃
1. **KL 散度爆炸**：
   在预训练 Warmup 阶段，监督损失包含 KL 散度：
   $$\mathcal{L}_{\text{KL}} = \sum_a \pi_{\text{learner}}(a) \cdot \left(\ln \pi_{\text{learner}}(a) - \ln \pi_{\text{expert}}(a)\right)$$
   当离线 DP 专家在逆势动作赋予 $-10^{10}$ 时，$\ln \pi_{\text{expert}}(a_{\text{adverse}}) \approx -10^{10}$。只要学生网络输出任何微小的正概率（如 0.09），单步 KL 散度损失直接飙升至 **`9.0 × 10^8`**，乘以权重 $\text{ada}=96.0$ 后总损失突破 **`8.6 × 10^{10}`**，梯度瞬间击穿网络。
2. **局部可观测性（POMDP）下的生存选择**：
   低层 Agent 的输入特征并不包含全知未来的宏观分段标签。在微观震荡中，网络无法 100% 确认当前属于牛市还是熊市。
   - 只要网络尝试做多，在熊市时刻就会被 $-10^{10}$ 极刑轰炸；
   - 只要网络尝试做空，在牛市时刻就会被 $-10^{10}$ 极刑轰炸；
   - **全空间中唯一永远不会触碰 $-10^{10}$ 惩罚的动作只有 Flat（平仓观望）**！
   - 于是，9 个子网络迅速在梯度轰炸下收敛至“躺平平仓”的纳什均衡，彻底丧失主动择时 Alpha。

### 8.5 Optuna 超参优化空间塌陷
- 在低层 Agent 输出 82% Flat 的前提下，高层 VAE 无论如何路由，调仓行为都无法捕捉市场波段。
- Optuna 在 400 次试验中，所有参数组合的年化回报率全被压缩在 `0.005 ~ 0.011` 的极窄死水区间（无法形成任何参数梯度）。
- Optuna 最终只能选择将 `hysteresis_exit_ratio` 推高至 `0.774`（极度保守滞后），以极力降低手续费损耗为最优解，从而产出了表面上盈利微薄（+1.59%）但实际无交易能力的残疾策略。

### 8.6 历史时序断层与旧修复局限性
在早先的快速排查中，尝试通过软配置将 `enable_trend_entry_lock` 默认值设为 `False` 并暴露 CLI 参数。但由于：
1. **此前训练时序在前，代码修改在后**：Trial 336 运行的模型是在修改之前（10:13~13:36）训练生成的，模型权重早已被 $-10^{10}$ 毒害；
2. **软开关保留死代码与极端惩罚隐患**：$-10^{10}$ 惩罚逻辑仍然保留在 `futures_util.py` 中，并未真正物理删除；如果一个 feature 存在根本性机理缺陷导致必须设为 `False`，根据精益架构与 CLAUDE.md 规范，必须彻底物理删除该特性，杜绝技术债务与防御性代码堆叠。


---

## 9. `enable_trend_entry_lock` 机制溯源、坏死归因与全链路物理彻底删除 (Definitive Elimination of Trend Entry Lock)

### 9.1 为什么当初要引入 `enable_trend_entry_lock`？
该特性起源于 **ADR 0051**（Commit `9b205fe`，2026-10-08）：
1. **DP 专家的全知作弊（Oracle Bias）**：
   - 离线 DP 算法通过全时序反向逆序递推（Backward Induction）求解最优 Q 表，天然具备全知未来价格的“上帝视角”。
   - 在单边大牛市（如 `fu2503`，标的暴涨 +36.7%）中，夜盘一旦发生 20~30 点的局部回调，全知的 DP 算法会在波峰最高点开空、波谷最低点平空翻多，以榨取每一个微观波段的最大收益。
2. **低层 Agent 模仿学习误入歧途**：
   - 低层强化学习智能体仅具备局部技术特征视界，在 Warmup 阶段通过 KL 散度向 DP 专家表学习时，误学到了在牛市回调中逆势做空的行为模式。
   - 导致在实盘评测中，`fu2503` 频繁逆势开空（692 步），净亏损 -126 点。
3. **设计初衷**：
   - ADR 0051 试图建立“因果体制单向开仓硬锁”，强制在离线 DP 表中：牛市体制（`regime_grid_id % 3 == 2`）严禁开空，熊市体制（`regime_grid_id % 3 == 0`）严禁开多；并在高层在线路由执行端进行镜像动作掩码拦截。

### 9.2 为什么判定 `enable_trend_entry_lock` 是不可逆坏死特性？
实证与数学推导证明，该特性的数理机理与强化学习基本规律完全冲突：
1. **POMDP 状态混叠（State Aliasing）的不可解矛盾**：
   - 低层网络按解耦原则（ADR 0037/0041）绝对不能输入全知宏观体制标签 `regime_grid_id`；
   - 局部 10 分钟 K 线的微观量价形态在牛市回调与熊市下跌中高度相似甚至完全重合；
   - 在同一微观观测下，DP 专家在牛市要求做多并给做空打 $-10^{10}$，在熊市要求做空并给做多打 $-10^{10}$。学生网络面临相差 $10^{10}$ 数量级的剧烈矛盾目标。
2. **极端惩罚引发梯度核爆**：
   - Warmup 的 KL 散度监督下，学生网络输出微小正概率即可引发 $8.6 	imes 10^{10}$ 的巨额损失与 $10^8$ 量级梯度冲击，彻底摧毁神经网络权重。
3. **纳什避险崩溃（全网瘫痪躺平）**：
   - 做多在熊市会死，做空在牛市会死，全空间中唯一永远不会触碰 $-10^{10}$ 极刑的动作只有 **Flat（空仓观望）**；
   - 9 个子网络全部退化收敛为 82% 躺平空仓，彻底丧失主动择时 Alpha。
4. **高层硬切破坏连续性**：
   - 高层的顺势交易本应由高层 VAE 路由选择对应体制子网络（如牛市选牛市槽位专家）和滞后退出机制来实现。高层推断层的硬拦截（`TREND_ENTRY_LOCK`）强行切断持仓，违背动作持续性设计。

因此，`enable_trend_entry_lock` 是一个设计机理错误的坏死特性，必须坚决彻底连根拔除，决不能仅用 `False` 开关伪掩盖。

### 9.3 全链路物理彻底删除清单 (Exhaustive Physical Deletion Manifest)
遵循 CLAUDE.md「无向后兼容包袱、彻底替换废弃代码、无防御性代码」规范，已对以下文件完成物理删除：

1. **底层 DP 表求解算子 (`FineFT/env/env_class/futures_util.py`)**：
   - 物理删除 `create_optimal_q_table` 中的 `enable_trend_entry_lock` 与 `trend_entry_penalty` 参数；
   - 物理删除 `elif (enable_trend_entry_lock and ...): q_table[...] = -effective_trend_penalty` 全部代码分支，**彻底拔除 $-10^{10}$ 悬崖惩罚**；
   - 物理删除 `create_optimal_q_table_from_df` 中的对应形参与透传。
2. **底层 Q 表配置生成器 (`FineFT/RL/DiHFT/low_level/qtable_config.py`)**：
   - 物理删除 `build_optimal_qtable_kwargs` 中的 `enable_trend_entry_lock` 与 `trend_entry_penalty` 参数与字典映射。
3. **共享数据管理器 (`FineFT/RL/DiHFT/low_level/shared_data_manager.py`)**：
   - 物理删除调用 `create_optimal_q_table_from_df` 时传递的对应参数。
4. **底层预训练入口与脚本 (`parallel_weight_advantage_pretrain.py`, `train_commodity_fu_10.sh`)**：
   - 物理删除 `--enable_trend_entry_lock` 与 `--trend_entry_penalty` CLI 参数、实例属性及参数字典透传；
   - 训练脚本中完全移除对应参数。
5. **高层执行路由与优化器 (`vae_routing_util.py`, `vae_routing_optuna.py`)**：
   - 物理删除高层 CLI 参数 `--enable_trend_entry_lock`；
   - 物理删除 `vae_routing_config`、`vae_routing_heuristic`、`vae_risk_aware_routing` 中的 `enable_trend_entry_lock` 与 `trend_entry_lock_count`；
   - 物理删除 `get_action` 中的 `5c. Directional Trend Action Mask Interception` 拦截分支；
   - 物理删除 `vae_routing_optuna.py` 中的参数透传。
6. **单元测试集清理**：
   - 物理删除整套过时的 DP 趋势锁测试文件 `FineFT/tests/env/test_trend_entry_lock.py`；
   - 清理 `FineFT/tests/rl/test_trend_entry_lock_and_trailing_stop.py` 中的测试用例与参数断言，保留追踪止盈核心测试；
   - 清理 `test_vae_routing_non_main_defense.py`、`test_vae_routing_allow_reverse_position.py`、`test_contract_level_stop_loss_and_circuit_breaker.py`、`test_vae_routing_action_persistence.py` 中的无用参数。
   - **验证结论**：全量测试套件通过（56 项 RL 测试 + 10 项数据审计测试 100% PASS）。

### 9.4 高层因果顺势架构重构：基于逆市损失比例阈值的动态强制顺势 (Loss-Governed Forced Trend Alignment)

#### 9.4.1 设计初衷与思想蜕变
针对用户提出的关键指导原则：**“高层的顺势不应该删除应该保留。但是应该根据损失来限制顺势，比如逆市损失超过多少比例然后强制顺势”**，我们对高层风控体系进行了深层升级：
- **旧版设计的硬伤**：从步数 0 开始无条件硬性封杀所有逆势动作，完全无视盘面实际盈亏与微观对冲需求，导致策略死板僵化，且在状态泄漏时酿成灾难；
- **新版设计的精髓**：
  1. **允许正常的逆势高频博弈**：当低层微观策略在牛市急跌中捕捉到反弹做空机会时，系统不预设偏见，不阻断开仓；
  2. **逆市浮盈自由奔跑**：只要逆势波段处于盈利或保本状态（`unrealized_pnl >= 0`），风控引擎绝不提前干预；
  3. **逆市亏损精确截断并强制顺势**：一旦逆势持仓浮亏触及预设阈值（`adverse_loss_rate >= trend_loss_threshold`），风控引擎判定“微观逆势博弈失败，必须顺应宏观大势”，立即强行平仓出场，并在当前体制内锁死逆势再入，只允许顺势交易（“强制顺势”）。

#### 9.4.2 动态强制顺势的精确算法机理
1. **逆市持仓识别与真实本金亏损率计算**：
   在宏观单边行情中（牛市 `slope_index == 2` 或熊市 `slope_index == 0`）：
   - 牛市持有空头（`current_position < 0`）或熊市持有多头（`current_position > 0`）被定义为**逆市持仓（Adverse Position）**；
   - 采用对齐实际占用保证金的资本分母：
     $$	ext{effective\_capital} = rac{|	ext{current\_position}| 	imes 	ext{current\_markprice}}{\max(	ext{current\_leverage}, 1.0)}$$
     $$	ext{adverse\_loss\_rate} = rac{-	ext{current\_unrealized\_pnl}}{	ext{effective\_capital}}$$
2. **Step 3b：超阈值强制平仓止损 (Adverse Liquidation)**：
   当 `adverse_loss_rate >= self.trend_loss_threshold`（默认阈值 `0.015`，即保证金亏损达到 1.5% 时）：
   - 立即清空动作持续性（`remaining_persist = 0`）；
   - 执行 `_defensive_action` 强行平仓至 Flat，截断逆势继续失血；
   - 记录决策原因 `ActionDecisionReasons.TREND_ENTRY_LOCK`；
   - 激活宏观体制锁定标记：`self.trend_locked_slope = slope_index`。
3. **Step 5c：锁定期内拦截逆势开仓与强制顺势通行 (Regime Guard)**：
   - 当 `self.trend_locked_slope == slope_index` 时：若智能体后续候选动作试图再次逆势开仓（牛市想开空、熊市想开多），被 Step 5c 无情拦截并强制置为 Flat；
   - **顺势开仓畅通无阻**：若候选动作符合宏观趋势（牛市开多、熊市开空），则直接通行！真正实现了**“逆市亏损超限后强制顺势”**。
4. **体制转移自愈与跨合约防泄漏**：
   - 当宏观体制切换（例如转为震荡 `slope_index == 1` 或大势反转）时，`self.trend_locked_slope` 自动清零解除；
   - 在 `reset_routing_state` 中同步清空 `trend_locked_slope`，彻底根除跨合约幽灵污染。

#### 9.4.3 参数配置与全量验证
- **核心 CLI 参数**：
  - `--enable_trend_entry_lock True`：启用高层因果顺势风控引擎；
  - `--trend_loss_threshold 0.015`：逆势持仓占用保证金损失率阈值（默认 1.5%）；
- **单元测试验证**：
  - `FineFT/tests/rl/test_trend_entry_lock_and_trailing_stop.py` 补充 4 项全新测试用例（覆盖小亏放行、超阈值平仓锁定、逆势拦截、顺势通行、体制切换自愈与跨合约重置），全套 74 项测试 **100% PASS**。


---

### 9.5 训练重跑指令与预期验证标准
在底层 DP 表 $-10^{10}$ 悬崖惩罚被彻底物理拔除、宏观漂移门完成修复、移动止盈状态泄漏与保证金分母修复完毕、高层因果顺势护栏基于损失比例重构生效的全新状态下，执行全流程重训：
```bash
./run_fu_10min_pipeline.sh -s 3,4,5,6,7,8
```
- **预期验证标准**：
  1. **低层 Q-Net 训练指标**：Warmup 阶段 loss 平稳下降至正常区间（$10^{-2} \sim 10^{-1}$），无 $10^{10}$ 级巨损，梯度范数 $< 10$；
  2. **模型组装 Q 值健康度**：Q(Flat)、Q(Long)、Q(Short) 恢复动态相对优势，彻底打破 Flat 的 +60~+90 垄断；
  3. **动作分布恢复**：全测试集 Flat 比例从 81.7% 下降至正常合理区间（30%~50%），Long/Short 积极参与波段交易；
  4. **高层表现指标**：12 支合约胜率恢复至 $\ge 90\%$，组合净收益恢复并超越基线（目标 $> +8\%$）。
