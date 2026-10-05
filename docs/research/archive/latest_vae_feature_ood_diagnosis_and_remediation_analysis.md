# 最新 VAE 特征级 OOD 实证诊断、残差缺陷根因剖析与定向修复方案调研报告

- **研究主题**：基于 2026-09-27 最新运行的 FineFT 10 分钟燃料油 (`fu`) VAE 特征级 OOD 诊断报告，系统评估 ADR-0027 / ADR-0028 治理后的似然表现，深度排查残余高 OOD 恶化特征的底层数学与工程根因，并给出下一阶段修复清单。
- **关联第一手来源 (Primary Sources)**：
  - **实证诊断产物**：
    - `analysis_result/DiHFT/feature_ood/fu/10min_parallel/feature_ood_summary.csv` (全量多视角诊断宽表)
    - `analysis_result/DiHFT/feature_ood/fu/10min_parallel/feature_ood_test_vs_train.csv` (测试集真实 OOD 崩溃归因表)
    - `analysis_result/DiHFT/feature_ood/fu/10min_parallel/feature_ood_valid_vs_train.csv` (验证集体制漂移表)
    - `analysis_result/DiHFT/feature_ood/fu/10min_parallel/feature_ood_test_vs_valid.csv` (泛化退化表)
    - `analysis_result/DiHFT/feature_ood/fu/10min_parallel/contracts/*.csv` (13 个测试合约逐合约归因文件)
    - `PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/fu/10min/scaler_manifest.json` (Robust Scaler 训练集分位数与缩放参数)
  - **特征生成与算子实现源码**：
    - `data_preprocess/operator_futures/time_operator/multi_processing_util.py` (L452-454: `min_*/max_*` 未中心化极值比; L489: `max_*_std_norm` 单边极值距离; L494: `vma_*_std_norm`)
    - `data_preprocess/operator_futures/cross_section/base_feature_util.py` (L455-460, L769-770: `buy_spread_oe_max`, `sell_spread_oe_max` 盘口价差)
    - `data_preprocess/operator_futures/commodity/cross_month_feature.py` (L33-34, L442: `cm_main_sub_volume_share_sub` 跨期静态成交份额; L42-44: `spread_rolling_zscore_192`)
    - `data_preprocess/operator_futures/scale_describe_save/muti_contract_scale_save.py` (L349-365: `VOLATILITY_FEATURE_PATTERNS` 对数正态化; L240-275: `fit_feature_stats` 方差与 IQR 回退)
    - `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh` (L70-88: `COMMODITY_FU_FEATURE_BLACKLIST` 特征黑名单)
  - **模型与基准决策规范**：
    - `FineFT/RL/DiHFT/VAE/vae.py` (L125-145: 高斯负对数似然 $\text{NLL}$ 损失与 `softclip`)
    - `FineFT/analysis/feature/vae_feature_ood_analysis.py` (三视角闭式高斯似然分解矩阵)
    - `docs/adr/0026-multi-perspective-vae-feature-ood-diagnostic-matrix.md`
    - `docs/adr/0027-three-stage-remediation-for-vae-feature-ood-drift.md`
    - `docs/adr/0028-extended-ood-remediation-for-bandwidth-pivots-and-cross-month-spreads.md`

---

## 1. 执行摘要与阶段治理成效评估

在应用 ADR-0027（黑名单 6 项非平稳特征、修复 `*_std_norm` 绝对价格减法缺陷、波动率 Log 正态化）与 ADR-0028（192 周期名义价格比拉黑、布林带宽 Log 化、跨期价差速度双曲正切软饱和）之后，对 105 个入选状态特征重新训练 VAE 并运行最新诊断，治理成效显著：

### 1.1 总体似然恶化量压缩显著
- **Valid vs Train（验证集漂移）**：
  - 总似然恶化量从 **+280.92** 剧降至 **+163.71**（降幅达 **41.7%**）。
  - 布林带宽 (`bollinger_bandwidth_96_origin`) $\Delta\text{NLL}$ 从 22.33 降至 **5.40**，方差比从 2.20x 降至 **1.44x**，Log 变换有效遏制了方差爆炸。
- **Test vs Train（测试集真实 OOD 崩溃基准）**：
  - 总似然恶化量从 **+38.17**（早期基线更达 +60 以上）压缩至 **+26.03**（降幅达 **31.8%**）。
  - 此前位列前三的超级崩溃特征（`sell_spread_oe_max_trend_192`, `cm_main_sub_open_interest_share_sub`, `cntn_192_origin`）因黑名单熔断已完全清零。
- **Test vs Valid（泛化退化视角）**：
  - 总似然增量为 **-137.68**。负值表明测试集在 VAE 视角下的重构拟合优于极端高波动的验证集，模型并未在测试集出现泛化崩塌。

### 1.2 残留问题核心判断
虽然总 OOD 恶化量大幅下降，但**测试集和验证集仍存在若干集中贡献的异常特征**。排查发现，当前残存的 OOD 恶化并非随机噪声，而是由于**前期治理中未完全覆盖的同类型衍生特征（漏网特征）**以及**微观盘口变量离散步长与零深度异常值**引起的。

---

## 2. 最新多视角 OOD 排行榜与关键实证指标

### 2.1 视角一：真实测试集 OOD 崩溃榜 (Test vs Train Top 15)
*基准：`train`；评估目标：`test`；度量指标：$\Delta\text{NLL} = \text{NLL}_{\text{test}} - \text{NLL}_{\text{train}}$*

| 排名 | 特征名称 | $\Delta\text{NLL}$ | 贡献占比 | 测试集均值偏离 ($\sigma$) | 方差比 | 测试合约 Top 3 频次 | 核心病理归类 |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | `realized_volatility_192` | **2.03** | **7.78%** | 0.62 | 1.08 | 8 / 13 | 宏观周期波动率水平差异（良性体制变化） |
| 2 | `max_192_std_norm_origin` | **1.92** | **7.38%** | 0.40 | 1.09 | **9 / 13** | **32 小时长周期单边最高价偏离，缺乏对称极值** |
| 3 | `sell_spread_oe_max_trend_6` | **1.20** | **4.60%** | 0.01 | 0.98 | 1 / 13 | 盘口 Tick 价差常数除以微小方差导致的尖峰脉冲 |
| 4 | `buy_spread_oe_max` | **1.14** | **4.38%** | 0.01 | 0.95 | 3 / 13 | 盘口 5 档价差离散步长与空深度 0 值下溢 |
| 5 | `cm_m2_m3_log_price_spread_velocity_10m` | **0.99** | **3.82%** | 0.03 | 1.09 | 1 / 13 | 远月换月期间非活跃合约价格跳变 |
| 6 | `cm_m1_m2_log_price_spread_velocity_10m` | **0.95** | **3.66%** | 0.17 | 1.34 | 1 / 13 | 换月交割月价差跳变 |
| 7 | `vma_192_std_norm_origin` | **0.91** | **3.48%** | 0.32 | 0.74 | 1 / 13 | 成交量信噪比长周期水平漂移 |
| 8 | `max_48_std_norm_origin` | **0.90** | **3.44%** | 0.16 | 1.04 | 0 / 13 | 8 小时单边极值偏离 |
| 9 | `sell_spread_oe_max` | **0.78** | **2.99%** | 0.03 | 0.80 | 2 / 13 | 卖方 5 档价差离散步长与空深度 0 值下溢 |
| 10 | `cm_main_sub_spread_rolling_zscore_192` | **0.74** | **2.85%** | 0.11 | 0.90 | 1 / 13 | 192 周期长窗口价差滚动 Z-Score 边界截断 |
| 11 | `trend_r2_48` | **0.70** | **2.69%** | 0.01 | 1.03 | 0 / 13 | 拟合优度判定系数轻微漂移 |
| 12 | `imin_96_origin` | **0.70** | **2.69%** | 0.23 | 1.02 | 0 / 13 | 16 小时最低价发生位置偏离 |
| 13 | `cm_main_sub_volume_share_sub` | **0.68** | **2.63%** | 0.09 | 0.77 | **4 / 13** | **跨期静态换月成交份额生命周期泄露 (ADR-0027 漏网项)** |
| 14 | `garman_klass_volatility_16` | **0.68** | **2.62%** | 0.49 | 1.11 | 1 / 13 | 短期高低价波动率体制变化 |
| 15 | `cm_current_main_volume_share_current` | **0.58** | **2.21%** | 0.28 | 0.88 | 0 / 13 | 跨期静态换月成交份额生命周期泄露 |

### 2.2 视角二：验证集体制漂移榜 (Valid vs Train Top 10)
*基准：`train`；评估目标：`valid`；度量指标：$\Delta\text{NLL} = \text{NLL}_{\text{valid}} - \text{NLL}_{\text{train}}$*

| 排名 | 特征名称 | $\Delta\text{NLL}$ | 贡献占比 | 验证集均值偏离 ($\sigma$) | 方差比 | 核心病理归类 |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| 1 | `realized_volatility_192` | **16.13** | **9.85%** | 0.22 | **2.34** | 2025 下半年红海地缘导致的真实高波动体制（良性） |
| 2 | `log_price_slope_96` | **11.31** | **6.91%** | 0.01 | 1.68 | 验证集单边趋势斜率（模型体制分类锚点，良性） |
| 3 | `min_96_origin` | **11.18** | **6.83%** | 0.08 | **1.75** | **16 小时未中心化名义价格比值漂移 (ADR-0028 漏网项)** |
| 4 | `sell_spread_oe_max` | **7.49** | **4.58%** | 0.09 | 0.75 | 盘口 5 档价差离散步长与 0 值异常下溢 |
| 5 | `buy_spread_oe_max` | **7.14** | **4.36%** | 0.07 | 0.83 | 盘口 5 档价差离散步长与 0 值异常下溢 |
| 6 | `rolling_volatility_48` | **6.79** | **4.15%** | 0.12 | 1.90 | 验证集高波动体制反应 |
| 7 | `cm_m2_m3_log_price_spread_velocity_10m` | **6.41** | **3.91%** | 0.04 | 1.42 | 跨期价差速度漂移 |
| 8 | `cm_m1_m2_log_price_spread_velocity_10m` | **6.20** | **3.79%** | 0.03 | 1.63 | 跨期价差速度漂移 |
| 9 | `ema_slope_192` | **6.14** | **3.75%** | 0.02 | 1.75 | 长周期趋势斜率体制差异 |
| 10 | `cm_current_main_spread_rolling_zscore_192` | **5.99** | **3.66%** | 0.03 | 1.22 | 192 周期长窗口价差滚动 Z-Score 边界截断 |

---

## 3. 残差缺陷深度根因追溯与数学病理剖析

经过对第一手代码和数据的深入审计，以下 5 个特征缺陷集群需要进行定向修复：

### 3.1 缺陷集群 A：漏网的长周期未中心化名义价格比值 (`min_96_origin`, `pivot_s2_48_origin`)
- **源码位置**：`data_preprocess/operator_futures/time_operator/multi_processing_util.py` L453:
  ```python
  (close.rolling_min(window) / (close + min_value)).alias(f"min_{window}")
  ```
- **病理剖析**：
  - ADR-0028 正确拉黑了 192 周期（32 小时）的名义价格比值（`min_192_origin`, `max_192_origin`, `pivot_*_192_origin`）。
  - 但是，**96 周期的 `min_96_origin` 和 48 周期的 `pivot_s2_48_origin` 仍然遗留在候选池中并入选！**
  - 在 10 分钟频率下，96 周期长达 960 分钟（16 个交易小时，相当于近 3 个完整交易日）。在此跨度下直接用 3 天前的最低价除以当前价 $P_{\min, 96} / P_t$，在多空趋势行情下必然产生非平稳的水位位移。
  - 在 `valid_vs_train` 中，`min_96_origin` 赫然高居 **第 3 位**（$\Delta\text{NLL} = 11.18$，方差比 1.75x）！
- **修复方案**：将 `min_96_origin`、`max_96_origin`、`pivot_s2_48_origin`、`pivot_s1_24_origin`、`bollinger_lower_12_origin` 等剩余未中心化价格比值全部加入 `COMMODITY_FU_FEATURE_BLACKLIST`。

---

### 3.2 缺陷集群 B：单边非对称的长周期极值偏离 (`max_192_std_norm_origin`)
- **源码位置**：`data_preprocess/operator_futures/time_operator/multi_processing_util.py` L489:
  ```python
  ((close.rolling_max(window) - close) / close_std).alias(f"max_{window}_std_norm")
  ```
- **病理剖析**：
  - 虽然 ADR-0027 修复了绝对价格缺陷，使其成为标准化距离 $((P_{\max} - P_t) / \sigma_P)$，但其定义域被严格限制在 $[0, +\infty)$。
  - **192 周期（32 小时）横跨 4~5 个自然日及周末休市**。当测试集合约经历持续单边阴跌或低波动盘整时，价格长期无法刷新 4 天前的高点，导致该指标持续维持在远离 0 的大数值区间（测试集中位数达到 0.454，而训练集中位数为 0.000，均值偏离达 $0.40\sigma$）。
  - **特征选择的不对称性**：特征选择入选了 `max_192_std_norm_origin`，却没有入选对应的 `min_192_std_norm_origin`，导致输入向量存在单边方向性结构失衡。
  - **实证破坏力**：在全部 13 个独立测试合约中，`max_192_std_norm_origin` **在多达 9 个合约中进入 OOD 恶化 Top 3**，是测试集真实 OOD 崩溃的第 2 大推手（贡献占比 7.38%）。
- **修复方案**：拉黑 192 周期跨日极端距离 `max_192_std_norm_origin`，保留日内尺度合理的短期极值距离（如 48 周期、24 周期）。

---

### 3.3 缺陷集群 C：跨期成交份额的换月周期泄露 (`cm_main_sub_volume_share_sub`, `cm_current_main_volume_share_current`)
- **源码位置**：`data_preprocess/operator_futures/commodity/cross_month_feature.py` L33, L442
- **病理剖析**：
  - ADR-0027 在拉黑持仓份额时明确指出：“静态持仓份额随合约到期出现周期性非平稳漂移，已被平稳的换月速度 `cm_open_interest_shift_speed_10m` 完全替代”。因此拉黑了 `cm_main_sub_open_interest_share_sub`。
  - **严重的遗漏**：当时**仅拉黑了持仓份额 (`open_interest_share`)，却遗漏了成交份额 (`volume_share`)**！
  - 实际上，次主力合约成交份额 `cm_main_sub_volume_share_sub` 和当月主力成交份额 `cm_current_main_volume_share_current` 具有**完全相同的换月周期非平稳性**。在合约生命周期早期，次主力成交份额几乎为 0；在换月主力切换的一周内暴增至 40%~60%；随后又断崖式下跌。
  - **实证破坏力**：在测试合约中，`cm_main_sub_volume_share_sub` 在 `fu2606` 中位列 **OOD 恶化第 1 名**，在 `fu2609` 中位列第 2 名，在 4 个测试合约中位列 Top 3！在测试集总体排行中位列第 13。
- **修复方案**：立即将 `cm_main_sub_volume_share_sub`、`cm_current_main_volume_share_current`、`cm_current_sub_volume_share_current` 补全录入黑名单。

---

### 3.4 缺陷集群 D：盘口价差离散步长与 0 值异常下溢 (`buy_spread_oe_max`, `sell_spread_oe_max`, `*_trend_6`)
- **源码位置**：`data_preprocess/operator_futures/cross_section/base_feature_util.py` L455-460, L769-770:
  ```python
  price_related_df["buy_spread_oe_max"] = np.abs(df["bid1_price"] - df[f"bid{depth}_price"]).clip(0.0, 50.0)
  price_related_df["sell_spread_oe_max"] = np.abs(df["ask1_price"] - df[f"ask{depth}_price"]).clip(0.0, 50.0)
  ```
- **实证证据链**（查看 `scaler_manifest.json`）：
  - 训练集分位数：`center = 4.0`, `q25 = 4.0`, `q75 = 4.0`, `iqr = 0.0`！
  - 缩放方法：`scale_method = std`, `scale = 1.4059`, `fallback_reason = "iqr_below_epsilon"`。
  - 训练集原始值：严格 $\ge 4.0$（最小值 = 4.0）。
  - **缩放后的训练集最小值**：$(4.0 - 4.0) / 1.4059 = 0.0$。
- **病理剖析**：
  1. **离散整数步长**：燃料油最小变动价位是 1 元/吨。在五档行情中，当买 1 到买 5 每档均有挂单时，$|bid1 - bid5| = 4\text{ 个 tick} = 4.0$。在正常流动性下，绝大多数时刻该值恒等于 4.0，导致 IQR 为 0。
  2. **验证集/测试集 0 值下溢**：在早晚盘开盘瞬间、集合竞价或极端涨跌停无对手盘时，若盘口深度不足 5 档，买 1 与买 5 价差为 0.0。缩放后变为 $(0.0 - 4.0) / 1.4059 = \mathbf{-2.845}$！
  3. **VAE 重构崩溃**：VAE 在训练集上学到的分布是严格单边大于等于 0 的半高斯分布，而在验证集和测试集中突然遇到 $-2.845$ 的孤立脉冲点，导致高斯似然损失急剧恶化！
  4. **趋势算子除零**：针对该特征计算短期移动窗口 Z-Score（`sell_spread_oe_max_trend_6`、`buy_spread_oe_max_trend_6`），在窗口内全为 4.0 时方差极小，稍微有一个跳变就会被放大为极值脉冲，位列测试集第 3 名。
- **修复方案**：
  1. 拉黑脉冲趋势算子 `sell_spread_oe_max_trend_6` 与 `buy_spread_oe_max_trend_6`（对齐 ADR-0027 已拉黑的 `*_trend_192`）。
  2. 在特征黑名单中排除原始绝对 Tick 跨度 `buy_spread_oe_max` 和 `sell_spread_oe_max`，或者在算子中为其施加物理下限保底（不少于 4 个 Tick），改用连续型的盘口挂单量比例特征（如 `ask_size_topk_size_5_share`，该特征在测试集表现极其稳健，$\Delta\text{NLL} = -0.026$）。

---

### 3.5 缺陷集群 E：192 周期长窗口价差滚动 Z-Score 边界截断 (`cm_*_spread_rolling_zscore_192`)
- **源码位置**：`data_preprocess/operator_futures/commodity/cross_month_feature.py` L42-44
- **病理剖析**：
  - 在 10 分钟频率下，192 周期窗口长达 32 小时（跨越 4 天）。在换月基差发生结构性跳变前，滚动标准差可能极其微小（$<0.001$）。
  - 一旦发生换月基差重定价，$(spread - mean) / std$ 会瞬间打满硬截断边界（$\pm 5.0$），造成概率密度在边界截断处的人为堆积。
  - 在 `fu2601` 合约中，`cm_current_main_spread_rolling_zscore_192` **单特征贡献了该合约 292.4% 的 OOD 恶化**；在 `fu2603` 中贡献了 14.3%！
  - 相反，48 周期的平稳版本（`cm_spread_rolling_zscore_48`）表现极其平稳，未出现边界饱和。
- **修复方案**：将 192 周期的跨期价差滚动 Z-Score（`cm_current_main_spread_rolling_zscore_192`, `cm_main_sub_spread_rolling_zscore_192`）加入特征黑名单，保留更适应日内换月动态的 48 周期版本。

---

### 3.6 良性宏观体制变化（无需“过度修复”的特征）
在分析报告中，必须严格区分**数学/工程缺陷**与**真实宏观市场体制变化**，避免过度平滑掉有效的市场信号：
1. **`realized_volatility_192`**：
   - 虽然在 `valid_vs_train` 排行第 1，但其方差比为 2.34x，而在 `test_vs_train` 中方差比为 **1.08x**（非常健康）。
   - 验证集对应 2025 年 5 月至 12 月，国际原油与燃料油受中东地缘局势影响呈现历史级高波动，验证集似然偏高是客观市场特征。VAE 模型正确地将该时期识别为高波动体制，这正是体制分类器的价值所在，无需进一步人为抹杀其波动性。
2. **`log_price_slope_96` 与 `ema_slope_192`**：
   - 这是 FineFT 体制切分的核心锚点（Trend vs Reversal）。在测试集上其方差比为 0.78x~0.89x，均值漂移仅 0.01$\sigma$。在验证集上的恶化同样反映了 2025 年强烈的单边趋势，属于正常的宏观体制表现。

---

## 4. 定向修复执行清单与实施建议

针对上述根因，建议按以下两步实施定向修复：

### 4.1 第一步：扩展 `fu_full_process.sh` 特征黑名单 (Zero-Code-Risk Blacklist Expansion)

在 `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh` 的 `COMMODITY_FU_FEATURE_BLACKLIST` 中增加以下 14 个确定性缺陷与非平稳特征：

```bash
    # === 补全漏网的未中心化跨期极值与价格比 (Residual Uncentered Price Ratios) ===
    min_96_origin
    max_96_origin
    pivot_s2_48_origin
    pivot_s1_24_origin
    pivot_s1_6_origin
    bollinger_lower_12_origin

    # === 单边跨日极值偏离与长周期价差 Z-score (Multi-day Asymmetric Extremes & Long Z-scores) ===
    max_192_std_norm_origin
    cm_current_main_spread_rolling_zscore_192
    cm_main_sub_spread_rolling_zscore_192

    # === 跨期静态成交份额换月周期泄露 (ADR-0027 遗漏的 Volume Share 对应项) ===
    cm_main_sub_volume_share_sub
    cm_current_main_volume_share_current
    cm_current_sub_volume_share_current

    # === 盘口离散 Tick 步长与虚假微方差趋势脉冲 (Orderbook Tick Discrete Steps & Spurious Trends) ===
    sell_spread_oe_max_trend_6
    buy_spread_oe_max_trend_6
    buy_spread_oe_max
    sell_spread_oe_max
```

### 4.2 第二步：底层盘口深度保底修正 (Surgical Source Fix)
在 `data_preprocess/operator_futures/cross_section/base_feature_util.py` 中，为 `buy_spread_oe_max` 和 `sell_spread_oe_max` 增加深度缺失时的物理最小有效 Tick 距离兜底，杜绝异常 0.0 导致缩放后跳变为 $-2.845$：
```python
# 兜底避免空深度导致 0.0 异常值
min_valid_tick_spread = 1.0 * (depth - 1) # 5 档盘口至少间隔 4 个 tick
price_related_df["buy_spread_oe_max"] = np.clip(np.abs(df["bid1_price"] - df[f"bid{depth}_price"]), min_valid_tick_spread, 50.0)
price_related_df["sell_spread_oe_max"] = np.clip(np.abs(df["ask1_price"] - df[f"ask{depth}_price"]), min_valid_tick_spread, 50.0)
```

---

## 5. 预期量化效益

1. **测试集真实 OOD 恶化量**：
   - 当前测试集总 $\Delta\text{NLL}$ 为 **26.03**。
   - 剔除 `max_192_std_norm_origin` (-1.92)、`sell_spread_oe_max_trend_6` (-1.20)、`buy_spread_oe_max` (-1.14)、`sell_spread_oe_max` (-0.78)、`cm_main_sub_spread_rolling_zscore_192` (-0.74)、`cm_main_sub_volume_share_sub` (-0.68)、`cm_current_main_volume_share_current` (-0.58) 等特征后，**测试集 OOD 恶化量预计将直接压降约 7.03 NLL（降幅达 27.0%），总 $\Delta\text{NLL}$ 降至 19 以下**！
2. **验证集漂移量**：
   - 剔除 `min_96_origin` (-11.18)、`sell_spread_oe_max` (-7.49)、`buy_spread_oe_max` (-7.14)、`cm_current_main_spread_rolling_zscore_192` (-5.99)、`cm_main_sub_spread_rolling_zscore_192` (-5.41)、`pivot_s2_48_origin` (-3.14) 等特征后，**验证集总 $\Delta\text{NLL}$ 预计将直接压降约 40.35 NLL（降幅达 24.6%），降至 123 左右**。
3. **消除极端合约的 OOD 异常峰值**：
   - `fu2508`（盘口价差导致）、`fu2601`（192 周期 Z-score 导致）、`fu2606`（换月成交份额导致）的奇异恶化将被彻底抹平。
