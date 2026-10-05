# FineFT 特征工程优化、5-10分钟短线特征过滤与双流解耦架构调研报告

- **报告编号**：RES-2026-0929-01
- **研究主题**：5-10分钟级别期货交易下的特征工程平稳化、宏观特征剪枝、VAE与Low-level Agent双流解耦决策，以及强化学习智能体专用 OOD 检验体系设计。
- **关联第一手来源 (Primary Sources)**：
  - **实证诊断产物**：
    - `analysis_result/DiHFT/feature_ood/fu/10min_parallel/feature_ood_valid_vs_train.csv` (验证集体制漂移表)
    - `analysis_result/DiHFT/feature_ood/fu/10min_parallel/feature_ood_test_vs_train.csv` (测试集 OOD 崩溃归因表)
    - `analysis_result/DiHFT/feature_ood/fu/10min_parallel/feature_ood_summary.csv` (全量多视角诊断宽表)
    - `analysis_result/DiHFT/feature_ood/fu/10min_parallel/contracts/*.csv` (13个测试合约逐合约下钻表)
    - `analysis_result/DiHFT/high_level_heurstic/fu/10min_parallel/best_result.csv` (启发式路由最优参数表)
  - **代码实现第一手源码**：
    - `FineFT/RL/DiHFT/VAE/vae.py` (高斯对数似然 NLL 损失函数与 `softclip`)
    - `FineFT/RL/DiHFT/high_level/vae_routing_util.py` (VAE 双轴分位数路由机制)
    - `FineFT/model/low_level.py` (`ensemble_Qnet` 与 `Qnet` 架构)
    - `FineFT/analysis/feature/vae_feature_ood_analysis.py` (VAE 特征级高斯 NLL 分解引擎)
    - `data_preprocess/operator_futures/time_operator/multi_processing_util.py` (时间序列特征算子)
    - `dataset/10min/fu/state_features.npy` (当前使用的 90 维状态特征集合)
  - **架构决策与历史文档**：
    - `docs/adr/0020-remedy-feature-engineering-underflow-and-spurious-indicators.md` (算子除零与公式缺陷修复)
    - `docs/adr/0026-multi-perspective-vae-feature-ood-diagnostic-matrix.md` (多视角 VAE OOD 诊断矩阵)
    - `docs/research/vae_multi_perspective_ood_features_root_cause_and_remediation_guide.md` (前序 OOD 实证调研)

---

## 1. 调研背景与问题陈述

在 FineFT 框架下，高层路由采用双轴 VAE（Slope 趋势轴与 Volatility 波动率轴）评估当前市场机制置信度，并根据阈值（`rule_base_threshold`）在正常交易与防御性平仓（`_defensive_action`）之间切换。

近期在燃料油（`fu`）10 分钟级别的实际评测中，暴露出三大互相交织的核心矛盾：
1. **收益率被严重稀释至 0.5% 左右**：全周期超过 98% 的时间处于防御性空仓状态，实际持仓时间极短（如 Trial 17 仅持仓 0.36% 的步长）。
2. **高层 VAE 发生剧烈 OOD 报警**：验证集/测试集相对于训练集的对数似然从 $+8.87$ 暴跌至 $-30$ 左右，导致查表得到的分位数天然被砸在地表（单步权重中位数仅 0.16，平滑后仅 0.015），被迫触发拒识防御。
3. **特征时空尺度的严重错配**：虽然输入数据是 10 分钟采样，但特征工程中存在大量 $w=96$、$192$ 甚至 $240$ 步长的特征（相当于 16~40 个交易小时，即 3~5 个交易日的宏观周级别指标），在跨越 2 年的日历时间后发生了显著的宏观方差膨胀和均值漂移。

为此，本报告针对用户提出的四个核心课题展开系统性调研并提供落地设计规范。

---

## 2. 课题一：优化特征的方案（减少非合理 OOD）

### 2.1 非合理 OOD 的四大底层成因分析

通过审查 `data_preprocess/` 下的算子实现与 `feature_ood_valid_vs_train.csv` 实证数据，导致非合理 OOD 的机制根因如下：

1. **跨年宏观方差膨胀与全局静态标准化失效**：
   - 当前特征采用 2023 年历史数据的全局均值 $\mu_{\text{train}}$ 和标准差 $\sigma_{\text{train}}$ 进行静态 Z-score 缩放：$z = (x - \mu_{\text{train}}) / \sigma_{\text{train}}$。
   - 2025 年新行情中，市场整体波动中枢放大，`realized_volatility_192` 的方差膨胀为训练集的 **2.34 倍**，导致标准化后的特征值频繁出现 $+3.5 \sim +4.5$ 的极端值。
   - 高斯 VAE 的 NLL 损失包含平方项 $\frac{(x - \mu)^2}{2\sigma^2}$，当极端值出现时，单个特征的重构误差以二次方级数暴增数十倍。
2. **波动率类特征的右偏厚尾未做正态化**：
   - 波动率类指标（如 `realized_volatility`, `rolling_volatility`）天然属于右偏的对数正态分布（Log-normal Distribution）或卡方分布，下界为 0，右侧有长拖尾。
   - 直接用线性 Z-Score 标准化无法改变其偏度，导致 VAE 的高斯假设（即假设隐空间与观测空间服从高斯分布）被严重破坏。
3. **单边计数与累积量缺乏回正机制**：
   - `cntn_192_origin`（下跌 Bar 计数）、`cntd_96_origin`（主卖额差）等指标属于非平稳单边累积指标。在 2024-2025 年单边下行行情中，均值漂移高达 $0.70\sigma$，直接形成永久性 OOD。
4. **历史公式缺陷遗留（名义价格泄露）**：
   - 在 `data_preprocess/operator_futures/time_operator/multi_processing_util.py` 中，部分 `max_*_std_norm` 算子错误地使用名义价格直接除以短期标准差：$x = \text{max\_price} / \sigma$，未减去当前价格或基准均价，导致资产基础价格中枢（如 3000 元 vs 2500 元）直接泄露，引发显著的基准偏移。

### 2.2 系统性特征优化方案矩阵

| 优化技术 | 适用特征类别 | 数学公式 / 实现逻辑 | 预期降 OOD 效果 |
| :--- | :--- | :--- | :--- |
| **动态滚动标准化 (Rolling Z-Score)** | 所有波动率、成交量、价差特征 | $z_t = \frac{x_t - \text{SMA}_K(x)_t}{\text{Std}_K(x)_t + \epsilon}$，取 $K \in [24, 48]$ | **彻底消除跨年均值与方差膨胀**，使新数据强制符合 $\mathcal{N}(0, 1)$ |
| **对数正态化 (Log Transformation)** | `realized_volatility_*`, `rolling_volatility_*`, `volume` | $x' = \log(x + \epsilon)$ 或对数差分波动率 | 将右偏厚尾压缩为近似标准正态分布，抑制平方损失爆炸 |
| **百分位分位数变换 (Rolling Quantile Transform)** | 宽幅振荡指标 (`rsv`, `bollinger_bandwidth`, `imax/imin`) | 将绝对数值映射到 $[0, 1]$ 均匀分布区间：$q_t = \text{PercentileRank}_{K}(x_t)$ | 彻底消除极值离散点对高斯似然的冲击 |
| **差分/比率化去趋势 (Relative Differencing)** | `cntn`, `cntp`, `macro_trade_imbalance` | 改为正负比例：$\frac{\text{cntp} - \text{cntn}}{\text{cntp} + \text{cntn} + \epsilon}$，天然有界 $[-1, 1]$ | 消除单边长波累积趋势漂移 |
| **公式缺陷修正 (Stationary Normalization)** | `max_*_std_norm`, `min_*_std_norm` | 改为无量纲相对收益：$\frac{\text{max\_price} - \text{close}}{\text{close}}$ 或 $\frac{\text{max\_price} - \text{ma}}{\sigma}$ | 修复名义价格除法，恢复相对无量纲性质 |

---

## 3. 课题二：5-10分钟级别特征过滤方案（剔除宏观多日特征）

### 3.1 时间尺度错配的根本原理

在 10 分钟 K 线级别下，一个交易日（含日盘与夜盘）通常仅有约 35 ~ 45 个 Bar（6~7.5 个交易小时）。特征滞后步数 $w$ 的真实物理时间换算如下：

$$T_{\text{hours}} = \frac{w \times 10\text{ min}}{60\text{ min}} = \frac{w}{6}\text{ 小时}$$

- $w = 2 \sim 6$：$20 \sim 60$ 分钟（即时订单流与盘口微观结构）；
- $w = 12 \sim 24$：$2 \sim 4$ 小时（半个交易日内的日内动量与日内波段）；
- $w = 48$：约 8 小时（跨越 1 个完整交易日，日内与跨日过渡区）；
- $w = 96$：约 16 小时（**跨越 2 ~ 3 个交易日**）；
- $w = 192$：约 32 小时（**跨越 4 ~ 5 个交易日，整整一周**）；
- $w = 240$：约 40 小时（**接近两周的宏观周期**）。

**结论**：5-10 分钟级别的高频与短线策略，核心 Alpha 来源是**日内盘口供需不平衡、微观价差摆动、即时动量冲击与日内均值回归**。引入 $w \ge 96$ 的多日/跨周宏观特征，不仅无法对 10 分钟内的微观下单提供有效增量信息，反而将宏观大周期的跨期漂移强加给模型，是导致 OOD 崩溃的罪魁祸首。

### 3.2 特征过滤与剪枝清单

基于实证诊断结果（`feature_ood_valid_vs_train.csv`）与时间尺度分析，将现有 90 个特征严格分类筛选：

#### 3.2.1 坚决淘汰清单 (淘汰 19 个跨日宏观长周期特征，贡献了 >50% 的 OOD)

所有 $w \ge 96$ 的长周期宏观指标一律剔除：
1. `realized_volatility_192` (OOD 贡献榜第 1，贡献度 12.52%)
2. `ema_slope_192` (OOD 贡献榜第 2，贡献度 9.71%)
3. `log_price_slope_96` (OOD 贡献榜第 3，贡献度 7.57%)
4. `bollinger_bandwidth_96_origin` (OOD 贡献榜第 4，贡献度 5.97%)
5. `vma_192_std_norm_origin` (OOD 贡献榜第 9，贡献度 2.19%)
6. `cntd_96_origin` (OOD 贡献榜第 10，贡献度 2.13%)
7. `macro_trade_imbalance_continuous_240` (跨越两周的宏观资金流)
8. `cvd_slope_192`
9. `cvd_slope_96`
10. `sell_volume_oe_trend_192`
11. `wvma_192_origin`
12. `wvma_96_origin`
13. `imax_96_origin`
14. `imin_96_origin`
15. `rsv_96_std_norm_origin`
16. `corr_192_origin`
17. `relative_amount_192`
18. `trend_to_noise_96`
19. `log_return_vol_quantile_192`

此外，剔除**跨期静态持仓份额类特征**（如 `cm_main_sub_open_interest_share_sub` 等），该类特征包含合约交割生命周期的时间趋势泄露。

#### 3.2.2 优化保留清单 (保留 14 个单日过渡特征，需经滚动标准化)

对于 $w = 48$（单日）特征，仅在进行滚动 Z-Score 处理后予以保留：
- `rolling_volatility_48`、`log_price_slope_48`、`trend_r2_48`、`trend_to_noise_48`、`signed_efficiency_48`、`macro_trade_imbalance_continuous_48` 等。

#### 3.2.3 核心主力保留清单 (保留 37 个日内短线特征 + 20 个微观结构/时段特征)

完全适合 5-10 分钟短线交易的核心特征群（共约 55 维）：
1. **盘口与价格微观动量**：`wap_1_log_return_2`, `ask1_price_trend_2/6/24`, `wap_1_trend_2`, `wap_2_trend_2`, `wap_balance`
2. **订单簿深度与不平衡度**：`imblance_volume_oe_trend_2/16`, `ask_size_topk_size_5_share`, `bid_size_topk_size_5_share`, `trade_direction_persistence_20m`
3. **短期日内波动率与通道**：`garman_klass_volatility_16`, `bollinger_bandwidth_12_origin`, `cntp_16_origin`, `rsv_24_std_norm_origin`
4. **会话与日内进度上下文**：`trading_minute_progress`, `morning_session`, `afternoon_session`, `night_session`, `is_opening_30m`, `is_closing_30m`, `prev_day_contract_role_tier`

---

## 4. 课题三：VAE 与 Low-level Agent 特征分离决策

### 4.1 核心问题：两者是否应该使用不同特征集合？

**调研明确结论：强烈建议解耦分离（Decoupled Dual-Stream State Representation）。**

当前系统中，高层 VAE 与底层 `ensemble_Qnet` 强制绑定共享相同的 `state_features.npy`（90维）。这种“一刀切”设计违反了分层强化的信息解耦原则。

### 4.2 分开 vs 不分开的深度全景对比

| 评估维度 | 方案 A：特征分离（推荐方案） | 方案 B：特征不分离（当前现状） |
| :--- | :--- | :--- |
| **功能定位匹配度** | **完美匹配**：<br>• VAE 专注宏观/中观机制分类（低维粗粒度）；<br>• Agent 专注微观执行（高维细粒度）。 | **互相冲突**：<br>• 微观特征扰乱 VAE 的先验分布；<br>• 宏观特征干扰 Agent 的短线出入场。 |
| **OOD 假阳性控制** | **极佳**：VAE 输入压缩至 12~16 维核心体制指标，消除维度灾难，对数似然平稳，误拒识率下降 80%+。 | **极差**：90 维高维累加使得 VAE 对任何微弱扰动极端敏感，高斯 NLL 频繁爆表。 |
| **交易 Alpha 表达能力** | **完全释放**：Agent 可尽情引入 LOB 订单流、高频买卖差、微观动量，无需顾虑 VAE 报警。 | **严重受限**：为了迁就 VAE 不报错，必须阉割高频短线指标；或者为了保留指标让 VAE 锁死。 |
| **计算复杂度与延迟** | **推理显著加快**：VAE 前向计算（3个模型×2个轴）输入维度从 90 降至 15，前向时间降低 ~70%。 | **较高**：每个 Step 需对 90 维大向量执行 6 次全连接编解码前向。 |
| **系统与工程复杂度** | **略有增加**：数据预处理需输出两套特征清单（`vae_features.npy` 与 `agent_features.npy`），环境需支持双状态切片。 | **极低**：单一特征数组，维护成本低，代码结构简单。 |
| **调优与迭代解耦** | **独立演进**：调整底层 Agent 特征无需重新训练 VAE；调整 VAE 机制特征不影响底层策略。 | **牵一发而动全身**：改动任何一个特征，必须从 VAE、Low-level 到 High-level 全流程重训。 |

### 4.3 双流特征解耦架构设计规范

```
                       +----------------------------------------+
                       |        原始 10-Minute 行情与盘口数据     |
                       +----------------------------------------+
                                           |
                   +-----------------------+-----------------------+
                   |                                               |
                   v                                               v
     +---------------------------+                   +---------------------------+
     |   VAE 体制特征流 (12~15维)  |                   |  Agent 交易特征流 (50~70维) |
     +---------------------------+                   +---------------------------+
     | • 趋势强度 (EMA斜率, R2)   |                   | • 盘口挂单量与深浅不平衡度   |
     | • 波动等级 (短期GK波动率)   |                   | • 微观即时价差与微观动量     |
     | • 流动性分级 (成交额占比)  |                   | • 日内时间切片与进度特征     |
     | • 经滚动Z-Score强平稳化    |                   | • 历史持仓收益与反向惩罚特征 |
     +---------------------------+                   +---------------------------+
                   |                                               |
                   v                                               v
     +---------------------------+                   +---------------------------+
     |     High-level VAE 路由   |                   |   Low-level ensemble_Qnet |
     |    (Regime Gating & OOD)  |                   |   (Micro Action Selection)|
     +---------------------------+                   +---------------------------+
                   |                                               |
                   +-----------------------+-----------------------+
                                           |
                                           v
                             +---------------------------+
                             |    最终交易动作决策 Output |
                             +---------------------------+
```

### 4.4 “有 VAE” 与 “无 VAE” 特征选择哲学的深层数学差异

在量化分层强化学习（Hierarchical RL）系统中，“是否引入 VAE 作为高层机制路由器/状态表征模型”从根本上决定了特征选择的数学目标、容错边界与工程架构：

| 评估维度 | 无 VAE 架构（传统单体 RL Agent） | 有 VAE 分层架构（双流机制路由系统） |
| :--- | :--- | :--- |
| **优化目标** | **收益与动作效用最大化**：<br>$\max_\theta \mathbb{E}\left[\sum \gamma^t R(s_t, a_t)\right]$ 或 $\max Q(s, a)$ | **双重解耦独立目标**：<br>• VAE：最大化观测似然证据下界 $\text{ELBO} \approx \mathbb{E}[\log p(x|z)] - D_{\text{KL}}$<br>• Agent：在当前机制上下文约束下的局部动作最优探索 |
| **对非平稳与分布漂移的容忍度** | **高容忍度**：只要特征与次周期收益/买卖方向保持相对单调或排序关系（Rank IC > 0.02），即使宏观特征发生均值位移或方差放大，Q 网络的非线性层仍能通过排序学习做出正确动作。 | **极低容忍度（VAE 侧）**：高斯观测假设使得重构负对数似然 $\text{NLL} \propto \sum \frac{(x_i - \mu_i)^2}{2\sigma_i^2}$ 对极端值具有**二次方放大效应**。单个宏观特征轻微漂移（如 $\Delta \mu = 1.0\sigma$）就会让似然暴跌数十点，导致置信度归零并误触发防御拒识。 |
| **输入维度空间** | **支持高维宽表（50 ~ 100 维）**：深度神经网络具备强大的表征自压缩能力，通常依赖正则化（Dropout / L2）或注意力机制吸收大量弱特征。 | **严控低维紧凑表（10 ~ 15 维）**：高维高斯分布存在“球壳集中效应”（Curse of Dimensionality）。90 维下所有样本点均处于各向同性高斯球面的极薄边缘，似然对数方差剧烈发散，密度估计失效。 |
| **特征选取主导指标** | **收益敏感度（Alpha Power）**：<br>• 信息系数（IC / Rank IC）<br>• 互信息（Mutual Information with Returns）<br>• 特征归因重要性（SHAP / Tree Gain） | **生成平稳性与机制可聚类性（Stationarity & Separability）**：<br>• 跨期 OOD 漂移度（$\Delta\text{NLL}$、Wasserstein 距离）<br>• 机制判别 F-Score / 方差分析<br>• 数学有界性（Boundedness）与天然无量纲 |
| **系统架构风险** | **策略过拟合**：可能在特定市场周期学到伪规律，回测表现优异但实盘泛化收益衰减。 | **级联误拒识瘫痪**：如果让 VAE 吞入微观高频或未过滤特征，VAE 的 OOD 崩溃会直接掐断底层 Agent 的交易权限，导致策略 98%+ 时间处于防守状态而完全丧失收益能力。 |

### 4.5 为 VAE 专属特征设立的四大黄金量化准则

为 VAE 挑选输入特征时，**绝对不能以“能不能预测收益”作为第一评价指标**，必须强制满足以下四大统计假设准则：

1. **跨期分布绝对平稳（Invariance & Low OOD Drift）**：
   - **量化阈值**：在测试集对比训练集的检验中，单特征 $\Delta\text{NLL} < 0.5$，方差比 $\text{Var}_{\text{test}} / \text{Var}_{\text{train}} \in [0.8, 1.25]$，双样本 Kolmogorov-Smirnov 检验统计量 $D < 0.15$。
   - **筛选方法**：利用已跑通的 `FineFT/analysis/feature/vae_feature_ood_analysis.py` 输出，一票否决任何跨越交易日（$w \ge 48$）的长周期指标与未经平稳化的绝对量指标。
2. **天然无量纲与数学强有界（Dimensionless & Bounded）**：
   - 优先选择取值范围在 $[-1, 1]$ 或 $[0, 1]$ 之间具有物理/几何明确边界的特征：
     - 回归拟合优度：$R^2 \in [0, 1]$（如 `trend_r2_24`）；
     - 价格效率比：$\text{Signed Efficiency Ratio} = \frac{\Delta P}{\sum |\Delta p_i|} \in [-1, 1]$；
     - 日内分位数位置：$\text{RSV} = \frac{P - L_n}{H_n - L_n} \in [0, 1]$；
     - 交易时间进度：`trading_minute_progress` $\in [0, 1]$。
   - 波动率指标禁止使用原始标准差，必须转换为对数比率或相对于短期中枢的超额比：$\ln(\sigma_{\text{GK}} / \text{MA}_{12}(\sigma_{\text{GK}}))$。
3. **宏观机制判别力（Regime Separability）**：
   - 特征必须在预定义的市场机制（如“强趋势 vs 宽幅震荡”、“高波动 vs 低波动”）下具备统计显著的分布位移，确保隐空间 $z$ 能够学到有物理意义的流形聚类。
   - **量化检验**：计算候选特征在不同机制标签下的方差分析（ANOVA）$F\text{-statistic} > 50$ 且 $p\text{-value} < 10^{-5}$，或 Wasserstein 距离在不同机制间具备显著分离度。
4. **低共线性与严控维度（Low Collinearity & Compactness, $\le 15$ 维）**：
   - 特征矩阵两两相关系数严格控制在 $|r| < 0.60$ 以内。
   - 总体维度严格限制在 12 ~ 15 维以内，避免高斯似然维数灾难。

### 4.6 推荐的 VAE vs Low-level Agent 正交特征清单（10分钟级别）

结合商品期货 10 分钟线的微观特性，设计以下完全解耦的双流特征架构：

```
                    [ 原始 10-min 行情数据 (WAP, OHLCV, 盘口) ]
                                    │
           ┌────────────────────────┴────────────────────────┐
           ▼                                                 ▼
【流一：VAE 宏观机制特征】(12~14维)                  【流二：Agent 微观执行特征】(50~60维)
  核心目标：稳定识别宏观态、零OOD误拒识                 核心目标：捕捉买卖不对称性、精细寻找入场点
 ────────────────────────────────                   ────────────────────────────────
 1. 趋势纯度与平滑度 (3维):                         1. 微观即时价差与动量 (15维):
    • trend_r2_24 (拟合优度 R^2 ∈ [0,1])               • log_return_1, log_return_3, log_return_6
    • signed_efficiency_24 (位移路程比 ∈ [-1,1])       • ask1_price_trend_2/6, bid1_price_trend_2/6
    • trend_to_noise_24 (信噪比)                       • cm_spread_rolling_zscore_12
 2. 相对波动分层 (3维):                             2. 订单簿深度与资金流动态 (15维):
    • log(garman_klass_volatility_16)                  • wap_balance, flow_imbalance_6
    • bollinger_bandwidth_12_origin                    • ask_size_topk_size_5_share, bid_size_topk...
    • rsv_24_std_norm_origin                           • trade_direction_persistence_20m
 3. 多空微观分歧度 (3维):                           3. 短期超买超卖与均线摆动 (15维):
    • trade_direction_persistence_20m                  • RSI_12, KDJ_12, CCI_12
    • wap_balance (买卖加权偏离比)                      • 价格对短期EMA偏离率 (WAP - EMA_12) / EMA_12
 4. 交易日内拓扑与时段 (3维):                       4. 账户状态与高层机制感知 (5~8维):
    • trading_minute_progress (日内进度 0~1)           • VAE 输出的隐表征 z (2~4维)
    • is_opening_30m, is_closing_30m                   • VAE 路由置信度与机制概率 Softmax 权重
                                                       • 历史持仓收益与反向惩罚特征
```

### 4.7 落地实操：四步自动化特征筛选与接入管线

为从当前现有的 90 维特征全集平滑演进到双流解耦体系，制定以下工程管线：

1. **第一步：硬性时间窗口截断（Window Truncation）**
   - 在特征工程算子层直接废弃所有 $w \ge 48$ 的跨日累积窗口；
   - 10 分钟周期下，最长统计窗口锁死在 $w=24$（4小时/半个交易日），彻底切除跨周大宗商品周期漂移对日内模型的污染。
2. **第二步：OOD 似然初筛黑名单（Likelihood Pre-filtering）**
   - 运行 `./FineFT/script/analysis/feature/vae_feature_ood_fu_10.sh` 批量分析验证集与测试集合约；
   - 建立自动化过滤准则：单特征 $\Delta\text{NLL} > 0.5$ 或方差比 $\text{Var}_{\text{ratio}} \notin [0.7, 1.4]$ 的特征，一律移出 VAE 输入候选池。
3. **第三步：机制聚类有效性与相关性剪枝（Correlation & Clustering Pruning）**
   - 针对候选平稳特征计算皮尔逊相关矩阵；
   - 采用层次聚类（Hierarchical Clustering）自底向上合并，在每个相关度 $|r| > 0.60$ 的子簇中，仅保留机制标签 F-score 最高或 Wasserstein 距离最大的一项特征，将 VAE 维度强制压缩至 12~14 维。
4. **第四步：双流模型接入与置信度归一化修复（Dual-Stream Integration & Softmax Fix）**
   - 数据预处理阶段输出 `vae_state_features.npy` (12维) 与 `agent_state_features.npy` (55维)；
   - **关键修复**：在 `FineFT/RL/DiHFT/high_level/vae_routing_util.py` 中，对 VAE 输出的双轴分位数必须引入 Softmax 或 Temperature 归一化，将权重和约束在 1.0 附近，彻底杜绝当前 `volatility_weights sum=0.027` 永远被 `threshold=0.20` 防御拒识锁死的致命 Bug。


---

## 5. 课题四：Low-level Agent 专用 OOD 检验体系与脚本设计

### 5.1 为什么必须为 Low-level Agent 独立开发 OOD 检验脚本？

**数学本质的不可替代性**：
1. **模型范式不同**：
   - VAE 是**无监督概率生成模型（Density Estimator）**，其本质是估计概率分布 $p(x)$。OOD 的数学定义明确为样本落在低密度区（通过重建损失 $\text{NLL}$ 衡量）。
   - Low-level Agent（`ensemble_Qnet`）是**判别式值函数逼近器（Value-based Reinforcement Learning）**。它不拟合特征分布 $p(s)$，而是拟合状态-动作映射 $Q(s, a) \approx \mathbb{E}[R_t | s, a]$。
2. **失效模式不同**：
   - VAE 的 OOD 是“输入特征长得不像训练集”；
   - 强化学习 Agent 的 OOD 是**“外推认知不确定性爆炸（Epistemic Uncertainty Explosion）”**或**“Q 值高估/崩塌导致的动作盲目跳变”**。即使某个状态的特征密度很高，如果 Agent 在该状态下的策略未收敛或动作值发散，策略依然处于严重的强化学习 OOD 状态。
3. **因此**：现有基于高斯 NLL 的 VAE OOD 脚本完全无法用于评估 Agent 的决策可靠性，**必须开发独立的强化学习策略级 OOD 检验工具**。

### 5.2 Low-level Agent OOD 检验的三大核心数学范式

依托仓库现有的 `ensemble_Qnet`（由 $M$ 个独立 Q 网络构成的集成架构，`FineFT/model/low_level.py`），可天然落地业界最经典的基于认知不确定性的检测方法：

#### 范式一：集成 Q 值分歧度（Ensemble Q Epistemic Variance / Disagreement）

在训练集内（In-Distribution），不同初始化或 Bootstrap 的集成子网络对最优动作的估计高度一致；进入 OOD 区域后，不同网络的函数外推剧烈分歧：

$$\text{Var}_{\text{ensemble}}(Q(s, a)) = \frac{1}{M} \sum_{m=1}^M \left( Q_m(s, a) - \bar{Q}(s, a) \right)^2$$

定义贪心动作的认知不确定性：
$$U_{\text{greedy}}(s) = \text{Var}_{\text{ensemble}}\left(Q(s, \arg\max_a \bar{Q}(s, a))\right)$$

当 $U_{\text{greedy}}(s)$ 显著超过训练回放池（Replay Buffer）的 95% 分位数时，判定 Agent 处于策略 OOD。

#### 范式二：最优动作决策分歧率（Action Flipping & Disagreement Rate）

统计集成内 $M$ 个独立 Q 网络在单步所推荐的最优动作的一致性：

$$a_m^*(s) = \arg\max_a Q_m(s, a), \quad m=1, \dots, M$$

$$\text{Disagreement}(s) = 1 - \frac{\max_{a} \sum_{m=1}^M \mathbb{I}(a_m^*(s) = a)}{M}$$

- 若 $M$ 个网络全票通过选择同一动作，$\text{Disagreement} = 0$（决策极度置信）；
- 若动作产生剧烈分歧（如 3 个选多、3 个选中立、3 个选空），$\text{Disagreement} \to 1.0$（高度 OOD）。

#### 范式三：Replay Buffer 特征隐空间马氏距离（Latent Mahalanobis Distance）

提取 `ensemble_Qnet` 倒数第二层特征表征 $h(s) \in \mathbb{R}^{D_h}$，基于训练 Replay Buffer 计算均值向量 $\mu_h$ 和协方差矩阵 $\Sigma_h$：

$$D_M(s) = \sqrt{(h(s) - \mu_h)^T (\Sigma_h + \lambda I)^{-1} (h(s) - \mu_h)}$$

结合高斯收缩估计（Ledoit-Wolf）保证矩阵求逆数值稳定性，提供严密的隐空间距离度量。

### 5.3 新检验脚本架构设计规范 (`low_level_agent_ood_analysis.py`)

建议新建分析工具：`FineFT/analysis/feature/low_level_agent_ood_analysis.py`。

#### 输入与产物规范
- **输入参数**：
  - `--model_path`：`trained_model.pkl`（包含 `ensemble_Qnet` 检查点）；
  - `--train_buffer_dir`：训练阶段 Replay Buffer 离线缓存；
  - `--eval_data_path`：验证/测试合约 Feather 数据文件；
  - `--output_dir`：`analysis_result/DiHFT/agent_ood/...`。
- **输出产物**：
  - `agent_ood_summary.csv`：汇总逐合约、逐时段的集成方差均值、动作分歧率与马氏距离；
  - `agent_epistemic_uncertainty_timeseries.png`：绘制回测过程中 Q 值方差曲线与行情走势对照图；
  - `action_agreement_heatmap.pdf`：各状态下的动作集成一致性混淆矩阵。

---

## 6. 实施路线图与落地建议 (Actionable Roadmap)

1. **第一阶段：特征剪枝与平稳化修复（立竿见影）**
   - 立即从 `state_features.npy` 中移除 19 个 $w \ge 96$ 的跨日宏观长周期特征；
   - 修复 `max_*_std_norm` 等公式缺陷，全面启用对数波动率与滚动 Z-score。
2. **第二阶段：双流特征解耦改造**
   - 生成专门面向 VAE 的体制特征集 `vae_features.npy`（精选 12~15 维短周期趋势与波动率特征）；
   - 保留 Agent 的 55 维高频短线交易特征集 `agent_features.npy`；
   - 更新 `initiate_base_env` 与 `vae_routing_util.py` 支持双流状态切片。
3. **第三阶段：开发并部署 Low-level Agent 专用 OOD 检验工具**
   - 实现 `FineFT/analysis/feature/low_level_agent_ood_analysis.py`，建立集成 Q 方差与决策一致性基准监控；
   - 形成高层 VAE（宏观体制 OOD）与底层 Agent（微观决策 OOD）的双重立体风控体系。
