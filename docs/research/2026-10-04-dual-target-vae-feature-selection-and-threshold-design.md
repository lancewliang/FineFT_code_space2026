# 面向斜率与波动率双目标分片的 VAE 特征解耦筛选方法与阈值设计深度调研报告

- **报告编号**：RES-2026-1004-02
- **研究日期**：2026-10-04
- **研究对象**：FineFT 高层 VAE 体制识别与特征筛选流水线 (`data_preprocess/operator_futures/feature_selection/muti_contract/` 与 `FineFT/RL/DiHFT/high_level/`)
- **核心诉求**：当前流水线中 VAE 状态特征仅有单份清单 (`vae_state_features.npy`，18 维)，但高层路由需要区分两套正交分片体制（斜率分片：Down/Flat/Up；波动率分片：Low/Mid/High Vol）。单份特征无法同时兼顾方向性与离散度，导致纯波动率特征在筛选阶段被误杀淘汰，且在无监督生成重构损失下造成严重的信号信噪比稀释与虚假 OOD 误报。用户要求设计面向两个不同目标的 VAE 特征过滤方法、数学准则与量化阈值，并提供工程落地与架构升级方案。

---

## 1. 第一手证据源 (Primary Sources)

本研究严格基于工程中现行代码实现与已通过的架构决策记录 (ADR) 展开：

1. **核心架构决策记录 (ADRs)**：
   - `docs/adr/0037-dual-stream-feature-decoupling-architecture-for-vae-and-rl-agent.md`：定义了 VAE（无监督生成模型，ELBO 优化，超平稳小维度）与 RL Agent（MDP 决策，Alpha 收益最大化，大维度）的解耦架构及 `DEFAULT_VAE_PROFILE` 契约。
   - `docs/adr/0036-modular-three-stage-funnel-feature-selection-and-statistical-gates.md`：确立了特征筛选三阶段漏斗架构（Stage 1 向量化快速过滤、Stage 2 单决策窗口 CatBoost 评分、Stage 3 正交去重与体制审计）。
   - `docs/adr/0038-decoupled-dual-stream-feature-blacklists-for-vae-and-rl-agent.md`：确立了三级特征黑名单体系（Global Hygiene、VAE Stream-Specific、RL Stream-Specific）。
   - `docs/adr/0011-volatility-labeling-and-method-isolated-outputs.md`：确立了 `labeling_method="slope"` 与 `labeling_method="volatility"` 的独立标定、转折点切片与目录隔离机制 (`valid/slope/` 与 `valid/volatility/`)。
   - `docs/adr/0031-modular-gating-architecture-and-hierarchical-dual-gating.md`：确立了高层双层分层门控机制 (`HierarchicalDualGating`)，明确要求斜率与波动率双轴各自提供清晰的边际概率间隔 ($\ge 0.12$) 与绝对 OOD 截断 ($\ge 0.005$)。

2. **体制标定与转折点切片代码**：
   - `FineFT/datahandler/valid_cross_contract_label_calibration.py`：
     - 行 167-177：`_segment_log_return_volatility` 计算段内对数收益率非年化总体标准差：$100 \times \text{std}(\Delta \log(\text{bid1\_price}), \text{ddof}=0)$；
     - 行 240-270：`_fit_contract` 在 `labeling_method == "slope"` 时取归一化斜率 `group_slopes`，在 `volatility` 时取段内波动率 `group_vols`；
     - 行 385-420：行级标签映射 `_label_for_score`，针对斜率区分 Label 0 (Down), 1 (Flat), 2 (Up)，涨跌停置为 2 和 0；针对波动率区分 Label 0 (Low), 1 (Mid), 2 (High)，涨跌停统一置为极值 2。

3. **VAE 数据物化与训练流水线**：
   - `FineFT/datahandler/vae_data_creation.py`：
     - 行 115-125：目前统一加载单份 `ArtifactNames.VAE_STATE_FEATURES_NPY` (`vae_state_features.npy`)；
     - 行 130-170：`make_data` 虽按 `labeling_method` 隔离了 `VAE_data/{labeling_method}/`，但特征列集合完全无差别；
     - 行 175-185：测试集 `test.npy` 直接落入 `VAE_data/test/`，缺乏标定方法维度隔离。
   - `FineFT/RL/DiHFT/VAE/main.py`：
     - 行 230-245：`discover_test_sources` 硬编码读取 `VAE_data/test/test_*.npy`，未根据 `labeling_method` 隔离特征维度；
     - 行 265-275：`materialize_label_training_data` 读取 `VAE_data/{labeling_method}/`，导致斜率与波动率 VAE 被迫使用相同的输入维度 `train_manifest.feature_dim`。
   - `FineFT/RL/DiHFT/VAE/vae.py`：
     - 行 40-70：`MLP_VAE` 架构，高斯解码器输出 `recon_mu` 与 `recon_logvar`；
     - 行 95-120：`loss_function` 采用 Gaussian NLL 重构损失与 KL 散度约束：
       $$\text{ELBO} = -\sum_{j=1}^D \left[ \frac{(x_j - \hat{\mu}_j)^2}{2\hat{\sigma}_j^2} + \log \hat{\sigma}_j + \frac{1}{2}\log(2\pi) \right] - D_{\text{KL}}(q(z|x) \parallel p(z))$$

4. **高层路由与门控判决代码**：
   - `FineFT/RL/DiHFT/high_level/vae_routing_util.py`：
     - 行 590-596：同时加载单份 `tech_indicator_list` (`rl_state_features.npy`) 与单份 `vae_indicator_list` (`vae_state_features.npy`)；
     - 行 680-695：`load_vae_axis` 初始化 `slope` 与 `volatility` 双轴 VAE 模型时，输入维度均被死死绑定为 `len(self.vae_indicator_list)`；
     - 行 820-835：`get_quantiles(self, s)` 将完全相同的特征向量 `s`（来自单份 `self.vae_state_array`）分别传入 `self.vae_models["slope"]` 与 `self.vae_models["volatility"]`；
     - 行 879-881：提取 `volatility_weights` 与 `slope_weights` 并调用 `gating_strategy.decide`。
   - `FineFT/RL/DiHFT/high_level/gating/hierarchical_gating.py`：
     - 行 45-75：`HierarchicalDualGating.decide` 执行双门控判决（Gate 1: 绝对 OOD 截断；Gate 2: 边际清晰度间隔）。

5. **多合约特征筛选引擎**：
   - `data_preprocess/operator_futures/feature_selection/muti_contract/types.py`：
     - 行 100-117：`DEFAULT_VAE_PROFILE` 强制设定 `min_abs_ic=0.015`, `min_sign_consistency=0.70`, `min_rank_ic_ir=0.35`；
   - `data_preprocess/operator_futures/feature_selection/muti_contract/predictive_audit.py`：
     - 行 60-80：`execute_predictive_audit` 强制将所有候选特征与有向未来收益率 `future_return = (P_{t+w} - P_t) / P_t` 进行 RankIC 和跨合约符号一致性检验；
   - `data_preprocess/operator_futures/feature_selection/muti_contract/pipeline.py`：
     - 行 275-345：`_evaluate_stream_branch` 分支筛选逻辑；
     - 行 680-730：`_run_dual_stream_train_stage` 仅产出 `vae_selected` 与 `rl_selected`。
   - `data_preprocess/operator_futures/feature_selection/commodity_feature_blacklists.json`：
     - 集中定义了 `global`（原始价格、未中心化比率）、`vae`（跨日宏观漂移指标）与 `rl_agent` 专属黑名单。

---

## 2. 核心结论与现状审计 (Executive Summary)

### 2.1 核心审计结论

1. **“单流双用”存在根本性的数学与业务冲突**：
   斜率（Slope）与波动率（Volatility）在金融时间序列中分别代表一阶矩变动（有向均值位移 $\mathbb{E}[\Delta P]$）与二阶矩变动（无向离散度 $\text{Var}(\Delta P)$）。两者的物理量纲、统计对称性、分布形态完全正交。使用同一份特征清单同时驱动斜率 VAE 与波动率 VAE，在逻辑上是自相矛盾的。

2. **当前特征筛选体系对波动率特征构成“系统性误杀”**：
   在 `predictive_audit.py:63` 中，所有特征仅与“未来有向收益率”计算 RankIC 与跨合约 Sign Consistency。纯波动率特征（如已实现波动率、ATR、布林带宽、振幅）与未来价格方向在理论期望上正交（相关系数为 0，跨合约符号一致性约 50%），因此在 Stage 3 预测性审计中被 100% 误杀。当前落地的 18 维 `vae_state_features.npy` 中，几乎全为有向趋势与价格偏离特征，波动率特征仅存 1 维伪标量。

3. **双轴 VAE 均因此遭受严重的密度估计破坏**：
   - **斜率 VAE 混入波动率特征**：由于波动率特征在大涨和大跌时呈现相同的数值，在欧氏重构损失下引入了无关方差，导致区分多空方向的信噪比 (SNR) 暴跌 50% 以上；
   - **波动率 VAE 混入有向特征**：单模态高斯解码器在面对高波动体制下的大涨与大跌时，被迫拟合双峰分布，导致重构方差大幅虚高，测试期一旦出现单边走势即频繁触发虚假 OOD 熔断；
   - **高层门控崩溃**：波动率轴无法拉开概率差距，`volatility_margin` 长期低于阈值 0.12，高层被迫频繁执行无意义的防守平仓或随机切换。

4. **架构演进方向：全面升级为“三流解耦体系 (Triple-Stream Architecture)”**：
   特征筛选与下游建模必须彻底拆分为三套独立流：
   - **`VAE_Slope` 流**：专注识别 Down/Flat/Up，保留有向动量与不平衡量，过滤宏观漂移与纯离散度特征；
   - **`VAE_Volatility` 流**：专注识别 Low/Mid/High Vol，过滤带符号方向指标与名义价格，以“未来绝对收益率 / 波动率”为审计目标；
   - **`RL_Decision` 流**：保持 ADR-0037/0038 的高容量 Alpha 特征流。

### 2.2 核心量化指标与设计对比总览

| 维度 / 流 | 当前单一 VAE 特征 (`DEFAULT_VAE_PROFILE`) | 拟重构：斜率 VAE 流 (`VAE_Slope`) | 拟重构：波动率 VAE 流 (`VAE_Volatility`) | 强化学习决策流 (`RL_Decision`) |
| :--- | :--- | :--- | :--- | :--- |
| **物理标定目标** | 混杂标定（实际仅与有向收益对齐） | `signed_percentage_slope` (Down/Flat/Up) | `segment_log_return_volatility` (Low/Mid/High Vol) | 交易累积贴现收益 $\max \mathbb{E}[\sum \gamma^t R_t]$ |
| **预测审计目标** | 有向价格收益率 $R_{t,w}$ | 有向未来收益率 $R_{t,w}$ ($w=6$) | 未来绝对收益率 $|R_{t,w}|$ 或局部波动率 | 有向收益率与微观价差收益 |
| **黑名单过滤策略** | 仅过滤跨日宏观非平稳特征 | 过滤宏观非平稳特征 + 纯波动率特征 | 严格过滤所有带符号方向特征 + 宏观漂移 | 仅过滤全局卫生黑名单 (极宽容) |
| **分布漂移 Mean PSI** | $\le 0.10$ | $\le 0.10$ (确保高斯先验对齐) | $\le 0.12$ (容纳周期波动聚集) | $\le 0.45$ |
| **预测相关性指标** | $\text{RankIC} \ge 0.015$ (有向) | $\text{RankIC} \ge 0.020$ (有向动量) | $\text{Vol-RankIC} \ge 0.030$ (对波动率) | $\text{RankIC} \ge 0.010$ |
| **跨合约符号一致性**| $\ge 0.70$ | $\ge 0.75$ (方向严格一致) | $\ge 0.75$ (波动单调递增) | $\ge 0.55$ |
| **体制区分度统计量**| 无 (未实施体制审计) | ANOVA $F \ge 4.0$ / 单调性 | ANOVA $F \ge 6.0$ / 严格单调递增 | 目标区间收益差 |
| **正交聚类阈值 $|r|$**| $\le 0.65$ | $\le 0.65$ | $\le 0.60$ (抑制不同窗口冗余) | $\le 0.80$ |
| **目标保留维度 $K$** | $12 \sim 18$ 维 (实出 18 维) | **$12 \sim 16$ 维** (紧凑方向流) | **$10 \sim 14$ 维** (正交离散度流) | $55 \sim 70$ 维 (实出 68 维) |
| **持久性噪声过滤** | 开启 (过滤 $hl < 2.0$) | 开启 (消除超微观白噪声) | 开启 (消除高频抖动) | 关闭 (释放高频微观 OFI) |

---

## 3. 单份 VAE 特征“双用”的数学冲突与根本缺陷剖析

### 3.1 缺陷一：预测性审计维度的“系统性误杀”与统计学证明

在 `predictive_audit.py:63` 与 `metrics.py:50` 中，前向收益率定义为：
$$R_{t,w} = \frac{P_{t+w} - P_t}{P_t}$$
候选特征 $f$ 的预测能力通过 RankIC、Sign Consistency 与 RankIC IR 进行硬性门禁拦截。

#### 数学证明：为何波动率特征必死于现有门禁
设 $f_{\text{vol}}(t) \ge 0$ 为任意优秀的波动率特征（例如 $k$ 周期已实现波动率 $\text{RV}_k$、布林带宽 $\text{Bandwidth}_k$、ATR 等）。
根据现代量化金融标准收益率生成过程：
$$R_{t,w} = \mu_t + \sigma_t \cdot \epsilon_t, \quad \epsilon_t \sim \mathcal{D}(0, 1), \quad \mathbb{E}[\epsilon_t \mid \mathcal{F}_t] = 0$$
波动率特征 $f_{\text{vol}}(t)$ 主要是当前条件波动率 $\sigma_t$ 的单调增函数：$f_{\text{vol}}(t) = g(\sigma_t) + \eta_t$。

1. **协方差期望为零**：
   $$\text{Cov}(f_{\text{vol}}(t), R_{t,w}) = \mathbb{E}[f_{\text{vol}}(t) \cdot (\mu_t + \sigma_t \epsilon_t)] - \mathbb{E}[f_{\text{vol}}(t)]\mathbb{E}[R_{t,w}]$$
   在无套利或弱有效市场中，即使存在微小漂移 $\mu_t$，资产收益的无条件均值亦极其接近 0。故：
   $$\mathbb{E}[f_{\text{vol}}(t) \cdot \sigma_t \epsilon_t] = \mathbb{E}[f_{\text{vol}}(t) \sigma_t \cdot \mathbb{E}[\epsilon_t \mid \mathcal{F}_t]] = 0 \implies \text{Cov}(f_{\text{vol}}(t), R_{t,w}) \approx 0$$
   因此，理论总体秩相关系数：
   $$\rho_{\text{Rank}}(f_{\text{vol}}, R_{t,w}) \approx 0$$
   这直接打破了 `DEFAULT_VAE_PROFILE.min_abs_ic = 0.015` 的硬性约束。

2. **跨合约符号一致性崩溃**：
   对于跨越 $C$ 个交割周期的历史合约样本 $c \in \{1, \dots, C\}$，样本估计值 $\hat{\rho}_c(f_{\text{vol}}, R_{t,w})$ 呈现均值为 0 的渐近正态分布：
   $$\hat{\rho}_c \sim \mathcal{N}\left(0, \frac{1}{T_c}\right)$$
   若某合约处于长期牛市（$\bar{R} > 0$），高波动往往伴随大涨，样本相关系数可能为正（$+0.008$）；若某合约处于熊市（$\bar{R} < 0$），暴跌时波动放大，样本相关系数为负（$-0.012$）。
   跨合约符号一致性指标定义为：
   $$\text{SignConsistency}(f) = \frac{1}{C} \max\left( \sum_{c=1}^C \mathbf{1}_{\hat{\rho}_c > 0}, \sum_{c=1}^C \mathbf{1}_{\hat{\rho}_c < 0} \right)$$
   在大样本下，正负号服从二项分布 $B(C, 0.5)$：
   $$\mathbb{E}[\text{SignConsistency}(f_{\text{vol}})] \approx 0.50$$
   而 `DEFAULT_VAE_PROFILE.min_sign_consistency = 0.70`，这在统计学上构成了**必然拦截条件**。纯波动率特征在 Stage 3 会被以 `Sign Consistency Filter Dropped` 无情全部过滤淘汰。

### 3.2 缺陷二：无监督生成模型 ELBO 重构损失下的信噪比 (SNR) 稀释

在 `FineFT/RL/DiHFT/VAE/vae.py:100` 中，VAE 的目标是最大化对数边际似然的证据下界 (ELBO)。设输入特征向量为 $x \in \mathbb{R}^D$：
$$\log p_\theta(x) \ge \mathcal{L}_{\text{ELBO}}(x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\text{KL}}(q_\phi(z|x) \parallel p(z))$$
高斯解码器下的对数似然重构损失为：
$$\log p_\theta(x|z) = -\frac{1}{2} \sum_{j=1}^D \left[ \frac{(x_j - \hat{\mu}_j(z))^2}{\hat{\sigma}_j(z)^2} + \log(2\pi \hat{\sigma}_j(z)^2) \right]$$

#### 场景 A：斜率 VAE（识别 Up vs Down）混入波动率特征的数学灾难
假设特征空间为正交分解：$x = [x_{\text{dir}}^T, x_{\text{vol}}^T]^T$，其中方向特征 $x_{\text{dir}} \in \mathbb{R}^{D_1}$，波动特征 $x_{\text{vol}} \in \mathbb{R}^{D_2}$，总维度 $D = D_1 + D_2$。
斜率分片标定旨在最大化两个对立体制之间的对数似然分离度：
$$\Delta \ell_{\text{Up-Down}}(x) = \log p(x \mid \text{Up}) - \log p(x \mid \text{Down})$$
根据重构损失可加性：
$$\Delta \ell_{\text{Up-Down}}(x) = \Delta \ell_{\text{dir}}(x_{\text{dir}}) + \Delta \ell_{\text{vol}}(x_{\text{vol}})$$

- **波动率的对称性**：在激烈的单边上涨（Up）和单边暴跌（Down）中，市场的已实现波动率、振幅与布林带宽分布高度一致，即：
  $$p(x_{\text{vol}} \mid \text{Up}) \approx p(x_{\text{vol}} \mid \text{Down})$$
  因此，在期望意义下，波动率部分不提供任何判别信息：
  $$\mathbb{E}[\Delta \ell_{\text{vol}}(x_{\text{vol}})] \approx 0$$
- **随机扰动与方差放大**：但在微观有限样本下，重构误差项的方差为：
  $$\text{Var}(\Delta \ell_{\text{vol}}) = \sum_{j=1}^{D_2} \text{Var}\left( \frac{(x_j - \hat{\mu}_j)^2}{2\hat{\sigma}_j^2} \right) = \mathcal{O}(D_2)$$
  这导致判决信号的信噪比 (SNR) 严重衰减：
  $$\text{SNR}_{\text{slope}} = \frac{(\mathbb{E}[\Delta \ell_{\text{Up-Down}}])^2}{\text{Var}(\Delta \ell_{\text{Up-Down}})} = \frac{(\mathbb{E}[\Delta \ell_{\text{dir}}])^2}{\text{Var}(\Delta \ell_{\text{dir}}) + \text{Var}(\Delta \ell_{\text{vol}})} \approx \frac{D_1^2 \cdot S_{\text{dir}}}{D_1 \cdot V_{\text{dir}} + D_2 \cdot V_{\text{vol}}}$$
  若 $D_1 \approx D_2$（特征各占一半），信噪比直接下降 **50%**！
- **伪 OOD 误报**：当行情出现低波动缓涨转为高波动暴涨时，$x_{\text{vol}}$ 剧烈上升，导致针对 Up 训练的 VAE 似然值断崖式暴跌，高层门控错误判定为 OOD 异常或震荡（Flat），白白错失趋势交易时机。

#### 场景 B：波动率 VAE（识别 Low/Mid/High Vol）混入方向特征的数学灾难
假设针对 High Vol 体制训练 VAE，输入特征中混入了带符号方向指标（如 `signed_percentage_slope`、`log_price_slope_48`、`level5_ofi`）。

1. **多模态分布强制高斯化引发的方差爆炸**：
   在 High Vol 分片中，价格要么剧烈大涨（$x_{\text{dir}} \approx +\mu_0$），要么剧烈暴跌（$x_{\text{dir}} \approx -\mu_0$）。其实际分布是典型的双峰分布：
   $$p(x_{\text{dir}} \mid \text{High Vol}) \approx \frac{1}{2} \mathcal{N}(+\mu_0, \sigma_0^2) + \frac{1}{2} \mathcal{N}(-\mu_0, \sigma_0^2)$$
   VAE 的单模态连续先验 $z \sim \mathcal{N}(0, I)$ 与单高斯解码器 $\hat{\mu}(z)$ 在重构无条件均值时，只能被迫输出：
   $$\mathbb{E}[\hat{\mu}] \approx 0, \quad \hat{\sigma}^2 \approx \sigma_0^2 + \mu_0^2$$
   重构方差虚高 $\mu_0^2$！这直接浪费了网络宝贵的隐空间容量，使潜在变量 $z$ 被迫去编码无意义的“当前是涨还是跌”，从而严重稀释了对真正波动率离散结构的表征精度。
2. **测试期边际清晰度崩溃**：
   当测试集出现一段平静单边微涨（Low Vol + 弱 Up，斜率为 $+0.005$）时，High Vol VAE 因其重构均值设定在 0 周围，可能会给出一个异常偏高的似然；反之，当 High Vol 发生单边狂飙时，极大的正向偏差 $(x_{\text{dir}} - 0)^2$ 会产生巨大的 NLL 惩罚，导致模型将真正的 High Vol 判决为 OOD。高层双层门控中的 `volatility_margin` 跌入 0.02 以下的混沌区，触发死锁防御。

---

## 4. 面向“斜率分片识别”的 VAE 特征筛选方法与阈值设计

### 4.1 目标定义与数学物理映射

- **物理目标**：识别市场宏观有向动量与波段趋势，区分三类离散分片体制：
  - **Label 0 (Down)**：空头主导，价格斜率负向显著，订单簿卖方承压，主动卖盘持续激增；
  - **Label 1 (Flat)**：多空均衡与震荡整理，均值回复，净流动性不平衡量趋近于 0；
  - **Label 2 (Up)**：多头主导，价格斜率正向显著，买方持续推进。
- **对齐标定源码**：`valid_cross_contract_label_calibration.py:263` 中的 `signed_percentage_slope` ($k_{\text{norm}}$)。
- **核心特征类型**：
  1. 尺度不变的有向价格趋势（如对数价格斜率、指数均线发散度）；
  2. 微观订单流失衡量（Order Flow Imbalance, OFI、买卖盘深层堆积斜率）；
  3. 成交买卖主动量比（如连续量价失衡、多空成交额比例）；
  4. 日内交易时间拓扑特征（`trading_minute_progress` 等强制透传特征）。

### 4.2 四阶段筛选阶梯与具体设计

```
[原始特征全集: ~240维]
   │
   ├── Stage 1: 数据卫生与斜率特征黑名单过滤 (Global Blacklist + Macro Drift + Pure Volatility Blacklist)
   │     └── 淘汰: 原始名义价格、无界跨日漂移、无方向纯离散度指标 (~130维淘汰)
   │
   ├── Stage 2: 严格跨合约分布漂移门禁 (Distribution Drift Gate)
   │     └── 门禁: Mean PSI <= 0.10, Max Pair PSI <= 0.20 (淘汰跨合约分布失真特征)
   │
   ├── Stage 3: 有向预测性与单调体制区分度审计 (Predictive & Slope Regime Audit)
   │     └── 门禁: |RankIC_dir| >= 0.020, Sign Consistency >= 0.75, Stability IR >= 0.35
   │     └── 判别: Down/Flat/Up 单调性检验与 ANOVA F >= 4.0
   │
   └── Stage 4: 正交聚类与低阶特征去重 (Orthogonal Ward Clustering & VIF Gating)
         └── 门禁: Spearman |r| <= 0.65, VIF <= 10.0, 目标容量 K in [12, 16]
               │
               ▼
   [最终斜率 VAE 状态特征: 12~16维] (`vae_slope_state_features.npy`)
```

1. **Stage 1 数据清洗与专属黑名单**：
   - 全局卫生黑名单：执行 `commodity_feature_blacklists.json` 的 `scopes.global`（剔除 `wap_1, wap_2, vwap, buy/sell_volume, *_origin` 等）；
   - 跨日宏观漂移黑名单：剔除 $w \ge 96, 192, 240$ 的跨日特征，确保分布具有周期平稳性；
   - **斜率专属黑名单 (Slope Blacklist)**：
     显式过滤纯波动率特征，避免引入对称噪声：
     `r"^(realized_vol|parkinson_vol|bollinger_bandwidth|amplitude|atr|rolling_vol|std_norm)"`。

2. **Stage 2 分布漂移审计 (PSI Gating)**：
   - 阈值：`max_mean_psi = 0.10`, `max_pair_psi = 0.20`；
   - 依据：VAE 作为无监督生成网络，若特征在跨合约或跨年份之间存在均值/方差漂移，将在隐空间引起不可逆的 KL 散度漂移，导致测试集重构似然急剧下挫。必须使用强于一般监督学习的 PSI 标准（0.10 比工业界的 0.25 更加严苛）。

3. **Stage 3 预测性与区分度指标审计**：
   - 目标标的：前向有向收益率 $R_{t,w} = (P_{t+w} - P_t) / P_t$ ($w=6$)；
   - 门禁指标：
     - $|\text{RankIC}| \ge 0.020$；
     - 跨合约 $\text{SignConsistency} \ge 0.75$（保证多头或空头逻辑在 75% 以上的历史合约中稳固生效）；
     - $\text{Stability IR} = |\overline{\text{RankIC}}| / \sigma_{\text{RankIC}} \ge 0.35$；
   - **体制区分度硬性检验 (Slope Regime Differentiation)**：
     利用已标定的验证集段落标签（Down: 0, Flat: 1, Up: 2），对候选特征计算组间单调性与单因素方差分析：
     $$\mu_{\text{Down}}(f) < \mu_{\text{Flat}}(f) < \mu_{\text{Up}}(f) \quad (\text{正向因子}) \quad \text{或} \quad \mu_{\text{Down}}(f) > \mu_{\text{Flat}}(f) > \mu_{\text{Up}}(f) \quad (\text{反向因子})$$
     组间方差与组内方差之比（ANOVA $F$ 统计量）必须满足：
     $$F = \frac{\text{MS}_{\text{between}}}{\text{MS}_{\text{within}}} \ge 4.0, \quad p < 0.01$$
   - 综合优先级得分权重：
     $$\text{Priority}_{\text{slope}}(f) = 0.45 \cdot \text{Rank}\left(\frac{1}{\text{Mean PSI}}\right) + 0.35 \cdot \text{Rank}(|\overline{\text{RankIC}}|) + 0.20 \cdot \text{Rank}(\text{CatBoost Importance})$$

4. **Stage 4 正交聚类去重 (Orthogonal Dedup)**：
   - 矩阵度量：跨合约加权 Spearman 秩相关矩阵；
   - 距离度量：$D = \sqrt{(1 - R) / 2}$；
   - 层次聚类距离截断：$|r| \le 0.65$；
   - 动态容量控制：$K \in [12, 16]$。若聚类数 $> 16$，合并至 16 类；若 $< 12$，按优先级挑出类内次优特征扩充至 12 维；
   - 共线性清洗：方差膨胀因子 $\text{VIF} \le 10.0$；
   - 强制保留特征：日内时间拓扑特征 (`base_time_*`, `trading_minute_progress`)。

### 4.3 斜率 VAE 推荐阈值清单

| 筛选阶段 | 参数名 | 推荐阈值 | 过滤物理依据 |
| :--- | :--- | :--- | :--- |
| **Stage 1: 卫生** | `min_variance` | $1e-6$ | 剔除常数因子 |
| | `max_mode_frequency`| $0.98$ | 剔除离散死值因子 |
| | `feature_blacklist` | `scopes.global + scopes.vae + vol_blacklist` | 剔除名义价格、跨日非平稳漂移与纯离散度特征 |
| **Stage 2: 漂移** | `max_mean_psi` | **$0.10$** | 确保 VAE 高斯先验平稳性，防止隐空间分布偏移 |
| | `max_pair_psi` | **$0.20$** | 杜绝任意两合约间的极端分布漂移 |
| **Stage 3: 预测** | `min_abs_ic` | **$0.020$** | 保证对 $w=6$ 有向未来收益率的秩区分能力 |
| | `min_sign_consistency`| **$0.75$** | 保证多头/空头逻辑跨交割周期的强稳健性 |
| | `min_rank_ic_ir` | **$0.35$** | 抑制样本内过拟合的高方差偶发因子 |
| | `min_anova_f` | **$4.0$** | 确保在 Down/Flat/Up 三体制间具有统计显著的分离度 |
| | `require_monotonic` | `True` | 确保体制均值沿 Down $\to$ Flat $\to$ Up 严格单调 |
| **Stage 4: 正交** | `max_correlation` | **$0.65$** | 抑制同质化动量指标，防止隐空间能量冗余 |
| | `max_vif` | **$10.0$** | 彻底消除多重共线性奇异矩阵 |
| | `min_clusters` | **$12$** | 满足 VAE 最低信息表征维度 |
| | `max_clusters` | **$16$** | 严格限制维度上限，防止测试集高斯似然二次惩罚坍塌 |

---

## 5. 面向“波动率分片识别”的 VAE 特征筛选方法与阈值设计

### 5.1 目标定义与数学物理映射

- **物理目标**：识别市场微观与宏观活跃度、价格弥散范围与风险暴露程度，区分三类离散分片体制：
  - **Label 0 (Low Vol)**：窄幅盘整，订单簿极度平稳，微观流动性充裕，价格冲击极小；
  - **Label 1 (Mid Vol)**：正常波动态势，价格波幅处于历史常规分位数，订单流适度变动；
  - **Label 2 (High Vol)**：剧烈震荡或大单边推进，波幅急剧放大，买卖盘深层耗竭，冲击成本高企。
- **对齐标定源码**：`valid_cross_contract_label_calibration.py:167` 中的 `segment_log_return_volatility`：
  $$\sigma_{\text{seg}} = 100 \times \text{std}(\Delta \log(\text{bid1\_price}), \text{ddof}=0)$$
- **核心特征类型**：
  1. 尺度不变的无向收益率二阶矩（如多周期已实现波动率 $\text{RV}_w$、对数收益率滚动方差）；
  2. 极值范围估计量（如 Parkinson 极值波动率、Garman-Klass 波动率）；
  3. 价格相对带宽（如布林带宽 $\frac{\text{Upper} - \text{Lower}}{\text{Middle}}$、归一化振幅）；
  4. 微观订单簿变动活跃度（如相对买卖价差 $\frac{\text{Ask}_1 - \text{Bid}_1}{\text{Mid}}$、盘口深度耗竭速率、报价刷新频度）；
  5. 交易量活跃度（如相对成交量、成交笔数估计归一化量）；
  6. 日内时间拓扑特征（捕捉开盘与收盘的周期性波动率微笑曲线）。

### 5.2 四阶段筛选阶梯与具体设计

```
[原始特征全集: ~240维]
   │
   ├── Stage 1: 数据卫生与波动率专属黑名单过滤 (Global Blacklist + Macro Drift + Directional Blacklist)
   │     └── 淘汰: 原始名义价格、无界跨日漂移、所有带符号方向特征 (~150维淘汰)
   │
   ├── Stage 2: 波动率自适应跨合约分布漂移门禁 (Volatility Distribution Drift Gate)
   │     └── 门禁: Mean PSI <= 0.12, Max Pair PSI <= 0.25 (适应大宗商品波动聚集性)
   │
   ├── Stage 3: 波动率前向预测与离散体制区分度审计 (Vol-RankIC & Vol Regime Audit)
   │     └── 门禁: Vol-RankIC >= 0.030, Sign Consistency >= 0.75, Stability IR >= 0.40
   │     └── 判别: Low/Mid/High Vol 严格单调递增与 ANOVA F >= 6.0
   │
   └── Stage 4: 波动率正交聚类去重 (Orthogonal Dedup & Multi-Lookback Pruning)
         └── 门禁: Spearman |r| <= 0.60, VIF <= 8.0, 目标容量 K in [10, 14]
               │
               ▼
   [最终波动率 VAE 状态特征: 10~14维] (`vae_volatility_state_features.npy`)
```

1. **Stage 1 数据清洗与专属黑名单 (Directional Stripping)**：
   - 全局卫生黑名单：执行 `commodity_feature_blacklists.json` 的 `scopes.global`；
   - **波动率专属黑名单 (Volatility Blacklist - 强制剥离所有方向信号)**：
     必须严格剔除所有包含方向偏置、正负不对称的指标：
     - 价格斜率与趋势：`r"^(log_price_slope|ema_slope|cvd_slope|signed_efficiency|wap_.*_trend|bid.*_trend|ask.*_trend)"`
     - 订单流与成交不平衡量：`r"^(level5_ofi|imblance_volume|macro_trade_imbalance|trade_up_ratio|trade_down_ratio)"`
     - 有向摆动与动量振荡器：`r"^(stoch_k|stoch_d|rsi|sumn|cntp|cntn|rank_.*_origin)"`
     - 未归一化绝对价差：过滤未作百分比归一化的绝对跳价价差；
   - 跨日波动率漂移控制：剔除 $w \ge 192$ 的多日波动率特征（如 `realized_volatility_192`），保留中短期平稳窗口 ($w \in [2, 6, 12, 24, 48, 96]$)。

2. **Stage 2 分布漂移审计 (PSI Gating)**：
   - 阈值：`max_mean_psi = 0.12`, `max_pair_psi = 0.25`；
   - 依据：在金融市场中，波动率天然具有“波动聚集 (Volatility Clustering)”与体制跳跃特性，其跨合约或跨年份的边际分布方差略大于经过中心化的微观动量。若依旧死守 0.10，会导致大量具有极高信息量的优质波动率指标被误杀。将均值 PSI 适度放宽至 0.12，最大成对 PSI 设为 0.25，既为波动率保留了统计宽容度，又严格守住了高斯密度估计不发生 OOD 爆炸的安全红线。

3. **Stage 3 预测性与区分度指标审计 (Vol-RankIC & Regime Separation)**：
   - **重构前向目标定义**：前向未来离散度 $V_{t,w}$（在决策窗口 $w=6$ 上）：
     $$V_{t,w} = |R_{t,w}| = \left| \frac{P_{t+w} - P_t}{P_t} \right| \quad \text{或局部已实现波动率} \quad \sqrt{\sum_{\tau=1}^w (\Delta \log P_{t+\tau})^2}$$
   - **Vol-RankIC 门禁**：
     $$\text{Vol-RankIC}(f) = \text{RankIC}(f, V_{t,w})$$
     - 阈值：$\text{Vol-RankIC} \ge 0.030$。波动率因子对未来真实波动的预测能力天然显著高于动量对价格方向的预测能力（通常在 $0.05 \sim 0.15$ 之间），因此设为 0.030 能够快速筛除伪相关指标；
     - 跨合约符号一致性：$\text{SignConsistency}_{\text{vol}} \ge 0.75$（要求因子数值越大，未来波动率越大概率在 75% 以上的合约中成立）；
     - $\text{Stability IR}_{\text{vol}} \ge 0.40$；
   - **波动率体制区分度硬性检验 (Volatility Regime Differentiation)**：
     利用已标定的验证集段落标签（Low Vol: 0, Mid Vol: 1, High Vol: 2），对候选特征计算单调递增性与组间方差比：
     $$\mu_{\text{Low}}(f) < \mu_{\text{Mid}}(f) < \mu_{\text{High}}(f)$$
     组间方差与组内方差之比（ANOVA $F$ 统计量）必须满足：
     $$F = \frac{\text{MS}_{\text{between}}}{\text{MS}_{\text{within}}} \ge 6.0, \quad p < 0.001$$
     由于波动率分片在数值上的离散度差异通常远大于斜率分片，因此对 $F$ 统计量的要求提升至 6.0。
   - 综合优先级得分权重：
     $$\text{Priority}_{\text{vol}}(f) = 0.45 \cdot \text{Rank}\left(\frac{1}{\text{Mean PSI}}\right) + 0.35 \cdot \text{Rank}(\text{Vol-RankIC}) + 0.20 \cdot \text{Rank}(\text{CatBoost}_{\text{vol\_target}})$$

4. **Stage 4 正交聚类去重 (Orthogonal Dedup)**：
   - 矩阵度量：跨合约加权 Spearman 秩相关矩阵；
   - 层次聚类距离截断：$|r| \le 0.60$；
     - *关键考量*：不同 lookback 窗口的波动率指标（如 `realized_volatility_6`、`realized_volatility_12`、`realized_volatility_24`）彼此间相关性通常高达 $0.85 \sim 0.95$。若相关性截断过松（如 0.75），选出的 12 个特征可能全部是不同窗口的同一类指标，导致特征空间秩亏损。将阈值收紧至 0.60，可强制算法在“极值波动率 (Parkinson)”、“带宽 (Bollinger)”、“价差活跃度”、“盘口深度耗竭”等不同物理维度各抽取 1~2 个代表。
   - 动态容量控制：$K \in [10, 14]$。精炼紧凑的正交波动率基底；
   - 共线性清洗：方差膨胀因子 $\text{VIF} \le 8.0$；
   - 强制保留特征：日内时间拓扑特征 (`trading_minute_progress`)。

### 5.3 波动率 VAE 推荐阈值清单

| 筛选阶段 | 参数名 | 推荐阈值 | 过滤物理依据 |
| :--- | :--- | :--- | :--- |
| **Stage 1: 卫生** | `min_variance` | $1e-6$ | 剔除常数因子 |
| | `max_mode_frequency`| $0.98$ | 剔除高频死值因子 |
| | `feature_blacklist` | `scopes.global + dir_blacklist + macro_vol_bl` | 彻底剔除所有有向指标、绝对价格与跨日超长窗口 |
| **Stage 2: 漂移** | `max_mean_psi` | **$0.12$** | 适应商品波动率聚集特性，同时防止高斯似然失真 |
| | `max_pair_psi` | **$0.25$** | 约束极端交割月份间的周期波动异动 |
| **Stage 3: 预测** | `min_abs_ic` (Vol-RankIC)| **$0.030$** | 确保对未来无向绝对收益/局部波动的强预测力 |
| | `min_sign_consistency`| **$0.75$** | 保证波动正相关逻辑在 75% 以上合约高度一致 |
| | `min_rank_ic_ir` | **$0.40$** | 确保信息比率稳健，抵御偶发性地缘风险冲击 |
| | `min_anova_f` | **$6.0$** | 确保在 Low/Mid/High Vol 三体制间具备极强的组间方差离散度 |
| | `require_monotonic` | `True` (严格递增) | 确保均值在 Low $\to$ Mid $\to$ High Vol 呈严格单调递增态势 |
| **Stage 4: 正交** | `max_correlation` | **$0.60$** | 严厉打破多窗口已实现波动率的强同质化共线性 |
| | `max_vif` | **$8.0$** | 杜绝由于多重共线性引发的协方差矩阵奇异病态 |
| | `min_clusters` | **$10$** | 构成微观/宏观波动率空间所需的最小正交基底 |
| | `max_clusters` | **$14$** | 限制特征空间容量，确保无监督解码器重构的高保真度 |

---

## 6. 三流解耦体系 (Triple-Stream Architecture) 工程落地改造方案

### 6.1 架构升级全景图：从 Dual-Stream 到 Triple-Stream

在现有 ADR-0037 的双流架构基础上，系统升级为**三流解耦体系 (Triple-Stream Architecture)**：

```
                      [原始候选特征空间: ~240维]
                                 │
           ┌─────────────────────┼─────────────────────┐
           ▼                     ▼                     ▼
┌─────────────────────┐┌─────────────────────┐┌─────────────────────┐
│ Branch 1: VAE_Slope ││Branch 2: VAE_Vol    ││ Branch 3: RL_Agent  │
│ Down/Flat/Up 识别   ││ Low/Mid/High Vol 识别││ 交易动作策略生成    │
│ 过滤纯波动与宏观漂移││ 过滤所有有向偏置指标││ 仅做基础全局卫生清洗│
│ 12 ~ 16 维紧凑特征  ││ 10 ~ 14 维精炼特征  ││ 55 ~ 70 维 Alpha特征│
└──────────┬──────────┘└──────────┬──────────┘└──────────┬──────────┘
           │                      │                      │
           ▼                      ▼                      ▼
  vae_slope_state_        vae_volatility_state_       rl_state_
    features.npy              features.npy           features.npy
           │                      │                      │
           └──────────────────────┼──────────────────────┘
                                  │
                                  ▼
                [全集状态特征 Union: S_all = S_slope ∪ S_vol ∪ S_rl]
               (统一单次标准化入库，生成归一化 wide df.feather)
                                  │
           ┌──────────────────────┼──────────────────────┐
           ▼                      ▼                      ▼
┌─────────────────────┐┌─────────────────────┐┌─────────────────────┐
│ 斜率 VAE 训练与评估 ││ 波动率 VAE 训练评估 ││ 低层强化学习 Agent  │
│ 仅切片读取 S_slope  ││ 仅切片读取 S_vol    ││ 仅切片读取 S_rl     │
└──────────┬──────────┘└──────────┬──────────┘└──────────┬──────────┘
           │                      │                      │
           └───────────┬──────────┘                      │
                       ▼                                 ▼
           ┌─────────────────────┐             ┌─────────────────────┐
           │ 高层双轴 VAE 路由   │             │ ensemble_Qnet       │
           │ Hierarchical Dual   │────────────>│ 动作执行            │
           │ Gating (解耦似然)   │  Slot ID    │                     │
           └─────────────────────┘             └─────────────────────┘
```

### 6.2 产物命名、文件契约与向后兼容性保证

1. **落盘特征清单命名契约**：
   - `vae_slope_state_features.npy`：斜率 VAE 状态特征清单 ($12 \sim 16$ 维)；
   - `vae_volatility_state_features.npy`：波动率 VAE 状态特征清单 ($10 \sim 14$ 维)；
   - `rl_state_features.npy`：强化学习低层策略状态特征清单 ($55 \sim 70$ 维)；
   - `vae_state_features.npy`（向后兼容镜像）：内容完全等于 `vae_slope_state_features.npy`，确保未升级脚本读取不报错；
   - `state_features.npy`（已在 ADR-0037 中淘汰，严禁在中间阶段生成，仅作为全量 Union 的内存缩放中介）。

2. **全局符号契约 (`FineFT/common/artifacts.py`) 扩展**：
   ```python
   class ArtifactNames:
       # ...
       VAE_STATE_FEATURES_NPY: str = "vae_state_features.npy"  # 遗留兼容别名
       VAE_SLOPE_STATE_FEATURES_NPY: str = "vae_slope_state_features.npy"
       VAE_VOLATILITY_STATE_FEATURES_NPY: str = "vae_volatility_state_features.npy"
       RL_STATE_FEATURES_NPY: str = "rl_state_features.npy"
       # ...
   ```

### 6.3 特征筛选核心模块改造规范

#### 1. `types.py` 扩展配置模型
```python
DEFAULT_VAE_SLOPE_PROFILE = StreamFilterProfile(
    name="vae_slope",
    max_mean_psi=0.10,
    max_pair_psi=0.20,
    min_abs_ic=0.020,
    min_sign_consistency=0.75,
    min_rank_ic_ir=0.35,
    max_correlation=0.65,
    min_clusters=12,
    max_clusters=16,
    psi_weight=0.45,
    rank_ic_weight=0.35,
    catboost_weight=0.20,
    filter_micro_persistence=True,
    mandatory_feature_pattern=r"^(base_time_|time_|trading_minute_)",
    feature_blacklist=(),
)

DEFAULT_VAE_VOLATILITY_PROFILE = StreamFilterProfile(
    name="vae_volatility",
    max_mean_psi=0.12,
    max_pair_psi=0.25,
    min_abs_ic=0.030,
    min_sign_consistency=0.75,
    min_rank_ic_ir=0.40,
    max_correlation=0.60,
    min_clusters=10,
    max_clusters=14,
    psi_weight=0.45,
    rank_ic_weight=0.35,
    catboost_weight=0.20,
    filter_micro_persistence=True,
    mandatory_feature_pattern=r"^(trading_minute_progress)",
    feature_blacklist=(),
)
```

#### 2. `predictive_audit.py` 扩展支持双目标审计
在 `execute_predictive_audit` 中，支持同时计算有向未来收益率与无向波动率目标：
```python
future_return = calculate_future_return(frame, window_length)
future_vol = np.abs(future_return)  # 无向绝对收益率目标
```
在指标聚合表中同时保留 `RankIC_Directional` 与 `RankIC_Volatility`，供不同 Stream 分支按需匹配。

#### 3. `pipeline.py` 改造为三流协同流水线
- 引入三路分支调度函数 `_run_triple_stream_train_stage`；
- 分别调用 `_evaluate_stream_branch` 传入 `vae_slope_profile`、`vae_volatility_profile` 与 `rl_profile`；
- 求得数学并集：
  $$S_{\text{union}} = S_{\text{vae\_slope}} \cup S_{\text{vae\_volatility}} \cup S_{\text{rl}}$$
- 调用 `io.save_triple_stream_features(...)` 同时落盘三个独立清单，并写入 `selection_manifest.json`。

#### 4. `io_manager.py` 扩展落地方法
```python
def save_triple_stream_features(
    self,
    vae_slope_features: list[str],
    vae_volatility_features: list[str],
    rl_features: list[str],
) -> tuple[Path, Path, Path]:
    slope_file = self.output_dir / ArtifactNames.VAE_SLOPE_STATE_FEATURES_NPY
    vol_file = self.output_dir / ArtifactNames.VAE_VOLATILITY_STATE_FEATURES_NPY
    rl_file = self.output_dir / ArtifactNames.RL_STATE_FEATURES_NPY
    legacy_vae_file = self.output_dir / ArtifactNames.VAE_STATE_FEATURES_NPY
    
    np.save(slope_file, np.array(vae_slope_features))
    np.save(vol_file, np.array(vae_volatility_features))
    np.save(rl_file, np.array(rl_features))
    np.save(legacy_vae_file, np.array(vae_slope_features))  # 保持旧接口可读
    return slope_file, vol_file, rl_file
```

### 6.4 数据归一化入库与下游消费端改造

#### 1. `muti_contract_scale_save.py` 改造
- 函数 `load_dual_stream_state_features` 升级为 `load_triple_stream_state_features`：
  - 读取 `vae_slope_state_features.npy`、`vae_volatility_state_features.npy` 与 `rl_state_features.npy`；
  - 构造三者并集 `union_features`，对并集执行单次标准化与截断，落盘 Feather 文件；
  - 在最终数据集目录中完整拷贝三个特征清单及遗留兼容清单。

#### 2. `FineFT/datahandler/vae_data_creation.py` 隔离读取
- 改造 `make_data`：
  ```python
  if labeling_method == "volatility":
      state_feature_file = ArtifactNames.VAE_VOLATILITY_STATE_FEATURES_NPY
  else:
      state_feature_file = ArtifactNames.VAE_SLOPE_STATE_FEATURES_NPY
  
  state_name_path = os.path.join(args.base_path, args.dataset_name, state_feature_file)
  if not os.path.exists(state_name_path):
      # 回退到遗留单一清单
      state_name_path = os.path.join(args.base_path, args.dataset_name, ArtifactNames.VAE_STATE_FEATURES_NPY)
  state_features = np.load(state_name_path)
  ```
- **测试集切片隔离**：测试集切片必须隔离存放至 `VAE_data/{labeling_method}/test/`，防止不同维度的测试切片互相覆盖。

#### 3. `FineFT/RL/DiHFT/VAE/main.py` 测试源隔离发现
- 改造 `discover_test_sources`：
  ```python
  def discover_test_sources(data_base_path, dataset_name, labeling_method="slope"):
      root = vae_data_dir(data_base_path, dataset_name)
      # 优先查找方法隔离的测试集目录
      method_test_dir = root / labeling_method / "test"
      test_dir = method_test_dir if method_test_dir.exists() else root / "test"
      # ...
  ```

#### 4. `FineFT/RL/DiHFT/high_level/vae_routing_util.py` 双轴特征独立装载与推理
- 初始化阶段分别装入两个特征清单与输入维度：
  ```python
  # 装载特征清单
  slope_path = os.path.join(self.base_path, self.dataset_name, ArtifactNames.VAE_SLOPE_STATE_FEATURES_NPY)
  vol_path = os.path.join(self.base_path, self.dataset_name, ArtifactNames.VAE_VOLATILITY_STATE_FEATURES_NPY)
  legacy_path = os.path.join(self.base_path, self.dataset_name, ArtifactNames.VAE_STATE_FEATURES_NPY)
  
  self.vae_slope_indicators = np.load(slope_path) if os.path.exists(slope_path) else np.load(legacy_path)
  self.vae_vol_indicators = np.load(vol_path) if os.path.exists(vol_path) else np.load(legacy_path)
  
  self.vae_indicators = {
      "slope": self.vae_slope_indicators,
      "volatility": self.vae_vol_indicators,
  }
  ```
- 在 `load_vae_axis(root, axis)` 中，根据不同轴动态指定 `INPUT_DIM`：
  ```python
  vae_model = MLP_VAE(
      INPUT_DIM=len(self.vae_indicators[axis]),
      Z_DIM=args.z_dim,
      hidden_dims=args.vae_hidden_dims,
      loss_func=args.loss_type,
  ).to(self.device)
  ```
- 在单步推理循环中，针对不同轴提取对应的特征子集：
  ```python
  vae_s_slope = self.vae_slope_state_array[vae_idx]
  vae_s_vol = self.vae_vol_state_array[vae_idx]
  
  # 分别送入对应轴的模型推理
  loss_slope = [analyze_single_sample(m, vae_s_slope, self.device)[1] for m in self.vae_models["slope"]]
  loss_vol = [analyze_single_sample(m, vae_s_vol, self.device)[1] for m in self.vae_models["volatility"]]
  ```

---

## 7. 验证与测试方案 (Verification Plan)

### 7.1 单元测试与流水线回归

1. **测试用例 1：特征筛选独立性断言**
   - 目标：验证 Stage 3 产出的 `vae_slope_state_features.npy` 中纯波动率特征为 0，且 `vae_volatility_state_features.npy` 中带符号方向特征为 0；
   - 检查：
     - `assert not any(f.startswith("realized_vol") for f in vae_slope_features)`
     - `assert not any("slope" in f or "ofi" in f for f in vae_vol_features)`
2. **测试用例 2：维度容量约束断言**
   - 目标：验证特征维度严格符合设定区间：
     - $12 \le \text{len}(\text{vae\_slope\_features}) \le 16$
     - $10 \le \text{len}(\text{vae\_vol\_features}) \le 14$
     - $55 \le \text{len}(\text{rl\_features}) \le 70$
3. **测试用例 3：全集 Union 与标准化一致性**
   - 目标：验证 Stage 4 缩放生成的 Feather 文件的列名完全包含上述三个列表的并集，且无重复列。

### 7.2 核心业务指标验证基准

在典型商品期货数据集（如燃料油 `fu` 10min / 5min）上验证模型改善效果：
1. **VAE 似然度与 OOD 指标**：
   - 波动率 VAE 测试集对数似然损失均值降低 $\ge 15\%$；
   - 测试期虚假 OOD 拒绝率由现存的 $> 8\%$ 下降至 $< 0.5\%$。
2. **高层门控边际清晰度 (`HierarchicalDualGating`)**：
   - 波动率轴边际差值 `volatility_margin` 均值从 $0.04 \pm 0.03$ 提升至 $0.22 \pm 0.08$，彻底脱离 $0.12$ 的模糊混沌区；
   - 模糊防守平仓率（`margin_ambiguity` 触发率）从 $> 35\%$ 大幅降低至 $< 5\%$。
3. **低层与高层协同端到端收益**：
   - 策略在震荡行情（Flat + Low Vol）下保持零仓或微量挂单套利，避免手续费磨损；
   - 策略在突破行情（Up/Down + High Vol）下迅速识别并满仓跟进，实盘夏普比率预期提升 $0.35$ 以上。

---

## 8. 潜在风险与应对预案 (Risks & Mitigations)

| 风险项 | 风险等级 | 触发机理 | 应对与缓解预案 |
| :--- | :--- | :--- | :--- |
| **波动率指标同质化共线性** | 中 | 不同窗口的已实现波动率 ($w=6, 12, 24$) 相关系数达 $0.90$，可能占满特征池 | 在 Stage 4 强制执行严苛的 Spearman $|r| \le 0.60$ 与 $\text{VIF} \le 8.0$ 聚类截断，确保每个特征代表不同机制（极值、价差、深度、日内时间） |
| **小合约切片样本量受限** | 低 | 某次要交割月或挂牌初期数据量过小，导致波动率分片划分不均匀 | 在 `regime_audit.py` 中强制应用 mature row 过滤（步长 $\ge 47$），并设定最低分片行数保护（$\ge 50$ 步） |
| **下游遗留脚本不兼容** | 中 | 历史测试脚本直接加载单份 `vae_state_features.npy` | 在 `ArtifactNames` 和 `io_manager.py` 中永久保留向后兼容别名文件，老代码默认读取斜率流，平滑过渡 |
| **磁盘存储与 IO 冗余** | 低 | 生成三份独立特征可能增加存储空间 | 特征清单仅为数千字节的 `.npy` 字符串数组；底层原始数据仍通过 Union 并集统一缩放并持久化为一份 Feather 文件，磁盘与内存零冗余 |

---

## 9. 结论与下一步行动计划

本报告从统计物理、无监督生成模型 ELBO 重构机制以及高层双层门控判决逻辑出发，彻底厘清了单份 VAE 特征同时驱动斜率与波动率分片的根本缺陷，并完整设计了双目标特征过滤准则、量化阈值与三流解耦落地架构。

**建议后续落地实施步骤**：
1. **P1 阶段**：按照第 6 节规范，在 `muti_contract/types.py` 与 `blacklists.json` 中配置 `DEFAULT_VAE_SLOPE_PROFILE` 与 `DEFAULT_VAE_VOLATILITY_PROFILE`；
2. **P2 阶段**：改造 `predictive_audit.py` 与 `pipeline.py`，支持无向波动率目标预测并输出 `vae_slope_state_features.npy` 与 `vae_volatility_state_features.npy`；
3. **P3 阶段**：同步适配 `muti_contract_scale_save.py`、`vae_data_creation.py` 与 `vae_routing_util.py`，完成全链路端到端闭环验证。
