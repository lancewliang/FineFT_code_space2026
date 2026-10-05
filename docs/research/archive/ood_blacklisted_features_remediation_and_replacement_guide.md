# OOD 黑名单特征的数学成因、改造可用化方法与等价替代特征体系深度调研报告

- **研究主题**：商品期货多体制强化学习 (FineFT) 与 VAE 观测表征中，因分布外漂移 (Out-Of-Distribution, OOD) 被列入黑名单特征的根因剖析、平稳化改造方案及高信息量替代特征体系
- **关联代码与第一手来源 (Primary Sources)**：
  - 特征计算与算子定义源码：
    - `data_preprocess/operator_futures/commodity/cross_month_feature.py` (跨期期限结构价差与角色特征)
    - `data_preprocess/operator_futures/commodity/base_time_feature.py` (日内时间、到期与日历周期特征)
    - `data_preprocess/operator_futures/commodity/daily_mixed_frequency_feature.py` (日级别混频统计特征)
    - `data_preprocess/operator_futures/commodity/weekly_mixed_frequency_feature.py` (周级别混频统计特征)
    - `data_preprocess/operator_futures/cross_section/base_feature_util.py` (订单簿微观深度增量、价差与成交状态)
    - `data_preprocess/operator_futures/time_operator/time_operator_util.py` & `multi_processing_util.py` (滚动窗口极值、波幅与动量算子)
    - `data_preprocess/operator_futures/commodity/schema.py` (状态特征与回报/风控执行特征契约解耦)
  - 筛选、缩放与流水线管理：
    - `data_preprocess/operator_futures/feature_selection/muti_contract/pipeline.py` (特征筛选与黑名单最高优先级熔断)
    - `data_preprocess/operator_futures/scale_describe_save/muti_contract_scale_save.py` (分层分级 Robust 缩放与截断)
    - `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh` (`COMMODITY_FU_FEATURE_BLACKLIST` 全量黑名单定义)
  - 深度学习模型、损失函数与 OOD 诊断源码：
    - `FineFT/RL/DiHFT/VAE/vae.py` (高斯对数似然 $\text{NLL}$ 损失计算与方差下限软截断 `softclip`)
    - `FineFT/analysis/feature/vae_feature_ood_analysis.py` (特征级闭式高斯 NLL 贡献分解分析引擎)
  - 核心架构决策记录 (ADRs)：
    - `docs/adr/0018-vae-feature-level-ood-analysis.md` (特征级高斯 NLL 闭式方差贡献分解机制)
    - `docs/adr/0019-remedy-vae-ood-features-via-blacklist-and-rolling-stationary-replacements.md` (期限结构与跨期价差 OOD 熔断与平稳化替代)
    - `docs/adr/0020-remedy-feature-engineering-underflow-and-spurious-indicators.md` (无符号整型溢出、混频步进跳跃与涨跌停状态特征解耦)
    - `docs/adr/0022-mitigate-vae-feature-ood-drift.md` (生命周期漂移与滚动极值边界锁定 OOD 消除)
    - `docs/adr/0023-root-cause-feature-truncation-and-physical-bounding.md` (未归一化深度与成交量阶数膨胀物理有界截断)
  - 实证分析数据：
    - `analysis_result/DiHFT/feature_ood/fu/10min_parallel/feature_ood_summary.csv` (全量特征实证 $\Delta\text{NLL}$ 诊断表)

---

## 1. 调研执行摘要 (Executive Summary)

在 FineFT 多体制表征与强化学习框架中，截至当前流水线（`fu_full_process.sh`），共有 **70 余个特征** 因触发严重分布外漂移（OOD）被直接列入 `COMMODITY_FU_FEATURE_BLACKLIST` 进行硬性剔除。

**核心结论与直接回答**：
> **这些因 OOD 被拉黑的特征绝大多数完全可以被改进，且蕴含着不可替代的高价值经济学信号（如期限结构基差动量、多级别挂单微观弹性、大周期持仓与成交异动、极端价格边界缓冲等）。**
> 
> 之所以之前直接拉入黑名单，是因为在初始工程实现中，这些特征存在**“绝对价格/成交量尺度泄露”、“长周期离散阶梯跳跃”、“非平稳宏观体制反转”、“强趋势边界锁死”或“稀疏极端事件破坏高斯假设”**等数学缺陷，导致 VAE 的高斯负对数似然（Gaussian NLL）发生灾难性二次惩罚爆炸。
> 
> 通过应用**“动态基准自适应标准化（Rolling Z-Score）”、“相对比例/份额不变性（Scale-Invariant Share）”、“连续距离/弹性平滑（Continuous Distance/Elasticity）”、“有界非线性映射（Bounded Mapping）”**以及**“状态表征与执行屏蔽解耦（State-Shield Decoupling）”**等量化特征工程方法，这些被拉黑的特征不仅可以被“复活”为严格平稳的优质特征，还可以衍生出更具区分度的一组全新替代特征。

---

## 2. 为什么特征在 VAE/RL 中会发生 OOD 崩溃？（数学本质）

在深入各特征分类前，必须首先明确特征在本项目深度学习架构下“被判 OOD 死刑”的底层数学机制。

### 2.1 高斯负对数似然 (Gaussian NLL) 的二次惩罚放大效应
根据 `FineFT/RL/DiHFT/VAE/vae.py:24-29` 与 `FineFT/analysis/feature/vae_feature_ood_analysis.py:44-48`，VAE 解码器计算特征 $x_j$ 的高斯负对数似然损失：
$$\text{NLL}_j = \frac{1}{2} \left( \frac{x_j - \mu_j(z)}{\sigma_j(z)} \right)^2 + \ln \sigma_j(z) + \frac{1}{2}\ln(2\pi)$$

其中，VAE 重构对数方差 $\ln \sigma^2$ 通过 `softclip` 约束在 $[-6.0, 0.0]$（即标准差 $\sigma_j \approx [0.05, 1.0]$）。
- **正常状态**：当特征经过 RobustScaler 变换并在训练分布内时，$|x_j - \mu_j| \le 1.0$，二次惩罚项 $\frac{1}{2} \left( \frac{x_j - \mu_j}{\sigma_j} \right)^2 \approx 0.5 \sim 2.0$。
- **OOD 状态**：一旦测试样本出现均值漂移（如 $\Delta \mu = 0.5$）或极端长尾（如 $x_j = 5.0$），且网络对该特征预测的标准差较小（如 $\sigma_j \approx 0.05$）：
  $$\text{Penalty} = \frac{1}{2} \left( \frac{5.0}{0.05} \right)^2 = \frac{1}{2} \times 100^2 = 5000$$
  **单个特征产生的 NLL 惩罚（高达 5000）将瞬间淹没其余所有 110+ 个正常特征的总梯度与似然**，使网络陷入局部表征崩溃，体制识别（Regime Clustering）完全失效。

### 2.2 金融特征 OOD 的五大根因类型
分析全部被拉黑的 70+ 个特征，其 OOD 成因严格归为以下 5 种结构性矛盾：
1. **宏观结构性体制反转 (Macro Regime Inversion)**：绝对价差水平受宏观供需周期影响，从贴水（Backwardation）反转为升水（Contango），均值发生符号级反转。
2. **绝对物理尺度膨胀 (Scale & Volume Expansion)**：挂单手数、成交笔数随着交易所制度改革、量化做市参与度提升及年份推进而自然膨胀，分布支撑集向右不可逆漂移。
3. **低频混频广播步进阶跃 (Sampling Mismatch & Step Pulses)**：日度/周度特征直接复制给 10 分钟 Bar，在日内呈现长时间水平直线，在换日换周瞬间出现脉冲跳跃，破坏局部连续性与差分平稳性。
4. **强趋势边界锁定 (Boundary Locking in Trends)**：滚动滑动窗口极值索引在单边行情中持续停留在窗口两端（0 或 $W$），将连续平滑分布打成离散双峰。
5. **稀疏离散极端事件对连续似然的破坏 (Extreme Event Sparsity)**：涨跌停等事件在训练集中发生概率为 0，网络学习到的残差方差极小；一旦测试集发生，高斯似然直接炸裂。

---

## 3. 全量黑名单特征分类、改造方法与替代特征详解

我们将 `fu_full_process.sh` 中的所有被拉黑特征划分为 7 大物理集群，并对每一个集群给出：
- **原始数学定义与缺陷剖析**
- **平稳化改造方案（复活原特征）**
- **新增高级替代特征体系**

---

### 集群一：跨期价差与期限结构价格水平 (`cm_*_log_price_ratio`, `cm_*_relative_price_spread`)

#### 1. 包含特征
- `cm_current_main_log_price_ratio`
- `cm_current_main_relative_price_spread`
- `cm_current_sub_log_price_ratio`
- `cm_current_sub_relative_price_spread`
- `cm_main_sub_log_price_ratio`
- `cm_main_sub_relative_price_spread`

#### 2. 原始定义与 OOD 根因 (Primary Source: `cross_month_feature.py:168-175`, ADR-0019)
- **源码公式**：
  $$\text{log\_price\_ratio} = \ln\left( \frac{P_{\text{contractA}}}{P_{\text{contractB}}} \right)$$
  $$\text{relative\_price\_spread} = \frac{P_{\text{contractA}} - P_{\text{contractB}}}{P_{\text{contractA}}}$$
- **根因分析**：
  商品期货（如高硫燃料油 FU）的期限结构是由宏观现货基本面、地缘政治与炼厂开工率决定的。
  在 2024 年验证集中，FU 处于强 Backwardation（近月高于远月），$\ln(P_{\text{current}} / P_{\text{main}})$ 均值为 $+0.035$；
  但在 2025/2026 年测试集由于供给过剩转为 Contango（近月低于远月），均值直接转为 $-0.025$。
  在经过标准差约 $0.01$ 的 RobustScaler 处理后，测试集样本整体漂移了超过 $5\sim 6$ 个标准化单位，在 `feature_ood_summary.csv` 中造成了高达 90% 的初始 NLL 爆炸。

#### 3. 平稳化改造方案 (如何复活)
不能直接使用跨期价差的“绝对水平”，必须使用“**相对于近期期限结构基准的动态偏离度**”：

1. **滚动跨期基差动态 Z-Score (`cm_spread_rolling_zscore_W`)**：
   $$S_t = \ln(P_t^{(A)} / P_t^{(B)})$$
   $$Z_t^{(W)} = \frac{S_t - \text{SMA}_W(S)}{\text{Std}_W(S) + \epsilon}$$
   - **参数建议**：窗口 $W \in [24, 72]$ 根 10 分钟 Bar（相当于 1 至 3 个交易日）。
   - **平稳性证明**：无论宏观处于 Contango 还是 Backwardation，$Z_t^{(W)}$ 均值严格恒为 0，方差严格恒为 1，衡量的是“短线跨期价差是否相对于近期的期限结构基准出现超买/超卖”，具备严格的跨年份分布平稳性。
2. **双曲正切有界基差偏离 (`cm_spread_tanh_norm`)**：
   $$\tilde{S}_t = \tanh\left( \frac{S_t - \text{EMA}_{48}(S)}{2 \cdot \text{ATR}_{48} / P_t} \right) \in (-1, 1)$$
   - 利用 ATR（真实波动率）动态调整基差偏离敏感度，并将值域严格锁定在 $(-1, 1)$。

#### 4. 高级替代特征体系
- **年化滚动展期收益率斜率变化 (Roll Yield Velocity)**：
  $$\Delta \text{RollYield}_t = \Delta \left( \frac{\ln(P_t^{(\text{near})} / P_t^{(\text{far})})}{\Delta T} \right)$$
  消除静态斜率差异，只捕捉期限结构的动态变动速度。
- **跨期蝶式曲率比率 (Butterfly Term Curvature Ratio)**：
  $$\text{Curvature}_t = \frac{2 \ln P_t^{(\text{sub})} - (\ln P_t^{(\text{current})} + \ln P_t^{(\text{far})})}{\text{ATR\_pct}_{24}}$$
  衡量期限结构的凸度变化，该度量在无套利均衡下高度平稳。
- **跨期相对持仓份额与成交份额 (`cm_*_open_interest_share`)**：
  天然有界在 $[0, 1]$ 之间，ADR-0019 已验证其实证漂移为负（更优于基线），是完美的平稳替代品。

---

### 集群二：高滞后混频多日/多周指标 (`prev_*` 与 `prev_*_quantile_rank`)

#### 1. 包含特征
- 23 个原始 `prev_*` 特征：
  `prev_2_day_trade_up_ratio`, `prev_5_day_trade_imbalance`, `prev_2_day_turnover_rate`, `prev_week_twap_deviation_pct`, `prev_15_day_open_interest_change`, `prev_6_week_open_interest_change` 等。
- 30 个 `prev_*_quantile_rank` 特征：
  `prev_5_day_trade_imbalance_quantile_rank`, `prev_15_day_turnover_rate_quantile_rank` 等。

#### 2. 原始定义与 OOD 根因 (Primary Source: `daily_mixed_frequency_feature.py:115-125`, ADR-0019, ADR-0020)
- **源码公式**：
  $$\text{open\_interest\_change} = \frac{\text{OI}_{\text{end}} - \text{OI}_{\text{start}}}{\text{OI}_{\text{start}}}$$
  $$\text{turnover\_rate} = \frac{\text{Volume}}{\text{OI}_{\text{end}}}$$
- **根因分析**：
  1. **分母坍塌导致的巨幅长尾**：新合约刚挂牌或远月非主力合约的 `OI_start` 可能仅有数十手，主力换月流入时，$\Delta \text{OI} / \text{OI}$ 产生 $10000\%$（100倍）的异常爆炸；在换月衰亡期，`OI_end` 趋近于 0，换手率计算除以微小值发散。
  2. **采样频率不匹配与步进阶跃 (Sampling Frequency Mismatch)**：
     日度和周度特征在 10 分钟 Bar 上直接前向填充（Forward Fill），导致日内 192 根 Bar 特征值完全相同，日换算瞬间产生台阶跳跃。
  3. **滚动分位数阶梯死锁**：在 ADR-0020 中，工程上曾尝试使用 192 根 Bar 窗口计算 `prev_*_quantile_rank`。由于 192 根 10 分钟 Bar 仅跨越 3 个交易日，这意味着 192 个样本中实际上只有 3 个不同的离散取值，导致超过 **26% 的样本值完全并列相等**，分位数阶梯化严重，彻底摧毁了连续概率假设。

#### 3. 平稳化改造方案 (如何复活)
1. **分母正则化与对数平滑换手率**：
   $$\text{SafeTurnover}_{t, W} = \ln\left( 1.0 + \frac{\sum_{i=0}^W \text{Volume}_{t-i}}{\text{OI}_t + \text{MedianDailyVol}} \right)$$
   加入品种全局中位数成交量 `MedianDailyVol` 作为分母保护底线，消除非主力期分母趋零异常。
2. **连续日内滚动窗口替代离散混频广播**：
   彻底废弃“跨日静态广播”，改为**真正的日内连续滑动多尺度算子**：
   - 将 1 日、2 日、5 日的时间窗口分别映射为日内 10 分钟 Bar 的连续滑动窗口：$W \in \{24, 48, 120, 240\}$ 根 Bar。
   - 每一根 10 分钟 Bar 均平滑更新滚动成交量占比与持仓变动率：
     $$\text{OI\_change\_pct}_{t, W} = \frac{\text{OI}_t - \text{OI}_{t-W}}{\text{OI}_{t-W} + \epsilon}$$
     再施加物理截断 $[-0.2, 0.2]$，杜绝日间跳阶断层。
3. **指数衰减加权成交失衡 (EMA Order Flow Imbalance)**：
   $$\text{ImbalanceEMA}_{t, \alpha} = \alpha \cdot \left( \frac{\text{Vol}_{\text{buy}} - \text{Vol}_{\text{sell}}}{\text{Vol}_{\text{buy}} + \text{Vol}_{\text{sell}} + \epsilon} \right)_t + (1 - \alpha) \cdot \text{ImbalanceEMA}_{t-1, \alpha}$$
   设置半衰期分别为 4 小时、1 天、3 天，实现连续、无记忆突变的多周期资金流度量。

#### 4. 高级替代特征体系
- **横截面相对分位数排名 (Cross-Sectional Rank)**：
  如果必须要使用多日持仓变化与换手率，**严禁使用时间序列分位数**，应采用**全合约横截面排名**：
  $$\text{CSRank}(x_i) = \frac{\text{Rank}_{t}(x_i) - 1}{N_t - 1} \in [0, 1]$$
  - **性质**：无论全市场总体成交量如何暴涨暴跌，横截面排名的均值在任意时刻严格恒为 $0.5$，方差恒定，具有数学上的绝对平稳性。
- **持仓量/成交量长短期比率 (Volume Term Ratio)**：
  $$\text{VTR}_t = \frac{\text{SMA}_{24}(\text{Volume})}{\text{SMA}_{192}(\text{Volume}) + \epsilon}$$
  通过长短期均线比值度量流动性聚集，消除了合约间与年份间的绝对规模差异。

---

### 集群三：静态日历与合约生命周期编码 (`contract_month_sin/cos`, `contract_life_remaining_ratio`)

#### 1. 包含特征
- `contract_month_sin`
- `contract_month_cos`
- `contract_life_remaining_ratio`

#### 2. 原始定义与 OOD 根因 (Primary Source: `base_time_feature.py:21-25`, ADR-0019, ADR-0022)
- **源码公式**：
  $$\text{contract\_month\_sin} = \sin\left( \frac{2\pi \cdot \text{delivery\_month}}{12} \right)$$
  $$\text{contract\_life\_remaining\_ratio} = \frac{\text{expire\_date} - \text{current\_date}}{\text{total\_life\_days}}$$
- **根因分析**：
  1. **合约内静态常量陷阱**：对于合约 `fu2405`，交割月份为 5，在其存续的 365 天内，`contract_month_sin` 在每一根 10 分钟 Bar 上恒等于 $\sin(10\pi/12) = 0.5$。由于训练集（2309-2412）与测试集（2501-2609）按时间切分，测试集某一特定主力合约月份在训练集中对应样本的分布权重并不对等，直接造成静态域漂移（Domain Shift）。
  2. **确定性线性斜坡漂移**：`contract_life_remaining_ratio` 是从 1.0 单调下降至 0.0 的直线。若测试集集中于某几个特定季度的合约前半程，其特征均值将显著偏离训练集的均值（0.5 左右），在 VAE 似然评估中产生持续的单向偏差。

#### 3. 平稳化改造方案 (如何复活)
1. **相对换月倒计时对数平滑 (Log Time-to-Roll)**：
   交易者真正关注的不是物理到期日，而是“**距离下一次主力换月还有多久**”：
   $$\text{DaysToRollNorm}_t = \tanh\left( \frac{\text{DaysToNextRoll}_t}{15.0} \right) \in [0, 1)$$
   - 当距离换月大于 15 天时饱和于 1.0，临近换月 3 天内平滑衰减至 0，形成循环往复的周期信号。
2. **反向存续期平方根衰减 (Inverted Maturity Discount)**：
   $$\tau_t = \frac{1}{\sqrt{\text{DaysToExpiry}_t + 1.0}}$$
   仅在到期前最后 5 个交易日快速上升，其余时间保持接近 0 的稳定基线，反映到期流动性枯竭冲击。

#### 4. 高级替代特征体系
- **主力/次主力角色层级独热编码 (`contract_role_tier`) (已验证)**：
  ADR-0017 与 ADR-0019 保留的 `prev_day_contract_role_tier`（取值为 `0: other, 1: sub, 2: main`）。无论年份与交割月份如何轮换，每个交易日主力、次主力与非主力的结构比例恒定，分布完全平稳。
- **微观流动性生命周期份额 (Liquidity Maturity Share)**：
  $$\text{LMS}_t = \frac{\text{OI}_{\text{current}, t}}{\max(\text{OI}_{\text{main}, t}, \text{OI}_{\text{sub}, t}) + \epsilon} \in [0, 1]$$
  动态反映当前合约处于“孕育期”、“主力鼎盛期”还是“交割衰退期”，用流动性事实替代死板的日历天数。

---

### 集群四：涨跌停极端事件比率 (`limit_*`)

#### 1. 包含特征
- `limit_up_single_sided_ratio`
- `limit_down_single_sided_ratio`
- `limit_up_ask_depth_ratio_5`
- `limit_down_bid_depth_ratio_5`
- `limit_depth_imbalance_ratio_5`

#### 2. 原始定义与 OOD 根因 (Primary Source: `schema.py:48-60`, ADR-0020)
- **源码公式**：
  $$\text{limit\_up\_single\_sided\_ratio} = \frac{\text{AskDepth}_5 == 0 \text{ \& } P == P_{\text{upper}}}{\text{TotalDepth}}$$
- **根因分析**：
  在正常交易日中，涨跌停属于千分之一级别的罕见黑天鹅事件，在训练集数据中几乎 100% 取值为 0.0。
  VAE 解码器学习到的重构残差方差 $\sigma$ 逼近软下限 $\exp(-3) \approx 0.05$。一旦在测试集或压力测试中发生封板，该特征值跳升至 1.0，在二次项 $\frac{1}{2}((1.0 - 0.0)/0.05)^2 = 200$ 下直接导致状态表征似然崩塌。

#### 3. 平稳化改造方案 (如何复活)
1. **波动率单位的连续距离缓冲 (Continuous Volatility Distance to Limit)**：
   不要使用“是否触及涨跌停”的二元比率，改用“**当前价格距离涨跌停板还有多少个 ATR 的安全缓冲距离**”：
   $$\text{DistToUpperLimit}_t = \frac{P_{\text{upper\_limit}, t} - P_t}{\text{ATR}_{24}(P) + \epsilon}$$
   $$\text{DistToLowerLimit}_t = \frac{P_t - P_{\text{lower\_limit}, t}}{\text{ATR}_{24}(P) + \epsilon}$$
   - **平稳性与信息量**：在日常交易中，该指标平滑波动在 $5.0 \sim 20.0$ 之间；当价格逼近涨跌停时，平稳下降至 $0 \sim 2$。既消除了 0 处的点质量脉冲，又为神经网络提供了极为有价值的“剩余做多/做空空间”连续信息。
2. **S型软屏障逼近度 (Sigmoid Limit Proximity)**：
   $$\text{LimitProximity}_t = \text{Sigmoid}\left( -k \cdot \frac{P_{\text{upper}} - P_t}{\text{ATR}_{24}} \right) \in (0, 1)$$
   仅在距离封板不足 2 个 ATR 时平滑响应，杜绝突变跳跃。

#### 4. 高级替代特征体系
- **架构解耦：状态特征与风控执行屏蔽解耦 (State-Shield Decoupling)**：
  正如 ADR-0020 已经确立的优雅架构：**严禁将二元涨跌停指示器送入连续 VAE 状态网络，但保留在强化学习的环境执行层（Action Masking 与 Risk Shield）**。环境在检测到 `limit_up` 时直接通过规则屏蔽买入 Action 并施加 penalty，无需强求状态表征网络去拟合这一极端奇异点。

---

### 集群五：未归一化的微观挂单深度增量与成交笔数 (`_size_*_increments`, `ntrade_*`)

#### 1. 包含特征
- 10 个盘口增量特征：
  `ask_size_topk_size_1..5_increments`, `bid_size_topk_size_1..5_increments`
- 6 个成交笔数特征：
  `ntrade_estimated`, `ntrade_estimated_up_udnorm`, `ntrade_estimated_down_udnorm`, `ntrade_estimated_flat_udnorm`, `ntrade_estimated_updown_imbalance_udnorm`, `ntrade_estimated_updownflat_vol_udnorm`

#### 2. 原始定义与 OOD 根因 (Primary Source: `base_feature_util.py:474-476`, ADR-0023, 实证 `feature_ood_summary.csv`)
- **源码公式**：
  $$\text{ask\_size\_topk\_size\_k\_increments} = \text{AskSize}_k - \text{AskSize}_1$$
  $$\text{ntrade\_estimated\_up\_udnorm} = \frac{\text{ntrade\_up}}{\text{ntrade\_total} + \epsilon}$$
- **根因分析**：
  1. **绝对手数膨胀**：$AS_k - AS_1$ 是以“手（Lots）”为单位的物理绝对值。近几年商品市场整体流动性增长、算法高频做市商报价厚度增加，2023 年盘口增量中位数为 20 手，2026 年测试集增至 150 手，在 `feature_ood_summary.csv` 中居于 OOD 榜首（贡献率高达 130%）。
  2. **低流动性时段离散化断层**：在夜盘或非主力合约，单个 10 分钟 Bar 的成交总笔数 `ntrade_total` 可能只有 1 笔或 2 笔。此时 `ntrade_estimated_up_udnorm` 只能取 $0.0, 0.5, 1.0$ 这几个离散点；而在主力活跃时段，成交数千笔，其值呈现均值 0.5 的连续钟形分布。这种体制间的形态跳跃引发似然崩塌。

#### 3. 平稳化改造方案 (如何复活)
1. **无量纲多档挂单分布份额 (Orderbook Depth Share)**：
   彻底剔除绝对手数，改用各档挂单在全深度中的相对占比：
   $$\text{AskQueueShare}_k = \frac{\text{AskSize}_k}{\sum_{j=1}^5 \text{AskSize}_j} \in [0, 1]$$
   - **尺度不变性**：无论市场处于几百手还是几万手流动性环境，5 档份额之和恒为 1，单档份额天然有界在 $[0, 1]$，完美抵抗流动性规模膨胀。
2. **贝叶斯先验拉普拉斯平滑成交失衡 (Bayesian Smoothed Imbalance)**：
   针对低成交笔数的离散跳跃，在分子分母引入伪计数先验（Pseudo-counts $C=10$）：
   $$\text{SmoothTradeImbalance} = \frac{\text{ntrade\_up} - \text{ntrade\_down}}{\text{ntrade\_total} + C} \in (-1, 1)$$
   - 当样本仅有 1 笔成交时，失衡度为 $1 / 11 \approx 0.09$，被安全收缩至 0 附近；只有成交充分活跃时才展现强方向性，彻底消除离散脉冲。

#### 4. 高级替代特征体系
- **多档微观订单流失衡 (Multi-Level Order Flow Imbalance, OFI)**：
  $$\text{LevelImbalance}_k = \frac{\text{BidSize}_k - \text{AskSize}_k}{\text{BidSize}_k + \text{AskSize}_k} \in [-1, 1]$$
  各档买卖对比在 $[-1, 1]$ 之间完全自平稳。
- **对数订单簿深度弹性斜率 (Orderbook Elasticity)**：
  $$\text{DepthElasticity} = \frac{\ln\left(\sum_{j=1}^5 \text{AskSize}_j\right) - \ln(\text{AskSize}_1)}{\ln(\text{AskPrice}_5) - \ln(\text{AskPrice}_1)}$$
  反映单位价格深度拓宽的速度，是高频微观结构的标准无量纲特征。
- **相对成交频度活跃度 (Relative Trading Frequency)**：
  $$\text{TradeIntensity}_t = \ln\left( 1.0 + \frac{\text{ntrade}_t}{\text{EMA}_{48}(\text{ntrade}) + \epsilon} \right)$$
  消除了跨年度趋势增长，仅反映短线交易热度的突然放大。

---

### 集群六：长窗口极值索引与边界锁定 (`imin_192`, `imax_192`)

#### 1. 包含特征
- `imin_192_origin`, `imin_192`
- `imax_192_origin`, `imax_192`

#### 2. 原始定义与 OOD 根因 (Primary Source: `multi_processing_util.py:443`, `time_operator_util.py:221`, ADR-0022)
- **源码公式**：
  $$\text{imin}_{192} = \frac{\text{argmin}_{0 \le i < 192}(P_{\text{low}, t-i})}{192}$$
- **根因分析**：
  192 根 10 分钟 Bar 对应约 32 个交易小时（4 个交易日）。
  在震荡行情中，最低点均匀分布在窗口各处，特征值呈现均匀分布（均值 0.5）；
  但在持续单边暴跌的大牛市或大熊市中，每一根新 Bar 都在刷新低点，`argmin` 持续为 191（标准化后为 1.0）；而在暴涨行情中，最低点持续停留在 192 根 Bar 之前的左边界（标准化后为 0.0）。
  在 ADR-0022 中，该特征在测试集中均值漂移高达 0.35，产生 +1.47 的 $\Delta\text{NLL}$，在单特征中贡献率高达 146.81%。

#### 3. 平稳化改造方案 (如何复活)
1. **相对强弱极值位置 (Relative Strength Value, RSV / Stochastic Oscillator)**：
   不要记录极值出现的“时间索引”，而要记录“**当前价格在近期极值区间内的相对高低位置**”：
   $$\text{RSV}_{t, W} = \frac{P_t - \min_{0 \le i < W}(P_{t-i})}{\max_{0 \le i < W}(P_{t-i}) - \min_{0 \le i < W}(P_{t-i}) + \epsilon} \in [0, 1]$$
   - **优势**：即便在强趋势行情中，随着行情的推进和回踩，RSV 也能平滑波动在 $[0, 1]$ 之间，天然有界且不会在边界死锁。
2. **波动率标准化的极值距离 (Volatility-Normalized Extremum Distance)**：
   $$\text{DistToHigh}_{t, W} = \text{clip}\left( \frac{P_t - \max_{0 \le i < W}(P_{t-i})}{\text{ATR}_W + \epsilon}, -4.0, 0.0 \right)$$
   $$\text{DistToLow}_{t, W} = \text{clip}\left( \frac{P_t - \min_{0 \le i < W}(P_{t-i})}{\text{ATR}_W + \epsilon}, 0.0, 4.0 \right)$$
   提供连续的突破距离，有界且完全平稳。

---

### 集群七：离散价差对数差分与除零算子 (`spread_oe_max_log_return_2`, `roc_*_std_norm`, `vstd_*`)

#### 1. 包含特征
- `buy_spread_oe_max_log_return_2`, `sell_spread_oe_max_log_return_2`
- `roc_6..192_std_norm_origin`, `roc_6..192_std_norm`
- `vstd_24_origin` (在调研报告中超界率达 10.20%)

#### 2. 原始定义与 OOD 根因 (Primary Source: `multi_processing_util.py:481`, ADR-0020, ADR-0023)
- **源码公式**：
  $$\text{spread\_log\_return} = 1000 \times \ln\left( \frac{\text{Spread}_t}{\text{Spread}_{t-2}} \right)$$
  $$\text{vstd}_W = \frac{\sigma_{\text{vol}, W}}{\text{Volume} + 10^{-12}}$$
  $$\text{roc\_std\_norm} = \frac{P_t - P_{t-W}}{\sigma_{\text{price}, W} + 10^{-12}}$$
- **根因分析**：
  1. **离散整数状态强加连续对数收益率公式**：盘口最小价差以 Tick（最小变动价位）为离散单位（如 1 tick, 2 ticks）。买一卖一价差从 1 tick 跳到 2 ticks 是离散整数跳跃，用连续资产价格对数收益率公式乘以 1000 会产生 $\ln(2/1) \times 1000 = 693$ 的巨大脉冲尖刺。
  2. **分母过小除零爆炸**：在夜盘休市边缘或午间，成交量仅有 0 手或 1 手，$\text{Volume} \to 0$ 导致 `vstd` 瞬间膨胀至上万；在横盘僵滞期价格完全不波动时，$\sigma_{\text{price}} \to 0$ 导致 `roc_std_norm` 崩溃。

#### 3. 平稳化改造方案 (如何复活)
1. **Tick 单位离散价差对数水平**：
   $$\text{SpreadTickLog}_t = \ln\left( \frac{\text{AskPrice}_1 - \text{BidPrice}_1}{\text{TickSize}} \right)$$
   直接度量价差占用几个 Tick 的对数，取值在 $[0.0, 3.0]$ 之间，完全平稳连续。
2. **算子层分母下限保护与物理截断**：
   参考 `docs/research/feature_long_tail_truncation_hierarchy_and_threshold_analysis.md:200-210`：
   ```python
   # 安全分母并施加物理截断
   safe_volume = pl.when(volume > 1.0).then(volume).otherwise(1.0)
   expr = (pl.col("__volume_std") / safe_volume).clip(0.0, 10.0).alias(f"vstd_{window}")
   ```
3. **正则化动量比率 (Regularized Return-over-Volatility)**：
   $$\text{NormROC}_{t, W} = \frac{(P_t - P_{t-W}) / P_{t-W}}{\max(\sigma_{\text{ret}, W}, 0.001)}$$
   强制设置波动率底线为千分之一，彻底杜绝除零崩溃。

---

## 4. 特征平稳化改造与替代对照全景总表

| 序号 | 原始黑名单特征组 | 核心 OOD 根因 | 改造后的可用新特征公式 | 替代特征推荐 | 预期值域与平稳性保障 |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | `cm_*_log_price_ratio`<br>`cm_*_relative_price_spread` | 升贴水宏观体制反转，均值发生符号级跳变 | **Rolling Basis Z-Score**:<br>$Z_t = \frac{S_t - \text{SMA}_{48}(S)}{\text{Std}_{48}(S) + \epsilon}$ | `cm_*_open_interest_share`<br>展期收益率变化率 | 均值严格恒为 0，方差为 1，杜绝均值漂移 |
| **2** | `prev_*_open_interest_change`<br>`prev_*_turnover_rate` | 远月换月分母趋零爆炸；日间广播台阶跳跃 | **日内连续滑动窗口**:<br>$\text{OI\_pct}_{t, W} = \text{clip}\left(\frac{\Delta \text{OI}_W}{\text{OI} + \epsilon}, -0.2, 0.2\right)$ | 横截面排名 `CSRank`<br>对数平滑换手率 | 连续无台阶，消除了日度广播造成的 192 样本并列阶梯 |
| **3** | `prev_*_quantile_rank` | 192 根 Bar 仅 3 个离散日值，并列率 >26% | 废弃单特征时序分位数，改用 **EMA 动态衰减加权失衡** | 横截面排名 `CSRank` | 严格均匀分布于 $[0, 1]$，彻底消除离散阶梯死锁 |
| **4** | `contract_month_sin/cos`<br>`contract_life_remaining_ratio` | 合约内为静态常量，造成跨时间样本域漂移 | **换月倒计时软饱和**:<br>$\tanh(\text{DaysToNextRoll} / 15.0)$ | `prev_day_contract_role_tier`<br>流动性份额 `LMS` | 随换月周期动态循环，与特定日历月份完全解耦 |
| **5** | `limit_*` (涨跌停指示器) | 罕见离散极端事件，训练集 0 发生率炸飞似然 | **连续 ATR 距离**:<br>$\text{Dist} = \frac{P_{\text{limit}} - P_t}{\text{ATR}_{24} + \epsilon}$ | 环境动作遮罩 (Action Masking)<br>风控屏障解耦 | 连续平滑、全时段非零，且为 RL 提供明确的边界空间度量 |
| **6** | `ask/bid_size_*_increments`<br>`ntrade_estimated` | 市场容量与高频参与度增长导致绝对手数膨胀 | **微观盘口分布份额**:<br>$\text{Share}_k = \frac{\text{Size}_k}{\sum_{j=1}^5 \text{Size}_j}$ | 多档 OFI 失衡度<br>对数深度弹性斜率 | 天然有界于 $[0, 1]$，完全与市场整体绝对交易规模脱钩 |
| **7** | `ntrade_*_udnorm` | 低频交易时段仅 1~2 笔成交，比率离散跳变 | **拉普拉斯平滑失衡度**:<br>$\frac{\text{up} - \text{down}}{\text{total} + 10.0}$ | 相对交易频度 $\ln(1 + N/\bar{N})$ | 消除低成交量时的 0.0/1.0 极端跳变，连续平滑收缩 |
| **8** | `imin_192`<br>`imax_192` | 单边强趋势下 argmin 索引持续锁死在边界 0 或 192 | **相对强弱极值位 (RSV)**:<br>$\frac{P_t - \min_W P}{\max_W P - \min_W P + \epsilon}$ | ATR 极值距离<br>去趋势震荡指标 | 震荡行情与单边行情均平稳映射在 $[0, 1]$，无边界死锁 |
| **9** | `spread_oe_max_log_return_2` | 离散 Tick 阶跃强加连续对数收益率导致脉冲尖刺 | **Tick 规范化价差对数**:<br>$\ln((\text{Ask}_1 - \text{Bid}_1) / \text{TickSize})$ | ATR 归一化盘口价差 | 平滑连续，对价差拓宽具备对数压缩性 |
| **10**| `vstd_*`<br>`roc_*_std_norm` | 成交量趋零或价格僵滞导致分母微小除零发散 | **下限保护与物理截断**:<br>$\text{clip}\left(\frac{\sigma}{\max(V, V_{\text{floor}})}, 0.0, 10.0\right)$ | 正则化 Sharpe 动量 | 根绝除零溢出，物理边界硬封顶 |

---

## 5. 建议的落地实施路径 (Implementation Roadmap)

如果要在后续流水线版本中全面释放这 70+ 个被拉黑特征的潜力，建议按照以下阶段推进：

### 阶段一：算子层修复与平稳化改造 (Low-Hanging Fruits)
1. **微观深度份额与弹性 (`Share` & `Elasticity`)**：
   在 `base_feature_util.py` 中增加 `ask/bid_size_topk_share_1..5`，用无量纲的份额直接替换已被拉黑的 `increments`。
2. **基差动态 Z-Score (`cm_spread_rolling_zscore`)**：
   在 `cross_month_feature.py` 中为 `cm_current_main` 和 `cm_main_sub` 增加 48-bar 滚动均值与方差标准化输出，释放期限结构对短线均值回归的高价值信号。
3. **极值 RSV 指标 (`rsv_24`, `rsv_192`)**：
   在 `time_operator_util.py` 中引入标准的 RSV 算子，平稳替换容易锁死的 `imin`/`imax`。

### 阶段二：混频连续化与架构解耦 (Architecture Refactoring)
1. **废弃低频时序分位数**：
   在 `mixed_frequency_feature.py` 中移除所有通过日内 192-bar 算出来的 `prev_*_quantile_rank`，代之以全市场合约横截面排名 `CSRank` 或连续日内 EMA 失衡指标。
2. **涨跌停距离化**：
   在 `base_feature_util.py` 中为状态空间增加 `dist_to_limit_up_atr` 与 `dist_to_limit_down_atr`，让智能体具备连续的价格边界感知，同时保留原有的 `PRICE_LIMIT_RATIO_FEATURE_COLUMNS` 仅用于执行层的 Action Masking。

### 阶段三：自动化 OOD 检验验证 (Verification Loop)
运行 `FineFT/analysis/feature/vae_feature_ood_analysis.py` 进行再检验：
- 确保所有新改造特征的 $\Delta\text{NLL}_{\text{test vs valid}} \le 0.05$。
- 确保测试集样本在 $[-5.0, 5.0]$ 之外的长尾超界率低于 $0.05\%$。
