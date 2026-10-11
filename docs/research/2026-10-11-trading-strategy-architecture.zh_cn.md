# FineFT / DiHFT 商品期货交易策略系统架构设计与工程规范文档

- **文档版本**：v1.0 (Production Architecture Specification)
- **归档日期**：2026-10-11
- **研究与架构团队**：FineFT Quantitative Research & Strategy Infrastructure Team
- **文档密级/性质**：核心策略架构与实施工程白皮书
- **对应系统代码库**：`/FineFT` (`FineFT/RL/DiHFT/`, `FineFT/env/`, `data_preprocess/`)
- **关联主要架构决策 (ADRs)**：ADR 0001, ADR 0002, ADR 0005, ADR 0006, ADR 0007, ADR 0012, ADR 0013, ADR 0014, ADR 0031, ADR 0037, ADR 0041, ADR 0045, ADR 0046, ADR 0047, ADR 0048, ADR 0049, ADR 0050
- **理论基石与论文**：*"FineFT: Efficient and Risk-Aware Ensemble Reinforcement Learning for Futures Trading"* (arXiv:2512.23773)

---

## 1. 策略架构总览与核心设计哲学 (Strategy Architecture Overview & Core Philosophy)

### 1.1 业务背景与面临的核心挑战
商品期货市场具备**非平稳性强（Non-stationarity）**、**杠杆率高（High Leverage）**、**交易摩擦剧烈（High Transaction Friction）**以及**微观噪声信噪比极低（Low SNR）**的鲜明特征。在将深度强化学习（DRL）应用于高频与波段商品期货交易（如国内商品交易所的燃料油 `fu` 等品种）时，业界传统方案通常面临三大系统性瓶颈：

1. **单模型泛化崩溃与左尾爆仓（Single Model Collapse & Left-Tail Ruin）**：
   单一强化学习策略在经历长周期单边暴跌或暴涨行情时，由于视野受限与极端非平稳分布外漂移（Out-of-Distribution, OOD），极易陷入“逆势摸顶”或“连续扛单做多”的灾难性状态，单笔交易即可导致账户巨幅失血（如 `fu2411` 历史单笔做多扛单 18 小时亏损 93%）。
2. **高频过度调仓与摩擦吞噬黑盒（Overtrading & Friction Cannibalization）**：
   微观强化学习 Agent 缺乏物理与数学上的持仓惯性，在横盘震荡行情中频繁追涨杀跌。历史实测表明，策略捕获的盘面择时毛利润（Gross PnL）虽可高达 +81.1%，但高达 1,696 次的无谓调仓产生天量资金换手，导致手续费与订单簿滑点吞噬了 **96.87%** 的总毛利，最终净利几近归零。
3. **市场宏观体制与微观执行的语义错配（Regime-Execution Mismatch）**：
   低层策略执行需要高维微观 Alpha 特征，而宏观大盘的趋势与波动率体制判定则需要低维、平稳、对称性分明的统计特征。若将两类特征混合喂入单一网络，会导致特征互相污染与门控振荡失效。

### 1.2 FineFT / DiHFT 三阶段分层强化学习体系概览
为彻底根治上述结构性矛盾，FineFT 构建了一套**三阶段分层风险感知集成强化学习架构（Three-Stage Risk-Aware Ensemble Reinforcement Learning Pipeline）**：

```
[原始高频行情 / LOB 五档快照]
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│ 特征工程与三流解耦管线 (Triple-Stream Feature Decoupling)       │
├──────────────────┬──────────────────────┬───────────────────────┤
│ 趋势斜率流       │ 波动率流             │ 微观 RL 决策流        │
│ vae_slope (14维) │ vae_volatility (12维)│ rl_state (68维)       │
└─────────┬────────┴──────────┬───────────┴───────────┬───────────┘
          │                   │                       │
          ▼                   ▼                       │
┌─────────────────────────────────────────┐           │
│ Stage II: 宏观表征与双轴 VAE 密度估计   │           │
│  - Slope MLP_VAE (趋势体制似然评估)     │           │
│  - Volatility MLP_VAE (波动能级似然评估)│           │
└────────────────────┬────────────────────┘           │
                     │ (似然与重构误差)               │
                     ▼                                │
┌─────────────────────────────────────────┐           │
│ Stage III: 风险感知元路由与执行引擎     │           │
│  - 迟滞带门控 (Hysteresis Gating)       │           │
│  - 动作持久化与防御抢占 (Persistence)   │           │
│  - 三级硬核风控 (止损/冷静期/断路器)    │           │
└────────────────────┬────────────────────┘           │
                     │ (选定专业 Slot 子网络)         │
                     ▼                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage I: 微观执行策略集群 (Low-Level Policy Ensemble)           │
│  - 5 路张量输入 (State, Time, PrevAct, TradingInfo, ActMask)    │
│  - 4x4 专业化子智能体网格 (Ensemble Q-Net)                      │
│  - 动态规划离线专家教师 (DP Teacher Q-Table) 预训练引导        │
└────────────────────┬────────────────────────────────────────────┘
                     │ (目标动作执行)
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 真实物理交易环境与订单簿撮合 (Futures Trading Environment)     │
│  - 五档真实滑点模拟 & 佣金计算                                  │
│  - 非对称方向性调仓惩罚 (Asymmetric Turnover Penalty)           │
│  - 逐 Step 交易诊断与摩擦损耗解耦 (Diagnostic Triad)           │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 核心工程设计准则
- **Fail-Fast 契约优先**：严格遵循 `CLAUDE.md` 规范，杜绝一切隐式兜底、容错捕获与 `getattr`/`hasattr`/`isinstance` 保护，接口参数必须显式强类型化。
- **确定性风控底线高于算法寻优**：止损、冷静期与断路器作为系统基础设施级物理护栏，绝不交由黑盒优化器在特定样本上过拟合。
- **全链路数学自洽**：离线 DP 专家轨迹计算、在线环境 Step 步进以及微观策略网络损失函数必须共享完全相同的奖励塑形与惩罚项，杜绝策略分布偏移。

## 2. 特征工程与三流特征解耦架构 (Feature Engineering & Triple-Stream Decoupling)

### 2.1 统计矩与对称性矛盾 (Statistical Moment & Symmetry Contradiction)
依据 ADR 0037 与 ADR 0041，FineFT 早期将高层 VAE 密度估计特征与低层 RL 策略输入统一混用的做法遭遇了根本性的统计学矛盾：
1. **宏观体制切片物理定义**：
   - **趋势斜率切片 (Slope Slices)**：由归一化巴特沃斯（Butterworth）滤波后的分段价格斜率决定，划分为下跌（Down）、震荡（Flat）、上涨（Up）。
   - **波动能级切片 (Volatility Slices)**：由分段对数收益率的总体标准差决定，划分为低波动（Low Vol）、中波动（Mid Vol）、高波动（High Vol）。
2. **特征对称性与单调性冲突**：
   - **趋势斜率识别**：物理上必须依赖**奇对称特征（Odd-Symmetric Features）**，满足 $f(-r) pprox -f(r)$，且在不同状态间呈现单调递增的均值漂移：
     $$\mu_{\text{Down}} < \mu_{\text{Flat}} < \mu_{\text{Up}}$$
   - **波动率识别**：物理上必须依赖**偶对称特征（Even-Symmetric Features）**，满足 $f(-r) pprox f(r) > 0$，反映价格能量的离散度与尺度不变性：
     $$\mu_{\text{Low}} < \mu_{\text{Mid}} < \mu_{\text{High}}$$
   - 若将二者混合评估，纯波动率指标（如 ATR、布林带宽度、Parkinson 波动率）在预测未来方向性收益率 $R_{t, w=6}$ 时，其相关系数期望为 0、符号一致性仅为 0.5，会被特征筛选硬过滤器 100% 抹杀；反之，若强行保留，高波动下单边大涨大跌的双峰分布将直接打爆高斯负对数似然（Gaussian NLL），导致 VAE 误报假 OOD。

### 2.2 三流特征规格定义与独立过滤漏斗 (ADR 0041)
为根治上述矛盾，系统确立了**无向后兼容的三流特征解耦体系（Triple-Stream Feature Decoupling Architecture）**，在 Stage 3 特征筛选阶段物理切分为三个独立的特征流：

| 特征流名称 | 物理产物文件名 | 目标维度 | 统计目标与预测目标 | 漂移门控 (PSI) 与预测门控 | 黑名单与特征域 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **趋势斜率流** (`DEFAULT_VAE_SLOPE_PROFILE`) | `vae_slope_state_features.npy` | $12 \sim 16$ 维 | 预测未来 6 步方向性收益 $R_{t,w=6}$ | $\text{PSI} \le 0.10$, $\|\text{RankIC}\| \ge 0.020$, 符号一致性 $\ge 0.75$, 方差分析 $F \ge 4.0$ ($p < 0.01$) | 黑名单：通用漂移特征 + 纯波动率特征（`realized_vol_*`, `atr_*`, `bollinger_*` 等） |
| **波动能级流** (`DEFAULT_VAE_VOLATILITY_PROFILE`) | `vae_volatility_state_features.npy` | $10 \sim 14$ 维 | 预测未来 6 步绝对收益/离散度 $V_{t,w=6} = \|R_{t,w=6}\|$ | $\text{PSI} \le 0.12$, $\text{Vol-RankIC} \ge 0.030$, 符号一致性 $\ge 0.75$, 方差分析 $F \ge 6.0$ ($p < 0.001$) | 黑名单：带符号方向性指标（`*_trend`, `macd_*`, `rsi_*` 等）+ 微观订单流失衡 |
| **微观决策流** (`DEFAULT_RL_PROFILE`) | `rl_state_features.npy` | $55 \sim 70$ 维 | 强化学习决策执行，捕捉微观盘口 Alpha | $\text{PSI} \le 0.25$, $\|\text{RankIC}\| \ge 0.020$, 稳定性 $\text{IR} \ge 0.30$, 容忍 Spearman $\|r\| \le 0.80$ | 黑名单：空（全面放行微观订单流不平衡量 OFI、持仓量分位数、深度消耗率） |

### 2.3 物理数据归一化与统一内存宽表切片机制
在数据物化与缩放阶段（`data_preprocess/operator_futures/scale_describe_save/muti_contract_scale_save.py`）：
1. **单一宽表物化**：系统计算三流特征的数学并集：
   $$\mathcal{S}_{\text{union}} = \mathcal{S}_{\text{vae\_slope}} \cup \mathcal{S}_{\text{vae\_vol}} \cup \mathcal{S}_{\text{rl}}$$
2. **Train-Only 自适应 Robust 缩放**：
   对训练集每个特征计算中位数 $Q_{50}$ 与四分位距 $\text{IQR} = Q_{75} - Q_{25}$，通过中心化与缩放将其变换为无量纲得分：
   $$z = \frac{x - Q_{50}}{\text{IQR} + \epsilon}$$
3. **软饱和截断**：引入双曲正切软裁剪 $x_{\text{scaled}} = M \cdot \tanh(z / M)$，物理抑制极端异常点（Outliers）同时保持梯度平滑。
4. **内存切片零拷贝交付**：下游模型仅持久化一份宽表 Feather 文件，在内存加载时由各自的特征索引列表切出独立的张量输入，消除了磁盘多副本冗余。

## 3. Stage I：低层执行策略集群架构 (Low-Level Execution Policy Architecture)

### 3.1 状态与观察空间：五路张量输入体系 (ADR 0002)
低层执行策略采用价值基 Q 学习架构，由单一策略网络 `Qnet` 或集成网络 `ensemble_Qnet` 实现（`FineFT/model/low_level.py`）。为消除部分可观测性（POMDP）带来的“持仓状态盲区”与“倒手逻辑死循环”，网络接收五路结构化物理张量输入：

```
                    ┌─────────────────────────┐
State (N_STATES) ──►│ Linear(fc1) + ReLU      │──┐
                    └─────────────────────────┘  │
                    ┌─────────────────────────┐  │
Time (TIME_DIM=2) ─►│ Linear(fc_time)         │──┼──► Concat ──► Linear(fc2) ──► Linear(out) ──► Q-Values
                    └─────────────────────────┘  │                                                     │
                    ┌─────────────────────────┐  │                                                     ▼
PrevAct (1) ───────►│ Linear(fc3) + ReLU      │──┤                                            Mask Operation
                    └─────────────────────────┘  │                                    Q + (Mask - 1) * 1e12
                    ┌─────────────────────────┐  │                                                     │
TradingInfo (4) ───►│ Linear(fc_trading)+ReLU │──┘                                                     ▼
                    └─────────────────────────┘                                               Masked Q-Values
                                                                                                       ▲
AvaliableAction ───────────────────────────────────────────────────────────────────────────────────────┘
```

1. **`state` [batch, N_STATES]**：
   由 Stage 3 特征漏斗筛选出的微观 Alpha 因子矩阵（通常 55~70 维），包含微观 LOB 不平衡量、短周期动量、VWAP 偏差及量价波动因子。
2. **`time` [batch, TIME_INFO_DIM=2]**：
   交易时段生命周期倒计时（`[funding_count_down_hour, funding_count_down_minute]`），向智能体显式注入日内交易时钟与换日边界压力。
3. **`previous_action` [batch, 1]**：
   上一决策时间步实际生效的离散动作编号，为 Q 函数提供马尔可夫决策转移的动作基准。
4. **`trading_info` [batch, TRADING_INFO_DIM=4]**（ADR 0002 强契约）：
   包含当前持仓微观维度的 4 项核心物理特征：
   - **`position_exposure`**：当前真实仓位相对最大可用仓位的无量纲比例，$\text{pos} / \max(|\text{pos}|) \in [-1.0, 1.0]$；
   - **`single_holding_return_rate`**：当前单次回合（从开仓起算）的浮动收益率；
   - **`single_holding_max_drawdown`**：当前单次回合历经的历史最大浮亏/回撤幅度；
   - **`duration_norm`**：归一化持仓时长，$\min(\text{holding\_duration} / 180, 1.0)$。
   - *物理不变性*：当处于空仓（`position == 0`）时，该张量全量严格归零。
5. **`avaliable_action` [batch, N_ACTIONS]**：
   物理动作合法性二值掩码（1 为合法，0 为非法）。网络在前向推理末端执行物理阻断：
   $$Q_{\text{masked}}(s, a) = Q(s, a) + (\text{avaliable\_action}(a) - 1) \times 10^{12}$$
   彻底剥夺非法动作（如保证金不足开仓、违规连续反手等）被选中的可能性。

### 3.2 动作空间、杠杆与反向开仓语义 (ADR 0001, ADR 0007)
1. **离散仓位档位映射**：
   系统采用固定档位的离散持仓模型，由 `position_choices`（默认 9 档）与 `leverage_choice`（如 5x 杠杆）决定：
   $$\text{PositionList} = \{-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0\}$$
   对应动作编号 $a \in \{0, 1, 2, \dots, 8\}$，其中 $a=4$ 对应空仓 Flat。
2. **反向开仓（Reverse Position）语义**（ADR 0001）：
   - 在高频期货交易中，从多头直接跨零点反手翻空（$\text{old\_pos} > 0 \land \text{new\_pos} < 0$）必须分解为“先平多仓、再开空仓”两个独立物理过程。
   - 系统采用**尽力而为（Best-effort）语义**：平仓必定首先执行成功；若反向开仓因保证金不足或订单簿深度枯竭受阻，仓位回退停留在 Flat（0 仓位），绝不产生非法的中间态。

### 3.3 离线动态规划 (DP) 专家教师示范 Q 表
在低层强化学习探索前，系统通过后验全知视角（Omniscient Hindsight）在历史 K 线数据上运行动态规划（`FineFT/env/env_class/futures_util.py:create_optimal_q_table_from_df`），构建离线全局最优专家 Q 表作为预训练导师：

1. **贝尔曼反向归纳递推方程（Bellman Backward Induction）**：
   设时间步 $t \in [T-1, T-2, \dots, 0]$，当前步持仓对应动作 $a_{\text{curr}}$，下一步转移目标动作 $a_{\text{next}}$：
   $$Q(t, a_{\text{curr}}, a_{\text{next}}) = r(t, a_{\text{curr}}, a_{\text{next}}) + \gamma \max_{a^{\prime}} Q(t+1, a_{\text{next}}, a^{\prime})$$
   边界条件为终局收益 $Q(T, \cdot, \cdot) = 0$。
2. **全链路惩罚镜像（DP Synchronization）**：
   根据 ADR 0049 与 ADR 0050，DP 教师在计算单步转移奖励 $r(t, a_{\text{curr}}, a_{\text{next}})$ 时，**必须同步扣除与在线交易环境完全一致的名义货值换手惩罚项**。若 DP 教师未引入换手损耗，生成的专家轨迹将充斥微幅日内超高频倒手，导致预训练策略学到错误的“过度换手”行为模式。

### 3.4 双阶段训练目标与损失函数体系
低层智能体采用**“离线监督预训练 (Warmup) + 多体制局部加权探索 (Diverse Train)”**的两阶段训练范式（`FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`）：

#### 阶段一：教师引导监督预训练 (Warmup Pretraining)
联合最小化 Bellman TD 误差与模仿教师策略分布的 KL 散度：
$$\mathcal{L}_{\text{pretrain}} = \mathcal{L}_{\text{TD}} + \text{ada} \cdot \mathcal{L}_{\text{KL}}$$
- **Bellman TD 损失**（采用 SmoothL1 / Huber 损失提高对厚尾离群点的鲁棒性）：
  $$\mathcal{L}_{\text{TD}} = \frac{1}{B} \sum_{i=1}^B \sum_{k=1}^N \text{SmoothL1}\Big(Q^{(k)}(s_i, a_i) - \big(r_i + \gamma \max_{a^{\prime}} Q_{\text{target}}^{(k)}(s_i^{\prime}, a^{\prime})\big)\Big)$$
- **监督者 KL 散度损失**：
  $$\mathcal{L}_{\text{KL}} = \frac{1}{B} \sum_{i=1}^B \sum_{k=1}^N \sum_{a} \pi^{(k)}(a|s_i) \left( \log \pi^{(k)}(a|s_i) - \log \pi_{\text{teacher}}(a|s_i) \right)$$
  其中教师概率分布由 DP 最优 Q 表通过 Softmax 导出：$\pi_{\text{teacher}}(a|s) \propto \exp(Q_{\text{teacher}}(s, a) / \tau)$。
- **自适应权重衰减**：监督系数 $\text{ada}$ 从初始值 $\text{ada}_{\text{init}}=256$ 随训练步数线性退火至 $\text{ada}_{\text{min}}$。

#### 阶段二：多样化局部加权训练 (Diverse Ensemble Training)
当网络具备基本交易常识后，进入集成多样化分化阶段（FineFT Paper Algorithm 2）。系统构建集成误差距离矩阵：
$$\text{ETD}_{i, j} = \Big| Q^{(i)}(s, a) - \big( r + \gamma \max Q^{(j)}(s^{\prime}, a^{\prime}) \big) \Big|$$
通过 `construct_paper_weight_matrix` 计算局部邻域加权衰减矩阵 $W$，仅对邻近子网络的误差进行反向传播：
$$\mathcal{L}_{\text{diverse}} = \frac{1}{B} \sum_{b=1}^B \sum_{i, j} W_{i, j} \cdot \text{ETD}_{i, j}$$
该机制打破了传统集成网络趋同退化的缺陷，迫使 $N$ 个集成子网络自发分化，分别专精于不同行情形态。

### 3.5 二维体制空间子模型选优与集成组装算法 (pick_agent, ADR 0005, ADR 0006)
在全量候选模型产出后，系统不采用传统的全集平均（Mean Ensemble），而是运行**二维网格遴选算法**（`FineFT/analysis/pick_agent/FineFT_two_dimensional_agent_selector.py`）：

1. **二维离散体制网格划分**：
   以波动率分箱（Volatility Bin，0~3）与趋势斜率分箱（Slope Bin，0~3）构建 $4 \times 4 = 16$ 个正交体制槽位（Slots）。
2. **严苛风险与统计门控**（`SelectionConfig`）：
   每个槽位在遍历所有候选 `(epoch, bin_index)` 时必须通过多重门控：
   - 最小联合测试合约数覆盖：$\ge \text{min\_joint\_contracts}$；
   - 正收益合约比例：$\text{positive\_ratio} \ge 40\%$；
   - 置信下界（Lower Confidence Bound, LCB）门控：
     $$\text{LCB} = \mu_{\text{return}} - z \cdot \frac{\sigma_{\text{return}}}{\sqrt{N_{\text{contracts}}}} \ge 0.0$$
   - 最坏初始仓位浮亏门控：$\text{min\_worst\_initial\_position\_return} \ge -1.0$（排除开局即被秒杀的脆弱模型）。
3. **模型装配与空槽位防守策略**：
   - 通过门控的最优子网络参数被抽取并拼装进 `model.pth`；
   - 若某特定槽位无任何候选模型能通过风控门控，该槽位被严格标记为 `empty_model`（空模型槽位），在运行时强制映射为 Flat 空仓防守（`intended_runtime_behavior: flat_position`）。

## 4. Stage II：宏观体制表征与 VAE 密度估计架构 (Stage II: Regime & VAE Modeling)

### 4.1 MLP_VAE 模型网络结构与潜在空间编码
高层宏观体制表征由变分自编码器 `MLP_VAE` 实现（`FineFT/RL/DiHFT/VAE/vae.py`），用于对非平稳高频金融特征进行无监督生成式密度建模：

```
                ┌────────────────────────────────────────────────────────┐
                │ Encoder (识别模型 q_phi(z|x))                           │
输入特征 x ────►│ Linear(fc1) -> ReLU -> fc_enc (多层感知机)            │
                │  ├──► Linear(fc21) ──► 均值向量 mu                     │
                │  └──► Linear(fc22) ──► 对数方差 logvar                │
                └──────────────────────────┬─────────────────────────────┘
                                           │ 重参数化技巧 (Reparameterization Trick)
                                           │ z = mu + eps * exp(0.5 * logvar), eps ~ N(0, I)
                                           ▼
                ┌────────────────────────────────────────────────────────┐
                │ Decoder (生成模型 p_theta(x|z))                         │
隐变量 z ──────►│ Linear(fc3) -> ReLU -> fc_dec                          │
                │  ├──► Linear(fc4) ──► 重构均值 recon_mu                │
                │  └──► Linear(fc5) ──► 重构对数方差 recon_logsigma      │
                └────────────────────────────────────────────────────────┘
```

1. **输入与隐空间配置**：
   - `INPUT_DIM`：动态自适应特征维度（趋势斜率流通常为 14 维，波动率流通常为 12 维）；
   - `Z_DIM`：低维流形潜空间维度（通常为 6~10 维）；
   - `hidden_dims`：对称前馈隐藏层结构（默认 `[500, 200]`）。
2. **重参数化技巧（Reparameterization Trick）**：
   为了使网络能够通过反向传播（Backpropagation）端到端优化隐变量采样过程，引入独立高斯噪声 $\epsilon \sim \mathcal{N}(0, I)$：
   $$z = \mu + \epsilon \odot \exp\left(0.5 \cdot \log \sigma^2\right)$$

### 4.2 证据下界 (ELBO) 损失函数与数值防护
VAE 的训练目标是最大化边际对数似然 $\log p(x)$ 的变分证据下界（Evidence Lower Bound, ELBO）：
$$\mathcal{L}_{\text{ELBO}}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}\left[\log p_\theta(x|z)\right] - D_{\text{KL}}\left(q_\phi(z|x) \parallel p(z)\right)$$
最小化目标函数为负 ELBO（$\mathcal{L}_{\text{VAE}} = -\mathcal{L}_{\text{ELBO}} = \text{ReconLoss} + \text{KLD}$）：

1. **高斯负对数似然重构损失 (Gaussian NLL Loss)**：
   假设解码器输出服从独立高斯分布 $\mathcal{N}(\mu_{\text{recon}}, \sigma_{\text{recon}}^2)$：
   $$\text{ReconLoss} = \sum_{i=1}^{D} \left( 0.5 \cdot \left(\frac{x_i - \mu_{\text{recon}, i}}{\sigma_{\text{recon}, i}}\right)^2 + \log \sigma_{\text{recon}, i} + 0.5 \log(2\pi) \right)$$
2. **防数值爆炸双向软截断（`softclip` 防护）**：
   为了彻底防止重构方差在未见过特征上发生指数级数值溢出或下溢导致 NaN，`vae.py` 对对数方差实施平滑软裁剪：
   $$\log \sigma_{\text{clip}} = -\text{softclip}\big(-\text{softclip}(\log \sigma, -6.0), 0.0\big)$$
   确保重构方差严格保持在稳健的数值边界内。
3. **KL 散度正则项 (KLD)**：
   强制后验隐分布逼近标准正态先验 $\mathcal{N}(0, I)$：
   $$D_{\text{KL}} = -0.5 \sum_{j=1}^{Z} \left( 1 + \log \sigma_j^2 - \mu_j^2 - \exp(\log \sigma_j^2) \right)$$

### 4.3 双轴独立训练与多体制生成式建模
根据 ADR 0041 的三流解耦规范，Stage II 在数据流与模型实例上完全物理隔离为两个正交轴：
1. **Slope VAE 轴**：
   - 训练数据：从 `vae_slope_state_features.npy` 提取；
   - 划分体制标签：按 Butterworth 价格斜率划分为 Down、Flat、Up 三类；
   - 分别独立训练 3 个独立的 `MLP_VAE` 模型，分别拟合三种趋势形态下的特征流形概率分布。
2. **Volatility VAE 轴**：
   - 训练数据：从 `vae_volatility_state_features.npy` 提取；
   - 划分体制标签：按对数收益率标准差划分为 Low Vol、Mid Vol、High Vol 三类；
   - 分别独立训练 3 个独立的 `MLP_VAE` 模型，分别拟合三种波动能级下的特征流形概率分布。

### 4.4 分布外 (OOD) 漂移检测数学原理
在在线推理时，针对当前市场观测样本 $x_t$，将样本分别送入各个体制的 VAE 进行前向传播：
1. **未归一化对数概率估算**：
   $$\log \hat{p}_k(x_t) = -\text{Loss}_{\text{VAE}, k}(x_t) = -\big( \text{ReconLoss}_k(x_t) + \text{KLD}_k(x_t) \big)$$
2. **物理直觉与 OOD 判别**：
   - 若当前行情属于体制 $k$（如常态震荡），则体制 $k$ 的 VAE 解码器能够精确重构该特征，重构误差极低，$\log \hat{p}_k(x_t)$ 处于高位；
   - 若市场爆发非平稳黑天鹅事件或极端异常行情（如崩盘闪崩），所有已知体制的 VAE 均无法有效重构该样本，重构误差呈现爆炸式上升，即：
     $$\max_k \log \hat{p}_k(x_t) < \text{OOD\_Threshold}$$
   - 该数学性质直接为 Stage III 的高层风控门控提供了绝对尾部风险断路依据。

## 5. Stage III：高层风险感知路由与执行引擎 (Stage III: Meta-Routing & Execution Engine)

### 5.1 门控策略架构 (Gating Strategy Architecture, ADR 0031, ADR 0049)
Stage III 负责在每个决策时刻整合双轴 VAE 似然评估结果，动态决定是激活特定的专业化低层智能体 Slot，还是执行风险防守退出（`FineFT/RL/DiHFT/high_level/gating/`）：

#### 方案一：绝对双阈值门控与迟滞带 (`AbsoluteThresholdGating`, ADR 0049)
针对传统单一临界阈值引发的“微幅波动频繁洗盘踢平”问题，系统确立了**非对称迟滞带（Hysteresis Band）机制**：
1. **开仓门槛与平仓容忍门槛解耦**：
   - 设定开仓置信度门槛：$T_{\text{vol}}^{\text{enter}}$ 与 $T_{\text{slope}}^{\text{enter}}$；
   - 引入平仓衰减比率 $\alpha_{\text{exit}} \in [0.50, 0.80]$（由 Optuna 自动寻优），派生退出防守阈值：
     $$T_{\text{vol}}^{\text{exit}} = \alpha_{\text{exit}} \cdot T_{\text{vol}}^{\text{enter}}, \quad T_{\text{slope}}^{\text{exit}} = \alpha_{\text{exit}} \cdot T_{\text{slope}}^{\text{enter}}$$
2. **状态感知动态决策逻辑**：
   - **空仓状态**（$\text{current\_position} == 0$）：严格要求 VAE 似然必须双双跨越开仓高门槛（$\max \text{vol} \ge T_{\text{vol}}^{\text{enter}} \land \max \text{slope} \ge T_{\text{slope}}^{\text{enter}}$），才允许激活对应体制的子智能体建仓；
   - **持仓状态**（$\text{current\_position} \neq 0$）：放宽考核容差，只要似然维持在 $T^{\text{exit}}$ 之上，继续允许持仓波段滚动，彻底消灭了临界抖动导致的频繁磨损；
   - **跌破防守线**：若得分跌破 $T^{\text{exit}}$，确认市场体制发生破位或进入高风险 OOD，立即输出 `is_defensive = True`，触发强制平仓。

#### 方案二：分层双重门控 (`HierarchicalDualGating`, ADR 0031)
将尾部极端黑天鹅与微观震荡模糊性解耦为两道串联防线：
- **Gate 1: 绝对尾部风险断路器 (Catastrophic OOD)**：
  若 $\max \text{vol} < \text{OOD\_Threshold}$ 或 $\max \text{slope} < \text{OOD\_Threshold}$，判定市场进入未见过的灾难性异动，立即触发 `ood_circuit_breaker` 强平防守。
- **Gate 2: 相对模式清晰度过滤 (Mode Ambiguity Filter)**：
  计算归一化概率分布的 Top-1 与 Top-2 裕度差：$\text{Margin} = P_{(1)} - P_{(2)}$。若 $\text{Margin} < \text{Margin\_Threshold}$，判定当前市场处于震荡杂波或体制摇摆期，触发 `margin_ambiguity` 平仓观望。

### 5.2 动作持久化与防御优先抢占机制 (ADR 0045)
为消除低层策略在微观尺度的超高频无谓倒手，并与低层强化学习 `--action_persistence 3` 训练假设对齐，系统在执行引擎核心（`FineFT/RL/DiHFT/high_level/vae_routing_util.py`）中设计了**动作持久化与防御优先抢占机制**：

```
决策时刻步进
    │
    ├──► 1. 检查是否存在高层防御信号 (OOD / 破位 / 非主力合约 / 空模型槽位)?
    │        └──► [是] 立即打断持仓锁定 (remaining_persist = 0)
    │                  强制执行规则平仓 (DEFENSIVE_PREEMPTION / DEFENSIVE_RULE_CLOSE)
    │
    ├──► 2. 处于持仓锁定中 (remaining_persist > 0)?
    │        ├──► [动作仍物理合法] 复用上一动作 (ACTION_PERSISTENCE)，跳过神经网络前向推理
    │        └──► [动作因深度枯竭非法] 打断锁定，重新推断 (ACTION_UNAVAILABLE_BREAK)
    │
    └──► 3. 锁定倒计时归零 (remaining_persist == 0) 或处于空仓?
             └──► 重新调用底层神经网络执行完整推断 (POLICY_INFERENCE)
                  若输出非平仓动作，重新激活锁定倒计时 (remaining_persist = action_persistence - 1)
```

1. **非 Flat 持仓跨 Slot 锁定**：当策略建立多头或空头仓位后，物理锁定维持 $K$ 步（默认 3 步，即 30 分钟）。在此期间，高层宏观路由在非防御状态下的微小网格切换不会打断当前持仓，跳过 60% 以上的前向推理计算开销。
2. **防御最高优先权抢占 (Defensive Preemption)**：一旦高层门控触发防御平仓信号或三级风控事件，无论持久化倒计时剩余多少，立即清零（`remaining_persist = 0`）并强行平仓至 Flat，坚守风控底线。
3. **空仓不锁定**：Flat 动作不触发持续性，每步重新评估建仓机会。

### 5.3 三级确定性安全风控护栏体系 (ADR 0047)
为了彻底根除单边极端行情中的“无刹车连续扛单巨亏”与“止损后抄底接飞刀”，系统在执行引擎构建了三级确定性物理风控体系：

| 风控层级 | 机制名称 | 核心判定参数与数学公式 | 触发动作与状态转移 | 治理痛点 |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | **标的价格相对变动率硬止损** (方案 A) | `--stop_loss_return_threshold 0.015`<br>$\text{loss\_rate} = \frac{-\text{unrealized\_pnl}}{\|\text{position}\| \cdot P_{\text{mark}}} \ge 1.5\%$ | 打断持仓锁定（`remaining_persist = 0`），强制规则平仓至 Flat，记录 `last_stopped_position` 与 `HARD_STOP_LOSS` | 斩断长周期单边暴跌中的死扛巨亏（如截断 `fu2411` 的 93% 扛单亏损） |
| **Tier 2** | **止损后单向同向开仓冷静期** (方案 2) | `--stop_loss_cooldown_steps 12`<br>（默认 12 步 = 2 小时） | 激活冷静期计数器；若策略试图开仓同向仓位（$\text{target\_pos} \cdot \text{last\_stopped} > 0$），强制拦截置 Flat 并记录 `STOP_LOSS_COOLDOWN`；允许反向或保持 Flat | 防止止损后数步内立即再度“追高/抄底接飞刀”送人头 |
| **Tier 3** | **连续止损单合约熔断断路器** (方案 4) | `--circuit_breaker_consecutive_stops 2`<br>`--circuit_breaker_cooling_steps 72`<br>（默认 72 步 = 12 小时） | 维持单合约连续止损计数器。连续触发 2 次硬止损且中途无正常平仓盈利时，强制进入熔断休眠；期间全量强制置 Flat 并记录 `CIRCUIT_BREAKER_SUSPENSION` | 彻底休眠极端非平稳单边合约，杜绝全天候失血 |

### 5.4 Optuna 超参数自适应寻优框架 (ADR 0046, `vae_routing_optuna.py`)
Stage III 采用轻量高效的 Optuna 贝叶斯优化管线在验证集上寻找最优门控参数组合：
1. **离线 VAE 分位数特征预计算**（ADR 0046）：
   在启动 Optuna 搜索前，系统预先对全量验证集执行双轴 VAE 前向推理并持久化为预计算分位数矩阵（`ensure_precomputed_vae_quantiles`）。各并发 Trial 仅需通过 $O(1)$ 的内存查表获取 VAE 评分，彻底消除了 GPU 重复推理开销。
2. **多进程 SQLite 共享并发架构**：
   使用 32 核 CPU 进程并行采样，通过 SQLite 统一管理 `optuna_study.db`，设置 60 秒 Busy Timeout 杜绝锁争用。
3. **核心寻优搜索空间**：
   - 滚动平滑窗口长度：`window_length` $\in [50, 150]$；
   - 衰减因子：`gamma` $\in [0.92, 0.98]$；
   - 基础开仓门槛：`rule_base_threshold` $\in [0.20, 0.50]$；
   - 迟滞平仓衰减比率：`hysteresis_exit_ratio` $\in [0.50, 0.80]$。
4. **确定性最佳试验复现**：
   寻优结束后，自动加载全局最优 Trial 参数，开启 `save_artifacts = True` 完整导出标准测试集轨迹与逐帧审计数据。

## 6. 交易环境仿真撮合与摩擦损耗解耦体系 (Environment Simulation & Friction Decomposition)

### 6.1 订单簿五档撮合仿真与物理成本核算 (`FineFT/env/`)
FineFT 针对商品期货市场特性构建了高保真回测与训练环境（`FineFT/env/env_class/base_env.py` 与 `commodity_env.py`）：
1. **订单簿五档深度吃单模拟（LOB Depth Execution）**：
   在市价买入/卖出下单时，环境依据快照行情提供的 `ask_prices`, `ask_qtys`, `bid_prices`, `bid_qtys` 进行逐档深度穿透撮合。
   - 若目标开仓量超出当前一档挂单量，剩余成交量必须穿透至二档乃至深档成交，真实产生物理订单簿滑点（`slippage`）；
   - 若整档深度依然不足以容纳目标仓位，根据 ADR 0001 截断至物理允许的最大档位，杜绝无限制超额成交。
2. **交易所佣金费率模型**：
   按国内商品期货真实交易规则配置买卖双边手续费率（商品期货默认 `buy_fee_rate = 0.0001`, `sell_fee_rate = 0.0003`），开平仓实时从 `wallet_balance` 中扣除佣金成本。
3. **账户资产净值核算体系**：
   - 钱包余额：$\text{WalletBalance}_{t} = \text{WalletBalance}_{t-1} + \text{RealizedPnL}_t - \text{CommissionFee}_t$；
   - 未结浮盈：$\text{UnrealizedPnL}_t$ 由当前持仓均价与标记价格（$P_{\text{mark}}$）实时差额计算；
   - 账户总资产：$\text{TotalAsset}_t = \text{WalletBalance}_t + \text{UnrealizedPnL}_t$；
   - 维持保证金核算：根据多档持仓规模阶梯保证金字典严格校验，一旦总资产跌破维持保证金阈值，触发强制平仓清算。

### 6.2 非对称方向性调仓惩罚体系 (Asymmetric Turnover Penalty, ADR 0049, ADR 0050)
针对强化学习在单边大行情中“顺势建仓受阻（Zero-Action Trap）”与震荡市中“超低摩擦诱发高频刮头皮（Scalping）”的天然矛盾，系统确立了**非对称方向性调仓惩罚机制**：

```
当前时间步行级宏观体制判定: slope_bin = regime_grid_id % 3
           │
           ├──► [slope_bin == 2: 上涨体制]
           │        ├──► new_pos > old_pos (顺势做多/加仓): 执行基准费率 base_rate (1 bps)
           │        └──► new_pos < old_pos (逆势平多/做空): 执行阻尼费率 adverse_rate (6 bps)
           │
           ├──► [slope_bin == 0: 下跌体制]
           │        ├──► new_pos < old_pos (顺势做空/加空): 执行基准费率 base_rate (1 bps)
           │        └──► new_pos > old_pos (逆势平空/做多): 执行阻尼费率 adverse_rate (6 bps)
           │
           └──► [slope_bin == 1: 横盘震荡体制]
                    ├──► |new_pos| > |old_pos| (开仓建立风险敞口 0 -> ±1) 或 反手: 执行阻尼费率 adverse_rate (6 bps)
                    └──► |new_pos| == 0 (平仓防守退出 ±1 -> 0): 执行基准费率 base_rate (1 bps)
```

1. **数学形式与参数化**：
   $$\text{Penalty}_{\text{turnover}} = \text{Rate}(\text{regime}, \Delta \text{pos}) \cdot |\Delta \text{position}| \cdot P_{\text{mark}} \cdot \text{ContractUnit}$$
   其中基准费率默认 $\text{turnover\_base\_rate} = 0.0001$（1 bps），逆势阻尼比率默认 $\text{turnover\_adverse\_ratio} = 6.0$，派生逆势费率为 6 bps（0.0006）。
2. **纯 Reward Shaping 定位**：
   该惩罚项**仅作用于强化学习单步奖励标量 $r_t$ 与离线 DP 专家 Q 表递归更新**，绝不扣减真实物理账户的 `wallet_balance` 与总资产，保证实盘账本核算的绝对真实性。
3. **全链路闭环效果**：
   - 顺势建仓零心理负担，彻底消除“全空仓不动点”；
   - 强行锁死趋势波段中的持仓惯性，大幅拉长平均持仓步数（实测从 15.7 步暴拉至 81.4 步）；
   - 在震荡市中惩罚无序试盘，促使策略自发学会空仓观望防守。

### 6.3 交易诊断三件套与全维度摩擦拆解引擎 (ADR 0048, `trading_diagnostics.py`)
为彻底打破量化交易回测中的“利润去向黑盒”，系统开发了标准化的全维度交易诊断计算器 `TradingDiagnosticsCalculator`（`FineFT/analysis/diagnostics/trading_diagnostics.py`），并在全部回测流水线中强制生成**诊断三件套产物（Diagnostic Triad Artifacts）**：

1. **核心摩擦解耦方程（第一手实证原则）**：
   对回测轨迹逐 Step 进行严格物理量化分解：
   $$\text{Portfolio Net Return} = \text{Gross PnL} - \text{Commission Fees} - \text{Slippage Losses}$$
   $$\text{Friction Consumption Ratio} = \frac{\text{Commission Fees} + \text{Slippage Losses}}{\text{Gross PnL}} \times 100\%$$
2. **标准交付三件套规格**：
   - **`diagnostics_summary.json`**：全量高精度结构化指标字典（包含投资组合收益、夏普比率、最大回撤、索提诺比率、卡尔玛比率、总交易次数、换手倍数、摩擦吞噬率、胜率等）；
   - **四张标准化 CSV 维度明细表**：
     - `contract_pnl_friction.csv`：逐合约毛利润、净利润、手续费、滑点与资金换手倍数；
     - `contract_behavior_risk.csv`：逐合约持仓分布、平均持仓周期、调仓次数、硬止损与断路器事件明细；
     - `macro_routing_distribution.csv`：逐合约高层 VAE 路由各 Slot 宏观分流占比；
     - `best_result.csv`：最终全组合聚合性能指标。
   - **`diagnostics_summary.md`**：自动排版的研究简报，直接呈现量化评级（PASS / FAIL）与对比分析。

---

## 7. 核心源码映射与工程契约索引表 (Source Code Mapping & Contract Index)

| 架构层级 / 核心模块 | 核心源码文件路径 | 关键类 / 函数 / 实体 | 关联 ADR / 规范 | 核心职责与设计契约 |
| :--- | :--- | :--- | :--- | :--- |
| **特征工程** | `data_preprocess/operator_futures/` | `muti_contract_scale_save.py`<br>`predictive_audit.py` | ADR 0037<br>ADR 0041<br>ADR 0042 | 生产趋势流、波动率流与微观 RL 流三流解耦特征，实现统一宽表物化与 GPU 矩阵快速审计 |
| **Stage I 模型** | `FineFT/model/low_level.py` | `Qnet`<br>`ensemble_Qnet` | ADR 0002<br>ADR 0007 | 实现五路张量结构化输入处理与动作合法性掩码物理阻断 |
| **Stage I 训练** | `FineFT/RL/DiHFT/low_level/` | `weight_advantage_pretrain.py`<br>`parallel_weight_advantage_pretrain.py`<br>`parallel_diverse_train.py` | ADR 0012<br>ADR 0014<br>ADR 0049<br>ADR 0050 | 实施离线 DP 导师监督预训练、集成局部加权多样化分化训练与异步贪心评估 |
| **Stage I 选优** | `FineFT/analysis/pick_agent/` | `FineFT_two_dimensional_agent_selector.py` | ADR 0005<br>ADR 0006 | 基于 4x4 波动率与趋势斜率网格，执行 LCB 稳健性遴选，装配输出 `model.pth` 与 selection manifest |
| **Stage II VAE** | `FineFT/RL/DiHFT/VAE/` | `vae.py` (`MLP_VAE`)<br>`process.py`<br>`summary.py` | ADR 0018<br>ADR 0026<br>ADR 0041 | 双轴无监督变分自编码器密度建模，提供 softclip 数值保护与 ELBO 重构对数似然估算 |
| **Stage III 门控** | `FineFT/RL/DiHFT/high_level/gating/` | `AbsoluteThresholdGating`<br>`HierarchicalDualGating` | ADR 0031<br>ADR 0049 | 提供双阈值迟滞带防抖动门控与绝对 OOD / 模式裕度分层熔断门控 |
| **Stage III 执行** | `FineFT/RL/DiHFT/high_level/` | `vae_routing_util.py`<br>(`vae_risk_aware_routing`) | ADR 0045<br>ADR 0047 | 调度动作持久化机制、防御优先抢占、三级确定性硬风控底线与专业子网络推理 |
| **Stage III 寻优** | `FineFT/RL/DiHFT/high_level/` | `vae_routing_optuna.py`<br>`optuna_trail.py` | ADR 0046 | 基于 VAE 分位数预计算矩阵与多进程 SQLite 共享，自适应优化门控超参数 |
| **交易环境撮合** | `FineFT/env/env_class/` | `base_env.py`<br>`commodity_env.py`<br>`futures_util.py` | ADR 0001<br>ADR 0049<br>ADR 0050 | 订单簿五档吃单真实滑点模拟、反向开仓尽力而为语义、非对称方向性调仓惩罚与 DP 专家表生成 |
| **诊断与归因** | `FineFT/analysis/diagnostics/` | `trading_diagnostics.py` | ADR 0048 | 自动解耦毛利、滑点与佣金磨损，无缝导出 JSON、CSV、Markdown 诊断三件套交付物 |

---

## 8. 总结与演进路线 (Conclusion & Future Roadmap)

FineFT / DiHFT 交易策略架构通过严格的**三流特征解耦**、**多体制微观智能体集群专业化**、**双轴生成式 VAE 宏观表征**以及**具备迟滞带和三级硬核风控的高层元路由引擎**，成功在高度非平稳与高摩擦的商品期货市场中实现了：
1. **100% 测试合约全胜率**（12/12 支验证合约全部实现绝对正收益）；
2. **摩擦吞噬率从历史 96.87% 骤降至 12.90%**；
3. **资金换手倍数压降 97.7%**（从 722.5x 降至 16.36x），平均持仓周期拉长 5.2 倍；
4. **彻底截断长周期极端行情的单边爆仓亏损**。

未来演进重点将持续推进：
- **宏观时间尺度扩展**：向微观 RL 状态注入长周期日级别大趋势显式特征，补充强趋势单向开仓硬锁（Trend Lock）；
- **动态自适应移动止盈 (Trailing Stop)**：构建基于 ATR 与峰值浮盈回撤比率的自适应移动止盈机制，防止大波段过山车行情中利润巨幅回吐。
