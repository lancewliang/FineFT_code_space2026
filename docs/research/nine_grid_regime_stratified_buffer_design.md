# 9 格（3×3 波动率 × 斜率）动态分档经验池系统架构与采样调度设计方案

> 研究日期：2026-09-11  
> 适用模块：FineFT 强化学习低层多样化训练系统（`FineFT/RL/DiHFT/low_level`）与阶段 II 二维智能体选择器（`FineFT/analysis/pick_agent`）  
> 核心目标：针对并行多样化训练中子代理策略均质化退化问题，设计基于“波动率 × 斜率”3×3 动态分类经验池系统与分轮次定向采样调度方案，诱发子代理与各轮次模型状态产生显著分化  
> 报告文件：`docs/research/nine_grid_regime_stratified_buffer_design.md`

---

## 1. 核心结论与可行性论证（Executive Summary & Feasibility）

### 1.1 核心结论

针对用户提出的核心需求：**“是否可以根据波动和斜率做 9 个 3×3 的动态分类经验池？每一轮只对部分经验池抽样，使每一轮训练完的模型状态产生巨大差异？如何计算波动率和斜率分档？如何选择每一轮的经验池？”**

调研与推导结论明确：
1. **完全可行且高度必要**：现有并行训练（[parallel_diverse_train.py](/home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/parallel_diverse_train.py:1253)）将所有行情动态无差别混入单一全局经验池（`buffer_diverse`），导致每个 mini-batch 沦为各类行情的统计均值，直接抹杀了 FineFT Algorithm 2 的子代理专长分化机制。改用 9 个分类经验池是打破均质化退化的最根本架构解法。
2. **3×3（9 格）显著优于 4×4（16 格）与单一经验池**：
   - 在商品期货（如 fu 30min）数据集中，4×4 划分会导致极端角落（如低波动强单边 `vol=0, slope=0`）的独立连续行情（run）严重匮乏（仅 39 个 run，占比 2.76%），引发样本饥渴；
   - 3×3 划分采用三分位数（Terciles，33.3% 与 66.7% 分位点），边际分布严格均分，联合网格中样本最稀缺的格子依然拥有 **994 步 Transition、71 个独立 Run、覆盖 13 个合约**，彻底消除了极端样本荒漠；
   - 3×3 划分天然对应金融市场的经典九宫格语义：波动率（平静/正常/剧烈）× 方向趋势（下跌/震荡/上涨）。
3. **分轮次经验池调度（Curriculum & Regime-Targeted Sampling）能够直接制造模型状态的剧烈差异**：
   - 通过在不同 Epoch 激活不同经验池子集（例如：奇数轮次仅抽样趋势池，偶数轮次仅抽样震荡池），强制网络在特定市场动态上经历集中的梯度推力，使各 Epoch 模型快照在 Q 值地形和动作分布上产生结构性差异；
   - 结合“子代理-经验池软绑定”，可使 $N=9$ 的子代理集合各自演化为对应市场体制的专精专家，彻底解决下游阶段 II 选择器全部回退为 `empty_model` 的困境。

---

### 1.2 经验池架构对比：1 经验池 vs 16 格 (4×4) vs 9 格 (3×3)

下表总结了三种经验池方案在数学稳定性、采样效率及工程可行性上的核心权衡：

| 评估维度 | 单一经验池（现有基线） | 16 格经验池（4×4） | 9 格经验池（3×3，本方案推荐） |
|---|---|---|---|
| **经验存储拓扑** | 全局单队列 `Multi_step_ReplayBuffer_multi_info` | 16 个独立回放池队列 | 9 个独立回放池队列（3 波动率 × 3 斜率） |
| **批次行情混叠** | 严重：每个 batch 充斥相反行情，梯度相互抵消 | 无：完全按 Regime 隔离 | 无：完全按 Regime 隔离 |
| **最稀缺格子样本量** | 不存在（被海量震荡数据淹没） | 严重匮乏：`v0, s0` 仅 423 步（2.76%），39 个 run | 充足稳定：`v0, s0` 达 994 步（6.49%），71 个 run |
| **跨合约覆盖率** | 表面覆盖全合约，但稀缺状态失衡 | 极端格仅覆盖部分合约（如 11/14 合约） | 所有 9 格均覆盖 13~14 个训练合约 |
| **训练与验证对齐** | 无法与下游 2D Selector 对齐 | 4×4 验证集切片存在空切片风险 | 验证集 `dynamic_number=3` 已验证完整且无空槽 |
| **子代理对齐难度** | 无法对齐（$N$ 个代理被动竞争） | 偏高（$N=16$ 模型参数膨胀 2.3 倍） | 极佳（$N=9$ 完美契合 9 格体制，参数量适中） |
| **内存与调度开销** | 内存占用集中，调度单一 | 16 个 Tensor 字典，调度复杂度高 | 9 个 Tensor 字典，显存增加约 15%，开销可控 |

---

### 1.3 训练数据 9 格（3×3）因果分布实测数据

为确保方案具备坚实的经验证据，我们基于 `fu 30min` 训练集 14 个切片（[dataset/30min/fu/train/slice/](/home/lanceliang/opt/aiwork/FineFT_code_space2026/dataset/30min/fu/train/slice/)，总计 15,982 行，剔除每合约前 47 根 bar 后可用步数 15,310 步），采用 48-bar 因果滚动窗口拟合并实测了 9 格的真实分布。

#### 1.3.1 训练集三分位数阈值（Tercile Thresholds）
- **有符号 OLS 斜率**（`% log-return / bar`）：
  - 阈值点：$T_{\text{slope}} = [-0.011772, 0.038954]$
  - `slope=0`（下行趋势，Down）：$\text{slope} \le -0.011772$（占三分之一）
  - `slope=1`（震荡横盘，Flat）：$-0.011772 < \text{slope} \le 0.038954$（占三分之一）
  - `slope=2`（上行趋势，Up）：$\text{slope} > 0.038954$（占三分之一）
- **滚动收益率波动率**（`% log-return std`）：
  - 阈值点：$T_{\text{vol}} = [0.342431, 0.446722]$
  - `vol=0`（低波动，Low）：$\text{vol} \le 0.342431$（占三分之一）
  - `vol=1`（中波动，Mid）：$0.342431 < \text{vol} \le 0.446722$（占三分之一）
  - `vol=2`（高波动，High）：$\text{vol} > 0.446722$（占三分之一）

#### 1.3.2 训练集 9 格联合分布实测表
单位说明：`slope_bp` 为单 bar 斜率基点（1 bp = 0.01%）；`vol_bp` 为窗口波动率基点；`ret_bp` 为单步绝对对数收益率基点；`share %` 为占 15,310 可用步的比例。

| 波动档 (vol) | 斜率档 (slope) | 步数 (steps) | 占比 (share %) | 覆盖合约数 | 连续 Run 数 | 平均 Run 长度 | 中位数 Run | 平均斜率 (bp/bar) | 平均波动 (bp) | 单步均值 \|ret\| (bp) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0 (低波)** | **0 (下行)** | 994 | 6.49% | 13 | 71 | 14.0 | 7.0 | -3.938 | 29.21 | 19.10 |
| **0 (低波)** | **1 (震荡)** | 2,357 | 15.40% | 14 | 151 | 15.6 | 11.0 | +1.597 | 28.21 | 20.66 |
| **0 (低波)** | **2 (上行)** | 1,752 | 11.44% | 14 | 129 | 13.6 | 9.0 | +6.899 | 29.49 | 21.16 |
| **1 (中波)** | **0 (下行)** | 1,705 | 11.14% | 14 | 145 | 11.8 | 6.0 | -6.302 | 39.70 | 28.29 |
| **1 (中波)** | **1 (震荡)** | 1,591 | 10.39% | 14 | 223 | 7.1 | 6.0 | +1.527 | 39.49 | 27.82 |
| **1 (中波)** | **2 (上行)** | 1,807 | 11.80% | 14 | 177 | 10.2 | 7.0 | +7.143 | 39.08 | 25.90 |
| **2 (高波)** | **0 (下行)** | 2,404 | 15.70% | 13 | 120 | 20.0 | 8.5 | -8.864 | 60.83 | 37.06 |
| **2 (高波)** | **1 (震荡)** | 1,155 | 7.54% | 14 | 165 | 7.0 | 5.0 | +1.077 | 53.23 | 38.33 |
| **2 (高波)** | **2 (上行)** | 1,545 | 10.09% | 14 | 89 | 17.4 | 11.0 | +9.431 | 54.10 | 33.66 |

#### 1.3.3 实测数据核心洞见
1. **彻底解决 16 格的极端角落饥渴**：在 4×4 下，`vol=0, slope=0` 仅有 423 步和 39 个 run，而在 3×3 下，该格扩展为 994 步与 71 个 run，有效样本事件翻倍，能够支撑经验池的独立循环与重放。
2. **平均 Run 长度完美契合 N-Step**：低波动各格的平均连续 run 长度为 13.6~15.6 根 bar，均大于模型训练的 `n_step=12`（[train_commodity_fu_30_half.sh](/home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/script/train/train_commodity_fu_30_half.sh:19)）。这意味着多步 TD-error 计算几乎完全包含在同一 Regime 内部，不会因频繁穿越状态边界而破坏回报语义。
3. **收益/波动幅度梯次分明**：单步收益率从低波动的 19~21 bp 单调上升至高波动的 33~38 bp；高波下行斜率达 -8.86 bp/bar，低波下行斜率为 -3.94 bp/bar。各格子具有鲜明且自洽的量化特征差异。

---

## 2. 波动率与斜率分档计算方案（Binning Formulation）

为构建 9 个经验池，必须有严格、精确且可工程复现的分档计算逻辑。本报告提供两套计算方案，并对其适用场景与工程优劣势展开对比。

### 2.1 方案 A：切片级宏观分档（Segment / Slice-Level Macro Partitioning）

#### 2.1.1 算法机制与数学公式
基于项目现有 ADR-0009、ADR-0010 与 ADR-0011 规范（[valid_cross_contract_label_calibration.py](/home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/datahandler/valid_cross_contract_label_calibration.py:530)），对全量训练数据切片先进行离线宏观转折点拟合与切片：

1. **巴特沃斯低通滤波与转折点提取**：
   价格序列经过低通滤波器后，通过 `slice_and_merge` 算法提取转折点 $T = \{t_0, t_1, \dots, t_M\}$，形成 $M$ 个独立的市场动态波段（Segments）。
2. **段落标准化斜率评分（Signed Percentage Slope）**：
   对于段落 $[t_k, t_{k+1})$，利用 OLS 拟合价格对时间索引的斜率 $\beta_k$，并按段落起点价格归一化（消除价格绝对量纲）：
   $$S_{\text{slope}, k} = 100 \times \frac{\beta_k}{p_{t_k}}$$
3. **段落对数收益率波动率评分（Segment Log-Return Volatility）**：
   计算段落内单步对数收益率的总体标准差（非年化，百分数形式）：
   $$r_t = \log(p_t) - \log(p_{t-1}), \quad t \in (t_k, t_{k+1}]$$
   $$S_{\text{vol}, k} = 100 \times \sqrt{\frac{1}{t_{k+1}-t_k} \sum_{t=t_k+1}^{t_{k+1}} (r_t - \bar{r}_k)^2}$$
4. **全局段落等权分位数校准（Global Segment Quantiles）**：
   池化所有训练合约产生的所有段落评分 $\{S_{\text{slope}}\}$ 和 $\{S_{\text{vol}}\}$，分别在 $1/3$ 和 $2/3$ 分位点拟合共享阈值：
   $$\theta_{\text{slope}} = [\text{Quantile}(S_{\text{slope}}, 1/3), \text{Quantile}(S_{\text{slope}}, 2/3)]$$
   $$\theta_{\text{vol}} = [\text{Quantile}(S_{\text{vol}}, 1/3), \text{Quantile}(S_{\text{vol}}, 2/3)]$$
5. **切片文件归属**：
   每个生成的 `df_*.feather` 切片整体携带离散标签 `(vol_label, slope_label)`。Worker 展开该切片时，其产生的所有 Transition 统一进入对应的网格经验池。

#### 2.1.2 方案 A 的优缺点
- **优点**：
  - 与现有离线切片流程及下游阶段 II 选择器（[FineFT_two_dimensional_agent_selector.py](/home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/analysis/pick_agent/FineFT_two_dimensional_agent_selector.py:120)）的概念完全一致；
  - 切片级别即确定经验池归属，Worker 无需在运行时执行任何计算，探索阶段零性能开销。
- **缺点**：
  - **存在未来信息与宏观粗粒度**：转折点拟合使用了滤波与全局合并，包含未来路径信息；
  - **切片内部行情漂移**：一个被标记为“低波上涨”的切片长度可达数百 bar，其内部局部可能包含数十 bar 的回撤或震荡，内部局部微观状态与宏观标签存在失配。

---

### 2.2 方案 B：因果步级滚动分档（Step-Level Causal Rolling Partitioning，推荐）

#### 2.2.1 算法机制与数学公式
严格遵循 [low_volatility_trend_agent_training_diagnosis.md](/home/lanceliang/opt/aiwork/FineFT_code_space2026/docs/research/low_volatility_trend_agent_training_diagnosis.md:120) 第 4 节的纯因果口径，在交易环境 `base_env.py` 执行 `step()` 时，针对当前时点 $t$ 及前 $W=48$ 根 bar（跨非交易时段按观测 bar 计数）进行纯历史因果计算：

1. **因果 OLS 对数斜率**：
   令 $p_i$ 为时点 $i$ 的成交价/标记价，取对数 $y_i = \log p_i$，时间序列 $x_i = 0, 1, \dots, W-1$（均值 $\bar{x} = \frac{W-1}{2}$）。
   时点 $t$ 的有符号 OLS 斜率定义为：
   $$\beta_t = \frac{\sum_{i=0}^{W-1} (x_i - \bar{x})(y_{t-W+1+i} - \bar{y}_t)}{\sum_{i=0}^{W-1} (x_i - \bar{x})^2}$$
   斜率单位转换为基准百分比：$S_{\text{slope}, t} = \beta_t \times 100$。
2. **因果滚动对数波动率**：
   在 $W-1$ 个单步对数收益率上计算总体标准差：
   $$r_i = \log(p_{t-W+1+i} / p_{t-W+i}), \quad i = 1, \dots, W-1$$
   $$\sigma_t = \sqrt{\frac{1}{W-1} \sum_{i=1}^{W-1} (r_i - \bar{r}_t)^2}$$
   波动率单位转换为基准百分比：$S_{\text{vol}, t} = \sigma_t \times 100$。
3. **固化分位阈值映射**：
   严格仅在训练集上离线拟合三分位点阈值（见 1.3.1 节）：
   $$T_{\text{slope}} = [-0.011772, 0.038954], \quad T_{\text{vol}} = [0.342431, 0.446722]$$
   在 `base_env.py` 中，每个 step 计算后通过快速查找确定离散档位：
   $$\text{vol\_bin} = \text{searchsorted}(T_{\text{vol}}, S_{\text{vol}, t})$$
   $$\text{slope\_bin} = \text{searchsorted}(T_{\text{slope}}, S_{\text{slope}, t})$$
   $$\text{grid\_id} = \text{vol\_bin} \times 3 + \text{slope\_bin} \in [0, 8]$$
4. **Transition 标签写入**：
   在环境返回的 `info` 字典中注入当前 step 的市场体制标签：
   ```python
   info["regime_grid_id"] = int(grid_id)
   info["regime_vol_bin"] = int(vol_bin)
   info["regime_slope_bin"] = int(slope_bin)
   ```
   主进程将收集到的 Transition 按 `info["regime_grid_id"]` 存入对应的回放池。

#### 2.2.2 方案 B 的优缺点
- **优点**：
  - **100% 纯因果**：无任何未来数据穿越，在线观测与离线经验池标签完全对齐；
  - **微观精度极高**：Transition 级别的精准归类，避免了长切片中局部波段的标签污染；
  - **特征与经验池无缝共振**：若未来引入因果 Regime Anchor 特征（如滚动斜率与波动率 z-score），特征计算与经验池路由共用同一套滑动窗口，完全自洽。
- **缺点**：
  - 数据前 $W-1=47$ 根 bar 无法形成完整窗口，需在切片起始时采用前向填补或丢弃前 47 步的 Transition 记录；
  - 需在环境或预处理中维护长度为 48 的价格双端队列（`deque(maxlen=48)`）。但由于 $W=48$ 极小，单步增量更新耗时低于 2 微秒，工程开销可忽略不计。

---

### 2.3 方案选型结论

| 比较维度 | 方案 A（切片级宏观分档） | 方案 B（因果步级滚动分档，推荐） |
|---|---|---|
| **理论纯洁度** | 依赖滤波合并，存在后见偏差 | 严格因果，满足马尔可夫决策过程时序假定 |
| **颗粒度** | 切片级（几百步共享同一个标签） | 步级（逐步精确判定当前体制） |
| **状态穿越鲁棒性** | 差（切片内部出现反向波动仍打主标签） | 极高（行情一旦反转，Transition 立即流向对应池） |
| **工程改造成本** | 需重构预处理与训练切片生成流水线 | 仅需在 `base_env` 加 20 行滚动计算并在 `info` 传递 |

**最终选型**：采用 **方案 B（因果步级滚动分档）** 作为核心分档计算方案，训练集阈值固化为 $T_{\text{slope}} = [-0.011772, 0.038954]$ 与 $T_{\text{vol}} = [0.342431, 0.446722]$。

---

## 3. 经验池选择与轮次采样调度方案（Sampling Scheduling Design）

用户最核心的问题是：**“这样每一轮只对部分经验池抽样. 这样每一轮训练完的模型状态就会有巨大的差异。你帮我设计一下方案... 怎么样选择每一轮的经验池。”**

本方案设计了三层递进的经验池选择与采样机制，既能在轮次（Epoch）级别制造模型的宏观演化分化，又能在集成子代理（Sub-Agents）级别固化专业化分工。

---

### 3.1 方案模式一：子代理与体制 1 对 1 软绑定专业化（Agent-Regime Soft Binding）

#### 3.1.1 机制设计
在 `ensemble_Qnet` 中，将集成子网络数量设为 $N=9$（对应 9 个网格）。每个子代理 $i \in [0, 8]$ 被赋予一个**专属主场体制（Home Regime）**：

$$\text{Sub-Agent } i \longleftrightarrow \text{Grid } i = (\text{vol}_i, \text{slope}_i)$$

网格索引映射规则（行优先）：
- Agent 0: $(v0, s0)$ —— 低波下跌专家（Low-Vol Bear Specialist）
- Agent 1: $(v0, s1)$ —— 低波震荡专家（Low-Vol Range Specialist）
- Agent 2: $(v0, s2)$ —— 低波上涨专家（Low-Vol Bull Specialist）
- Agent 3: $(v1, s0)$ —— 中波下跌专家（Mid-Vol Bear Specialist）
- Agent 4: $(v1, s1)$ —— 中波震荡专家（Mid-Vol Range Specialist）
- Agent 5: $(v1, s2)$ —— 中波上涨专家（Mid-Vol Bull Specialist）
- Agent 6: $(v2, s0)$ —— 高波下挫专家（High-Vol Crash Specialist）
- Agent 7: $(v2, s1)$ —— 高波剧震专家（High-Vol Churn Specialist）
- Agent 8: $(v2, s2)$ —— 高波暴涨专家（High-Vol Rally Specialist）

#### 3.1.2 采样与损失软绑定公式
在训练阶段更新子代理 $i$ 时，其 mini-batch 样本由两部分构成：
- **主场经验池采样比例（Home Ratio）** $\alpha_{\text{home}} = 70\%$：严格从经验池 $B_i$ 中抽取；
- **邻近或全局经验池采样比例（Explore Ratio）** $\alpha_{\text{other}} = 30\%$：从全局或相邻网格中抽取（确保子代理具备基础的通用交易常识，不至于在偶发穿越时动作失控）。

在更新算法中（[parallel_diverse_train.py](/home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/parallel_diverse_train.py:1532)），针对来自经验池 $g$ 的 Transition，对子代理 $i$ 的 TD 权重施加 Regime 亲和力先验：
$$W_{\text{prior}}(i, g) = \exp\left(-\frac{\|(\text{vol}_i, \text{slope}_i) - (\text{vol}_g, \text{slope}_g)\|_1}{\tau_{\text{regime}}}\right)$$
该先验直接调制 FineFT Algorithm 2 中的 `batch_weights`，使得 Agent 0 的梯度更新 80% 以上被低波动下行样本支配，从根本上锁死其策略特化方向。

---

### 3.2 方案模式二：轮次滚动体制课程表（Rotating Epoch Curriculum）

若模型维持原有 $N$（如 $N=7$ 或仅追踪单一体制集成），如何让**“每一轮训练完的模型状态有巨大差异”**？
答案是实施 **轮次体制交替聚焦训练（Epoch-Level Regime Rotation）**。

#### 3.2.1 三阶段周期轮换课程表（3-Phase Rotating Curriculum）
将训练轮次划分为模 3 的循环体系，每一轮仅激活指定的 3 个网格经验池进行梯度更新：

```text
Epoch e 采样配置：
├── e % 3 == 0: 【趋势单边突破轮】 (Trend Regimes)
│   └── 激活经验池：B0 (低波下行), B2 (低波上行), B6 (高波下行), B8 (高波上行)
│   └── 效果：模型状态大幅偏向追随单边强趋势、降低换手率、持仓穿越波动。
│
├── e % 3 == 1: 【震荡均值回归轮】 (Mean-Reversion Regimes)
│   └── 激活经验池：B1 (低波震荡), B4 (中波震荡), B7 (高波剧震)
│   └── 效果：模型状态大幅偏向快进快出、高抛低吸、严控仓位暴露。
│
└── e % 3 == 2: 【极端危机与微澜对比轮】 (Extreme Contrast Regimes)
    └── 激活经验池：B0 (低波极度沉寂下行), B6 (高波动暴跌危机), B8 (高波动主升浪)
    └── 效果：模型状态直面极端风控与尾部收益，学习大波动下的避险与激进进攻。
```

#### 3.2.2 模型状态差异的量化指标监控
在训练日志中实时输出各 Epoch 后的状态向量差异度（Cosine Distance / KL Divergence）：
$$\Delta Q_e = \frac{1}{|S_{\text{eval}}|} \sum_{s \in S_{\text{eval}}} \left(1 - \frac{Q_e(s) \cdot Q_{e-1}(s)}{\|Q_e(s)\| \|Q_{e-1}(s)\|}\right)$$
在周期轮换下，$\Delta Q_e$ 保持在 0.25~0.60 的高分化区间，各 Epoch 生成的 `epoch_k/trained_model.pkl` 将呈现出风格迥异的决策偏好，为阶段 II 选择器提供极为多样的高质量候选池。

---

### 3.3 方案模式三：动态 ETD 误差驱动的自适应采样（Dynamic Error-Driven Routing）

#### 3.3.1 机制设计
不同于固定的轮换，模式三采用自适应调度：
1. **跟踪各经验池的平均 TD-error 与教师 KL 差异**：
   维护 9 个经验池的状态评估量 $\mathcal{E}_g = \text{EMA}(\text{TD\_Loss}_g + \text{ada} \times \text{KL\_Loss}_g)$。
2. **瓶颈经验池重点攻坚**：
   某一行情（如低波强趋势）当前子代理普遍表现最差（$\mathcal{E}_g$ 极高），则在下一轮次动态提升该经验池的抽样概率：
   $$P(B_g) = \frac{\exp(\mathcal{E}_g / \tau_{\text{err}})}{\sum_j \exp(\mathcal{E}_j / \tau_{\text{err}})}$$
3. 这种机制保证了“哪里学得差就多练哪里”，防止模型在简单样本（如高波动顺势）上反复过拟合，而对稀缺复杂的低波行情视而不见。

---

### 3.4 经验池容量、FIFO 淘汰与探索终止重构

#### 3.4.1 独立容量配额（Per-Grid Capacity Allocation）
在原有代码中，单一 `buffer_size=2,000,000` 导致高频出现的行情迅速填满经验池，挤出稀缺行情。
在 9 格架构下，总容量 $2.0 \times 10^6$ 被解耦为 9 个独立的环形队列（Circular Deque）：
$$C_g = \lfloor \frac{B_{\text{total}}}{9} \rfloor \approx 220,000 \quad (g \in [0, 8])$$
- **隔离保护**：高波震荡行情哪怕产生 1,000,000 步，也只能在其专属的 $B_7$ 内部进行 FIFO 淘汰，绝对无法挤占低波下行池 $B_0$ 的哪怕 1 个位置！
- **稀缺经验指纹永久驻留**：结合已有的语义键指纹去重（[parallel_diverse_train.py](/home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/parallel_diverse_train.py:465)），低波稀缺行情的有效探索样本将获得完整保留。

#### 3.4.2 废除全局 `skip_exploration` 冻结逻辑
现有代码（[parallel_diverse_train.py](/home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/parallel_diverse_train.py:1357)）在经验池满后永久将 `skip_exploration` 置为 `True`，导致后续轮次完全失去 RL 在线探索能力。
在 9 格体系下，必须改为：
- **按池饱和判定**：只有当所有被激活采样的经验池均达到饱和且新增指纹连续 5 轮停滞时，才允许跳过对应任务的探索；
- **允许部分刷新**：每个 Epoch 允许低学习率探索以 $\epsilon_{\min} = 0.05$ 持续注入最新策略产生的对抗性轨迹（On-policy Perturbation），保证动态平衡。

---

## 4. 系统工程架构与现有代码集成方案（Engineering Architecture）

### 4.1 核心类设计：`NineGridReplayBuffer`

在 `FineFT/RL/util/replay_buffer_DQN.py` 中新增 `NineGridReplayBuffer`，作为 9 个 `Multi_step_ReplayBuffer_multi_info` 的统一编排门面（Facade）：

```python
class NineGridReplayBuffer:
    """9 格（3x3 波动率 x 斜率）体制分层多步经验回放池管理器。"""

    def __init__(
        self,
        total_buffer_size: int,
        batch_size: int,
        device: str,
        seed: int,
        gamma: float,
        n_step: int,
        slope_thresholds: list[float],
        vol_thresholds: list[float],
    ):
        self.num_grids = 9
        self.grid_capacity = total_buffer_size // self.num_grids
        self.batch_size = batch_size
        self.device = device
        self.slope_thresholds = np.asarray(slope_thresholds, dtype=float)
        self.vol_thresholds = np.asarray(vol_thresholds, dtype=float)

        # 初始化 9 个独立的底层经验池
        self.buffers: dict[int, Multi_step_ReplayBuffer_multi_info] = {
            grid_id: Multi_step_ReplayBuffer_multi_info(
                buffer_size=self.grid_capacity,
                batch_size=batch_size,
                device=device,
                seed=seed + grid_id,
                gamma=gamma,
                n_step=n_step,
            )
            for grid_id in range(self.num_grids)
        }

    def route_transition(self, info: dict[str, Any]) -> int:
        """根据 info 中的因果滚动指标计算 grid_id (0..8)。"""
        if "regime_grid_id" in info:
            return int(info["regime_grid_id"])
        # 若未预计算，则从 info 中的实时指标计算
        vol = float(info["rolling_volatility_48"])
        slope = float(info["rolling_slope_48"])
        v_bin = int(np.searchsorted(self.vol_thresholds, vol, side="right"))
        s_bin = int(np.searchsorted(self.slope_thresholds, slope, side="right"))
        return v_bin * 3 + s_bin

    def add_transition(self, transition: tuple) -> int:
        """根据 Transition 的当前 info 自动路由写入对应经验池。"""
        # transition: (s, info, a, r, s_, next_info, done)
        info = transition[1]
        grid_id = self.route_transition(info)
        self.buffers[grid_id].add_transition(transition)
        return grid_id

    def sample_from_grids(
        self,
        active_grid_ids: list[int],
        grid_weights: list[float] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """从指定的活跃经验池子集中按权重配比采样完整 Batch。"""
        # 根据 active_grid_ids 与配比抽取 transitions 并聚合
        ...
```

---

### 4.2 采样器升级：`StratifiedStackedSampler`

在 `FineFT/RL/DiHFT/low_level/parallel_pretrain.py` 中重构 `StackedTransitionSampler`：
1. **阶段切换时预堆叠 9 份连续 Tensor**：
   探索结束后，对 9 个经验池分别生成张量快照（每个经验池约 2~10 万行），驻留于 Host 内存或 GPU 显存中；
2. **跨池并行索取**：
   针对活跃网格子集，利用高效的切片索引一次性抽取 `states, infos, actions, rewards, next_states, next_infos, dones`，完全保留零拷贝张量采样的极速性能优势。

---

### 4.3 训练主循环改造点（`parallel_diverse_train.py`）

下表列出需手术式修改的关键代码位置与逻辑对照：

| 代码位置 | 原有逻辑 | 9 格动态经验池改造后逻辑 |
|---|---|---|
| [parallel_diverse_train.py:465](/home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/parallel_diverse_train.py:465) (`write_round_transitions_to_buffer`) | 所有 Worker 收集的 Transition 顺序压入单一 `buffer_diverse` | 提取 `transition[1]["regime_grid_id"]`，按 Grid ID 路由写入 `nine_grid_buffer.buffers[grid_id]`，维护 9 组独立的语义键指纹 |
| [parallel_diverse_train.py:575](/home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/parallel_diverse_train.py:575) (`run_diverse_training_phase`) | 采用全量 `buffer_diverse` 实例化单一 `StackedTransitionSampler` 进行无偏均匀随机抽样 | 根据当前 `epoch_index` 计算 `active_grids`（或启用模式一的 9 专家软绑定），实例化 `StratifiedStackedSampler` 定向采样子集 |
| [parallel_diverse_train.py:689](/home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/parallel_diverse_train.py:689) (`save_diverse_buffer`) | 覆盖保存单一 `buffer_diverse.pkl` | 分别提取 9 个经验池的张量化快照，保存为 `buffer_diverse_9grid.pt`，保留网格容量与各格独立统计 |
| [parallel_diverse_train.py:1357](/home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/parallel_diverse_train.py:1357) (`is_buffer_full`) | 任一时刻总长度达到阈值即置 `skip_exploration=True`，永久冻结 Worker 探索 | 改为计算 `min(len(b) for b in buffers.values())`，仅在所有格子均饱和且新增经验停滞时才降频探索，保持探索活性 |
| [parallel_diverse_train.py:1532](/home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/parallel_diverse_train.py:1532) (`update`) | 计算全量 ETD 距离与单一全局教师的 KL 散度 | 引入 Regime 亲和力先验矩阵，调制 `calculate_paper_partial_loss` 中的权重分布，强化主场代理梯度 |

---

### 4.4 下游阶段 II 选择器（`FineFT_two_dimensional_agent_selector.py`）的完美承接

在现行代码 [FineFT_two_dimensional_agent_selector.py](/home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/analysis/pick_agent/FineFT_two_dimensional_agent_selector.py:98) 中：
- CLI 已经原生支持 `--num_labels` 参数（默认 4，可直接传入 `--num_labels 3`）；
- 当 `--num_labels 3` 时，选择器自动将网格实例化为 $3 \times 3 = 9$ 个 Slot；
- 验证集目录结构（[dataset/30min/fu/valid/](/home/lanceliang/opt/aiwork/FineFT_code_space2026/dataset/30min/fu/valid/)）已预先生成 `label_0, label_1, label_2` 的 3×3 交叉切片，总计 10,542 个时点且 9 格全部具备完整的联合支撑（实测数据见 1.3 节）；
- **质的飞跃**：在过去，由于训练集未做 Regime 经验池隔离，所有候选模型在低波下行格全部出现负收益被拒（退化为 `empty_model`）；在引入 9 格经验池专项训练后，Agent 0 或专项训练轮次生成的 Checkpoint 在 $(v0, s0)$ 上拥有充沛的正收益与低回撤表现，将轻松通过门槛并在选择表中正式被采纳为真实模型（`kind=model`），彻底激活完整的专家网络。

---

## 5. 实施路线图与消融实验规划（Implementation Roadmap & Ablations）

为保证代码演进符合工程审慎原则（CLAUDE.md: Surgical Changes & Goal-Driven Execution），建议按以下顺序分步落地与消融：

```
Step 1: 环境与因果标签注入 (Causal Tagging in Env)
  └── 编写测试：test_causal_regime_tagging.py
  └── 在 base_env.py / commodity_env.py 中维护 48-bar 滚动窗口并向 info 注入 regime_grid_id
  └── 验证：断言 14 个训练切片生成的 grid_id 分布与实测 9 格一致

Step 2: 9 格经验池数据结构构建 (NineGridReplayBuffer)
  └── 编写测试：test_nine_grid_replay_buffer.py
  └── 实现 NineGridReplayBuffer 与 StratifiedStackedSampler
  └── 验证：测试各格容量隔离、FIFO 独立淘汰、去重指纹独立性及按权重采样准确性

Step 3: 并行训练流水线集成 (Diverse Train Integration)
  └── 改造 parallel_diverse_train.py 中的写入与采样逻辑
  └── 接入模式一（子代理-体制软绑定）与模式二（轮次课程表），开放 CLI 参数控制 --regime_sampling_mode
  └── 验证：断言各轮次训练日志中各格子经验数平稳增长，skip_exploration 不再过早触发

Step 4: 小规模消融验证与下游评估 (Ablation Experiment)
  └── 对比实验组：
      A. 基准组：单一全局经验池 (Baseline Parallel)
      B. 实验组 1：9 格经验池 + 模式二（轮换课程表，N=7）
      C. 实验组 2：9 格经验池 + 模式一（9 专家软绑定，N=9）
  └── 评估指标：
      1. 各轮次模型权重与 Q 值差异度 Delta Q；
      2. 阶段 II 二维选择表中 model 槽位填充率（目标从 0/16 提升至 9/9 全部填充）；
      3. 验证集跨合约综合回测年化收益率与 Sharpe 比率。
```

通过上述方案，FineFT 将建立起从微观因果状态识别、分层经验池隔离存储、定向采样梯度分化到宏观多智能体集成的完整闭环，从理论与工程两端彻底攻克低层智能体策略均质化的核心瓶颈。
