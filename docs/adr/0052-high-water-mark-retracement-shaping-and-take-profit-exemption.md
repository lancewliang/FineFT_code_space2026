# 0052. 高水位浮盈回撤奖励塑形与保盈平仓换手豁免架构

为根治强化学习累积折扣回报对“资产与利润路径无感（Path Insensitivity）”导致底层 Agent 在波峰大过山车行情中眼睁睁回吐巨额浮盈（如 `fu2505` 回吐 51.4% 历史浮盈）的数学缺陷，同时化解 ADR 0050 非对称换手惩罚在上涨冲顶时将顺势平仓误判为逆势早退（施加 6 倍惩罚）造成的“死扛持仓无惩罚、主动平仓遭重罚”的相克困境，我们决定确立一套原生的强化学习马尔可夫增广奖励塑形与保盈平仓豁免架构，并与 ADR 0051 构筑“软硬双层深度防御体系”。

## Status

accepted

## Context

在 2026-10-08 针对燃料油长周期主力合约（`fu2505` 等）的实证诊断与数学根因审查中，确认了以下深层矛盾：
1. **线性累积期望回报的路径盲区**：强化学习优化目标 $J(\pi) = \mathbb{E}\left[\sum_{t=0}^T \gamma^t r_t\right]$ 对回报累加过程完全线性且对路径形态无感。稳健单调上升（$0 \to 100 \to 200 \to 340$）与剧烈坐电梯（$0 \to 300 \to 700 \to 340$）的总回报完全一致。环境从未对“自历史最高浮盈大幅回吐”扣减过惩罚项，导致策略没有动力在见顶回落初期主动保盈离场；
2. **ADR 0050 逆势早退惩罚的过度钝化副作用**：ADR 0050 为防止大牛市中频繁折腾下车，对顺势平仓动作（如多头平仓）施加了 6 倍基础换手惩罚（`adverse_turnover_penalty`）。这导致当行情见顶回踩时，Agent 平仓要立刻扣除 6 倍名义货值虚拟惩罚，而死扛持仓却无需支付任何额外换手罚款，模型被反向规训为“宁可死扛坐电梯、绝不主动平仓”；
3. **离线 DP 专家的状态维度爆炸制约**：离线 DP 专家表（`futures_util.py: create_optimal_q_table`）是基于当前离散仓位（3~9 维）的一步反向递归。若在 DP 中强行引入历史高水位与回撤追踪，状态空间需离散化展开，维度膨胀数百倍并破坏矩阵向量化求解；
4. **效用函数优化的工程不可行性**：直接优化非线性风险敏感指标（如微分索提诺比率或 CVaR 约束）破坏了标准贝尔曼最优性原理（Bellman's Principle of Optimality does not hold for variance/Sortino），必须重构为分布式价值网络（QR-DQN）或策略梯度，工程代价过于沉重。

## Considered Options

- **选项 1：纯效用函数 / CVaR 风险敏感优化**：
  *缺陷*：推翻现有的 `Weighted_Contexts_DQN` 与 GPU 显存直存经验回放池，破坏贝尔曼标量可加性，重构成本极大。
- **选项 2：DP 专家状态网格离散化增广**：
  *缺陷*：DP 求解状态空间膨胀 $100 \sim 1000$ 倍，导致多进程预计算内存暴增且耗时恶化数十倍。
- **选项 3：仅依靠 ADR 0051 执行层 Tier 4 移动追踪止盈外挂硬规则**：
  *缺陷*：底层神经网络本身仍处于病态激励下，未能自主学出回撤保盈行为，在未受外挂规则干预的场景下仍会严重回吐。
- **选项 4：马尔可夫增广高水位回撤奖励塑形 + 保盈平仓豁免 + 双层防御架构（选中）**：
  1. **单步环境高水位回撤二次型流血惩罚 (Holding Bleed Reward Shaping)**：
     - 在 `base_env.py` 中追踪单笔持仓的历史最高收益率 $R_{\text{max}} = \max_{\tau \le t} R_\tau$；
     - 激活门槛：$\theta_{\text{profit\_min}} = 0.08$（+8% 标的浮盈）；
     - 允许回撤死区：$\theta_{\text{allow}} = 0.15$（15% 回撤）；
     - 回撤惩罚函数：
       $$\text{Retracement}_t = \frac{R_{\text{max}} - R_t}{R_{\text{max}}}$$
       $$\text{penalty}_{\text{dd}} = \lambda_{\text{dd}} \cdot \max(0, \text{Retracement}_t - \theta_{\text{allow}})^2 \times (| \text{position}_t | \times P_t)$$
       基准系数设为 $\lambda_{\text{dd}} = 0.01$；
     - 扣费时钟：仅在智能体维持持仓（Hold Long / Hold Short）时按步持续扣罚，一旦执行平仓（Flat），当步惩罚归零且持仓终结。在贝尔曼反向传播中使 $Q(s, \text{Hold}) \ll Q(s, \text{Flat})$；
  2. **状态空间马尔可夫原位闭环 (`trading_info[2]`)**：
     - 保持现有 4 维输入契约 `[position_exposure, single_holding_return_rate, single_holding_max_drawdown, duration_norm]` 不变；
     - 将原有的全账户资产回撤原位校准为**即时浮盈回吐率 (Instant Profit Retracement Ratio)**；
     - 门槛截断式对齐：当未持仓或 $R_{\text{max}} < \theta_{\text{profit\_min}}$ 时，特征恒定输出 `0.0`；一旦激活，输出实时的 $\text{clip}(\text{Retracement}_t, 0.0, 1.0)$，实现状态观测与奖励扣罚的因果马尔可夫对齐，且无需修改下游网络输入维度；
  3. **保盈平仓换手豁免机制 (Take-Profit Penalty Exemption)**：
     - 宽口径触发：单笔持仓历史最高浮盈达到门槛（$R_{\text{max}} \ge \theta_{\text{profit\_min}}$）的平仓动作（Long $\to$ Flat 或 Short $\to$ Flat），认定为正当的获利退避（Take-Profit De-risking）；
     - 费率降级：不触发 ADR 0050 的 6 倍 `turnover_adverse_ratio` 惩罚，降级结算为 1 倍基础换手费率（`turnover_base_rate`），既保障保盈平仓不被抑制，又防止单步高频抖动；
  4. **离线 DP 专家表的解耦边界**：
     - DP 专家表保持仅受 ADR 0051 体制开仓硬锁约束，不引入回撤状态膨胀；回撤塑形专注于在线环境与 Stage I 多样化探索（Diverse Training）经验回放池，由 Agent 在真实因果采样中自主学得保盈平仓策略；
  5. **软硬双层深度防御架构 (Two-Tier Defense in Depth)**：
     - **Tier 1 软性自适应防线 (Native RL)**：在 15%~25% 回撤区间内，Agent 承受二次型流血惩罚，自主依据微观盘口流在最佳时机主动平仓；
     - **Tier 2 硬性物理底线 (ADR 0051 Execution Guard)**：若极端情况下 Agent 回撤至 25% 仍未离场，高层执行层 Tier 4 引擎物理强平兜底，确保全生命周期无死角锁死利润。

## Consequences

- **打破强化学习对利润路径的数学盲区**：将“高位回撤持续持仓”转化为剧烈的负向单步收益，使得贝尔曼方程自发学出“见顶回落主动平仓”的最优不动点；
- **消除奖励惩罚相克**：保盈平仓豁免机制彻底解放了在单边大势中被 6 倍逆势换手惩罚锁死的多头持仓，使 Agent 敢于在冲顶后从容止盈离场；
- **状态空间与网络架构零破坏**：原位校准 `trading_info[2]` 实现了纯马尔可夫对齐，下游所有 Q-Net、VAE 路由器与张量接口无需修改尺寸；
- **双层防御无缝协同**：底层原生算法自适应退出（15%~25%）与高层规则确定性兜底（25%）分工明确，形成高度自洽的软硬风控闭环。
