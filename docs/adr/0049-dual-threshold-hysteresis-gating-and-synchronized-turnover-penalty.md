# 0049. 门控迟滞带与强化学习全链路自洽换手惩罚架构

为了根治高层 VAE 路由临界单阈值抖动引发的“进场-强平-再进场”恶性循环，以及低层强化学习 Agent 缺乏持仓惯性导致的超高频无效翻转（在燃料油 10 分钟验证集上产生 1,696 次调仓，吞噬了 96.87% 的毛利润），我们决定在高层 VAE 门控中确立**非对称双阈值迟滞带（Hysteresis Band）**并将其平仓衰减比率纳入 Optuna 自动寻优，同时在低层 Agent 的离散动作转移中确立**名义货值比例换手惩罚项（Turnover Penalty）**并同步贯通至动态规划专家 Q 表（`create_optimal_q_table`）与环境单步奖励（`Base_Env.step`），实现全链路波段持仓惯性自洽。

## Status

accepted

## Context

在 DiHFT 算法针对燃料油 10 分钟验证集（`fu/10min_parallel`）的 200 次 Optuna 寻优产出的全局最优解（Trial 0）回测诊断中发现：
1. **毛利丰厚但净利严重被吞噬**：策略实际波段方向预测捕获的毛利润高达 **58,410.00 元（毛收益率 +81.1%）**，远超 5x B&H 基准（18,580 元，+25.8%）；但累计扣除手续费（26,008.87 元）和订单簿滑点（30,571.30 元）共 **56,580.16 元**，导致最终账户净利润仅存 **1,829.84 元（+2.54%）**；
2. **高层门控临界抖动破坏趋势**：现有 `AbsoluteThresholdGating` 采用单一阈值 $T_{\text{vol}}$ 与 $T_{\text{slope}}$（如 0.43 与 0.39）。当 VAE 得分在临界点微幅波动时，策略反复将持仓踢平（全期触发 `DEFENSIVE_RULE_CLOSE` 高达 6,752 步，占比 22.25%），几十分钟后又重新追高进场，产生海量无谓摩擦；
3. **低层 Agent 缺乏持仓惯性**：平均持仓周期仅 15.7 根 K 线（约 2.6 小时），单边大牛市（如 `fu2503` 上涨 +36.7%）被拆碎为 245 次微观日内调仓，甚至在牛市回调中频繁反手做空；
4. **约束前提**：本阶段保持 `action_persistence` 和交易费率基准不变，完全依靠高层决策迟滞与低层奖励塑形从算法根本上抑制过度换手。

## Considered Options

- **选项 1：固定常数迟滞带，低层 Agent 仅在环境端扣罚**：
  将平仓衰减比率固定为常数 0.65，不纳入 Optuna；换手惩罚仅在 `Base_Env.step` 中扣减，DP Q 表（`create_optimal_q_table`）不修改。
  *缺陷*：DP 专家轨迹依然充斥微幅高频翻转，与强化学习环境的惩罚目标相违背，导致专家 Warmup 预训练与强化学习迭代目标脱节割裂。
- **选项 2：完全解耦四独立门控阈值**：
  显式设定 $T_{\text{vol}}^{\text{enter}}, T_{\text{vol}}^{\text{exit}}, T_{\text{slope}}^{\text{enter}}, T_{\text{slope}}^{\text{exit}}$ 四个独立参数供 Optuna 搜索。
  *缺陷*：超参维度增加且极易采样出 $T^{\text{exit}} \ge T^{\text{enter}}$ 的非法无序解，搜索空间效率大幅降低。
- **选项 3：双阈值迟滞带 Optuna 联合寻优 + 全链路自洽换手惩罚（选中）**：
  1. **高层门控迟滞带（Hysteresis Band）**：
     - 保留 $T_{\text{vol}}^{\text{enter}}$ 与 $T_{\text{slope}}^{\text{enter}}$ 作为开仓高置信度门槛；
     - 引入平仓衰减比率 $\alpha_{\text{exit}} \in [0.50, 0.80]$，动态生成退出防守阈值：$T^{\text{exit}} = \alpha_{\text{exit}} \cdot T^{\text{enter}}$；
     - 空仓时（`current_position == 0`）严格执行 $T^{\text{enter}}$；持仓时（`current_position != 0`）宽容放行，仅当 VAE 评分跌破 $T^{\text{exit}}$ 时才触发防御性退出，彻底根除临界洗盘损耗；
     - 将 $\alpha_{\text{exit}}$（参数名 `params_hysteresis_exit_ratio`）纳入 Optuna 200 轮搜索空间，自适应捕捉不同标的与频度下的最优宽容容差；
  2. **极端破位抢先防御保留**：
     - 一旦确认跌破 $T^{\text{exit}}$，说明市场已脱离常态并进入深度 OOD/剧烈异动，立即打断当前持仓并清仓（`DEFENSIVE_PREEMPTION`），兼顾波段持仓与尾部避险；
  3. **名义货值换手惩罚（Turnover Penalty）**：
     - 在低层 Agent 中引入比例换手惩罚：$\text{Penalty} = \lambda_{\text{turnover}} \cdot |\Delta \text{position}| \cdot P_{\text{mark}} \cdot \text{ContractUnit}$，默认 $\lambda_{\text{turnover}} = 0.0002$（2 bps，约 6 元/手）；
     - **纯 Reward Shaping 定位**：惩罚仅作用于单步强化学习回报标量 $r_t$，不扣减物理账本 `wallet_balance`、`total_asset_history` 与强平逻辑，维护资产核算的客观真实性；
  4. **动态规划 Q 表全链路自洽（DP Q-Table Synchronization）**：
     - 在 `create_optimal_q_table()` 的贝尔曼反向更新中，同步在单步转移收益扣除相同换手惩罚项：
       $$r_{\text{DP}}(s, a, s') = \Delta \text{Balance} - \text{Penalty}_{\text{turnover}}$$
     - 确保 DP 离线最优路径天然具备持仓惯性，为低层 Q-Net 预训练注入自洽的高质量波段示范。

## Consequences

- **正面收益**：
  - **终结临界假平仓**：迟滞带将有效消除 6,700 余步中的绝大多数假突破强平，使持仓能够完整吃满日内 2~6 小时的主要波段；
  - **换手大幅收敛**：换手惩罚抑制了 Agent 对 0.1~0.2 个点微利波动的无意义翻转，预期换手次数压降 60% 以上；
  - **留存超额利润**：在 5.8 万元的毛利盘面上，预期可挽回 3.5 万元以上的摩擦成本，使净收益率从 +2.54% 质跃至 +30%~+50%；
  - **专家引导与环境完美对齐**：DP 生成的专家轨迹与实盘环境的目标函数高度统一，避免 Warmup 后的策略认知震荡。
- **权衡与约束**：
  - 在突发的快速中枢位移中，由于持仓出场阈值 $T^{\text{exit}}$ 调低，策略较单阈值模式会有微小的延后平仓反应，但该延迟所带来的微小浮亏远低于频繁来回翻转的确定性摩擦成本。
