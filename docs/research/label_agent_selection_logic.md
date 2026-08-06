# DiHFT/FineFT 框架中 Label 选择 Agent 逻辑与 12 大策略原型全量升级规范文档

**版本**: v5.0 (12大策略原型与全量架构规范版)  
**更新日期**: 2026-08-06  
**研究依据**: 代码库 primary sources、实测运行轨迹数据及 140 维特征工程定义

---

## 1. 架构概览与核心结论

在 FineFT / DiHFT 架构中，“Label 选择 Agent”经过对 `fu2409` 和 `fu2505` 实测运行轨迹的深采样拆解，确认了原有架构的核心缺陷：**无监督 VAE 状态似然匹配仅代表“几何特征长得像”，不代表“策略方向对且能赚钱”**。由于缺乏 Agent 策略原型打标与动作语义一致性防错，门控在主升浪中把暴涨行情分配给了“均值回归/高位做空型”的 Agent 4，导致 247 次逆势满仓做空并招致巨额亏损。

本规范整合了全新的 **“Label 原生语义 + 12 大 Agent 策略原型档案库 + PnL 记忆追踪 + 语义硬隔离 + 多因子打分排序”** 四位一体完整解决方案。

---

## 2. 12 大 Agent 策略原型分类体系 (Strategy Archetypes Taxonomy)

根据 `state_features.npy` 中定义的 140 维特征（如 `ma_192_origin`, `parkinson_volatility_16`, `limit_depth_imbalance_ratio_5` 等），低层训练出的 Agent 可被归类为以下 **12 大策略原型**：

| 策略分类 | 策略原型代码 | 核心交易逻辑与依赖指标 | 适用行情体制 |
| :--- | :--- | :--- | :--- |
| **一、 动量与趋势类** | **1. `Trend_Following`** | 顺价格斜率与均线建仓，盈亏比高 (`ma_192_origin`, `price_velocity_10m`)。 | 单边强趋势 |
| | **2. `Momentum_Acceleration`** | 抓二阶加速度爆破段，建仓极快 (`roc_2_std_norm_origin`, `ask1_price_log_return_2`)。 | 主升浪 / 主跌浪 |
| **二、 均值回归类** | **3. `Mean_Reverting`** | 超买超卖区反向建仓，胜率高 (`rsv_192_std_norm_origin`, `bollinger_lower_96_origin`)。 | 区间震荡 (在强牛市中易爆亏) |
| | **4. `Fade_Breakout`** | 猎杀假突破，量能未跟上时反向建仓 (`imax_48_origin`, `buy_spread_oe_max`)。 | 假突破 / 诱多诱空 |
| **三、 盘口微观结构类**| **5. `Order_Flow_Imbalance`**| 利用 L2 盘口买卖挂单失衡抢单 (`limit_depth_imbalance_ratio_5`)。 | 盘口流动性失衡期 |
| | **6. `Scalping_Grid`** | 高频双向开平仓，捕获微小买卖价差 (`wap_balance`, `ask1_size_n`)。 | 微观高频低波期 |
| **四、 持仓量驱动类** | **7. `Open_Interest_Drive`** | 增仓确认为真突破时跟进 (`prev_15_day_open_interest_change`, `oi_change_rate_norm_10m`)。 | 主力增仓破位期 |
| | **8. `Volume_Price_Divergence`**| 价格创新高但量能耗尽时提前布局反向单 (`price_oi_vol_interaction_10m`)。 | 趋势衰竭见顶/底期 |
| **五、 跨合约套利类** | **9. `Calendar_Spread_Arbitrage`**| 跨月价差偏离均值时买强卖弱 (`cm_m1_m2_relative_price_spread`, `cm_main_sub_log_price_ratio`)。 | 跨期价差均值回归期 |
| **六、 时段与事件类** | **10. `Volatility_Breakout`**| 波动率压缩后捕捉放量突破 (`parkinson_volatility_16`, `garman_klass_volatility_2`)。 | 波动率暴增突破期 |
| | **11. `Session_Time_Pattern`**| 利用开盘/收盘 30 分钟统计规律 (`is_opening_30m`, `is_closing_30m`)。 | 特定日内交易时段 |
| | **12. `Risk_Averse_Neutral`**| 趋向于 0 仓位观望（`micro_action = 4`），追求低回撤 (`rolling_volatility_24`)。 | OOD 离群高风险期 |

---

## 3. 升级架构四大核心组件

### 3.1 第一组件：Agents 策略档案库设计 (Agent Archetype Profile)
- **原理**:
  为每个 Label（`label_0` ~ `label_4`）建立专属的 Agent 策略档案智囊团。一个 Agent 在一个 Label 中可能同时归属于 1~3 种原型。总结每个 Label 维护包含 12 大策略原型的备选智囊团，每个策略原型仅挑选历史收益率最高的那个 Agent 进入档案库。

---

### 3.2 第二组件：PnL 记忆追踪器 (PnL Memory Tracker)
- **原理**:
  维护全局 `AgentPnLTracker` 动态记忆池，记录各 Agent 在环境中的近 50 步滚动收益 $r_{k, \tau}$ 与胜率，计算得分：
  $$S_{\text{pnl}}(k) = 0.7 \times \sum_{\tau \in 50} r_{k, \tau} + 0.3 \times \left(\text{WinRate}_k - 0.5\right)$$

---

### 3.3 第三组件：Candidate Generator 组件 (候选池生成与三重硬隔离)
- **输入**: 锁定门控 Label $L^*$ 以及 Label $L^*$ 智囊团中候选 Agents 产生的拟执行动作集合。
- **校验逻辑**:
  1. **动作与 Label 固有语义一致性校验 (Semantic Guard)**:
     检查 Agent 拟吐出的动作是否在 Label $L^*$ 的原生允许动作范围内（例如：对于 `label_4` 强多头，严禁吐出 `0~3` 做空动作；若 Agent 吐出做空动作，一票否决剔除！）。
  2. **近端 PnL 回撤安全界限校验**:
     剔除近 50 步滚动回撤 $> 20\%$ 的 Agent。
- **输出**: 生成安全候选 Agent 集合 $\mathcal{C}_t = \{k_1, k_2, \dots\}$。

---

### 3.4 第四组件：Meta Router 组件 (动态多因子打分与熔断)
- **打分规则**:
  对候选集 $\mathcal{C}_t$ 内的合格 Agent 进行软打分排序：
  $$\text{Score}(k) = 0.5 \times S_{\text{vae}}(k) + 0.5 \times S_{\text{pnl}}(k)$$
- **单合约熔断保护 (Circuit Breaker)**:
  当单合约累计最大回撤率 $> 15\%$ 时，强行切断路由，降级为规则平仓 (`macro_action = 5`)。
- **输出**: 选中综合得分最高的 `selected_agent_index` 以及对应的 `final_action`。

---

## 4. 标准四步路由流程 (Routing Pipeline Standard)

```text
[ 时间步 t 的 140维特征 s_t & 交易信息 info ]
                       │
                       ▼
 [ 步骤 1: 门控锁定最高分 Label ] ──▶ 结合 VAE 似然与 PnL 记忆选出置信度最高的 Label L*
                       │             (不能识别的离群 Label 依然保持原有的降级防守策略 macro_action=5)
                       ▼
 [ 步骤 2: 调取 Label L* 档案池 ] ──▶ 仅调取 Label L* 智囊团的所有候选 Agents 获取拟执行动作
                       │
                       ▼
 [ 步骤 3: Candidate Generator ] ──▶ 进行动作与 Label 固有语义一致性及 PnL 20% 回撤校验，生成安全候选池 C_t
                       │
                       ▼
 [ 步骤 4: Meta Router ] ─────────▶ 在 C_t 内计算 Score = 0.5*VAE + 0.5*PnL，检查单合约 15% 熔断，
                                     输出最高分 selected_agent_index 与 final_action
```

---

## 5. 方案实施与验证目标

1. **彻底杜绝逆势做空**:
   在 `fu2505` 的暴涨主升浪中，即使选中了 `label_4`，Candidate Generator 在步骤 3 会将 Agent 4 拟吐出的做空动作（247 次逆势做空）直接拦截一票否决，彻底关死爆亏出口。
2. **丰富策略多样性**:
   通过从 12 大策略原型中为各 Label 智囊团择优，显著提升了算法在趋势、震荡、盘口失衡等不同细分市场形态下的适应能力。
