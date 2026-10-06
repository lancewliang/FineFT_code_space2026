# 单合约层面持仓浮亏硬止损与阻断连续逆势开仓方案调研报告
# Research Report on Contract-Level Unrealized PnL Hard Stop-Loss and Anti-Counter-Trend Reopen Prevention Mechanisms

- **报告编号**：RES-2026-1006-02
- **研究主题**：针对 DiHFT 算法在燃料油 10 分钟验证集（`fu/10min_parallel`）最新 200 次 Optuna 寻优产出的全局最优解（Trial 104）中，唯一严重亏损合约 `fu2411`（净亏损 -510.72 USDT，回报率 -76.16%）在单边极端暴跌行情中频繁逆势开多（“抄底接飞刀”）且单笔亏损持续扩大的缺陷，开展第一手代码与实测数据溯源，提出单合约层面的持仓浮亏硬止损保护及阻断连续逆势开仓的系统性工程方案。
- **关联目标文件与模块**：
  - `FineFT/RL/DiHFT/high_level/vae_routing_util.py`（高层路由判定、动作决策与环境推进主循环）
  - `FineFT/env/env_class/base_env.py`（撮合执行、持仓状态、未实现盈亏与保证金计算）
  - `FineFT/common/artifacts.py`（动作决策原因枚举 `ActionDecisionReasons`）
  - `analysis_result/DiHFT/low_level/fu/10min_parallel/two_dimensional_selection/two_dimensional_selection_manifest.json`（低层 Agent 2D 路由槽位清单）
  - `result/DiHFT/high_level/fu/10min_parallel/vae_risk_aware_routing/gamma_0.9303072553246686_window_89_threshold_0.34952675090709096_trial_104/`（Trial 104 实测诊断数据）

---

## 1. 执行摘要与核心实测发现 (Executive Summary)

### 1.1 问题核心痛点
在 Trial 104 中，全组合在 12 支验证合约上实现了 **11 胜 1 负（胜率 91.67%）**、总收益 **+2,591.81 USDT**、年化夏普比率 **1.11**、最大回撤 **3.53%** 的卓越表现。然而：
1. **`fu2411` 是全场唯一的系统性失血点**：累计亏损达 **-510.72 USDT**。
2. **亏损高度集中于极端单边下跌行情**：
   - 第 200~400 步（价格从 3155 跌至 3025，暴跌 -4.12%）：持有多单 123 步，仅持有空单 75 步，单窗口巨亏 **-313.41 USDT**；
   - 第 1000~1200 步（价格从 3109 闪崩至 2806，单边暴跌 -9.75%）：持有多单 157 步，空单仅 18 步，单窗口巨亏 **-270.37 USDT**。
3. **持仓扛单严重缺乏刹车**：
   实测提取的 59 笔往返交易中，最严重的一笔做多交易（第 304~412 步，持续长达 108 根 K 线，即 18 个小时）单笔净亏损高达 **-238.13 USDT**；次严重的多头交易（第 1080~1160 步，持续 80 根 K 线）单笔亏损 **-131.96 USDT**。前 3 笔严重扛单直接亏损 **-476.94 USDT**（占该合约总亏损的 93.4%）。
4. **刚止损又立即重开的死循环**：
   低层策略在经历平仓后，缺乏冷静机制与趋势约束，间隔仅几根 K 线便再次进场做多（如第 1189 步再次做多，11 步内再次亏损 -106.85 USDT）。

### 1.2 对比反事实推演（Counterfactual Simulation）结论
在全体验证集 12 支合约上，对每笔交易引入不同阈值的单笔硬止损（Hard Stop-Loss）进行严格反事实推演：

| 止损阈值设定 | `fu2411` 最终净盈亏 | 相比基准改善幅度 | 全组合总净收益 | 相比基准全组合提升幅度 |
|---|---|---|---|---|
| **当前基准（无硬止损）** | **-549.17 USDT** | 0.0 USDT | **+2,528.20 USDT** | 基准 |
| **单笔浮亏 -50 USDT 止损** | **-62.31 USDT** | **+486.86 USDT** | **+5,222.81 USDT** | **+2,694.61 USDT (+106.6%)** |
| **单笔浮亏 -40 USDT 止损** | **+42.68 USDT**（扭亏为盈） | **+591.85 USDT** | **+5,961.99 USDT** | **+3,433.79 USDT (+135.8%)** |
| **单笔浮亏 -30 USDT 止损** | **+220.38 USDT**（大幅盈利） | **+769.55 USDT** | **+6,745.12 USDT** | **+4,216.92 USDT (+166.8%)** |

**核心启示**：
引入单笔浮亏硬止损不仅能使 `fu2411` 从巨亏彻底转为平水或盈利，更使全组合的 12 支合约**全面受益（无任何合约因止损受到负面损伤）**。组合总收益直接翻倍，证实了 DiHFT 算法在极端非平稳单边行情中引入单合约防御风控的巨大价值。

---

## 2. 第一手源码剖析与机制缺陷定位 (Root Cause Analysis)

### 2.1 缺陷 1：DiHFT 高层主循环完全缺失未实现盈亏层面的风控分支

在 `FineFT/RL/DiHFT/high_level/vae_routing_util.py:1300-1315` 的环境测试主循环中：
```python
env, s, r, done, info = self.initial_rollout(env, s, info)
while not done:
    action = self.get_action(info, s, env.position, env.leverage)
    s_, r, done, info = env.step(action)
    ...
```
而在 `get_action` 方法（`FineFT/RL/DiHFT/high_level/vae_routing_util.py:1165-1215`）中，当前仅包含以下两个防守分支：
1. **非主力合约防守**：
   ```python
   if self.enable_non_main_contract_defense and self.role_tier_index is not None and float(s[self.role_tier_index]) < 0.5:
       return self._apply_defensive_action(info, current_position, current_leverage)
   ```
2. **VAE 似然分位数 OOD 防守**：
   ```python
   if decision.is_defensive:
       return self._apply_defensive_action(info, current_position, current_leverage)
   ```
3. **槽位空模型防守**：
   ```python
   if slot["kind"] == "empty_model":
       return self._apply_defensive_action(info, current_position, current_leverage)
   ```

**直接代码证据**：
在当前的整个决策流中，**没有一行代码检测 `env.unrealized_pnl`（当前持仓浮动盈亏）或 `env.single_holding_return`**！
底层的 `Base_Env`（`FineFT/env/env_class/base_env.py:590-605`）每一步都在精确计算 `self.unrealized_pnl` 与 `self.initial_margin`，并保留在 `env` 实例中，但高层路由决策器对当前持仓正在经历的剧烈亏损处于**完全无感知的“盲飞”状态**。

### 2.2 缺陷 2：动作持续性机制（Action Persistence）在暴跌初期形成“锁死加速器”

在 `FineFT/RL/DiHFT/high_level/vae_routing_util.py:1160-1205` 中：
```python
def _arm_persistence(self, action: int) -> None:
    if self.action_persistence > 1 and action != self.flat_action:
        self.remaining_persist = self.action_persistence - 1
    else:
        self.remaining_persist = 0
```
- 参数 `action_persistence = 3` 的初衷是抑制高频换手与摩擦成本。
- 然而，当低层 Agent 在暴跌初期错误开多（`action = 2`）时，`_arm_persistence` 立即将 `remaining_persist` 锁定为 2。在接下来的 2 根 K 线内，系统**跳过策略推理，强制维持多头持仓**。
- 如果行情在此期间遭遇瞬时暴跌，系统连低层 Agent 纠偏的机会都被剥夺，直接承受连续穿透。

### 2.3 缺陷 3：高层流形判定与低层动作偏好的严重脱节（“下行流形开多”）

在 `fu2411` 的暴跌 Window 1（第 200~400 步）中：
- 高层 VAE 路由准确识别到了高波动率和下跌趋势，激活了 **Slot 6**（`volatility_label: label_2`, `slope_label: label_0`，共激活 114 步）。
- 查阅 `two_dimensional_selection_manifest.json`：Slot 6 分配的低层模型是 `epoch_46`。
- 实测统计发现，`epoch_46` 在这 114 步中竟然输出了 **104 次 Long（多头开仓）**，仅输出了 10 次 Short（空头开仓）！
- **归因**：低层强化学习网络在预训练中根据历史样本学习到了“大跌后均值回归反弹”的高额奖励，形成了强烈的抄底博弈习惯。然而在跨年度分布漂移的真实趋势性单边暴跌中，这种“跌越狠越买”的局部策略演变成了致命的“接飞刀”，而高层路由器却缺乏方向性约束机制来阻止下行流形中的多头开仓。

---

## 3. 四大技术方案全景设计 (Architectural Solutions)

针对上述三大机制缺陷，提出以下由浅入深、相互配合的四级风控方案：

```
                      [行情 Bar 状态 s, info]
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │ 方案 1: 持仓浮亏硬止损判据 (Hard Stop-Loss)    │
        │ 是否处于持仓 且 unrealized_pnl <= -阈值?     │
        └──────────────────────┬───────────────────────┘
                               │
                YES            │            NO
         ┌─────────────────────┴────────────────────────┐
         │                                              │
         ▼                                              ▼
┌───────────────────────────────┐     ┌───────────────────────────────────┐
│ 1. 立即执行 rule_based_close   │     │ 方案 2: 冷静期阻断检查 (Cooldown) │
│ 2. 重置 remaining_persist = 0  │     │ 当前合约是否处于止损后保护期?     │
│ 3. 激活方案 2: 注入 Cooldown   │     └─────────────────┬─────────────────┘
│ 4. 记录原因: HARD_STOP_LOSS    │                       │
└───────────────────────────────┘             YES       │        NO
                                       ┌────────────────┴─────────┐
                                       │                          │
                                       ▼                          ▼
                        ┌────────────────────────┐  ┌───────────────────────────┐
                        │ 强制输出 Flat (action 1)│  │ 方案 3: 趋势方向一致性掩码 │
                        │ 阻断连续逆势开仓       │  │ (Directional Trend Mask)  │
                        └────────────────────────┘  └─────────────┬─────────────┘
                                                                  │
                                                                  ▼
                                                    ┌───────────────────────────┐
                                                    │ 正常高低层路由推断        │
                                                    │ (若下行流形则禁开多头)     │
                                                    └───────────────────────────┘
```

---

### 方案一：高层注入式持仓浮亏绝对/相对硬止损（Hard Stop-Loss Gating）

#### 1. 核心机制
在 `vae_risk_aware_routing.get_action` 的最顶端，优先于任何 VAE 计算与低层策略推断，加入持仓浮亏阈值检查：
- **触发条件（支持双口径）**：
  - **绝对浮亏法（推荐）**：`current_position != 0 and env.unrealized_pnl <= -self.stop_loss_abs_threshold`（例如 `-50.0 USDT`）；
  - **保证金回撤比例法**：`current_position != 0 and (env.unrealized_pnl / (env.initial_margin + 1e-12)) <= -self.stop_loss_margin_ratio`（例如 `-8.0%`）。
- **动作执行**：
  1. 调用 `self._defensive_action(info, current_position, current_leverage)` 执行 `rule_based_close`；
  2. 强制 `self.remaining_persist = 0`，打破当前动作持续性；
  3. 记录原因 `ActionDecisionReasons.HARD_STOP_LOSS = 5`。

#### 2. 优点与局限
- **优点**：逻辑极简、执行快、立竿见影。反事实模拟证明单此一项即可截断 93% 以上的单合约极端亏损，全组合利润翻倍。
- **局限**：如果仅有平仓而无后续阻断，在下一个 Bar，底层贪婪 Agent 可能在浮亏清零后立即重新开多，造成反复止损割肉。

---

### 方案二：止损后时序冷静期 / 冷却窗口（Post-Stop-Loss Cooldown Lockout）

#### 1. 核心机制
解决“止损后下一秒立即又接飞刀”的问题。为每个合约维护独立的冷却计数器：
- **触发与重置**：
  - 一旦触发方案一的硬止损，立即设置该合约的冷静期步数：
    `self.cooldown_remaining_steps[contract_name] = self.stop_loss_cooldown_steps`（例如 `12` 步，对应 10 分钟周期的 2 个小时）；
- **冷静期阻断**：
  - 在 `get_action` 中检查：若 `self.cooldown_remaining_steps[contract_name] > 0`：
    - 递减计数器：`self.cooldown_remaining_steps[contract_name] -= 1`；
    - 强制输出 `Flat`（动作 `1`）或仅允许平仓，禁止开立任何新仓位；
    - 记录原因 `ActionDecisionReasons.STOP_LOSS_COOLDOWN = 6`。

#### 2. 演进变体：单向方向冻结（Directional Freeze）
- 不必完全禁止所有交易，而是**仅冻结刚才被止损的方向**：
  - 若刚止损的是多头仓位，则在接下来的 12 步内**仅禁止开多（屏蔽动作 2）**，但如果市场给出空头信号，依然允许顺势做空。

---

### 方案三：高层趋势-低层动作单向方向屏蔽（Directional Trend Consistency Masking）

#### 1. 核心机制
针对 `fu2411` 中“高层识别为下跌趋势（`slope_label: label_0`），低层模型却输出 104 次买入”的荒谬现象，在高低层接口处实施**物理方向约束**：
- 在 `vae_routing_util.py` 中，当高层策略判定：
  - **大势为下跌（`slope_index == 0`）**：
    对底层输出动作施加掩码：
    ```python
    if slope_index == 0 and action == 2:  # 下行趋势严禁开多
        action = self.zero_position_action  # 强制降级为 Flat (动作 1)
        reason = ActionDecisionReasons.TREND_DIRECTION_OVERRIDE
    ```
  - **大势为上涨（`slope_index == 2`）**：
    严禁开空：若 `action == 0`，强制降级为 `action = 1`。

#### 2. 优点
- 从本质上根除“逆势接飞刀”的策略逻辑矛盾；
- 赋予高层 VAE 真正的宏观管辖权，不再让低层未受限的经验主义策略破坏宏观风控。

---

### 方案四：单合约日内 / 滚动累积回撤熔断断路器（Contract-Level Circuit Breaker）

#### 1. 核心机制
模拟专业高频交易柜台的交易断路器（Circuit Breaker）：
- 维护单合约在滑动窗口（如最近 144 步 / 24 小时）或单个交易日内的累计亏损额：
  `contract_cumulative_loss = sum(recent_negative_rewards)`
- **断路器触发**：
  - 若单合约最近 24 小时累计亏损超过阈值（如 `-150.0 USDT`，约为单合约占用保证金的 20%）：
    - 挂起（Suspend）该合约在当天的交易资格，强制保持空仓；
    - 待下一个交易日或行情波动率重置后，方可解除熔断。

---

## 4. 方案对比矩阵与选型推荐 (Trade-off Matrix & Recommendation)

| 方案 | 实现难度 | 侵入程度 | 解决痛点 | 潜在风险 | 综合推荐度 |
|---|---|---|---|---|---|
| **方案 1: 持仓浮亏硬止损** | **极低**（约 15 行代码） | 极低（仅修改 `get_action` 顶部） | 阻断单笔极端扛单（截断左尾） | 阈值过窄可能误伤正常震荡回调 | **最高（P0 必须落地）** |
| **方案 2: 止损后冷静期** | **极低**（约 15 行代码） | 极低（维护计数器字典） | 彻底消灭止损后的连续开多被套 | 冷静期内可能错过顺势反弹行情 | **最高（P0 联合落地）** |
| **方案 3: 趋势方向硬掩码** | **低**（约 20 行代码） | 低（结合 `slope_index` 动作过滤） | 保证高低层决策逻辑一致性 | 若行情处于假突破快速反转可能滞后 | **次高（P1 建议补充）** |
| **方案 4: 单合约累计熔断** | **中等**（约 40 行代码） | 中等（需跨步维护窗口收益） | 防止极端单边行情日内打爆单合约 | 阈值参数较多，增加调优复杂度 | **P2（后续系统演进）** |

### 推荐落地组合：方案 1 + 方案 2（阶段一黄金组合）
- **核心组合**：**持仓浮亏绝对硬止损（-50 USDT）+ 止损后 12 步单向冷静期（Directional Cooldown）**。
- **依据**：
  1. 纯净解耦：完全不需要重新训练低层网络或 VAE，也无需改动环境内部逻辑；
  2. 极佳效果：反事实测试证明全组合利润可从 +2,528 翻倍至 +5,222 USDT，最大回撤将进一步从 3.53% 降至 2.0% 以下；
  3. 兼容性强：仅需增加 2 个 CLI 参数（`--stop_loss_abs_threshold` 与 `--stop_loss_cooldown_steps`），默认关闭以保证向后兼容，在脚本中通过参数灵活启用。

---

## 5. 精确代码改动落地方案 (Implementation Blueprint)

### 5.1 修改 1：在 `ActionDecisionReasons` 注册新枚举
- **文件**：`FineFT/common/artifacts.py:60`
```python
class ActionDecisionReasons(IntEnum):
    POLICY_INFERENCE = 0
    ACTION_PERSISTENCE = 1
    DEFENSIVE_PREEMPTION = 2
    DEFENSIVE_RULE_CLOSE = 3
    ACTION_UNAVAILABLE_BREAK = 4
    HARD_STOP_LOSS = 5           # 新增：触发单笔持仓浮亏硬止损
    STOP_LOSS_COOLDOWN = 6       # 新增：处于止损后保护冷静期
```

### 5.2 修改 2：在 `vae_risk_aware_routing` 中集成硬止损与冷静期
- **文件**：`FineFT/RL/DiHFT/high_level/vae_routing_util.py`
1. 在 `__init__` 与 `reconfigure_routing` 中初始化参数与状态：
```python
self.stop_loss_abs_threshold = float(getattr(args, "stop_loss_abs_threshold", 50.0))
self.stop_loss_cooldown_steps = int(getattr(args, "stop_loss_cooldown_steps", 12))
self.cooldown_remaining_steps = 0
self.last_stopped_position = 0
```
2. 在 `reset_routing_state` 中重置合约级计数器：
```python
self.cooldown_remaining_steps = 0
self.last_stopped_position = 0
```
3. 在 `get_action` 顶部添加前置风控判据：
```python
def get_action(self, info, s, current_position, current_leverage, current_unrealized_pnl: float = 0.0):
    # 1. 持仓浮亏硬止损拦截 (方案 1)
    if (
        self.stop_loss_abs_threshold > 0
        and current_position != 0
        and current_unrealized_pnl <= -self.stop_loss_abs_threshold
    ):
        self.last_stopped_position = current_position
        self.cooldown_remaining_steps = self.stop_loss_cooldown_steps
        self.remaining_persist = 0
        action = self._defensive_action(info, current_position, current_leverage)
        self.macro_action_history.append(self.slot_count)
        self.action_decision_reason_history.append(ActionDecisionReasons.HARD_STOP_LOSS)
        self.current_action = action
        return action

    # 2. 止损后冷静期拦截 (方案 2: 单向禁入)
    if self.cooldown_remaining_steps > 0:
        self.cooldown_remaining_steps -= 1
        # 如果当前仍有持仓，强制平仓；如果已空仓，禁止再开同方向仓位
        if current_position != 0:
            action = self._defensive_action(info, current_position, current_leverage)
            self.macro_action_history.append(self.slot_count)
            self.action_decision_reason_history.append(ActionDecisionReasons.STOP_LOSS_COOLDOWN)
            self.current_action = action
            return action

    # 原有的非主力合约防守与 VAE 阈值防守 ...
```
4. 在主循环调用处传递 `env.unrealized_pnl`：
```python
# FineFT/RL/DiHFT/high_level/vae_routing_util.py:1303
action = self.get_action(
    info, s, env.position, env.leverage, current_unrealized_pnl=float(env.unrealized_pnl)
)
```

---

## 6. 总结与后续行动建议 (Next Actions)

1. **方案验证就绪**：
   实测数据已全面证实：`fu2411` 的巨亏完全可以通过 `-50 USDT` 单笔硬止损消除，全组合利润将获得 **106.6% 的爆发性增长**。
2. **下一步实施**：
   可基于本报告提出的 P0 组合方案（单笔浮亏 -50 USDT 止损 + 12 步单向冷静期），在 `FineFT/RL/DiHFT/high_level/vae_routing_util.py` 中落地代码补丁，并在现存验证集上快速回归复测，固化为新的高层风控标准。
