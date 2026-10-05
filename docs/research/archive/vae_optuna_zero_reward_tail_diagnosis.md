# DiHFT 高层 VAE 路由 Optuna 搜索全零收益（Value: 0.0）根因诊断报告

- **诊断对象**: `FineFT/script/test/DiHFT/high_level/vae_optuna_fu_10.sh`
- **对应日志**: `log/DiHFT/fu/high_level/optuna/10min_parallel/optuna.log`
- **典型现象**: 几乎所有 Trial（包括 Trial 37 与最优 Trial 0）的目标值均为 `value: 0.0`，且所有合约回测收益均为 0。

---

## 1. 核心结论（Executive Summary）

在 `optuna.log` 中，所有 44 个已完成的 Trial 的评估结果均为 `value: 0.0`。该现象**并非模型代码 Bug 或网络崩溃**，而是由**时序数据分布漂移（OOD）与过于严苛的防守阈值区间共同导致的“全时段防守锁死”**。

具体因果链路如下：
1. **非主力合约直接拦截**：参数 `--enable_non_main_contract_defense` 拦截了所有 `prev_day_contract_role_tier < 0.5` 的步。在 12 个验证合约中，有 4 个合约（fu2412, fu2508, fu2511, fu2601）为纯非主力合约，被 100% 拦截；全体验证样本中有 **43.7%** 被非主力防守直接拦截。
2. **时序分布漂移导致 VAE 分位数极度偏低**：验证集（2024-07 至 2025-06）与训练集（2023-01 至 2024-07）之间存在明显的波动率与价差特征漂移（如 `realized_volatility_192` 的 NLL 差异达 -89.78）。导致验证样本在 6 个 VAE 模型下的对数似然暴跌至 -60 ~ -270，对应训练集基准 `id_logpx` 的分位数基本都落在 **0.000 ~ 0.010（0% ~ 1%）** 的极低区间。
3. **滚动窗口加权值远低于 Optuna 搜索阈值下限**：由于输入分位数极低，双轴的滚动平滑权重 `max(volatility_weights)` 仅在 0.001 ~ 0.03 左右（峰值不超 0.13），`max(slope_weights)` 仅在 0.01 ~ 0.08 左右。而 Optuna 的阈值搜索空间为 `[0.2, 0.5]`（均值 0.35），最低也不会低于 0.20。
4. **`OR` 条件判定致使 100% 触发规则防守**：路由决策条件为 `if max(volatility_weights) < threshold or max(slope_weights) < threshold:`。只要任一轴低于阈值即判定为高风险状态，进入 `_defensive_action`。因为权重几乎永远小于 0.20，该条件在验证集主力时段上 **100% 命中**。
5. **规则防守永不建仓，回测收益恒为 0**：防守动作 `rule_based_close` 在初始仓位为 0 时始终输出动作 `1`（Flat / Hold）。整个回测过程中没有一笔开仓成交，导致总收益为 0，胜率为 0，目标函数最终返回 `0.0`。

---

## 2. 证据链与一手源码溯源（Primary Source Evidence）

### 2.1 目标函数计算与 0 收益结算

- **源码位置**: `FineFT/RL/DiHFT/high_level/vae_routing_util.py:1039-1065`
```python
total_reward_sum = float(result_df[MetricColumns.REWARD_SUM].sum())
traded_mask = (result_df[MetricColumns.REWARD_SUM] != 0) | (result_df[MetricColumns.RETURN_RATE] != 0)
traded_count = int(traded_mask.sum())
portfolio_return_rate = total_reward_sum / (total_initial_capital + 1e-12)
win_rate = float((result_df[MetricColumns.RETURN_RATE] > 0).mean())
self.return_rate = portfolio_return_rate * win_rate
return self.return_rate
```
- **日志实测**:
  在 `log/DiHFT/fu/high_level/optuna/10min_parallel/optuna.log` 中：
  ```text
  [Test End] Multi-contract test completed | Contracts: 12 | Total Reward Sum: 0.0000 | Portfolio Return: 0.000000 | Win Rate: 0.0000 (0.0%) | Final Return Rate: 0.000000
  ```
  每个合约完成时均输出：
  ```text
  [Test Contract Done] [1/12] Completed 'fu2409': reward_sum=0.0000, require_money=0.0000, return_rate=0.000000
  ```
  因为没有任何交易产生，`total_reward_sum=0.0`，`win_rate=0.0`，因此目标函数严格返回 `0.0`。

---

### 2.2 防守机制双重拦截逻辑

- **源码位置**: `FineFT/RL/DiHFT/high_level/vae_routing_util.py:783-802`
```python
def get_action(self, info, s, current_position, current_leverage):
    # 拦截层 1：非主力合约防守
    if (
        self.enable_non_main_contract_defense
        and self.role_tier_index is not None
        and float(s[self.role_tier_index]) < 0.5
    ):
        action = self._defensive_action(info, current_position, current_leverage)
        self.macro_action_history.append(self.slot_count)
        self.action = action
        return action

    volatility_weights = self.calculate_axis_window_result("volatility")
    slope_weights = self.calculate_axis_window_result("slope")
    # 拦截层 2：双轴 OOD 规则阈值防守
    if (
        max(volatility_weights) < self.axis_thresholds["volatility"]
        or max(slope_weights) < self.axis_thresholds["slope"]
    ):
        action = self._defensive_action(info, current_position, current_leverage)
        self.macro_action_history.append(self.slot_count)
    else:
        # 正常路由给底层 Q 网络
        ...
```

#### 拦截层 1 实测数据：非主力合约占比
在 `dataset/10min/fu/valid` 的 12 个合约中，统计 `prev_day_contract_role_tier >= 0.5`（主力时段）：
- `fu2409.feather`: 94.8% 主力
- `fu2411.feather`: 100.0% 主力
- `fu2412.feather`: 0.0% 主力（全部被拦截层 1 防守）
- `fu2501.feather`: 79.9% 主力
- `fu2503.feather`: 76.9% 主力
- `fu2505.feather`: 64.6% 主力
- `fu2507.feather`: 75.9% 主力
- `fu2508.feather`: 0.0% 主力（全部被拦截层 1 防守）
- `fu2509.feather`: 51.9% 主力
- `fu2510.feather`: 25.6% 主力
- `fu2511.feather`: 0.0% 主力（全部被拦截层 1 防守）
- `fu2601.feather`: 0.0% 主力（全部被拦截层 1 防守）
全体验证集 30,363 步中，有 **13,276 步（43.7%）** 直接被拦截层 1 判定为非主力合约并强制平仓。

#### 拦截层 2 实测数据：时序漂移与分位数塌陷
对于剩余 56.3% 的主力时段样本（如 `fu2411`），计算 VAE 对数似然与分位数：
- 训练集基准 `id_logpx.npy` 分布（`result/DiHFT/vae_results/fu/10min_parallel/slope/label_0/summary.json`）：
  - 均值: `+27.81`
  - 中位数 (q50): `+30.38`
  - 1%分位数 (q01): `-31.44`
- 验证集样本在 6 个 VAE 模型上的实际推理损失 $L = -\text{NLL}$：
  - `slope label_0`: loss = `-64.68`，分位数 = `0.001772`
  - `slope label_1`: loss = `-153.63`，分位数 = `0.000226`
  - `slope label_2`: loss = `-216.86`，分位数 = `0.000107`
  - `volatility label_0`: loss = `-19.12`，分位数 = `0.005520`
  - `volatility label_1`: loss = `-85.22`，分位数 = `0.000427`
  - `volatility label_2`: loss = `-275.11`，分位数 = `0.000000`
- 经过 `calculate_rolling_window` 的指数加权平均后：
  - `fu2411`: `max(volatility_weights)` 均值仅 `0.0336`（最大 `0.1296`），`max(slope_weights)` 均值仅 `0.0854`
  - `fu2501`: `max(volatility_weights)` 均值仅 `0.0224`（最大 `0.2496`），`max(slope_weights)` 均值仅 `0.0367`
  - `fu2505`: `max(volatility_weights)` 均值仅 `0.0061`，`max(slope_weights)` 均值仅 `0.0114`

---

### 2.3 Optuna 搜索空间与采样阈值死锁

- **源码位置**: `FineFT/RL/DiHFT/high_level/vae_routing_optuna.py:76-85, 182-191`
```python
parser_all.add_argument("--rule_base_threshold_min", type=float, default=0.2)
parser_all.add_argument("--rule_base_threshold_max", type=float, default=0.5)

trial_args.slope_rule_base_threshold = trial.suggest_float(
    RoutingParamColumns.SLOPE_RULE_BASE_THRESHOLD,
    search_args.rule_base_threshold_min,
    search_args.rule_base_threshold_max,
)
trial_args.volatility_rule_base_threshold = trial.suggest_float(
    RoutingParamColumns.VOLATILITY_RULE_BASE_THRESHOLD,
    search_args.rule_base_threshold_min,
    search_args.rule_base_threshold_max,
)
```
- **日志实测**:
  在已完成的全部 44 个 Trial 中：
  - `slope_rule_base_threshold` 最小值为 `0.2297`
  - `volatility_rule_base_threshold` 最小值为 `0.2178`
  - **没有任何一个 Trial 的两个阈值同时小于 0.25**。
  - 在用户关注的 **Trial 37** 中：
    `'slope_rule_base_threshold': 0.3964, 'volatility_rule_base_threshold': 0.2880`
    而 `fu2411` 的 `max(volatility_weights)` 最大仅为 0.1296，永远无法达到 0.2880；
    `fu2501` 的 `max(slope_weights)` 最大仅为 0.1138，永远无法达到 0.3964。
- **因果判定**:
  因为采用了 `or` 逻辑判断（只要任何一个轴低于阈值就强制平仓），只要 `volatility` 或 `slope` 其中一个低于阈值就无法开仓。实际权重大多低于 0.10，直接导致在整个 30,363 步验证集中，**没有任何一步能同时跨过双轴阈值**，策略在整个评估周期内从未开过一次仓。

---

### 2.4 反证实验（Empirical Counter-Proof）

为了排除底层网络故障并验证此根因，我们将 `rule_base_threshold` 置为 `0.0`，在 `fu2411` 上运行单合约回测：
- **微观动作分布**:
  - `action 0 (Short)`: 862 步
  - `action 1 (Hold/Flat)`: 413 步
  - `action 2 (Long)`: 868 步
- **宏观路由分布**:
  - 成功激活了 8 个不同的宏观 Agent Slot（`slot 0, 2, 3, 4, 5, 6, 7, 8`）
- **回测指标**:
  - `reward_sum`: **+12.4255**
  - `require_money`: **661.8355**
  - `return_rate`: **+0.018774**（年化稳健盈利）

**反证结论**：底层集成强化学习网络完全具备交易能力和正向收益，零收益纯粹是由高层 VAE 路由的过高防守阈值区间与数据分布漂移共同引起的。

---

## 3. 为什么会出现这种漂移与脱节？

1. **时序跨度与市场状态变迁**：
   - 训练集时间跨度为 2023-01 至 2024-07，而验证集为 2024-07 至 2025-06。
   - 根据 `analysis_result/DiHFT/feature_ood/fu/10min_parallel/feature_ood_summary.csv`：
     - `realized_volatility_192`: 训练集均值 0.1136 (std 0.8951)，验证集均值 0.4587 (std 1.6378)，delta_nll = -89.78。
     - `sell_spread_oe_max_trend_192`: 训练集均值 0.0570，验证集均值 0.2258，delta_nll = -40.87。
     验证集燃料油市场的波动率水平和买卖盘挂单价差显著放大，超出了 VAE 在 2023 年历史数据上学习到的紧凑先验流形。
2. **理论预期与现实设定的错位**：
   - 算法设计初衷：在理想的独立同分布假设下，输入样本在真实分布下的分位数服从 $U(0, 1)$ 均匀分布，3 个标签取最大值的数学期望为 $3/(3+1) = 0.75$。因此原框架在设计时将 Optuna 阈值搜索空间设定在 `[0.2, 0.5]`（认为 0.2~0.5 已经是非常宽松的防守线）。
   - 现实困境：由于真实期货市场跨年度的时序分布漂移，未更新的 VAE 给出的分位数普遍塌陷至 0.01 甚至更低。原先被认为是“极度宽松”的 0.20 阈值，在漂移后的现实场景下变成了“100% 无法逾越的高墙”。

---

## 4. 优化建议与修复方案（Remediation）

若要使 Optuna 能够正常搜索出有效参数并释放底层策略的盈利能力，建议采取以下改动：

1. **调整 Optuna 阈值搜索下限与尺度**：
   在 `FineFT/RL/DiHFT/high_level/vae_routing_optuna.py` 或启动脚本中：
   ```bash
   --rule_base_threshold_min 0.0 \
   --rule_base_threshold_max 0.15
   ```
   或者采用对数采样（Log-scale），允许其在 `[0.001, 0.2]` 之间精细搜索。
2. **修改双轴防守判据的逻辑组合**：
   将 `or` 逻辑改为更宽松的融合形式（或允许单独只用单轴过滤）：
   ```python
   # 方案 A: 均值门限而非木桶效应的严苛 or
   if (volatility_weight + slope_weight) / 2.0 < threshold:
   # 方案 B: 仅在两轴均极端异常时才触发防守 (and)
   if max(volatility_weights) < vol_threshold and max(slope_weights) < slope_threshold:
   ```
3. **相对置信度（Softmax / Margin）替代绝对分位数门限**：
   不依赖跨时期可能剧烈漂移的绝对 `id_logpx` 分位数，而是直接计算各 Label 之间的相对概率分布：
   $$\pi_i = \frac{\exp(L_i / \tau)}{\sum_j \exp(L_j / \tau)}$$
   以最大概率与次大陆概率的置信差（Margin）作为防守依据。
