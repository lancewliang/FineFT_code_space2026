---
status: accepted
---

# 动态双主力准入控制、平滑降级与跨合约时间同步组合核算

针对商品期货（fu 30min）高层路由最终回测中多合约资金割裂、缺乏真实组合夏普与回撤，以及合约换月降级瞬间硬性强平冲击的问题，建立“动态双主力准入控制”、“平滑降级（Graceful Downgrade）”与“时间同步组合净值核算”机制。用户于 2026-10-19 访谈确认本决定。

### 1. 业务与物理背景
1. **日度主力与次主力稀缺性**：根据期货市场微观结构与上期所日度行情结算，在任意 30min Bar 时刻，全市场严格存在至多 1 支主力合约（`prev_day_contract_role_tier = 1.0`）与至多 1 支次主力合约（`prev_day_contract_role_tier = 0.5`）。
2. **硬性平仓痛点**：既有 `--enable_non_main_contract_defense` 策略在检测到 `tier < 0.5` 时，立刻调用规则平仓（`rule_based_close`）。这导致主力合约换月当日，老主力若持有多头/空头，会被立刻以市价强平割肉，无法享受策略正常的平仓获利节奏。
3. **资金割裂痛点**：原最终回测脚本对各个合约分别配置独立本金并串行运行，无法计算跨合约并发持仓下的真实组合最大回撤（MDD）与组合年化夏普比率（Annual SR）。

### 2. 核心架构决策

#### 2.1 动态双主力准入与平滑降级机制 (Graceful Downgrade)
1. **双模式防御配置**：保持 `--enable_non_main_contract_defense` 作为防御总开关，新增 `--non_main_defense_mode` 参数：
   - `graceful_hold_exit`（默认）：平滑降级退出模式。
   - `strict_instant_close`：原强制立即平仓模式（供对照消融）。
2. **平滑降级执行逻辑**：
   - 当合约处于非主力/非次主力状态（`tier < 0.5`）：
     - 若当前为空仓（`position == 0`）：**严格禁止开新仓**，强制保持空仓（`action = zero_position_action`），决策原因标记为 `DEFENSIVE_RULE_CLOSE`。
     - 若当前持有仓位（`position != 0`）：**禁止同向加仓与反手开仓**（若策略欲加仓则截断为保持原仓位 `hold`；若欲反手开仓则截断为平仓 `flat`）。
     - 允许策略自主发出同向持有或平仓指令，硬止损（Hard Stop Loss）与熔断机制照常执行。
   - 当合约处于主力或次主力状态（`tier >= 0.5`）：正常执行高层 VAE 路由与低层 Agent 策略推理，具有完整开仓、平仓与调仓权限。

#### 2.2 纯角色开仓准入制 (Pure Role Admission Control)
在换月交替期，老合约持仓等待平仓，新主力/次主力发出开仓信号。系统采用纯角色开仓准入制，避免硬性名额抢占导致的突发强平；由于老合约“只减不增、禁开新仓”，在老合约退出后自然收敛至至多 2 个活跃持仓。

#### 2.3 跨合约时间同步组合核算 (Time-Synchronized Portfolio Accounting)
1. **本金分配基准**：单合约以 6,000 元为独立本金，组合总资金池设定为 12,000 元（$6,000 \times 2$）。
2. **多合约时间轴对齐**：按全局统一的时间戳（Timestamp）对齐所有活跃合约的单步净盈亏（$\Delta PnL_{c, t} = \text{Reward}_{c, t} \times \text{contract\_unit}$）。
3. **全局组合净值合成**：计算全局连续资产曲线 $Asset_t = Asset_0 + \sum_{\tau=1}^t \sum_{c} \Delta PnL_{c, \tau}$，严格按照标准公式输出组合 TR、组合 MDD、组合 Annual SR 与持仓合约数时间序列。

### 3. 产物与交付界面
- 交付脚本：`FineFT/script/test/DiHFT/high_level/final_result_fu_30_main_sub.sh`。
- 默认评测阶段：`valid`，支持通过参数切换至 `test`。
- 输出产物隔离存储于：`result/DiHFT/final_result/${DATASET_NAME}/${EXPERIMENT_NAME}/main_sub/`，包含 `summary_metrics.json`、`summary_metrics.csv`、`portfolio_action_timeline.csv`、`portfolio_summary.md`，并保留逐合约标准历史与诊断轨迹。
