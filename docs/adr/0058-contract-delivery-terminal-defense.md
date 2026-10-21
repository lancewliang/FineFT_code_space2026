---
status: superseded
---

# 合约临近交割终局平仓与空仓锁定防御 (Contract Delivery Terminal Defense)

为阻断商品期货（如上期所燃料油 fu）临近交割月份流动性断崖萎缩、现货与多空对峙导致的基差剧烈异动以及开盘跳空深幅回撤风险（如 fu2409 在 2024 年 8 月中旬交割前夕发生的跳空与大幅亏损），在高层路由评估中建立“临近交割期终局平仓与空仓锁定防御”机制。用户于 2026-10-20 设计访谈确认此决定。

### 1. 业务与数据物理背景
1. **交易所规则与交易数据截断**：以上期所燃料油（fu）为例，燃料油合约（如 `fu2409`）的名义交割月为 9 月，其实际最后交易日为交割月前一月的最后一个交易日（8 月 30 日）。在数据预处理阶段（`main_contract.py:544`），为了剔除交割收尾流动性真空期，系统预先对各合约数据进行了尾部截断（`ordered_days[-11]`），导致回测数据集中合约往往在到期日前 10 个交易日左右终止（例如 `fu2409` 在验证集中仅存续至 2024-08-16）。
2. **生命周期尾部陷阱**：在合约临近到期退市的最后数个交易日内，投机资金大幅离场，盘口承接力锐减，点差走阔且频繁发生极端跳空，RL 策略模型在此类非平稳行情中极易连续受损。

### 2. 核心架构决策

#### 2.1 尾部交易日锚定 (Tail Trading Days Anchor)
以合约在当前评估数据集切片（DataFrame）中包含的全部唯一交易日序列 `unique_dates` 为基准。当设定防御天数为 $X$ 交易日（默认 $X = 5$ 交易日，即最后 1 周）时，将最后 $X$ 个交易日（`unique_dates[-X:]`）界定为交割终局风控期。该设计使风控完全自包含于评估切片，天然适应不同数据切片（Train / Valid / Test）和已执行前置截断的数据。

#### 2.2 终局平仓与空仓锁定行为 (Strict Instant Close & Lockout)
一旦时间步落入最后 $X$ 个交易日内：
1. **强制立即平仓**：若当前持有非零仓位（`position != 0`），立即解除动作持续性（`remaining_persist = 0`），通过规则平仓动作（`rule_based_close`）强制平仓至 0；
2. **全程锁定空仓**：若当前已为空仓（`position == 0`），严格禁止发起任何开仓推断，直接返回空仓动作（`zero_position_action`）；
3. **审计归因记录**：决策原因在动作历史中记录为 `DEFENSIVE_RULE_CLOSE`（若打断持续性则记录为 `DEFENSIVE_PREEMPTION`），宏观动作槽位标记为防御槽位（`slot_count`），完全兼容既有交易审计体系与下游诊断工具。

#### 2.3 CLI 参数与配置界面
- Python CLI 参数：
  - `--enable_delivery_month_defense`：布尔开关，启用交割终局平仓防御。
  - `--delivery_defense_days`：整型参数，指定到期前平仓的交易日天数，默认值为 `5`。
- Shell 启动脚本配置：
  - `ENABLE_DELIVERY_MONTH_DEFENSE=${ENABLE_DELIVERY_MONTH_DEFENSE:-1}`
  - `DELIVERY_DEFENSE_DAYS=${DELIVERY_DEFENSE_DAYS:-5}`
  - 同步集成于 `FineFT/script/test/DiHFT/high_level/final_result_fu_10_main_sub.sh` 及 `final_result_fu_30_main_sub.sh`。


2026-10-21 更新：边界与账户范围以 [ADR-0060](0060-contract-boundaries-and-tonne-rmb-accounting.md) 为准；原文保留记录历史决策。
