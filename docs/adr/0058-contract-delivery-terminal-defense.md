---
status: Reopened / Amended
---

# 合约真实生命周期与切片边界解耦架构 (Contract Lifecycle and Slice Boundary Decoupling Architecture)

> **修订说明 (2026-10-21)**：
> 本决策由 2026-10-20 原“切片末尾交割防御”正式重开并修订（Reopened / Amended），并与 [ADR-0060](0060-contract-boundaries-and-tonne-rmb-accounting.md) 共同确立真实生命周期与切片边界的三层解耦架构。
> **重开与物理删除原因**：
> 1. **交割真空在特征工程中已物理剔除**：在数据预处理阶段（`main_contract.py:544`），特征工程已经从每个合约完整生命周期中严格排除了最后 10 个交易日（`Contract Lifecycle Tail Exclusion: ordered_days[-11]`），高层回测切片内已物理不存在交割月的流动性枯竭与极端跳空。
> 2. **切片尾部不等于合约生命周期到期**：原 ADR-0058 简单锚定评估 DataFrame 尾部最后 5 日，误将样本切片分区边界（如 Valid 集终止点 2025-06-27）当成交割终局，导致正处于主力存续期的合约（如 fu2509、fu2510）被错误提前强平并错失盈利（诊断显示因此压制了 1,286.40 收益），且未来追加行情时会破坏历史决策掩码的前缀不变性（Prefix Invariance）。
> 因此，高层回测中原 `--enable_delivery_month_defense` 及硬性切片尾部平仓规则被彻底物理移除。

---

### 1. 三层解耦核心架构决策

为彻底解决“合约真实退市”、“特征数据可用范围”与“策略评估切片边界”混淆引发的决策漂移与过度风控，确立以下三层解耦标准：

#### 1.1 第一层：合约真实生命周期与前置交割物理裁剪 (Preprocessing Lifecycle Boundary)
- **职责归属**：离线数据预处理与特征工程层（`main_contract.py`）。
- **行为规范**：基于交易所官方合约规则（如上期所燃料油 fu 到期前一月最后交易日退市），以该合约完整的全部交易日序列为基准，统一裁剪剔除最后 10 个交易日（`ordered_days[-11]`）。
- **保证**：进入高层训练与评估的特征数据集已物理隔离交割期流动性断崖，策略无需且严禁在运行时重复推算“到期前 X 天防御”。

#### 1.2 第二层：运行时流动性角色与老仓平滑退出 (Runtime Dynamic Liquidity & Graceful Exit)
- **职责归属**：高层 VAE 路由与角色准入控制层（`vae_routing_util.py` / `ADR-0057`）。
- **行为规范**：
  1. 策略只允许在每日结算判定的主力（tier=1.0）与次主力（tier=0.5）合约上发起新增开仓。
  2. 当合约因流动性自然转移退化为非主力（tier < 0.5）时，触发平滑降级（`graceful_hold_exit`）：禁止任何加仓与反向开仓，允许跟随模型策略退出或触碰止损自然平仓退出，不再施加全局时间步硬性强平。

#### 1.3 第三层：评估切片窗口末真实盯市 (Evaluation Slice Mark-to-Market)
- **职责归属**：回测统计与组合绩效评估层（`calculate_metric.py` / `synchronized_portfolio_diagnostics.py`）。
- **行为规范**：
  1. 样本切片（Train / Valid / Test）结束点仅作为统计观测边界，最后一刻保留真实持仓状态与可用行情盯市估值（Mark-to-Market），绝对不触发虚构的瞬时平仓交易与清算手续费。
  2. **前缀不变性严格契约 (Prefix Invariance)**：策略在时刻 $t$ 的准入、风控与动作决策只能依赖 $t$ 及以前的信息，向后追加未来数据时，历史所有步的决策与风控掩码完全恒定不变。

---

### 2. 状态机与接口契约
- 高层不再暴露 `--enable_delivery_month_defense` 和 `--delivery_defense_days` 参数。
- `is_terminal_defense_step` 统一返回 `False`，仅作为向下兼容旧接口的轻量契约。
- 终局测试严格遵循 `fu_10min_terminal_boundary_check.py`：追加未来 2 天行情时，历史所有时间步的风控掩码与决策差异必须为 0。
