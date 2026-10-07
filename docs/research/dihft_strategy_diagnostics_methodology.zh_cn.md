# DiHFT 高频交易策略全维度诊断与归因分析方法论

## 1. 概述与核心目标

本方法论旨在为 FineFT 框架下的 DiHFT（分层深度强化学习高频交易系统，涵盖 VAE 宏观体制路由、低层微观 Q-Net、三级风控及执行层）提供一套系统化、严密且高效的量化策略诊断体系。

针对实验产物目录（例如 `analysis_result/DiHFT/high_level_heurstic/fu/10min_parallel/` 及对应的 `diagnostics/` 和 `result/DiHFT/final_result/`），本方法论解决以下核心问题：
1. **策略好坏评判**：如何在考虑多合约横截面、基准对比（Buy & Hold 1x/5x）、风险调整后收益与尾部回撤的前提下客观评判策略优劣？
2. **盈利成因解耦**：策略的净利润究竟来源于真实的盘面择时预测能力（Alpha），还是运气？
3. **失效根因诊断**：若策略亏损或不及预期，根本原因究竟是择时反向、摩擦损耗吞噬、持仓周期过短、宏观路由崩塌还是极端行情踩踏？
4. **针对性调优路线**：在确定根因后，应该调整高层路由门控、动作持续性、三级风控还是底层选模与训练奖励？
5. **高效分析实施与 Token 优化**：规划精准的文件读取清单与依赖链路，并依托一次性批处理工具（`batch_diagnostics_loader.py`）避免多文件串行逐个读取带来的上下文膨胀与分析延迟。

---

## 2. 策略好坏的量化评价标准 (Evaluation Standards)

评判一个高频策略的优劣不能仅看单一的回测总收益率（Total Return, TR），必须从**收益能力、基准相对优势、风险调整表现、截面稳健性**四个维度综合打分：

| 评价维度 | 核心量化指标 | 优秀 (Pass/Strong) | 亚健康 (Warning) | 不合格 (Fail) | 对应数据字段 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **净盈利能力** | 组合净回报率 (`portfolio_return_pct`) | $> +20\%$ | $0\% \sim +20\%$ | $\le 0\%$ | `portfolio.portfolio_return_pct` |
| **盘面择时能力** | 盘面毛回报率 (`portfolio_gross_return_pct`) | $> +40\%$ | $+15\% \sim +40\%$ | $< +10\%$ 或 $\le 0\%$ | `portfolio.portfolio_gross_return_pct` |
| **摩擦控制** | 摩擦损耗吞噬率 (`friction_consumption_ratio_pct`) | $< 50\%$ | $50\% \sim 75\%$ | $> 75\%$ | `portfolio.friction_consumption_ratio_pct` |
| **换手乘数** | 组合本金换手倍数 (`portfolio_turnover_multiple`) | $< 30\text{x}$ | $30\text{x} \sim 80\text{x}$ | $> 80\text{x}$ | `portfolio.portfolio_turnover_multiple` |
| **风险调整收益** | 平均年化夏普比率 (`mean_annual_sr`) | $> 2.0$ | $1.0 \sim 2.0$ | $< 1.0$ | `portfolio.mean_annual_sr` |
| **下行风险控制** | 平均最大回撤 (`mean_mdd_pct`) | $< 8\%$ | $8\% \sim 15\%$ | $> 15\%$ | `portfolio.mean_mdd_pct` |
| **卡尔玛比率** | 日频卡尔玛比率 (`mean_daily_cr`) | $> 2.5$ | $1.0 \sim 2.5$ | $< 1.0$ | `portfolio.mean_daily_cr` |
| **横截面一致性** | 多合约胜率 (`win_rate_pct`) | $\ge 85\%$ | $70\% \sim 85\%$ | $< 70\%$ | `portfolio.win_rate_pct` |
| **尾部单点风险** | 最差单合约亏损占比 (`worst_loss / capital`) | $< 5\%$ | $5\% \sim 12\%$ | $> 12\%$ | `min(df_pnl['net_pnl']) / capital` |

---

## 3. 盈利能力的成因与解耦框架 (Profitability Decomposition)

基于 ADR 0048 的损耗解耦理论，DiHFT 策略实现的最终净利润公式严格满足：
$$\text{Net PnL} = \text{Gross Market Timing PnL} - \text{Friction}$$
$$\text{Friction} = \text{Commission} + \text{Order Book Slippage}$$

### 3.1 盘面择时毛能力 (Alpha Timing Power)
- **概念**：假定无交易手续费且挂单即时在当前标记价（Mark Price）以零滑点撮合成交时，策略根据多空方向和价格变动捕捉到的理论毛利（Gross PnL）。
- **来源**：底层 Q-Net 智能体在微观价差/深度特征下的动作选择，以及高层 VAE 路由对当前市场体制（动量与波动率）的准确划分。
- **验证依据**：若毛利润显著高于 1x/5x Buy & Hold，说明模型具备强大的市场择时预测 Alpha。

### 3.2 摩擦成本吞噬机制 (Friction Breakdown)
- **手续费 (Commission)**：每笔交易按合约乘数与名义成交额计提（国内期货常用 0.0001~0.0005）。
- **订单簿滑点 (Slippage)**：由撮合引擎深度吃单产生（越过盘口价差造成的买卖损耗）。
- **摩擦吞噬率 (Friction Ratio)**：
  $$\text{Friction Ratio} = \frac{\text{Commission} + \text{Slippage}}{\text{Gross PnL}} \times 100\%$$
- **实证规律**：在多数高频策略中，模型预测方向正确率达标（毛收益率往往超过 50%~80%），但因为调仓频次过高，产生数千万元虚假换手，导致手续费与滑点将 $90\%+$ 的毛利吞噬殆尽（见 ADR 0048 实证）。

### 3.3 微观持仓姿态与执行机制 (Micro Posture & Execution)
- **多空平比例 (`long_pct`, `short_pct`, `flat_pct`)**：
  - 优秀策略在强趋势中坚定持仓（多头或空头占比鲜明），在横盘震荡或高不确定性时果断平仓观望（`flat_pct` 上升）。
  - 若 `flat_pct` 长期高达 80%+，属于不作为；若多空 50/50 且每隔 1~2 步反复反手，属于噪声震荡。
- **持仓平均周期 (`mean_holding_bars`, `median_holding_bars`)**：
  - 若平均持仓仅 2~3 个 bars（20~30分钟），不足以走完一个微观波段，往往无法覆盖双向进出场摩擦；
  - 优秀的 10min 策略应具备一定的持仓惯性（平均持仓 $\ge 8 \sim 15$ bars）。
- **动作持续性 (`action_persistence_steps`) 与推理跳过率**：
  - 依据 ADR 0045，非 Flat 持仓应物理锁定固定步数，跳过微观噪声推理，强行降低无效换手。

### 3.4 宏观 VAE 体制路由分流 (Macro Regime Routing)
- **体制九宫格覆盖度**：
  - 考察宏观路由在 9 个动力学槽位（Slot 0~8，由 `slope_bin` 0~2 与 `volatility_bin` 0~2 笛卡尔积构成）以及保底槽位（Slot 9 / Default）的步数分布。
  - **健康状态**：各槽位有合理的激活比例，顺应行情的动态迁移；
  - **退化状态**：单一 Slot 垄断超过 80% 步数（说明 VAE 分辨率失效或门控边界设错），或者大量步数跌入 Default/OOD Reject（说明特征严重分布外漂移）。

### 3.5 三级风控干预质量 (Risk Control Safeguards)
- **硬止损 (`hard_stop_losses`)**：标的价格相对持仓成本反向波动突破阈值（如 1.5%）时立即平仓，截断长尾损失。
- **单向冷静期 (`stop_loss_cooldown_steps`)**：被止损后短时间内禁止同向开仓，防止在瀑布行情中逆势抄底接飞刀。
- **断路器熔断 (`circuit_breaker_suspension_steps`)**：单合约连续止损达到设定次数（如 2 次）后进入强制休眠（如 72 步），保护账户本金。

---

## 4. 盈利不佳的根因诊断决策树 (Root Cause Diagnosis Tree)

当策略表现不佳时，按照如下决策树自顶向下分级诊断：

```text
[诊断起点: 输入 diagnostics_summary.json / contract_pnl_friction.csv]
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
   Gross PnL <= 0                Gross PnL > 0
 [场景 A: 盘面择时完全失效]               │
         │                             │
         ├─ 原因 A1: VAE严重OOD漂移     ├─────────────────────────────┐
         ├─ 原因 A2: 底层QNet反向预测   ▼                             ▼
         └─ 原因 A3: 选模指标过拟合  Net PnL <= 0                  Net PnL > 0
                                 [场景 B: 摩擦完全吞噬]                │
                                       │                             │
                                       ├─ 原因 B1: 调仓频次过高       ├─────────────────────────┐
                                       ├─ 原因 B2: 持仓周期过短       ▼                         ▼
                                       ├─ 原因 B3: 盘口滑点失控    MDD过大 / SR极低          表现优异 / 持续监控
                                       └─ 原因 B4: 对称换手未抑制 [场景 C: 尾部风险失控]     [场景 D: 达标策略]
                                                                     │
                                                                     ├─ 原因 C1: 单边行情扛单 (未设硬止损)
                                                                     ├─ 原因 C2: 止损后立即反扑 (无冷静期)
                                                                     └─ 原因 C3: 单合约严重失血 (熔断缺失)
```

### 场景 A：盘面择时毛利为负 ($Gross \le 0$) —— “方向看反了”
- **现象**：`total_gross_pnl <= 0`，策略即使在零手续费下也是亏钱的。
- **根因定位**：
  1. **低层模型能力缺失**：训练集未能学习到有效的短期动量与订单流特征；
  2. **选模指标陷阱**：在 `two_dimensional_selection` 阶段过度依赖未正则化的训练集回报率，选中了过度拟合的 checkpoint；
  3. **特征分布漂移 (OOD)**：验证集/测试集的波动率或偏度超出 VAE 训练时的极值范围，导致高层把“下跌”判定为“上涨”。
- **确认路径**：查看 `macro_routing_distribution.csv` 和 `two_dimensional_selection_manifest.json`，核对各槽位选出的模型及其在对应体制下的胜率。

### 场景 B：毛利丰厚但净利亏损 ($Gross > 0, Net \le 0$) —— “打工全给交易所”
- **现象**：`total_gross_pnl > 0` 且数值很大，但 `friction_consumption_ratio_pct > 100%`，最终 `total_net_pnl < 0`。
- **根因定位**：
  1. **微观高频倒手（Over-trading）**：`mean_holding_bars < 5`，每次价格稍有扰动就频繁平仓反手；
  2. **缺乏换手惩罚机制**：环境训练未接入换手惩罚，模型把微观噪声误当成盈利空间；
  3. **吃单滑点惩罚巨大**：在盘口价差大、深度浅的时段频繁下市价单。
- **确认路径**：查看 `contract_pnl_friction.csv` 中的 `est_commission`、`est_slippage` 与 `turnover_ratio`，以及 `contract_behavior_risk.csv` 中的 `mean_holding_bars`。

### 场景 C：整体盈利但回撤巨大 / 夏普过低 ($Net > 0, MDD > 20\%, SR < 1.0$) —— “赚小钱亏大钱”
- **现象**：大部分合约表现尚可（胜率可能达到 80%+），但个别月份合约出现断崖式下跌，拉垮整个组合夏普，MDD 飙升。
- **根因定位**：
  1. **单点灾难（如 ADR 0047 中的 `fu2411` 暴跌）**：模型在极端单边大跌中持续逆势抄底多头，一路扛单；
  2. **止损机制缺失或过宽**：未开启 `--stop_loss_return_threshold` 或阈值设得过大；
  3. **反复触碰止损后未冷却**：刚止损平仓，下一帧模型又发出做多指令，陷入“接飞刀-止损-再接飞刀-再止损”的死循环。
- **确认路径**：查看 `contract_pnl_friction.csv` 按 `net_pnl` 升序排列，找出垫底的 Worst Contracts；查看 `contract_behavior_risk.csv` 中的 `hard_stop_losses` 与 `circuit_breaker_suspension_steps`。

### 场景 D：收益极低且持仓极度被动 ($Net \approx 0, Flat\% > 75\%$) —— “过于保守踩空”
- **现象**：毛利和净利都在 0 附近徘徊，换手率极低，`flat_pct > 75%`。
- **根因定位**：
  1. **拒识门槛过严**：`ood_threshold` 设置过小，高层把正常行情误判为 OOD，长期强制 Flat；
  2. **边缘差值过大**：`slope_margin_threshold` / `volatility_margin_threshold` 设得过高，导致置信度不足而无法激活任何专业槽位；
  3. **断路器过于脆弱**：连续触发 1~2 次微小亏损即被长周期熔断。
- **确认路径**：查看 `macro_routing_distribution.csv` 中 `dynamic_default` 比例，以及 `high_level_agent_para.txt` 中的门控超参。

---

## 5. 策略调优与参数调整指南 (Actionable Tuning Playbook)

| 诊断结论 / 瓶颈 | 推荐调优杠杆 (Knobs) | 目标调整方向与操作建议 | 涉及代码与配置文件 |
| :--- | :--- | :--- | :--- |
| **摩擦吞噬过高 ($> 75\%$)** | 动作持续性 (`action_persistence`) | 从默认 `3` 调大至 `6` 或 `12`（锁仓 $1 \sim 2$ 小时，强制降低换手） | `final_result_fu_*.sh`<br>`high_level_heurstic_fu_*.sh` |
| **摩擦吞噬过高 ($> 75\%$)** | 非对称调仓惩罚 (ADR 0050) | 开启 `--turnover_adverse_ratio 6.0`，对逆势与震荡调仓施加 6 倍阻尼 | `parallel_diverse_train.py`<br>`futures_util.py` |
| **微观倒手 / 持仓过短** | 滞回退出比例 (`hysteresis_exit_ratio`) | 调大该参数（如从 `0.1` 提升至 `0.3`），防止在相邻状态间振荡切换 | `DiHFT_high_level_heurstic.py`<br>`gating.py` |
| **单合约极端暴跌扛单** | 标的硬止损 (`stop_loss_return_threshold`) | 缩紧止损阈值（如从 `0.02` 收窄至 `0.012`~`0.015`，即亏损 1.2%~1.5% 强制离场） | `final_result_fu_*.sh`<br>`vae_routing_final_result_macro_action.py` |
| **止损后频繁接飞刀** | 单向冷静期 (`stop_loss_cooldown_steps`) | 从 `6` 步延长至 `12`~`18` 步（禁止同向开仓 2~3 小时） | `final_result_fu_*.sh` |
| **单合约连续巨亏踩踏** | 断路器熔断 (`circuit_breaker_cooling_steps`) | 连续 2 次止损后，将休眠时间设为 `72` 步（12 小时交易时间不参与该合约） | `final_result_fu_*.sh` |
| **盘面毛利为负 ($Gross < 0$)** | 选模评价指标 (`selection_metric`) | 在 `high_level_heurstic` 中将 `--selection_metric` 从 `tr` 更改为 `annual_sr` 或 `daily_cr` | `high_level_heurstic_fu_*.sh` |
| **过于保守 / 长期 Flat** | 异常检测阈值 (`ood_threshold`) | 适当放宽门限（如从 `0.003` 调宽至 `0.008`~`0.010`），降低拒识率 | `high_level_agent_para.txt`<br>Optuna 搜索空间 |
| **宏观槽位单一垄断** | 动量/波动率边界阈值 (`margin_threshold`) | 减小边界裕度（如从 `0.15` 下调至 `0.08`），促使各槽位均衡激活 | `high_level_agent_para.txt` |

---

## 6. 分析必须读取的文件清单与依赖关系 (File Inventory)

为避免遗漏信息或无目的地反复查阅文件，建立如下标准文件清单：

```text
[实验根目录: analysis_result/DiHFT/high_level_heurstic/<dataset>/<exp>/]
    │
    ├── diagnostics/
    │   ├── diagnostics_summary.json      (核心：组合级总览、胜率、毛利、净利、摩擦拆解)
    │   ├── diagnostics_summary.md        (辅助：Markdown 全景简报)
    │   ├── contract_pnl_friction.csv     (核心：逐合约毛利、净利、佣金、滑点、夏普、回撤)
    │   ├── contract_behavior_risk.csv    (核心：持仓长短平分布、平均持仓周期、风控干预计数)
    │   └── macro_routing_distribution.csv(核心：九宫格 Slot 0~8 及 Default 分流分布)
    │
    ├── best_result.csv                   (核心：按 TR/SR/CR/MDD 选出的最优超参记录)
    ├── result.csv                        (参考：所有探索 Trial/Epoch 的全量指标)
    │
[执行与配置目录: result/DiHFT/final_result/<dataset>/<exp>/]
    │
    ├── high_level_agent_para.txt         (核心：当前正在运行的最优门控超参数配置)
    └── contracts/<contract>/             (按需采样：底层单步历史时序，如 micro/macro_action.npy)
    │
[选模元数据清单: analysis_result/DiHFT/low_level/<dataset>/<exp>/two_dimensional_selection/]
    │
    └── two_dimensional_selection_manifest.json (核心：九宫格各槽位选出的具体模型 Checkpoint 及指标)
    │
[相关 Python 核心代码]
    ├── FineFT/analysis/diagnostics/trading_diagnostics.py        (诊断指标计算与解耦实现)
    ├── FineFT/analysis/diagnostics/batch_diagnostics_loader.py   (批处理提取与自动诊断脚本)
    ├── FineFT/analysis/pick_agent/DiHFT_high_level_heurstic.py   (高层选优与 best_result 生成)
    ├── FineFT/RL/DiHFT/high_level/vae_routing_final_result_macro_action.py (最终回测执行与风控逻辑)
    └── FineFT/RL/DiHFT/high_level/vae_routing_util.py           (分级自适应门控与风控逻辑)
```

---

## 7. 结构化四阶段分析实施路径 (Analysis Pipeline)

在面对一个新完成的实验结果时，应按如下四阶段流程执行：

### 阶段 1：宏观总览与总分定位 (Tier 1 Macro Health Check)
- **目标**：在 30 秒内明确策略的生死状态（Pass / Warning / Fail）。
- **动作**：通过批处理脚本读取 `diagnostics_summary.json`，检查：
  - `portfolio_return_pct`（是否赚钱？）
  - `portfolio_gross_return_pct`（择时毛利是否充足？）
  - `friction_consumption_ratio_pct`（摩擦吞噬是否失控？）
  - `win_rate_pct` 与 `mean_annual_sr`（稳健性与夏普是否达标？）

### 阶段 2：损耗解耦与微观行为排查 (Tier 2 Friction & Behavior Profiling)
- **目标**：锁定收益损耗与风险累积的执行根因。
- **动作**：通过批处理脚本读取 `contract_pnl_friction.csv` 与 `contract_behavior_risk.csv`：
  - 检查多空平结构：是否存在过度偏向（如纯单边牛市中做空过多）？
  - 检查持仓周期：`mean_holding_bars` 是否过短？
  - 检查摩擦结构：手续费与滑点的比例（滑点过大说明订单簿深度浅，需要降低单次撮合规模或调整执行机制）；
  - 检查风控事件：硬止损与断路器熔断是否触发，是否成功截断了长尾亏损。

### 阶段 3：宏观路由与选模元数据溯源 (Tier 3 Routing & Manifest Trace)
- **目标**：检查高层 VAE 路由与底层 Q-Net 槽位协作是否健康。
- **动作**：
  - 查看 `macro_routing_distribution.csv`：是否有槽位未被激活，或单一槽位过度垄断？Default / Reject 占比是否过高？
  - 查看 `two_dimensional_selection_manifest.json`：各槽位对应的训练 Epoch 和候选得分，确认是否存在弱模型入选。

### 阶段 4：超参回溯与调优落地 (Tier 4 Parameter & Source Code Audit)
- **目标**：制定确切的调优修改清单并更新回测脚本。
- **动作**：
  - 对照 `best_result.csv` 与 `high_level_agent_para.txt`，分析指标权衡（例如以 TR 选优导致 MDD 恶化）；
  - 根据第 5 节的调优指南，修改 `final_result_fu_*.sh` 或 `high_level_heurstic_fu_*.sh` 中的风控与执行参数。

---

## 8. 一次性批处理读取脚本使用规范 (Batch Diagnostics Loader)

为了彻底消除逐个文件调用读取导致的上下文膨胀、Token 浪费和多轮等待，系统内置了高效聚合脚本：
`FineFT/analysis/diagnostics/batch_diagnostics_loader.py`。

### 8.1 运行指令

在 `finetf` conda 环境下执行：

```bash
# 模式 1: 紧凑人类/LLM可读文本简报（推荐，约 60~80 行，高密度展示全局结论与自动调优建议）
conda run -n finetf python FineFT/analysis/diagnostics/batch_diagnostics_loader.py \
    --analysis_path analysis_result/DiHFT/high_level_heurstic/fu/10min_parallel \
    --format text

# 模式 2: 完整结构化 JSON 格式（便于脚本进一步程序化解析或传参）
conda run -n finetf python FineFT/analysis/diagnostics/batch_diagnostics_loader.py \
    --analysis_path analysis_result/DiHFT/high_level_heurstic/fu/10min_parallel \
    --format json
```

### 8.2 脚本核心功能与输出概览
1. **单次 I/O 汇聚**：一次性读取 `diagnostics_summary.json`、`contract_pnl_friction.csv`、`contract_behavior_risk.csv`、`macro_routing_distribution.csv` 以及 `best_result.csv`；
2. **派生核心诊断指标**：
   - 自动计算净毛比（$\text{Net PnL} / \text{Gross PnL}$）；
   - 自动计算每千步调仓频率（`trades_per_1k_steps`）；
   - 自动提取 Dominant Slot（主导路由槽位及占比）；
   - 自动截取 Top-K 最优与最差合约（直接暴露尾部亏损合约）；
3. **内置规则诊断引擎与警报标志 (Diagnostic Flags)**：
   - `[EXCESSIVE_FRICTION_CONSUMPTION]`：摩擦吞噬率 $> 80\%$；
   - `[ULTRA_SHORT_HOLDING]`：平均持仓周期 $< 5$ 步；
   - `[DIRECTIONAL_ALPHA_WEAK]`：毛利为负或微弱；
   - `[FAT_TAIL_CONTRACT_RISK]`：单一合约亏损拖累超过总本金 $15\%$；
   - `[MACRO_SLOT_MONOPOLY]`：单一槽位垄断 $> 80\%$；
4. **生成即时调优建议**：根据触发的标志自动输出具体的调优参数指令（如 `--action_persistence`、`--stop_loss_return_threshold` 等）。

该批处理工具使得在分析任何策略实验结果时，仅需一次命令即可完成全方位体检，为后续决策提供完全自洽的量化支撑。

---

## 9. 核心 Python 源码批量提取与压缩工具 (Batch Code Loader)

策略诊断过程中，往往需要核对风控、撮合、动作持续性及门控选模的具体算法逻辑。若逐一读取 9 个相关的核心 Python 源码文件，将造成数万 Token 的极大浪费和严重的上下文漂移。

为此，系统配套开发了基于 AST 的代码批量提取与压缩工具：
`FineFT/analysis/diagnostics/batch_code_loader.py`。

### 9.1 默认纳入批量提取的核心 Python 代码清单
1. `FineFT/analysis/diagnostics/trading_diagnostics.py`：诊断指标计算与摩擦解耦算法
2. `FineFT/analysis/pick_agent/DiHFT_high_level_heurstic.py`：高层最优超参选优与评价指标提取
3. `FineFT/RL/DiHFT/high_level/vae_routing_final_result_macro_action.py`：回测撮合引擎与风控层实现
4. `FineFT/RL/DiHFT/high_level/vae_routing_util.py`：分级自适应门控与动作持续性执行逻辑
5. `FineFT/RL/DiHFT/high_level/gating/hierarchical_gating.py`：双阈值迟滞分级门控核心实现
6. `FineFT/analysis/pick_agent/FineFT_two_dimensional_agent_selector.py`：底层模型两维选模算法
7. `FineFT/common/artifacts.py`：工件文件名常量定义
8. `FineFT/common/metric_columns.py`：金融与行为指标列名定义
9. `FineFT/common/routing_params.py`：路由超参数键名定义

### 9.2 三种提取模式与运行指令

在 `finetf` conda 环境下执行：

#### 模式 1：AST 架构大纲与参数提取 (`outline` 模式，强烈推荐)
- **特点**：通过 Python 原生 AST 语法树遍历，自动提取所有模块说明、类继承关系、方法完整输入输出类型签名、顶级函数以及 `argparse` 定义的所有 CLI 参数（含默认值与帮助文档）。
- **Token 节约**：将 5,000+ 行源码压缩至 250 行精简大纲，**节省 95% 以上 Token 上下文**。
- **运行命令**：
```bash
conda run -n finetf python FineFT/analysis/diagnostics/batch_code_loader.py \
    --mode outline
```

#### 模式 2：语义关键字定向抽取 (`targeted` 模式)
- **特点**：无需通读全代码，仅针对诊断关键概念（如 `stop_loss`、`persistence`、`friction`、`gating`、`turnover`）抽取包含该逻辑的完整函数块或类定义，并带有准确的起止行号。
- **运行命令**：
```bash
conda run -n finetf python FineFT/analysis/diagnostics/batch_code_loader.py \
    --mode targeted \
    --keywords stop_loss persistence friction turnover
```

#### 模式 3：全量代码一键打包 (`full` 模式)
- **特点**：一次性将指定代码文件拼接打包，并生成标准代码块与带行号视图，避免多次发起文件读取请求。
- **运行命令**：
```bash
conda run -n finetf python FineFT/analysis/diagnostics/batch_code_loader.py \
    --mode full \
    --output analysis_result/code_bundle.md
```

---

## 10. 实验数据与代码双引擎批处理协同

至此，系统形成了完整的数据与代码“双引擎批处理”工作流：
1. **数据与指标引擎**：`batch_diagnostics_loader.py` 一键汇聚实验数据、摩擦损耗、胜率、回撤及风控触发统计；
2. **代码与逻辑引擎**：`batch_code_loader.py` 一键提取源码架构大纲、函数签名与关键风控算法；
3. **协同优势**：在启动任何策略分析任务时，仅需**两次极速调用**，即可将实验全貌与代码细节以最紧凑的格式注入上下文，彻底告别盲目翻查文件与 Token 膨胀。
