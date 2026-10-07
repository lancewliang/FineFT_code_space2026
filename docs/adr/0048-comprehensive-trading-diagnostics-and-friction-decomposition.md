# 0048. 全维度交易诊断、摩擦损耗归因与自动化研究产出物架构

为了彻底解决高层 VAE 路由与微观 RL 交易策略在回测分析中长期存在的“利润去向黑盒”（无法定量解耦模型择时毛利与手续费滑点磨损）以及“与 Buy & Hold 对比分析成本极高”（每次分析需临时手写 Python 脚本聚合向量）的痛点，我们决定在 `FineFT/analysis/diagnostics/` 构建标准化的全维度交易诊断模块 `trading_diagnostics.py`，并直接无缝集成到 `high_level_heurstic_fu_*.sh` 和 `final_result_fu_*.sh` 自动化流水线中。

## Status

accepted

## Context

1. **绝对收益落后与归因黑盒矛盾**：
   在商品期货 10min 验证集（12 支合约、30,351 步）的回测中，DiHFT 策略实现了 91.67% 的极高胜率和 5.55% 的极低最大回撤，但在绝对收益上（+2.54%）远低于单边牛市中 5x Buy & Hold 的表现（如 `fu2503` 5x B&H 达 +183.6%，而策略仅 +1.55%）。若缺乏底层摩擦拆解，研究者极易误判为“模型预测能力不足”；
2. **第一手实证揭示巨额摩擦吞噬**：
   深度数据挖掘表明，策略实际捕捉到的**盘面择时毛利润高达 58,410.00 元（毛收益率 +81.1%）**，是 5x B&H 总收益（18,580 元）的 3.1 倍。但由于 1,696 次高频调仓产生 5,201.7 万元天量换手，累积手续费（26,008 元）与订单簿深度滑点（30,571 元）合计达 **56,580.16 元**，吞噬了 **96.87%** 的总毛利；
3. **多维诊断碎片化与一次性脚本泛滥**：
   持仓时长分布、多空平微观分布、高层 VAE 宏观路由占比（Slot 0~9）以及三级风控事件（硬止损、单向冷静期、断路器熔断）分散存储在各个 `.npy` 历史轨迹中，缺乏一键自动化的归因报告生成工具。

## Considered Options

- **选项 1：维持现状，仅在临时需要时通过临时 Python 脚本提取**：
  *缺陷*：效率低下，跨合约指标聚合繁琐，极易因临时计算口径不一致（如合约乘数、保证金分母、日频步数）导致错误结论。
- **选项 2：仅在 `DiHFT_high_level_heurstic.py` 内部修改打印逻辑**：
  *缺陷*：无法服务于 `final_result` 测试集回测，也无法独立对历史 Arbitrary Trial 进行针对性复盘。
- **选项 3：独立模块化诊断引擎 + 流水线自动化嵌入（选中）**：
  1. **核心计算引擎 (`FineFT/analysis/diagnostics/trading_diagnostics.py`)**：
     - 提供纯 Python 标准类 `TradingDiagnosticsCalculator` 与 CLI 接口；
     - 严格遵循第一手计算原则：基于逐 step 价格跳动与仓位符号计算逐笔毛利（`gross_pnl`），对比逐 step 钱包余额变化精准解耦出真实手续费与滑点成本；
     - 全面聚合基准数据（1x B&H, 5x B&H, Alpha, Max Drawdown）；
     - 统计微观持仓结构（Long, Short, Flat 比例及平均/中位数持仓时间步数）；
     - 统计宏观 VAE 路由（Slot 0~8 及拒识 Slot 9 频次与占比）；
     - 统计风控拦截审计（ADR 0045 持久性跳过推理率、ADR 0047 硬止损、冷静期拦截、熔断步数）；
  2. **诊断三件套全景交付 (Diagnostic Triad Artifacts)**：
     - `diagnostics_summary.json`：全量高精度结构化归因指标（支持程序化二次消费与大模型无损解析）；
     - 标准化 CSV 表格：
       - `contract_pnl_friction.csv`（逐合约毛利、净利、手续费、滑点、换手总额、换手倍数）
       - `contract_behavior_risk.csv`（逐合约持仓分布、平均持仓周期、调仓次数、风控触发次数）
       - `macro_routing_distribution.csv`（逐合约 VAE 路由各 Slot 分流明细）
     - `diagnostics_summary.md`：自动排版的 Markdown 全景研究简报；
  3. **流水线无缝集成**：
     - 集成至 `high_level_heurstic_fu_*.sh`：在完成最优参数遴选后，自动针对选出的 Best Trial 运行全套诊断并落盘至 `analysis_result/DiHFT/high_level_heurstic/.../diagnostics/`；
     - 集成至 `final_result_fu_*.sh`：在测试集最终回测完成后，自动对 `result/DiHFT/final_result/.../` 运行诊断并落盘至 `result/DiHFT/final_result/.../diagnostics/`。

## Consequences

- **正面收益**：
  - **彻底破除黑盒，实现精准归因**：任何策略回测均能一眼看清究竟是“择时能力差”还是“换手磨损大”，直接为后续调大持久性（`action_persistence`）和费率校准提供量化证据；
  - **全自动零心智负担**：研究者执行常规的 `high_level_heurstic` 或 `final_result` 脚本后，即可自动在对应目录下获得完整的 JSON、CSV、Markdown 三件套产出物；
  - **支持独立 CLI 复盘**：支持传入任意历史回测目录 `--result_dir` 和数据目录 `--data_dir` 独立复盘生成诊断。
- **权衡与约束**：
  - 在大样本多合约（如上万步数据）下增加约 1~2 秒的纯 CPU 矢量化计算开销，对整体回测总时长几乎零影响。
