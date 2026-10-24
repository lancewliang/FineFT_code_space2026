# 07 独立回溯评估：现有资产可行性记录

检查日期：2026-10-24（Asia/Shanghai）。本记录只读取现有数据、manifest 与缓存；未运行训练、校准或历史回测。

当前资产不能支持正式独立结论。`dataset/10min/fu/train/fu2409.feather` 包含 2024-07-22 夜盘的 13 根 bar，而其原始记录的 `TradingDay=20240723`；valid 从 2024-07-23 开始，因此实际 TradingDay 跨越了 train/valid 边界。现有 feather 文件也没有显式 `trading_day` 列。除此之外，二维专家 Selection Manifest 的 `valid_root` 指向同一完整 valid 集，使它不能再作为独立迟滞校准段。

现有专家、VAE 与 valid quantile cache 是旧分区派生资产。它们没有经修正 TradingDay 分区重训的版本或内容哈希证明；cache 也没有正式来源 sidecar。名为 `final_result` 的既有输出实际使用 valid/absolute 路由，不能当作最终留出测试。当前运行时使用 independent-contract wallet 和事后组合汇总，不满足 ADR-0060 的共享执行账户条件。

恢复第 07 项前，实验负责人需要提供：

- TradingDay-atomic 的重建数据和版本化切分证据；
- 基于该切分重训且哈希锁定的专家、VAE、参考分布和特征资产；
- 在任何校准前确定的独立 fit、selection、calibration 与 final 时间段；
- 固定评分缓存及其来源清单；
- 符合 ADR-0060 的共享账户执行证据，或一份明确限定账户语义差异的研究设计；
- 在最终测试结果可见前固定的 switchback 时间窗和等待时间口径。

在这些证据到位前，已实现的冻结配置与审计可以用于受控行为验证，不能用于声称收益、回撤或交易摩擦的独立改善。
