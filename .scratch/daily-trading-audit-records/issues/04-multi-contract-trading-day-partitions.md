# 04: 生成多合约 TradingDay 分区

**What to build:** 让多日、多合约历史回测按照明确的 TradingDay 生成每日三份记录。夜盘事件使用行情中的 TradingDay 或显式映射文件归属，所有 ID、排序和独立合约账户身份在重复诊断时保持稳定。

**Blocked by:** 01: 建立最小审计级证据链.

**Status:** closed

- [x] 诊断入口支持行情内显式 TradingDay 和独立事件到 TradingDay 映射两种来源。
- [x] 缺少明确 TradingDay 归属时失败，不使用固定夜盘时钟或自然日期猜测。
- [x] 每个 TradingDay 目录准确包含当日的 Prediction、Trade 和 Account Records。
- [x] 夜盘、日盘、多日和多合约测试验证业务日期归属及稳定排序。
- [x] 每个账户明确标记为 independent_contract_wallet，跨合约记录不会被表述为共享组合账户。
- [x] 相同输入重复运行生成相同的记录 ID、关联关系和文件顺序。
