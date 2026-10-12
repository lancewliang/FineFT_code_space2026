# 01: 建立最小审计级证据链

**What to build:** 让一条普通策略推理经过模拟执行后，能够生成相互关联的 Prediction Record、Trade Record 和 Account Record，并为单个 TradingDay 发布三份可审阅的 CSV 和运行清单。用简单的开仓、持有、平仓路径贯通模型输出、执行事件、交易前后账户快照和离线诊断入口，同时固化领域术语和架构决策。

**Blocked by:** None (can start immediately).

**Status:** closed

- [x] 真实策略推理保存 short、flat、long 三个 Action Value、Action Value margin、路由证据、所选动作和冻结版本信息。
- [x] 账户审计历史包含初始快照及每个环境 step 后的快照，平仓不会清空已经记录的快照。
- [x] 普通开仓与平仓执行事件保存请求状态、实际状态、价格、数量、费用、盈亏证据及交易前后 Snapshot ID。
- [x] 离线诊断入口为一个 TradingDay 生成三份 CSV 和根级清单。
- [x] 三类记录使用确定性主键和可验证的外键连接。
- [x] 端到端测试从高层回测入口生成输入，再从离线诊断入口验证正式输出。
- [x] 领域词汇和 ADR 记录三类审计记录、显式 TradingDay、独立合约钱包和原子证据链的决策。
