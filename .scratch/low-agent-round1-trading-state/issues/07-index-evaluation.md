# 07: 迁移按索引独立评估入口及其检验测试

**What to build:** 使用户现用的按label/bin/context索引选择专家的独立评估入口，能够加载新状态模型、构造18维观测并执行完整评估循环，产出可核验的交易明细与指标。

**Blocked by:** 03: 让单进程 Low Agent 观察双向退出记忆与完整执行状态。

**Status:** ready-for-agent

用户已确认生成本工单。用户明确指定范围：`FineFT/RL/DiHFT/low_level/test_agent_index.py`，并覆盖其直接调用的模型、环境、评估shell入口与检验测试。可与04/05/06并行，不以完成老师构造或正式重训为前提。

- [ ] 更新ensemble模型构造、checkpoint加载及act_test交易输入；选定context实际执行真实18维Q网络前向，字段顺序、float32、设备与训练一致，旧4维checkpoint明确拒绝而非补零。
- [ ] 评估shell传入P、止损阈值、初始资金，入口解析后将其一致交给共享执行层；本轮P=2、止损3%、E_ref取传入初始资金，冷却/熔断固定关闭。
- [ ] 评估循环不再私有维护与模型快照不一致的remaining_persist、目标动作或止损pending；共享状态驱动请求、实际成交及next observation，风险抢占不重复执行/计时。
- [ ] 覆盖普通空仓、实际退出后空仓、多空记忆独立、止损退出拒单/重试、P步持续、动作不可用及跨合约/label/bin/reset隔离。
- [ ] 各initial_action评估初态具有可解释的一致持仓成本和单笔损益摘要；行情预热只读过去，预热不足明确处理，不读取未来或伪造退出记录。
- [ ] 交易明细、动作数/成交数、真实费用、损益及汇总指标仍按实际执行记录生成；保持existing输出开关及目录遍历行为，不以成功前向代替完成评估验收。
- [ ] 更新`FineFT/tests/rl/test_test_agent_index.py`中真实4维ensemble测试、stub和fixture到新契约；联动覆盖`test_qnet_trading_info.py`、`test_evaluation_stop_loss.py`及交易过程状态测试。断言实际字段值与状态演化，不能只改张量长度。
- [ ] 新维度checkpoint可用确定性测试权重完成CPU短轨迹评估，不依赖历史训练；GPU可用时覆盖相同输入设备路径，缺少GPU明确记录未验证。
- [ ] 检查同样构造ensemble或消费trading_info的平均评估等共享调用者，纳入必要的维度迁移和回归，避免按索引入口通过而其他评估入口损坏。


**范围依据：** [已确认实施范围与依赖](../spec.md)。
