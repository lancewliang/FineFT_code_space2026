# 01: 扩展双门控栈边界与公共审计契约

**What to build:** 建立可独立选择 legacy 或 tri-module 的高层门控边界，并让两套门控通过同一版本化决策与审计契约运行。现有 legacy 默认行为必须保持不变；tri-module 未就绪或资产不合法时必须明确 fail-closed，不能调用 legacy 作为隐藏回退。

**Blocked by:** None (can start immediately).

**Status:** ready-for-human

- [ ] 系统只接受 `legacy` 与 `tri_module` 两种高层门控栈取值，未指定时仍选择 legacy。
- [ ] 每次运行只构造并调用被选择的门控栈，不串联、投票或混用另一套门控逻辑。
- [ ] legacy 在相同输入和配置下保持现有可观察决策行为，且不要求任何 Utility Model 资产。
- [ ] tri-module 未注册、配置不完整或依赖资产校验失败时明确中止或禁止新增风险，不静默回退 legacy。
- [ ] 公共决策契约能表达门控栈、OOD 状态、路由证据、候选动作、拒绝基线、最终动作、效用、资格和拒绝原因。
- [ ] tri-module 专属字段在 legacy 输出中保持 null 或明确不适用，不触发额外候选推理来伪造数据。
- [ ] 运行配置快照、模型清单和审计记录均写出实际门控栈类型及契约版本。
- [ ] Utility Gate 的决策拒绝与交易所或执行层拒绝使用不同的类型和原因语义。
- [ ] 自动化测试覆盖默认 legacy、显式 legacy、显式 tri-module、未知取值、资产缺失和禁止静默回退。
