## Why

Stage II validation 回测中的低层 Agent 当前只使用环境原始可用动作集，无法按 Label 方向与强度约束逆向调仓，也缺少真实持仓成本支撑被拦截动作的止损决策。同时，涨跌停成交能力仍可被直接 `step` 调用绕过，会使行为轨迹偏离 Label 语义和真实商品期货成交约束。

## What Changes

- 新增独立的 `test_agent_index_with_guard.py` validation 测试入口，通过 CLI 显式提供每个 Label 的方向语义，不读取 `label_semantics.json`。
- 在环境外部新增逐步 Label 动作守卫，使用滚动逆 Label 动作配额将模型原始动作保留或修正为最终动作。
- 小幅 Label 默认逆向比例 40%，大幅 Label 默认 20%，涨跌停 Label 为 0；比例、窗口和逆向持仓止损阈值均可通过 CLI 调整。
- 被配额拦截时，同向或空仓保持不动；逆向持仓相对开仓价达到对称 3% 不利变动时尝试平仓，否则保持不动。
- Futures Trading Environment 新增当前持仓开仓价和当前持仓均价，按真实成交、滑点和已发生开仓税费维护，并通过环境属性与 `info` 对外暴露。
- Futures Trading Environment 依据当前时间步 `is_limit_down` / `is_limit_up` 实施涨跌停成交约束：跌停禁止卖出，涨停禁止买入。
- 涨跌停不可成交动作既从 `avaliable_action` 排除，也在直接交给 `step` 时被拒绝并保持实际仓位。
- 逐步明细区分模型原始动作与最终动作，并记录守卫原因、滚动配额、开仓价和持仓均价供审计。
- **BREAKING**: 原 type-index 守卫实验入口重命名为 `test_agent_index_with_guard.py`；本变更不自动迁移 shell、IDE 或其他调用方对旧名称的引用。

## Capabilities

### New Capabilities

- `fineft-label-action-guard`: 定义 Stage II validation 带守卫测试入口的 CLI Label 语义、滚动配额、动作降级和审计输出。
- `fineft-position-cost-state`: 定义 Futures Trading Environment 的当前持仓开仓价、当前持仓均价、真实开仓成交口径与 `info` 契约。
- `fineft-price-limit-execution`: 定义基于 `is_limit_down` / `is_limit_up` 的买卖不可成交规则、可用动作与 `step` 双重防护。

### Modified Capabilities


## Impact

- 影响区域：Futures Trading Environment 和 Stage II 低层 Agent validation 行为测试；不改变 Stage I 训练、Stage III Meta Router 或 Agent 选择逻辑。
- 环境结算结果需要暴露可区分开仓腿的成交数量、成交额和开仓税费元数据，以支持普通开仓、加仓、部分成交和反手的成本维护。
- 环境 `info` 新增独立成本价字段，不加入现有四维 Trading Process Feature，不改变 Q 网络输入或 checkpoint 契约。
- 无新外部依赖，无新 GPU 需求；每条行为轨迹仅增加固定大小滚动状态和少量审计列。
- 默认研究场景为商品期货 validation 切片（如 fu 的 10min/30min 实验）；沿用现有杠杆、交易费、订单簿深度、滑点、资金费、维持保证金和强平语义。
