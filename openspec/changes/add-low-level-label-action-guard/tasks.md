## 1. 当前持仓成本状态

- [ ] 1.1 新增 `FineFT/tests/env/test_position_cost_state.py`，通过 `Base_Env.reset/step` 先覆盖空仓/初始非零仓位、多空开仓、同向加仓、部分减仓、完全平仓、反手和只平未开的外部行为。
- [ ] 1.2 扩展 `FineFT/env/env_class/futures_util.py` 的 `WalletChangeResult` 与普通开仓/加仓/反手返回路径，暴露开仓腿实际数量、成交额和开仓税费，并保持现有六值 tuple-compatible 契约。
- [ ] 1.3 在 `FineFT/env/env_class/base_env.py` 维护 `current_holding_opening_price` / `current_holding_average_price`，实现开仓、加仓、减仓、平仓、反手和 Initial-action 生命周期，并通过属性及所有 `reset/step info` 分支暴露。
- [ ] 1.4 运行 `source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/env/test_position_cost_state.py FineFT/tests/env/test_commodity_env.py FineFT/tests/env/test_futures_reverse_position.py FineFT/tests/env/test_trading_process_features.py -q`，确认成本价、反手和四维 Trading Process Feature 回归通过。

## 2. 涨跌停成交双重防护

- [ ] 2.1 新增 `FineFT/tests/env/test_price_limit_execution.py`，通过 `Base_Env.reset/step` 先覆盖跌停禁卖、涨停禁买、普通行情不受限、绕过掩码和 Reverse Position 目标。
- [ ] 2.2 在 `FineFT/env/env_class/base_env.py` 的 reset 与 step 后续可用动作计算中，根据当前 `is_limit_down/is_limit_up` 排除需要禁止买卖方向的目标仓位。
- [ ] 2.3 在 `FineFT/env/env_class/base_env.py::step` 执行结算前增加同口径审核，对绕过 `avaliable_action` 的不可成交动作保持实际仓位且不产生税费、滑点或已实现盈亏。
- [ ] 2.4 运行 `source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/env/test_price_limit_execution.py FineFT/tests/env/test_limit_reward.py FineFT/tests/env/test_compute_limit_reward.py FineFT/tests/env/test_futures_reverse_position.py -q`，确认硬成交约束不改变 reward shaping 和非涨跌停反手语义。

## 3. Stage II 逐步 Label 动作守卫

- [ ] 3.1 将 `FineFT/RL/DiHFT/low_level/test_agents_type_index.py` 重命名为 `FineFT/RL/DiHFT/low_level/test_agent_index_with_guard.py`，保留原文件未提交内容，且不更新 shell、`.vscode/`、文档或其他调用方。
- [ ] 3.2 新增 `FineFT/tests/rl/test_test_agent_index_with_guard.py`，先覆盖 CLI 七类语义映射、完整性校验、参数边界、向下取整容量和轨迹起始完整配额。
- [ ] 3.3 在带守卫入口实现独立行为轨迹滚动状态，以最终动作分类逆向开仓/加仓/减仓，并排除保持与完全平仓的逆向分子计数。
- [ ] 3.4 在 Q 网络使用环境原始 `avaliable_action` 生成原始动作后应用守卫，实现 `allowed`、`quota_hold`、`stop_loss_close` 降级，使用环境开仓价完成多空对称阈值判断。
- [ ] 3.5 扩展带守卫逐步明细，记录 `proposed_action`、最终 `action`、守卫原因、当前逆 Label 数、容量、开仓价和持仓均价；换手率与动作变化改为最终动作口径，仓位指标保持执行后仓位口径。
- [ ] 3.6 增加完整行为轨迹测试，覆盖 20%/40%/0 配额、窗口滑出后恢复、原始/最终动作分离、被拦截保持、恰好 3% 止损、非主动止损、env 未完成目标时仍记录最终动作，以及涨跌停阻止守卫平仓。

## 4. 定向验证与回滚准备

- [ ] 4.1 运行 `source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_test_agent_index_with_guard.py FineFT/tests/env/test_position_cost_state.py FineFT/tests/env/test_price_limit_execution.py -q`。
- [ ] 4.2 运行 `source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_test_agent_index.py FineFT/tests/env/test_commodity_env.py FineFT/tests/env/test_futures_reverse_position.py FineFT/tests/env/test_trading_process_features.py FineFT/tests/env/test_limit_reward.py -q`。
- [ ] 4.3 运行 `source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/test_agent_index_with_guard.py FineFT/env/env_class/base_env.py FineFT/env/env_class/futures_util.py`。
- [ ] 4.4 运行 `openspec validate add-low-level-label-action-guard --strict`，并在实现交付中说明未运行真实 fu 10min/30min validation 全量回测（需要本地数据、checkpoint，运行时间远高于单元测试）。
- [ ] 4.5 回滚时恢复原 type-index 文件名、删除守卫与新审计列，停止消费成本价 `info` 字段和涨跌停防护；保留 `WalletChangeResult` 具默认值的向后兼容扩展，避免大范围调用方反向迁移。
