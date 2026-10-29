# 底层独立评估持仓止损

> 历史设计记录：后续用户要求去除止损记录、配置保存及按阈值分目录；当前设计以相邻的 low-level-evaluation-stop-loss-v2 规格为准。

Status: complete

## 已确认决策

2026-10-29，用户回答“q1 a 其他按推荐”：

- 仅底层 Agent 独立评估启用；不将本功能加入训练，不改变高层现有风控。
- 触发比例沿用既有高层口径：`loss_rate = -unrealized_pnl / (abs(position) * current_markprice)`；空仓不触发。
- “单笔”按当前持仓划分：同向加减仓不结束当前持仓，完全平仓或反手才结束。比例依当前实际持仓与浮亏计算，不额外引入批次账本。
- 第二轮用户回答“默认开启比例0.03 其它按推荐”：默认阈值 `0.03`（3%），保留 `0` 显式关闭能力。
- 每个可观察 step 在执行策略动作前检查；达到阈值打断动作保持，以全部平仓为目标，当步不反手。
- 一旦触发，持续请求退出，直到实际空仓；即使浮亏比例回到阈值以内也不撤销退出状态。期间禁止加仓和反手，保留真实盘口、涨跌停和部分成交约束。
- 触发和退出完成分开记录。止损退出后下一步恢复正常决策，允许同方向重新开仓；本次不加入冷静期或断路器。

第三轮用户回答“按推荐”，确认：

- 三个独立评估入口统一支持：`test_agent_index.py`、`evaluate_sub_agents.py`、`test_agent_average.py`。
- 带止损结果正常用于后续选模，明确标注规则贡献，关闭止损结果单独保存。
- 保存配置、触发次数、退出完成次数、退出原因，并隔离不同阈值的结果。

## 接口与产物

- 独立评估 CLI：`--stop_loss_return_threshold 0.03` 默认开启；`0` 关闭；拒绝负数、NaN、Inf。
- 批量评估 Python 接口：`evaluates(..., stop_loss_return_threshold=0.03)`。训练结束时调用该接口显式传入 `0.0`，保持训练所属评估的原有策略语义。
- Shell 评估与选模入口通过 `STOP_LOSS_RETURN_THRESHOLD` 覆盖，默认 `0.03`。
- 评估结果：`epoch_N/evaluations/stop_loss_0.03/`；关闭时为 `stop_loss_disabled/`；其余比例按浮点数规范字符串分目录。既有 `epoch_N/slope` 等历史评估不覆盖。
- 指定子 Agent 的结果位于上述目录的 `slope/` 或 `volatility/` 下；配置文件在阈值目录根部。NPY 与 CSV 中的 `stop_loss_summary` 逐数据文件保存计数、退出未完成状态与逐步退出事件。
- 每步交易明细附加触发比例、触发/请求/完成标志、决策原因和动作来源；止损规则步没有伪造的模型 Q 值。
- 批量评估返回逐回放风险摘要，并将结果保存为阈值目录内的 `<checkpoint_filename>.sub_agent_results.json`；日志按阈值隔离。
- 集成平均策略的 `average_analysis_result.npy` 按阈值隔离，每个文件单独记录风险摘要。
- 两维选模 CLI 默认只读取 `0.03` 目录，不回退混读历史结果；加载权重仍定位原始 `epoch_N/trained_model.pkl`。标准 3% 选模产物保持原路径，保障高层现有默认清单路径可用；其他比例的选模产物另分阈值目录。清单保存所用评估阈值。
- 两维选模 Python API 的 `SelectionConfig.stop_loss_return_threshold=None` 保留历史产物读取能力；显式传入数值才读取对应阈值目录，CLI 始终显式传入所选比例。
- 单维选模增加 `--label_type`（默认 volatility），读取对应标签类型及阈值目录，产物亦按阈值、标签类型隔离。

## 部分成交支持

真实盘口测试暴露原环境在部分平仓后，因实际剩余仓位不属于离散档位而断言失败。新增显式 `allow_partial_positions` 能力，默认 False，仅开启止损的独立评估启用。实际持仓、盈亏和交易状态保留真实数量；固定维度网络的仓位动作编码采用最近档位。退出未完成期间不进行策略推断，因此该编码不替代剩余数量，也不撤销退出锁定。

训练和高层环境的默认参数及风控行为不变。触发比例不混入 `single_holding_return_rate`，不追溯加入已实现的部分平仓损益，也不预估未来退出摩擦。

## 事实与约束

- 已有高层规则见 `docs/adr/0047-contract-level-hard-stop-loss-cooldown-and-circuit-breaker.md`。
- 当前比例是浮亏触发线，不保证最终成交亏损上限；跳空、深度和涨跌停约束仍然有效。
- 不可直接复用 `trading_info.single_holding_return_rate`，其分母与当前名义金额不同。
- 代码存在其他在途修改，后续实现须保留已有工作。

## 实施状态

2026-10-29，用户已确认设计并完成代码接入。

## 使用示例

在仓库根目录、已激活 `finetf` 环境下运行。以下示例仅评估指定 epoch 与标签类型，不启动训练：

```bash
conda activate finetf
EPOCH_START=35 EPOCH_END=35 LABEL_TYPE=slope STOP_LOSS_RETURN_THRESHOLD=0.03 \
  bash FineFT/script/test/DiHFT/low_level/test_util_fu_10.sh
```

关闭止损时将 `STOP_LOSS_RETURN_THRESHOLD` 设为 `0`；直接调用评估 Python 入口时对应 `--stop_loss_return_threshold 0`。选模时需使用同样的阈值，例如：

```bash
STOP_LOSS_RETURN_THRESHOLD=0.03 \
  bash FineFT/script/analysis/pick_agent/low_level_fu_10.sh
```

当前新代码只做了受控测试，没有运行用户真实历史数据的整轮评估，也没有重新训练、覆盖权重或发布模型。

## 验证记录

- 全部 Python 验证均在 `conda activate finetf` 后运行。
- 相关回归共 182 项通过：新止损边界与三个入口的集成测试、止损开启/关闭结果隔离、长动作保持抢占、真实盘口多空涨跌停拒绝/部分退出/立即重入、真实 checkpoint 推断与 Q 诊断、两维选模读取阈值目录并回到原 checkpoint、批量子进程评估、训练所属评估、环境测试全目录。
- 新增测试包含 shell 命令替身验证：默认 0.03、自定义 0.015、显式关闭 0 均能从评估与选模脚本转发；不启动真实训练和评估任务。
- 新模块和新测试通过 Ruff；涉及 Python 文件通过 `py_compile`；评估和选模 shell 通过 `bash -n`；`git diff --check` 通过。
- 排除了一项已有失败 `test_test_util_fu_10_script_configuration`：该断言要求 `numactl --cpunodebind=1 --preferred=1`，但本次修改前的 HEAD 脚本亦不包含它；本次保留该无关断言与既有脚本行为。此前直接运行此测试已观察到失败，最终回归使用 `-k 'not test_test_util_fu_10_script_configuration'`。
