# add-test-agent-csv-outputs

## 背景与目标

`FineFT/RL/DiHFT/low_level/test_agent_index.py` 当前将测试汇总结果保存为 `analysis_result.npy`。该格式不方便直接阅读，也缺少 `df_path` 与列表结果的显式对应关系。用户希望增加 CSV 输出，便于查看每个 `label / initial_action / bin_index` 组合的测试表现。

此外，用户希望在需要观察策略行为时，能够输出每个 epoch 对应的交易动作明细 CSV。该明细需要按时间步记录行情、动作、执行前后仓位、真实发生的交易次数、动作变更次数、真实手续费、真实已实现利润、真实滑点和账户价值等信息。

## 用户场景

- 运行 `test_agent_index.py` 后，直接打开 CSV 查看每组测试结果，而不是加载 `.npy`。
- 查看某个 epoch 的低层 agent 在验证集上的逐时间步动作，分析 action 变化和真实持仓变化是否一致。
- 对比真实已实现利润、手续费、滑点、现金、权益和名义资产价值，定位策略行为或环境结算问题。

## 设计方向

采用最小旁路输出设计：

1. 保留现有 `analysis_result.npy`。
2. 默认在同一 `epoch_path` 下新增 `analysis_result.csv`。
3. 汇总 CSV 保持每个 `(label, initial_action, bin_index)` 一行。
4. 汇总 CSV 增加 `df_path` 列，和 `reward_sum / df_length / turnover` 按顺序对齐。
5. 汇总 CSV 中列表字段使用 JSON 数组字符串，方便阅读和后续解析。

交易明细采用显式开关：

1. 新增 `--save_trading_detail_csv`。
2. 不传该参数时，不生成交易明细 CSV，避免默认输出过大的文件。
3. 传入该参数时，每个 epoch 在 `epoch_path` 下输出 `trading_action_detail_epoch_<epoch_num>.csv`。
4. 明细 CSV 每行对应一个具体时间步，覆盖所有 `label / df_path / initial_action / bin_index` 轨迹。
5. 为了真实记录手续费和已实现利润，需要在环境交易计算链路显式暴露每步 `commission_fee_step`、`realized_pnl_step` 和 `slippage_step`，不能从余额变化中反推。

## 关键决策

- 汇总 CSV 默认生成；交易明细 CSV 通过 `--save_trading_detail_csv` 显式开启。
- 汇总 CSV 字段为 `label, initial_action, bin_index, df_path, reward_sum, df_length, turnover`。
- 汇总 CSV 的 `df_path / reward_sum / df_length / turnover` 使用 JSON 数组字符串。
- 交易明细文件名为 `trading_action_detail_epoch_<epoch_num>.csv`。
- 交易明细 OHLCV 字段采用“存在即写”：`open/high/low/close/volume/mark_price` 中数据有哪列写哪列。
- 交易明细同时记录 action id、目标仓位/杠杆、执行前仓位/杠杆和执行后真实仓位/杠杆。
- `action_change_step` 统计 action id 是否相对上一时间步发生变化。
- `trade_count_step` 统计环境执行后真实 `position` 或 `leverage` 是否发生变化。
- `cumulative_action_change_count` 和 `cumulative_trade_count` 分别累计上述 step 级计数。
- `margin_balance` 使用环境权益口径：`wallet_balance + unrealized_pnl`。
- `notional_asset_value` 使用名义资产口径：`mark_price * position_after`。
- `total_value` 使用环境主口径，等于 `margin_balance`。
- 真实手续费、已实现利润和滑点必须由 env/交易计算显式提供；如果取不到，不能用 0 或余额差静默替代。

## 范围边界

**包含：**
- 在 `test_agent_index.py` 中新增默认汇总 CSV 输出。
- 在汇总结果中补充 `df_path`，使列表字段可解释。
- 新增 `--save_trading_detail_csv` 参数。
- 开启参数时输出每个 epoch 的交易动作明细 CSV。
- 为交易环境或交易计算 helper 补充真实手续费、真实已实现利润和真实滑点的显式记录/返回。
- 增加轻量单元测试覆盖汇总 CSV、明细 CSV 开关、字段格式、JSON 数组解析、动作变更计数和真实交易计数。

**不包含（本次）：**
- 不移除或重命名 `analysis_result.npy`。
- 不改变现有测试 shell 的默认行为，除非用户显式加入明细 CSV 参数。
- 不新增独立 npy/json 轨迹格式。
- 不强制所有数据集必须有完整 `open/high/low/close/volume` 字段。
- 不运行真实长测试或真实模型训练。
- 不修改与本需求无关的高层 agent、训练入口或分析脚本。

## 字段设计

汇总 CSV 字段：

```text
label,
initial_action,
bin_index,
df_path,
reward_sum,
df_length,
turnover
```

交易明细 CSV 字段：

```text
label,
df_path,
initial_action,
bin_index,
timestep,
timestamp,

open,
high,
low,
close,
volume,
mark_price,

action,
target_position,
target_leverage,
position_before,
leverage_before,
position_after,
leverage_after,
action_change_step,
trade_count_step,
cumulative_action_change_count,
cumulative_trade_count,

step_reward,
realized_pnl_step,
cumulative_realized_pnl,
commission_fee_step,
cumulative_commission_fee,
slippage_step,
cumulative_slippage,

wallet_balance,
unrealized_pnl,
margin_balance,
notional_asset_value,
cash_balance,
total_value
```

## 错误处理

- `analysis_result.csv` 写入失败时直接抛错。
- `trading_action_detail_epoch_<epoch_num>.csv` 只在传入 `--save_trading_detail_csv` 时写入。
- OHLCV 字段缺失不报错；存在的行情字段写入 CSV。
- 真实手续费、已实现利润、滑点字段取不到时应暴露错误或补齐 env 字段，不静默写入错误数据。
- 不因为交易明细 CSV 写入失败而主动删除已生成的 `analysis_result.npy` 或 `analysis_result.csv`。

## 验收标准

- [ ] `test_agent_index.py` 保留 `analysis_result.npy` 输出。
- [ ] `test_agent_index.py` 默认新增同目录 `analysis_result.csv`。
- [ ] `analysis_result.csv` 每个 `(label, initial_action, bin_index)` 一行。
- [ ] `analysis_result.csv` 包含 `df_path`，并与 `reward_sum / df_length / turnover` 顺序对齐。
- [ ] `df_path / reward_sum / df_length / turnover` 可通过 JSON 解析为数组。
- [ ] 新增 `--save_trading_detail_csv` 参数。
- [ ] 不传 `--save_trading_detail_csv` 时不生成交易明细 CSV。
- [ ] 传入 `--save_trading_detail_csv` 时生成 `trading_action_detail_epoch_<epoch_num>.csv`。
- [ ] 交易明细 CSV 每行对应一个时间步，并包含上下文、行情、动作、仓位、计数、费用、利润、滑点和账户价值字段。
- [ ] `action_change_step` 与 `cumulative_action_change_count` 按 action id 变化统计。
- [ ] `trade_count_step` 与 `cumulative_trade_count` 按真实 position/leverage 变化统计。
- [ ] `commission_fee_step` 和 `cumulative_commission_fee` 使用真实手续费，不从余额变化反推。
- [ ] `realized_pnl_step` 和 `cumulative_realized_pnl` 使用真实已实现利润，不从余额变化反推。
- [ ] 轻量单元测试覆盖汇总 CSV 和交易明细 CSV 的核心字段与边界。
- [ ] 按项目要求在 `conda activate finetf` 环境中运行相关 python/pytest 验证。

