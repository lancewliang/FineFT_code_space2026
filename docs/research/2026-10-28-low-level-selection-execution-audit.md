# 低层专家选拔与 expert_hysteresis 执行一致性审计

日期：2026-10-28  
范围：只读核对 `fu/10min_parallel` 当前两维低层选择产物、`test_agent_index.py` 的验证回测与 `expert_hysteresis` 生产路由。没有修改生产代码、配置或 Selection Manifest。

后续进展：本文提出的纯权益重算和完整路由对照已完成。26候选局部池的冠军未变；18组单专家替换及1组组合回放找到40.643%收益、3.583%最大回撤的研究候选。见[最新结果与优化顺序](2026-10-28-low-level-optimization-results.md)。下文保留审计时的事实与推论。

## 结论

选择器默认使用 `reward`，它大部分时间等于账户盯市权益变化（已包含手续费和执行滑点），但当前低层验证命令还启用了涨跌停奖励塑形。实测的 epoch 55 volatility 详情 CSV 中，999,603 条记录有 12,554 条奖励与权益变化不一致，差值绝对值最大 1.5 元、合计 -7,269.06 元；其中 bin 6 的 90,873 条记录有 947 条不一致，合计 -660.86 元。因此，选择器的“net_pnl”列名不能解读成纯现金/账户收益，塑形项确实改变了候选得分。这个差值在不同候选、方向和涨跌停状态下可正可负，不能据此断言当前候选被高估。

另一个实质差异是测试样本组织方式：低层评估把每个状态标签切片独立重置，并分别以空头、空仓、多头作为起始动作；高层运行则在完整合约时间线上延续仓位、账户、路由状态，并套用硬止损、冷静期、断路器和非主力合约限制。测试脚本与高层配置都使用 6 步动作持久性，但低层测试没有覆盖专家在路由切换时沿用前一动作的情形。因此，低层分数可以筛掉明显弱的专家，却不能单独证明被选中的专家在真实路由生命周期中能拿到同样的收益。

当前产物记录 `reward_metric="reward"`、`enable_hard_stop_loss_gate=false`，并且 `mean_net_return_main_*` 由同一 `net_pnl` 字段汇总；硬止损比率虽有计算和输出，但没有参与候选门禁。现有数据不足以声称需要改写选拔方法；需要用同一冻结配置做完整路由回放，确认低层候选排序能否转化成实际账户收益。

## 可复核事实

### 1. `reward` 是盯市权益增量加塑形，不等于纯粹现金净收益

- 选择器对 `reward_metric="reward"` 的处理，是直接把明细里的 `reward` 复制到 `net_pnl`（[FineFT_two_dimensional_agent_selector.py:906](../../FineFT/analysis/pick_agent/FineFT_two_dimensional_agent_selector.py:906)）。指标按每个合约和初始动作计算奖励和，再除以转移步数；当前 Manifest 使用 `contract_equal` 汇总方式（[FineFT_two_dimensional_agent_selector.py:1182](../../FineFT/analysis/pick_agent/FineFT_two_dimensional_agent_selector.py:1182)、[Manifest](../../analysis_result/DiHFT/low_level/fu/10min_parallel/two_dimensional_selection/two_dimensional_selection_manifest.json)）。
- 环境一般步骤的基础奖励为“钱包余额 + 未实现盈亏”的权益变化，减去 turnover penalty，再加涨跌停奖励（[base_env.py:1041](../../FineFT/env/env_class/base_env.py:1041)）。当前 `test_agent_index.py` 未传 turnover penalty，环境默认值为 0（[base_env.py:105](../../FineFT/env/env_class/base_env.py:105)、[base_env.py:111](../../FineFT/env/env_class/base_env.py:111)）。
- 低层验证脚本显式传入 6 步持久性、¥6,000 初始钱包、5 档盘口、0.03% 手续费以及多空估算滑点费率 0；没有传 `--no_enable_limit_reward`（[test_util_fu_10.sh:32](../../FineFT/script/test/DiHFT/low_level/test_util_fu_10.sh:32)）。而 `test_agent_index.py` 的 `--enable_limit_reward` 默认开启；只有缺少 `UpperLimitPrice` 或 `limit_up_single_sided_ratio` 列时才会关闭（[test_agent_index.py:202](../../FineFT/RL/DiHFT/low_level/test_agent_index.py:202)、[test_agent_index.py:852](../../FineFT/RL/DiHFT/low_level/test_agent_index.py:852)）。抽查的 fu2409 validation 切片两列均存在，环境另外要求的涨跌停列也由 `initiate_base_env` 映射传入（[base_initiate.py:69](../../FineFT/env/env_initiate/base_initiate.py:69)）。
- 高层生产回放调用同一个环境工厂时没有传 `enable_limit_reward`，该参数默认是 `False`（[vae_routing_util.py:2323](../../FineFT/RL/DiHFT/high_level/vae_routing_util.py:2323)、[base_initiate.py:35](../../FineFT/env/env_initiate/base_initiate.py:35)）。所以“低层选拔得分”与“高层实际运行奖励”的涨跌停项不一致。
- 我按每个 `(数据文件, 标签, 初始动作, 分箱索引)` 分组，用 CSV 中的“浮动总价值”算权益差分，再和“单步奖励”比较。epoch 55 volatility 详情的 999,603 条记录里有 12,554 条不同；差值总和 -7,269.063496 元。仅 bin 6（所有标签、初始动作和合约切片）有 947 条不同，差值总和 -660.864512 元。这是所有候选分箱的合计，不能直接等同于某一个 Slot 的选择分数。
- 结果重现命令（按仓库约定激活 `finetf` 环境）：

  ```bash
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate finetf
  python - <<'PY'
  import polars as pl

  path = "result/DiHFT/low_level/fu/10min_parallel/weights_advantage_pretrain/epoch_55/volatility/trading_action_detail_epoch_55.csv"
  keys = ["数据文件", "标签", "初始动作", "分箱索引"]
  df = pl.read_csv(path, columns=keys + ["时间步", "单步奖励", "浮动总价值"])
  df = df.sort(keys + ["时间步"]).with_columns(
      pl.col("浮动总价值").shift(1).over(keys).alias("previous_equity"),
      pl.col("时间步").min().over(keys).alias("first_step"),
  ).with_columns(
      pl.when(pl.col("时间步") == pl.col("first_step"))
      .then(pl.col("浮动总价值") - 6000.0)
      .otherwise(pl.col("浮动总价值") - pl.col("previous_equity"))
      .alias("equity_delta")
  ).with_columns(
      (pl.col("单步奖励") - pl.col("equity_delta")).alias("reward_diff")
  )
  print(df.select(
      pl.len().alias("rows"),
      (pl.col("reward_diff").abs() > 1e-6).sum().alias("different_rows"),
      pl.col("reward_diff").sum().alias("sum_reward_minus_equity_delta"),
  ))
  print(df.filter(pl.col("分箱索引") == 6).select(
      pl.len().alias("bin6_rows"),
      (pl.col("reward_diff").abs() > 1e-6).sum().alias("bin6_different_rows"),
      pl.col("reward_diff").sum().alias("bin6_sum_reward_minus_equity_delta"),
  ))
  PY
  ```

命令以 6,000 元作为每段低层测试的初始权益；这是当前低层验证脚本实际传入的本金。首次样本使用“当前总权益减初始权益”，后续样本按同一片段内前后总权益差计算。

### 2. 不应直接把 `reward_metric` 改成 `realized_net_pnl`

`realized_net_pnl` 分支确实会将 `realized_pnl - commission_fee - slippage` 作为选拔字段（[FineFT_two_dimensional_agent_selector.py:910](../../FineFT/analysis/pick_agent/FineFT_two_dimensional_agent_selector.py:910)）。但环境的 `realized_pnl_step` 是平仓后将旧浮盈、成交价和参考价合并后的实现盈亏；例如多头平仓公式已经包含成交价相对 mark 的差额（[futures_util.py:740](../../FineFT/env/env_class/futures_util.py:740)），并另记一项 `slippage_step`（[futures_util.py:751](../../FineFT/env/env_class/futures_util.py:751)）。因此再减一次滑点会重复扣除成交价差；同时只统计平仓实现盈亏会漏掉样本终点仍持有仓位的未实现盈亏。这个字段不等于完整的净账户权益变化，不能作为不经验证的替代指标。

要得到和账户实际变化匹配、又去除奖励塑形的比较值，应从同一逐步账户审计记录计算 `wallet_balance + unrealized_pnl` 的前后差，并从选拔分数中排除塑形项；历史详情 CSV 已包含“浮动总价值”，可用于核验权益变化。重新排名前应确认首步基准、片段边界、未平仓盯市和单位口径。

### 3. 低层验证与高层执行的持仓生命周期不同

- `test_agent_index.py` 对每个 feather 标签切片、初始动作和专家分箱分别新建环境；初始仓位由 `initial_action` 映射，初始保证金按切片首个 mark price 计算（[test_agent_index.py:803](../../FineFT/RL/DiHFT/low_level/test_agent_index.py:803)、[test_agent_index.py:833](../../FineFT/RL/DiHFT/low_level/test_agent_index.py:833)、[test_agent_index.py:857](../../FineFT/RL/DiHFT/low_level/test_agent_index.py:857)）。每段都从 `reset()` 开始（[test_agent_index.py:883](../../FineFT/RL/DiHFT/low_level/test_agent_index.py:883)），在所有可行动作初始仓位上重复评分（[test_agent_index.py:742](../../FineFT/RL/DiHFT/low_level/test_agent_index.py:742)）。
- 该低层测试循环只执行策略动作持久性和环境可用动作约束，没有调用高层 router 的 3% 浮亏硬止损、12 步方向冷静期、连续止损断路器或非主力合约防御（其循环见 [test_agent_index.py:894](../../FineFT/RL/DiHFT/low_level/test_agent_index.py:894)）。当前 Manifest 也明确 `enable_hard_stop_loss_gate=false`；门禁默认关闭，即便选择器计算了止损 episode ratio（[Manifest](../../analysis_result/DiHFT/low_level/fu/10min_parallel/two_dimensional_selection/two_dimensional_selection_manifest.json)、[selector.py:1777](../../FineFT/analysis/pick_agent/FineFT_two_dimensional_agent_selector.py:1777)）。
- 高层 router 在完整合约评估上创建单一环境（[vae_routing_util.py:2298](../../FineFT/RL/DiHFT/high_level/vae_routing_util.py:2298)、[vae_routing_util.py:2323](../../FineFT/RL/DiHFT/high_level/vae_routing_util.py:2323)）；动作会先经过硬止损及冷静期（[vae_routing_util.py:1929](../../FineFT/RL/DiHFT/high_level/vae_routing_util.py:1929)、[vae_routing_util.py:2099](../../FineFT/RL/DiHFT/high_level/vae_routing_util.py:2099)），再经过非主力合约的开仓、加仓、反手限制（[vae_routing_util.py:1974](../../FineFT/RL/DiHFT/high_level/vae_routing_util.py:1974)、[vae_routing_util.py:2125](../../FineFT/RL/DiHFT/high_level/vae_routing_util.py:2125)）。
- 6 步动作持久性配置是对齐的：低层验证脚本传 `--action_persistence 6`，当前高层冻结配置也为 6。但低层测试固定单一 `bin_index` 跑完整个切片；高层每步重新选路，选路改变后仍可能在剩余持久步内复用 `current_action`，而不是马上询问新 Slot 的模型（[vae_routing_util.py:2045](../../FineFT/RL/DiHFT/high_level/vae_routing_util.py:2045)）。低层指标没有覆盖这个“状态切换时继承旧动作”的情形。
- 开启的 `main_contract_sub_gate` 只是检查主力合约样本上的净回报是否为正（不足 100 步时不执行门禁），不是低层仿真中的交易限制（[selector.py:1757](../../FineFT/analysis/pick_agent/FineFT_two_dimensional_agent_selector.py:1757)）。实际禁止非主力新开仓/加仓的是高层 router。两者回答的问题不同。

### 4. 得分是每步均值筛选，不是组合净收益预测

当前 manifest 选择配置为 `contract_equal`、`lcb_z=1.0`、`min_slice_steps=30`、`enable_hard_stop_loss_gate=false` 和 `reward_metric="reward"`。选择器对每个合约和起始动作计算每步回报，再跨初始动作与合约聚合并计算 LCB（[selector.py:1198](../../FineFT/analysis/pick_agent/FineFT_two_dimensional_agent_selector.py:1198)、[selector.py:1227](../../FineFT/analysis/pick_agent/FineFT_two_dimensional_agent_selector.py:1227)、[selector.py:1287](../../FineFT/analysis/pick_agent/FineFT_two_dimensional_agent_selector.py:1287)）。对有联合标签的 Slot，使用 volatility-run 与 slope-run 两套已生成轨迹的较低 LCB（[selector.py:1677](../../FineFT/analysis/pick_agent/FineFT_two_dimensional_agent_selector.py:1677)、[selector.py:1684](../../FineFT/analysis/pick_agent/FineFT_two_dimensional_agent_selector.py:1684)）。

这有助于覆盖不同初始仓位和市场切片，但与单一连续账户中实际出现的仓位分布、Slot 转换、风控打断和资金占用并不等价。某 Slot 的每步分数为正，不代表组合一定取得同等比例的利润；最终仍须查看冻结 router 的真实账户权益曲线、回撤和成本。

## 最小真实回放对照

保持当前接受的 6,000 元基线、费用、模型、Selection Manifest、VAE 路由配置和 Trial 508 参数不变，先以当前 manifest 完整重放 valid，核对已复现的 **+34.679% 净收益、4.466% step MDD**。然后做一组诊断性单变量对照：只替换一个低层 Slot 的候选模型（优先挑低层得分较高但现有 router 实际表现偏弱的 Slot），仍通过完整 `expert_hysteresis` 路由重放整个 valid 时间线。同步记录该 Slot 的 dominant/active/queried 身份、策略建议动作、最终动作与动作原因、交易归属、净账户权益变化、手续费、滑点、硬止损/冷静期事件和最大回撤。当前 manifest 与替换候选必须跑相同输入及冻结配置；全 9 个 Slot 若需筛查可逐一替换，但一次只变一个。

在重选模型之前，先用已有 `trading_action_detail_epoch_55.csv` 对照“原始 reward 排名”与“按账户权益差分、去掉 limit shaping 后的排名”，确认候选排序是否实际改变。若排序没有改变，塑形偏差就不是这一轮收益不足的主要原因；若排序改变，再把变化最大的 Slot 带入上述完整 router 对照。所有 valid 对照只能用于诊断，不能当作未见数据证明；最终收益和 MDD 必须用之后冻结配置在独立 test 上报告。

当前证据支持的限制判断是：低层选拔 `reward` 不是纯现金净收益，但它保留了盯市账户盈亏并计入成本，只叠加了涨跌停奖励；低层评估没有模拟高层止损与路由切换，因此分数和最终组合收益之间有实质路径差异。哪些差异造成最终收益差、是否改变最优低层候选，必须由上述同输入完整路由对照决定。
