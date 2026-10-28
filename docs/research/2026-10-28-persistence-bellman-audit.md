# 动作保持与 Bellman 目标核查

日期：2026-10-28。只读源码及现有 train/valid 配置产物；未读 test、未跑训练、未改生产代码。

## 核心结论

有两件需要分开说：

1. **动作保持下的控制状态没有进Q网络。** 多样化训练的行为采样器会把非flat动作保持6步；但TD bootstrap在18步累计奖励后的下一状态直接对所有环境允许动作取最大Q，不知道此刻是否还在强制保持期。这是“真实执行受保持约束、价值目标按可自由换动作定义”的建模差异。若目标Q本来就是底层自由决策MDP的Q值，重复动作作为off-policy行为本身不必然错误；但实际部署也强制保持6步，因此训练Q的greedy策略和实际执行策略不同。要把持久动作本身视为策略约束，需要把剩余保持步数/当前强制动作加入状态，并在bootstrap时遵从该约束。
2. **发现更确定的多步TD折扣问题。** 训练轨迹先计算18步折现回报 `R_18 = Σ(k=0..17) γ^k r[t+k]`，终点取18步后的状态；但预训练和多样化训练的TD更新都再乘一次 `γ`，不是 `γ^18`。当前训练脚本给 `n_step=18, γ=0.992`，理应 bootstrap 系数约为0.865386，代码系数为0.992。此处不是保持机制推断，而是n-step奖励跨度和bootstrap折扣公式不一致。

**这两项都不能直接证明本次valid中的11笔止损或某一笔亏损是由它造成。** 它们值得作为训练实现问题优先审计/修正评估，但需要冻结新训练、独立时间段验证后才能建立收益因果关系。

后续已做实际函数级复现：调用现有compact n-step累计器、预训练回放的`calc_multistep_return`，再将合成样本送入原`update_pretrain`和`update`，在损失入口截取实际TD目标，没有运行优化器。两函数的1步与终止样本正确；18步非终止样本均使用单步折扣。正、负终点价值的反例都能复现，因此是未来价值权重偏大，不能笼统说所有Q都会偏正。见[可重跑探针](fu_10min_cause_20261028/training_target_probe.py)、[实际函数输出](fu_10min_cause_20261028/training_audit/training_target_probe.json)。带`--assert-correct`运行会有意触发断言失败，证明当前公式不满足n-step折扣要求。

## 一、确切的18步TD公式与实际训练入口

训练启动脚本给出`--n_step 18 --gamma 0.992`：[fu 10分钟训练脚本](../../FineFT/script/train/train_commodity_fu_10.sh#L20-L30)。同一模型目录内保存的[qtable诊断manifest](../../result/DiHFT/low_level/fu/10min_parallel/weights_advantage_pretrain/qtable_diagnostics/manifest.json)记录`gamma=0.992`、佣金`0.002`、turnover基础惩罚`0.002`和不利方向倍数`5.0`；它未记录`n_step`。训练主日志文件目前为0字节，因此`n_step=18`的历史独立日志证据缺失；下文对18步结论按仓库中该产物对应的fu10训练入口配置说明。

### 多样化训练

实际训练入口先采样再调用`parallel_diverse_train.run_diverse_training_phase`中的`update`：[训练阶段调用](../../FineFT/RL/DiHFT/low_level/parallel_diverse_train.py#L886-L908)。Rollout每一步都记录实际执行动作及状态，但非flat动作发出后，后续最多5步复用同一动作：[行为保持循环](../../FineFT/RL/DiHFT/low_level/parallel_diverse_train.py#L1122-L1141)。累计器对连续18步奖励逐项乘`gamma**k`，把终点设为`step t+18`的状态/`info`：[n-step构造](../../FineFT/RL/util/regime_stratified_replay_buffer.py#L249-L281)。

TD更新读入终点状态和`next_info`，调用`get_best_q`后对可执行动作取最大值，再形成目标：

```text
R18 = r0 + γ r1 + … + γ^17 r17
Y  = R18 + (1-done) * γ * max_a Q(s[t+18], a)
```

代码位置：[多样化训练TD目标](../../FineFT/RL/DiHFT/low_level/parallel_diverse_train.py#L2283-L2299)。对标准18步TD，bootstrap应为`γ^18`，即`R18 + γ^18 max Q(s[t+18],a)`。因为动作累计器没有把n步折扣率作为样本字段传到`update`，`update`只能乘固定单步`trainer.gamma`。

在`γ=0.992`时，`γ^18≈0.865386`。例如终点最大Q为10、18步内奖励和为0时，正确bootstrap是8.6539，当前公式是9.92，相差1.2661。这里仅用于说明公式差异，不代表实际训练Q值误差或交易损失。

### 预训练与DP教师

主训练先创建`Multi_step_ReplayBuffer_multi_info(n_step=trainer.n_step)`，再运行exhaustive warmup，之后清空预训练池、进入多样化训练：[经验池与预训练调用](../../FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py#L1002-L1009)、[warmup主调](../../FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py#L1090-L1098)、[转入diverse](../../FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py#L1141-L1158)。Warmup的DP教师策略按每个时间步的Q-table动作序列选动作，没有action_persistence覆盖：[教师动作序列](../../FineFT/RL/DiHFT/low_level/parallel_pretrain.py#L136-L160)。Q-table本身也是逐时间步向后递推，并在每一步对下一状态取max：[DP递推](../../FineFT/env/env_class/futures_util.py#L1421-L1425)。所以DP教师问题定义是“每根bar可以重新决策”的控制问题，不含6步强制动作保持状态。

预训练replay buffer把n步奖励按gamma逐步折现、next state设为窗口末状态：[预训练缓冲累积](../../FineFT/RL/util/replay_buffer_DQN.py#L412-L455)。`update_pretrain`仍使用`gamma * next_q`，没有`gamma**n_step`：[预训练TD目标](../../FineFT/RL/DiHFT/low_level/parallel_pretrain.py#L750-L773)。所以**多步bootstrap折扣不一致同时存在于预训练和多样化TD两阶段**；DP教师表的单步递推本身每步乘gamma，并没有“18步bootstrap”问题，因为它一次递推只跨一根bar。

## 二、动作保持假设：确认差异，不能过度定性

多样化训练把`remaining_persist`保存在worker局部变量中；它不在网络的状态输入、`info`字典、回放样本或next action mask里。回放确实存下每根bar的状态、动作、奖励及next info，并在18步终点存下环境动作mask；它不存保持计数：[累计样本字段](../../FineFT/RL/DiHFT/low_level/parallel_diverse_train.py#L1154-L1167)、[compact replay记录字段](../../FineFT/RL/util/regime_stratified_replay_buffer.py#L189-L212)、[重建next_info](../../FineFT/RL/util/regime_stratified_replay_buffer.py#L413-L422)。Q模型`get_best_q`对所有由环境mask允许的动作求max；这个mask表达盘口/保证金等动作可行性，不表达“仍须沿用上个动作”。[Q网络动作mask](../../FineFT/model/low_level.py#L28-L46)、[ensemble max](../../FineFT/model/low_level.py#L93-L106)。

这不是简单的“off-policy行为一定错”：若训练目标定义成每一状态可自由挑选动作的原始MDP，行为策略偶尔因持久规则重复动作，off-policy Q学习在有足够覆盖时仍可朝自由决策Q学习。问题在于运行时也用同一持久规则强制执行，于是部署策略并不等于这个自由决策贪心策略；此外执行约束下的状态少了保持计数，无法只靠现有`info`还原何时准许重决策。故结论是**策略/控制建模不一致，可能导致持仓时机与训练Q目标不匹配**，而非已证明的低层代码崩坏。

边界条件：保持动作不是“持仓方向天然不变”，而是重复一个目标动作；实际环境若动作不可执行会提前重新选，flat动作也不启动保持。训练、低层valid测试和router部署的保持长度都是6（[训练参数](../../FineFT/script/train/train_commodity_fu_10.sh#L25-L30)、[低层valid参数](../../FineFT/script/test/DiHFT/low_level/test_util_fu_10.sh#L33-L47)、[冻结配置](../../result/DiHFT/final_result/fu/10min_parallel/expert_hysteresis/7620ac5ca9979f71bd0152761fc593693546dd8d02b61b6829a8d7c474db02eb/valid/main_sub/frozen_config.json)），所以不存在此前担心的“训练3步、部署6步”数值错配。

## 三、训练交易摩擦会强烈影响“是否平仓”目标

同模型目录的qtable诊断manifest记载训练环境和DP表使用佣金率0.002、`turnover_base_rate=0.002`、`turnover_adverse_ratio=5`；训练主流程用trainer参数构造`env_kwargs`与`qtable_kwargs`：[训练配置传入环境和DP](../../FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py#L1020-L1068)。已有train入口也明确给`--transcation_cost 0.002 --turnover_base_rate 0.002 --turnover_adverse_ratio 5.0`。[入口参数](../../FineFT/script/train/train_commodity_fu_10.sh#L20-L30)。valid/runtime冻结配置手续费为0.0003，且高层路由器没有配置这套turnover惩罚（其环境构造只传佣金等参数，turnover走Base_Env默认0）；原valid选择脚本也传入0.0003。[低层valid费用](../../FineFT/script/test/DiHFT/low_level/test_util_fu_10.sh#L37-L46)、[冻结runtime配置](../../result/DiHFT/final_result/fu/10min_parallel/expert_hysteresis/7620ac5ca9979f71bd0152761fc593693546dd8d02b61b6829a8d7c474db02eb/valid/main_sub/frozen_config.json)、[runtime环境创建参数](../../FineFT/RL/DiHFT/high_level/vae_routing_util.py#L2323-L2340)。

按当前记账中的仓位变化1单位、参考价格3,300、无滑点估算（不将1单位解释成交易所标准手）：

| 动作 | 训练中显式费用/奖励扣分 | 冻结runtime显式费用 |
|---|---:|---:|
| 平掉1单位仓位，且该方向被分类为“不利离场” | 0.2%佣金=6.60元，另加1%方向换手奖励惩罚=33元；合计39.60元奖励影响 | 0.03%佣金=0.99元；无这项turnover惩罚 |
| 1单位多头反手到1单位空头，且为不利方向 | 交易名义量约6,600元，佣金约13.20元；仓位变化2单位，1%惩罚约66元；合计约79.20元奖励影响 | 佣金约1.98元；无这项turnover惩罚 |

计算依据：佣金按实际成交价值乘commission rate；turnover惩罚按`effective_rate * abs(new_position-old_position) * markprice`从step reward扣除。[费用实现](../../FineFT/env/env_class/futures_util.py#L558-L619)、[换手惩罚计算](../../FineFT/env/env_class/base_env.py#L670-L685)。方向倍率规则为上涨斜率下减仓/离场或做空使用5倍惩罚，下跌斜率下空头减仓/离场使用5倍惩罚；横盘下的减仓惩罚是基础倍率。[方向规则](../../FineFT/env/env_class/futures_util.py#L150-L185)。

需特别说明：33元/66元是**训练奖励中的人为惩罚**，不是交易所真的收走的手续费；训练佣金0.2%才是env实际从账户扣的费用。这种奖励设计明显让模型在特定趋势标签下更不愿意减仓/平仓；当它与动作保持6步及Q目标鼓励持有叠加时，可能放大“浮亏时继续持有”的倾向。当前尚未把实际11笔止损按其当时`regime_grid_id`和`old/new position`逐笔连接到Q margin/保持状态，因此不能把这个机制写成那些止损的已证实因果主因。

## 四、与实际亏损的因果边界

当前valid研究基线11个硬止损回合合计约亏1,188.90元，但这是结果分类，不是模型在开仓或亏损持仓时能预知的标签。上述TD折扣和奖励摩擦是训练实现/目标层证据；没有针对它们重训并冻结模型，也没有跨时间样本外比较。因此可以准确说“模型训练时的价值目标更偏向自由逐bar决策，且特定方向平仓在训练奖励里代价远高于部署费用”，不能说“这就是亏损根因”或“改了必然减少止损”。

优先级建议：先在独立代码分支修正/隔离n-step折扣目标，并让研究回放显式记录保持状态；然后只用train/valid重新训练并检查Q目标；最后冻结单个方案，用未参与训练和方案选择的后续时间段验证。方向性turnover奖励需要单独做基线对照，不能在同一次实验里与n-step、persistence状态一起改，否则无法辨因。
