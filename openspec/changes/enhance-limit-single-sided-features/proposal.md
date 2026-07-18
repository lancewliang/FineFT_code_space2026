# enhance-limit-single-sided-features

## 背景与目标

商品期货在涨跌停或触及涨跌停时可能出现合法单边盘口。例如涨停时 ask 侧挂单量全部为 0，bid 侧仍有挂单量。当前商品下采样逻辑允许这种行情状态，但 snapshot 截面特征仍按双边盘口都有挂量计算，导致 `sell_wap`、`buy_sell_wap_spread`、`ask*_size_n` 等特征在空侧总挂量为 0 时产生 NaN，并在 `time_feature_input` 质量校验阶段失败。

本变更目标是增强涨跌停/单边盘口下的特征计算，使合法单边盘口生成有限且语义明确的特征，同时把涨跌停价纳入商品 reward/execution 当前行情列。

## 用户场景

- 预处理 `fu2411` 等商品合约时，某些交易日出现涨停 ask 侧空盘口，特征生成不应因合法单边盘口产生 NaN。
- 商品模型训练和特征选择需要看到当前时刻的 `UpperLimitPrice`、`LowerLimitPrice`，用于解释价格边界和单边盘口状态。
- 下游 `time_feature`、feature selection、scale save 仍应保留 NaN/Inf fail-fast 校验，确保非法输入不会被静默吞掉。

## 设计方向

采用已确认的混合方案：A+ 边界增强并扩展列合同。

在商品 orderbook 下采样分支中保留 `LowerLimitPrice`、`UpperLimitPrice`，让它们随 `snapshot` 进入 `CONCURRENT_FEATURE`，并更新商品 reward/execution manifest，使两列作为当前行情/reward 列，不被 feature selection 当作 state candidate。

在 snapshot 截面特征生成中显式处理单边盘口：

- `ask_side_empty = sell_volume_oe <= 0`
- `bid_side_empty = buy_volume_oe <= 0`
- 空 ask 侧所有 `ask{i}_size_n` 输出 0
- 空 bid 侧所有 `bid{i}_size_n` 输出 0
- 空 ask 侧 `sell_wap = ask1_price`
- 空 bid 侧 `buy_wap = bid1_price`
- `buy_sell_wap_spread` 使用增强后的 `buy_wap - sell_wap`
- `wap_1`、`wap_2` 对分母为 0 的情况做显式保护，避免 NaN/Inf

新增 `ask_side_empty`、`bid_side_empty` 作为 snapshot 状态特征列，供模型显式识别单边盘口。双侧全空盘口不静默修复，应继续作为非法数据 fail-fast。

## 关键决策

- 空侧 WAP 使用本侧已有占位价：ask 侧空时 `sell_wap = ask1_price`，bid 侧空时 `buy_wap = bid1_price`。
- `LowerLimitPrice`、`UpperLimitPrice` 进入商品 reward/execution manifest，放在 orderbook depth columns 之后、derivative reference columns 之前。
- `ask_side_empty`、`bid_side_empty` 不作为 reward/execution 列，而是 snapshot/state candidate 列。
- 不放宽下游质量校验；增强后的合法单边盘口不应产生 NaN/Inf。
- 非涨跌停原因导致的双侧空盘口或关键价格缺失不做静默填充。

## 范围边界

**包含：**
- 增强 `process_snapshot_features` 的单边盘口计算。
- 新增 snapshot 特征列 `ask_side_empty`、`bid_side_empty`。
- 在商品 orderbook 下采样输出中保留 `LowerLimitPrice`、`UpperLimitPrice`。
- 更新商品 reward/execution columns manifest 和相关 feature selection / scale save 使用路径。
- 更新 snapshot 特征文档、expected columns、商品 reward/environment 文档。
- 增加 focused tests 覆盖涨停 ask 空、跌停 bid 空、正常双边盘口、双侧空非法、depth=5/depth=25。

**不包含（本次）：**
- 不重新定义涨跌停合法性判断；商品 downscale/quote 校验层继续负责识别合法单边盘口。
- 不为单边盘口伪造非零挂单量。
- 不改变 time feature 的 NaN/Inf fail-fast 策略。
- 不引入新的特征选择算法或模型训练逻辑。

## 验收标准

- [ ] ask 侧全空、bid 侧有量的 snapshot 输入生成有限特征，`ask*_size_n == 0`，`sell_wap == ask1_price`，`ask_side_empty == true`，`bid_side_empty == false`。
- [ ] bid 侧全空、ask 侧有量的 snapshot 输入生成有限特征，`bid*_size_n == 0`，`buy_wap == bid1_price`，`bid_side_empty == true`，`ask_side_empty == false`。
- [ ] 正常双边盘口的现有 snapshot 公式结果保持兼容。
- [ ] 双侧全空或关键价格缺失不会被静默修复，并会被测试覆盖为非法输入路径。
- [ ] `downscale_orderbook()` 输出包含 `LowerLimitPrice`、`UpperLimitPrice`。
- [ ] `get_reward_execution_columns(depth=5)` 包含 `LowerLimitPrice`、`UpperLimitPrice`，总数从 27 变为 29，并保持稳定顺序。
- [ ] 商品 feature selection 将 `LowerLimitPrice`、`UpperLimitPrice` 识别为 reward/execution 列，而不是 state candidate。
- [ ] 重新生成 `fu2411` 相关中间文件后，2024-10-08 单边盘口不再导致 `sell_wap`、`buy_sell_wap_spread`、`ask*_size_n` 出现 NaN。
- [ ] 使用 `conda activate finetf` 运行 focused tests 通过。
