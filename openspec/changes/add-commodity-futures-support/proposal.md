## Why

FineFT 当前默认使用 Binance 风格的加密货币期货数据输入：`book_snapshot_25`、
`derivative_ticker`、`trades` 和 `quotes`。本项目可用的上海/国内商品期货数据是一类
五档行情快照流，无法直接满足现有预处理与环境对 25 档盘口、资金费率、真实逐笔成交和
真实 quotes 的假设。

本变更为燃料油 `fu` 新增原生商品期货处理链路：保留真实 5 档盘口契约，只生成可解释的
估计特征，并在商品期货训练中关闭 funding、index、伪 25 档等不适用假设。

## What Changes

- 新增面向本地五档 CSV 文件的商品期货预处理分支。
- 新增按 `TradingDay` 拼接主力合约的流程：使用前一交易日成交量选择当日主力；当目标日
  主力缺失或无成交时，回退到当日成交量最大的可用品种主力月份合约。
- 基于同一份五档行情流重实现商品期货版本的 derivative ticker 下采样、orderbook 下采样
  和 base feature 下采样。
- 使用 `Volume`/`Turnover` 差分构建秒级成交估计，并用 tick rule 生成带 `_estimated`
  标记的方向特征。
- 从秒频五档快照派生 quote 特征，而不是依赖不存在的原始 `quotes` 文件。
- 调整 cross-section、time feature、feature selection、scale/save 和 merge 契约，使其
  支持可配置盘口深度和 reward/execution 列 manifest，不再依赖固定 25 档或前 106 列假设。
- 新增商品期货环境支持：5 档盘口执行、关闭 funding、右闭右标 bar、按品种配置买入/卖出
  手续费。
- **BREAKING** 仅对商品期货数据集生效：商品期货输出使用 depth=5，不生成 `ask6_price`
  等 6-25 档列；现有加密货币期货 depth=25 输出保持兼容。

## Capabilities

### New Capabilities

- `commodity-futures-support`: 覆盖本地商品期货数据接入、主力合约拼接、5 档预处理、特征
  生成、最终数据集产出和商品期货环境初始化。

### Modified Capabilities

- 无。现有加密货币期货能力的需求不变；本变更新增独立商品期货能力。

## Impact

- 影响数据预处理区域：`data_preprocess/operator_futures/**`、
  `data_preprocess/script_preprocess/future_upgraded/**`，以及相关 overview/integrity-check
  入口。
- 影响环境区域：`FineFT/env/env_class/**` 和 `FineFT/env/env_initiate/**`。
- 影响数据契约：商品期货最终数据集仍需提供 `df.feather`、`state_features.npy`，以及商品
  期货等价环境配置或 `maintenance_margin_ratio_dict.npy`。
- 不新增 GPU 依赖；预处理仍主要受 CPU/IO 限制。
- 不包含 `download_operator`；商品期货原始文件由用户放入本地规划目录。
