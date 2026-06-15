# data_preprocess 代码与特征分析

本报告分析目录：`data_preprocess/`。

结论先说清楚：这个目录不会在代码里硬编码最终模型特征列表。它先生成大量候选状态特征，再由 `operator_futures/feature_selection/*.py` 根据未来 `mark_price` 变化做选择，最后把选择结果写入 `state_features*.npy`。当前仓库里没有已生成的 `state_features*.npy` 或 `df*.feather` 数据文件，所以无法静态列出某个交易对最终实际保留下来的具体列名；只能从代码确定候选特征全集、选择规则、输出文件位置。

默认参数下，进入特征选择前的状态候选特征大约是：

- reward/environment 前置列：106 列
- 状态候选特征：3854 列
- 合并后的 `ALL_FEATURE` 总列数约：3960 列

上面的数量基于代码默认：

- 时间滚动窗口：`[2, 6, 12, 16, 24, 48]`
- 特征选择预测窗口：`[1, 6, 12]`
- orderbook top-k：`topk=5`
- 正常 trades 数据同时包含 `buy` 和 `sell`

## 总流程

主流程在 `script_preprocess/future_upgraded/total_process/util_process.sh`：

1. `run_derivative_ticker_downscaling`
   降采样 `derivative_ticker`，生成资金费率、指数价、标记价等 reward/environment 相关数据。

2. `run_orderbook_downscaling`
   降采样 `book_snapshot_25`，生成 25 档 bid/ask 价格和数量。

3. `run_downscale_process`
   从原始 `trades` 和 `quotes` 生成基础 OHLCV、成交计数、quote 变化计数、一档盘口 OHLC 等。

4. `run_cross_section_process`
   基于基础特征和 25 档 orderbook 生成截面特征：K 线形态、买卖/涨跌/bid-ask 归一化、orderbook 截面特征。

5. `run_merge_process`
   单日合并，分成：
   - `CONCURRENT_FEATURE`：当前时刻可用于环境/reward 的列
   - `FUTURE_FEATURE`：后续会 shift 的状态特征

6. `run_concat_process`
   拼接日期区间，并对 `FUTURE_FEATURE` 做 `shift(+1)`，避免当前状态直接看到同一时刻的未来信息。

7. `run_feature_creation_multiprocessing`
   基于拼接后的数据生成时间滚动特征。

8. `run_merge_and_clean`
   把横截面/基础特征和时间滚动特征合并为 `ALL_FEATURE`。

9. `run_ic_correlation` / `run_catboost_correlation` / `run_rank_correlation`
   做特征选择，输出 `df*.feather` 和 `state_features*.npy`。

10. `run_scale_save`
    按已选特征做尺度缩放，输出 FineFT 最终使用的 `df.feather` 和 `state_features.npy`。

注意：脚本存在一个不一致点。`run_all_process` 里只调用了 `run_ic_correlation`，但 `scale_save.py` 默认 `--ic_choice catboost`，会去读 `df_catboost.feather` 和 `state_features_catboost.npy`。如果只跑 IC，没有改 `--ic_choice ic`，最后一步会找不到 catboost 结果。

## 每个 Python 文件的功能

### `download_operator/`

| 文件 | 功能 |
| --- | --- |
| `download.py` | 调用 `tardis_dev.datasets.download` 下载单个 exchange/symbol/data_type/date range 的原始数据，并解压 `.gz`。API key 仍是占位文本，需要用户替换。 |
| `unzip.py` | 解压某个目录下的 `.gz`，如果解压后的文件已存在则删除 `.gz`。 |
| `unzip_check.py` | 递归解压目录下 `.gz` 文件。但当前文件末尾 `target_directory = None` 并直接调用，原样运行会失败，需要传入真实目录后再用。 |

### `operator_futures/`

| 文件 | 功能 |
| --- | --- |
| `util.py` | 日期文件匹配，以及根据列名正则查找 OHLCV/OHLC 特征组。时间特征生成依赖这里的 `find_ohlcv_groups` 和 `find_ohlc_groups`。 |
| `delete/delete.py` | 按日期删除预处理阶段的中间 feather 文件。 |
| `orderbook_25/down_scale_single_shot.py` | 从原始 `book_snapshot_25` 读取单日数据，按目标频率 resample 取 first，输出 25 档 `ask{i}_price/size`、`bid{i}_price/size`。 |
| `orderbook_25/down_scale_single_shot_base_other.py` | 从已有较细频率 feather 继续降采样到目标频率，避免直接处理超大 raw CSV。 |
| `derivative_ticker/down_scale_single_shot.py` | 从原始 `derivative_ticker` 读取单日数据，保留 `symbol/funding_timestamp/funding_rate/index_price/mark_price`，按目标频率取 first。 |
| `derivative_ticker/down_scale_single_shot_base_other.py` | 从已有较细频率 feather 继续降采样 derivative ticker。 |
| `features_related/feature_util.py` | 基础特征核心函数：quote 事件计数、quote 一档盘口 OHLC/TWAP/AWAP、trades OHLCV、buy/sell 分组、预处理 trades/quotes。 |
| `features_related/base_feature.py` | 单日入口：读取 raw `quotes` 和 `trades`，调用 `feature_util.py` 生成 `BASE_FEATURE/{symbol}/{freq}/{date}.feather`。 |
| `cross_section/base_feature_util.py` | 截面特征核心函数：K 线形态、成交/quote 归一化、25 档 orderbook 截面统计。 |
| `cross_section/create_feature.py` | 单日入口：读取 `BASE_FEATURE` 和 `DOWNSCALE_ORDERBOOK_25`，输出 `KLINE_FEATURE`、`QUOTES_FEATURE`、`SNAPSHOT_FEATURE` 三类截面特征。README 中提到的 `features_related/create_feature.py` 实际不存在，当前代码在这里。 |
| `merge_concat/merge.py` | 单日合并：当前 reward/environment 列与 snapshot 截面列写入 `CONCURRENT_FEATURE`，基础/截面状态列写入 `FUTURE_FEATURE`。 |
| `merge_concat/concat.py` | 日期区间拼接：对 `CONCURRENT_FEATURE` 和 `FUTURE_FEATURE` 分别拼接、去重、resample；对 future 特征做 `shift(+1)` 后与 concurrent 内连接。 |
| `time_operator/create_feature.py` | 单进程时间滚动特征版本，使用 `time_operator_util.py`。 |
| `time_operator/time_operator_util.py` | 单进程 OHLCV/OHLC Alpha158 风格窗口特征实现。 |
| `time_operator/create_feature_multi_processing.py` | 实际脚本调用的多进程时间特征入口；额外对 110 个单列价格/数量特征生成 `log_return` 和 `trend`。 |
| `time_operator/multi_processing_util.py` | 多进程窗口特征实现：OHLCV、OHLC、单列 price/volume 的时间特征，并做重复列删除。 |
| `merge_all/merge_clean.py` | 合并 `MERGE_CONCAT/CONCAT_FEATURE` 与 `TIME_FEATURE`，输出 `ALL_FEATURE/{symbol}/{freq}/{start}-{end}.feather`。文件名叫 clean，但代码没有删除 NaN 列，只是 inner join 后保存。 |
| `feature_selection/ic_correlation.py` | 用 Pearson IC 选择特征，输出 `ic_window_{w}.json`、`correlation.csv`、`df.feather`、`state_features.npy`。 |
| `feature_selection/rank_ic_correlation.py` | 用 Rank IC 选择特征，输出 `rank_ic_window_{w}.json`、`rank_correlation.csv`、`df_rank.feather`、`state_features_rank.npy`。 |
| `feature_selection/catbooost.py` | 用 CatBoostRegressor 的 feature importance 选择特征，输出 `cat_boost_feature_importance_{w}.csv`、`correlation_catboost.csv`、`df_catboost.feather`、`state_features_catboost.npy`。文件名有拼写错误：`catbooost.py`。代码强制 `task_type="GPU"`。 |
| `feature_selection/lasso_linear.py` | 用 `LassoCV` 对一步未来 `mark_price` 变化做线性选择，保留非零系数列，输出 `df_lasso.feather`、`state_features_lasso.npy`。 |
| `feature_selection/cor_util.py` | 贪心相关性去重：按重要性顺序遍历，选中一个特征后删除与它绝对相关性大于阈值的后续特征。 |
| `feature_selection/remove_duplicates_feature.py` | 复用已保存的 IC/RankIC/CatBoost 中间结果重新做相关性去重。CatBoost 分支读取 CSV 时没有 `index_col=0`，按当前代码可能无法正确 `loc` 到 feature-name index。 |
| `scale_describe_save/scale_save.py` | 读取 feature selection 的 `df*.feather` 和 `state_features*.npy`，对状态特征按 std/mean 做尺度缩放，输出最终 `SCALE_SAVE/.../df.feather`、`state_features.npy`、`df_describe.csv`。 |

### `over_view/`

| 文件 | 功能 |
| --- | --- |
| `checkout_download.py` | 检查 raw 下载目录里每个日期是否有 `book_snapshot_25/trades/derivative_ticker/quotes` 文件。 |
| `checkout_create_feature.py` | 检查预处理各阶段的 feather 文件是否按日期生成。 |
| `checkout_read_create_feature.py` | 多线程读取 feather，检查是否有 Arrow 文件损坏。 |
| `create_overview.py` | 拼接 derivative ticker，生成 mark price buy-and-hold 曲线 PDF 和概览 feather。 |

### `script_*`

| 目录 | 功能 |
| --- | --- |
| `script_download/download/*.sh` | 对 BTC/ETH/BNB/DOT 等 symbol 调 `download_operator/download.py` 的下载脚本。 |
| `script_download/unzip/unzip.sh` | 批量解压下载数据。 |
| `script_preprocess/future_upgraded/` | 当前主流程脚本。`1_downscale` 到 `8_scale_save` 是函数库；`total_process` 里是按 symbol/date/frequency 写好的运行入口。 |
| `script_preprocess/futures/` | 较旧的 BTCUSDT 示例脚本，功能与 `future_upgraded` 类似但组织方式更早。 |
| `script_preprocess/overview/all.sh` | 跑 overview 检查/概览相关脚本。 |

另一个路径问题：`future_upgraded/total_process/util_process.sh` 里 source 的是 `script/future_upgraded/...`，但仓库实际目录是 `script_preprocess/future_upgraded/...`。如果没有额外 symlink 或运行目录映射，直接执行会找不到被 source 的脚本。

## 生成了哪些特征

### 1. Reward/environment 前 106 列

代码位置：

- `operator_futures/orderbook_25/down_scale_single_shot.py`
- `operator_futures/derivative_ticker/down_scale_single_shot.py`
- `operator_futures/merge_concat/concat.py`
- `operator_futures/feature_selection/*.py`
- `operator_futures/scale_describe_save/scale_save.py`

特征选择代码统一把 `df.columns[:106]` 当作 reward/environment 列，不参与状态特征选择。

这 106 列按当前 merge 顺序是：

- `timestamp`
- `symbol`
- 25 档 orderbook 原始快照列：
  - `ask1_price` ... `ask25_price`
  - `ask1_size` ... `ask25_size`
  - `bid1_price` ... `bid25_price`
  - `bid1_size` ... `bid25_size`
- derivative ticker 列：
  - `funding_timestamp`
  - `funding_rate`
  - `index_price`
  - `mark_price`

注意：`df.columns[:106]` 是位置约定，不是按列名查找。如果上游 merge 顺序改变，这个约定会变得脆弱。

### 2. `BASE_FEATURE`：112 个原始状态候选列

代码位置：

- `operator_futures/features_related/base_feature.py`
- `operator_futures/features_related/feature_util.py`

`base_feature.py` 输出文件包含 `timestamp/exchange/symbol` 加 112 个特征列。后续 `concat.py` 会 drop `symbol/exchange`，timestamp 变 index，所以这 112 个进入状态候选。

Trades 全量 OHLCV/成交统计：

```text
open, high, low, close, volume, tradeval, ntrade, vwap, awap, twap,
ntrade_up, ntrade_down, ntrade_flat
```

Trades 按 `side=buy/sell` 分组：

```text
open_buy, open_sell, high_buy, high_sell, low_buy, low_sell,
close_buy, close_sell, volume_buy, volume_sell,
tradeval_buy, tradeval_sell, ntrade_buy, ntrade_sell,
ntrade_up_buy, ntrade_up_sell, ntrade_down_buy, ntrade_down_sell,
ntrade_flat_buy, ntrade_flat_sell,
awap_buy, awap_sell, twap_buy, twap_sell, vwap_buy, vwap_sell
```

Quote 事件计数：

```text
nquote,
nquote_bid, nquote_ask,
nquote_bid_up, nquote_bid_down, nquote_ask_up, nquote_ask_down,
nquote_bidsize, nquote_asksize,
nquote_bidsize_up, nquote_bidsize_down,
nquote_asksize_up, nquote_asksize_down,
nquote_bid_askflat, nquote_bidup_askflat, nquote_biddown_askflat,
nquote_ask_bidflat, nquote_askup_bidflat, nquote_askdown_bidflat
```

Quote 一档盘口 OHLC/TWAP/AWAP：

基础指标为：

```text
spread, mid, imblance_volume, makav_rev, makav_ori, bid, bidsize, ask, asksize
```

每个基础指标生成：

```text
open_{name}, high_{name}, low_{name}, close_{name}, twap_{name}, awap_{name}
```

这里 `imblance_volume` 是代码原拼写，不是 `imbalance_volume`。

### 3. `CROSS_SECTION/KLINE_FEATURE`：216 个 K 线形态特征

代码位置：

- `operator_futures/cross_section/create_feature.py`
- `operator_futures/cross_section/base_feature_util.py`

从 OHLC/TWAP/AWAP/VWAP 组合派生 K 线形态。

对有 VWAP 的三组：

```text
origin, buy, sell
```

每组生成 21 个：

```text
klen, kmid, kmid2, kup, kup2, klow, klow2, ksft, ksft2,
kotwap, kotwap2, kctwap, kctwap2,
koawap, koawap2, kcawap, kcawap2,
kovwap, kovwap2, kcvwap, kcvwap2
```

命名示例：

```text
klen, kmid, kcvwap2
klen_buy, kmid_buy, kcvwap2_buy
klen_sell, kmid_sell, kcvwap2_sell
```

对没有 VWAP 的 9 组 quote OHLC：

```text
spread, mid, imblance_volume, makav_rev, makav_ori, bid, bidsize, ask, asksize
```

每组生成 17 个，不含 `kovwap/kovwap2/kcvwap/kcvwap2`：

```text
klen_{name}, kmid_{name}, kmid2_{name}, kup_{name}, kup2_{name},
klow_{name}, klow2_{name}, ksft_{name}, ksft2_{name},
kotwap_{name}, kotwap2_{name}, kctwap_{name}, kctwap2_{name},
koawap_{name}, koawap2_{name}, kcawap_{name}, kcawap2_{name}
```

数量：`3 * 21 + 9 * 17 = 216`。

### 4. `CROSS_SECTION/QUOTES_FEATURE`：69 个归一化截面特征

代码位置：

- `operator_futures/cross_section/base_feature_util.py`

Buy/sell 归一化：

```text
volume_buy_bsnorm, volume_sell_bsnorm, volume_buysell_imbalance_bsnorm,
tradeval_buy_bsnorm, tradeval_sell_bsnorm, tradeval_buysell_imbalance_bsnorm,
ntrade_buy_bsnorm, ntrade_sell_bsnorm, ntrade_buysell_imbalance_bsnorm,
ntrade_up_buy_bsnorm, ntrade_up_sell_bsnorm, ntrade_up_buysell_imbalance_bsnorm,
ntrade_down_buy_bsnorm, ntrade_down_sell_bsnorm, ntrade_down_buysell_imbalance_bsnorm,
ntrade_flat_buy_bsnorm, ntrade_flat_sell_bsnorm, ntrade_flat_buysell_imbalance_bsnorm
```

Up/down/flat 归一化：

```text
ntrade_up_udnorm, ntrade_down_udnorm, ntrade_flat_udnorm,
ntrade_updown_imbalance_udnorm, ntrade_updownflat_vol_udnorm,
ntrade_buy_up_udnorm, ntrade_buy_down_udnorm, ntrade_buy_flat_udnorm,
ntrade_buy_updown_imbalance_udnorm, ntrade_buy_updownflat_vol_udnorm,
ntrade_sell_up_udnorm, ntrade_sell_down_udnorm, ntrade_sell_flat_udnorm,
ntrade_sell_updown_imbalance_udnorm, ntrade_sell_updownflat_vol_udnorm,
nquote_bid_up_udnorm, nquote_bid_down_udnorm, nquote_bid_updown_imbalance_udnorm,
nquote_ask_up_udnorm, nquote_ask_down_udnorm, nquote_ask_updown_imbalance_udnorm,
nquote_bidsize_up_udnorm, nquote_bidsize_down_udnorm, nquote_bidsize_updown_imbalance_udnorm,
nquote_asksize_up_udnorm, nquote_asksize_down_udnorm, nquote_asksize_updown_imbalance_udnorm,
nquote_bid_askflat_up_udnorm, nquote_bid_askflat_down_udnorm, nquote_bid_askflat_updown_imbalance_udnorm,
nquote_ask_bidflat_up_udnorm, nquote_ask_bidflat_down_udnorm, nquote_ask_bidflat_updown_imbalance_udnorm
```

Bid/ask 归一化：

```text
nquote_bid_abnorm, nquote_ask_abnorm, nquote_askbid_imbalance_abnorm,
nquote_bid_up_abnorm, nquote_ask_up_abnorm, nquote_ask_up_bid_imbalance_abnorm,
nquote_bid_down_abnorm, nquote_ask_down_abnorm, nquote_ask_down_bid_imbalance_abnorm,
nquote_bidsize_abnorm, nquote_asksize_abnorm, nquote_asksize_bid_imbalance_abnorm,
nquote_bidsize_up_abnorm, nquote_asksize_up_abnorm, nquote_asksize_up_bid_imbalance_abnorm,
nquote_bidsize_down_abnorm, nquote_asksize_down_abnorm, nquote_asksize_down_bid_imbalance_abnorm
```

### 5. `CROSS_SECTION/SNAPSHOT_FEATURE`：82 个 25 档 orderbook 截面特征

代码位置：

- `operator_futures/cross_section/base_feature_util.py`

价格相关：

```text
midprice, wap_1, wap_2, wap_balance,
sell_wap, buy_wap, buy_sell_wap_spread,
buy_spread_oe_max, sell_spread_oe_max
```

Top-k 大挂单相对 best price/size 的增量，默认 `topk=5`：

```text
ask_price_topk_size_1_increments ... ask_price_topk_size_5_increments
bid_price_topk_size_1_increments ... bid_price_topk_size_5_increments
ask_size_topk_size_1_increments ... ask_size_topk_size_5_increments
bid_size_topk_size_1_increments ... bid_size_topk_size_5_increments
```

挂单量相关：

```text
buy_volume_oe, sell_volume_oe, imblance_volume_oe
```

25 档归一化 size：

```text
ask1_size_n ... ask25_size_n
bid1_size_n ... bid25_size_n
```

数量：`9 + 20 + 3 + 50 = 82`。

### 6. `TIME_FEATURE`：默认 3375 个时间滚动特征

代码位置：

- `operator_futures/time_operator/create_feature_multi_processing.py`
- `operator_futures/time_operator/multi_processing_util.py`

实际脚本调用的是多进程版本。默认窗口：

```text
2, 6, 12, 16, 24, 48
```

#### 6.1 单列 price/volume 类时间特征：1320 个

输入单列共 110 个：

```text
bid1_price ... bid25_price
ask1_price ... ask25_price
buy_spread_oe_max, sell_spread_oe_max,
wap_1, wap_2, buy_wap, sell_wap,
mark_price,
buy_volume_oe, sell_volume_oe, imblance_volume_oe,
ask1_size_n ... ask25_size_n,
bid1_size_n ... bid25_size_n
```

每个窗口生成：

```text
{feature}_log_return_{w}
{feature}_trend_{w}
```

因为默认窗口没有 `w=1`，所以每个输入列是 `6 * 2 = 12` 个时间特征；`110 * 12 = 1320`。

#### 6.2 OHLCV 时间特征：651 个

OHLCV 分组：

```text
origin, buy, sell
```

每组生成：

```text
log_volume_{group}
```

每个窗口生成 36 个：

```text
roc_{w}, ma_{w}, std_{w}, beta_{w}, max_{w}, min_{w},
qtlu_{w}, qtld_{w}, rank_{w}, imax_{w}, imin_{w}, imxd_{w},
rsv_{w}, cntp_{w}, cntn_{w}, cntd_{w}, corr_{w}, cord_{w},
sump_{w}, sumn_{w}, sumd_{w}, vma_{w}, vstd_{w}, wvma_{w},
vsump_{w}, vsumn_{w}, vsumd_{w},
roc_{w}_std_norm, ma_{w}_std_norm, beta_{w}_std_norm,
max_{w}_std_norm, min_{w}_std_norm, qtlu_{w}_std_norm,
qtld_{w}_std_norm, rsv_{w}_std_norm, vma_{w}_std_norm
```

命名示例：

```text
log_volume_origin, roc_2_origin, ma_2_origin
log_volume_buy, roc_2_buy, ma_2_buy
log_volume_sell, roc_2_sell, ma_2_sell
```

数量：每组 `1 + 36 * 6 = 217`，三组共 `651`。

#### 6.3 OHLC 时间特征：1404 个

OHLC 分组：

```text
spread, mid, imblance_volume, makav_rev, makav_ori, bid, bidsize, ask, asksize
```

每个窗口生成 26 个：

```text
roc_{w}, roc_{w}_std_norm,
ma_{w}, ma_{w}_std_norm,
std_{w},
beta_{w}, beta_{w}_std_norm,
max_{w}, max_{w}_std_norm,
min_{w}, min_{w}_std_norm,
qtlu_{w}, qtlu_{w}_std_norm,
qtld_{w}, qtld_{w}_std_norm,
rank_{w}, imax_{w}, imin_{w}, imxd_{w}, rsv_{w},
cntp_{w}, cntn_{w}, cntd_{w},
sump_{w}, sumn_{w}, sumd_{w}
```

命名示例：

```text
roc_2_spread, ma_2_spread, beta_2_std_norm_spread
roc_48_asksize, cntd_48_asksize, sumd_48_asksize
```

数量：`9 * 6 * 26 = 1404`。

## 特征数学公式

这一节写的是代码实际生成特征的公式。为了避免把 3854 个时间特征机械重复展开，这里按“特征族模板”写公式。列名中的 `{name}`、`{side}`、`{w}`、`{level}` 直接按前文列名规则替换即可得到每一个具体特征。

### 统一记号

设目标频率对应的第 `t` 个时间桶为 `I_t`。代码使用 pandas `resample(target_freq)`，通常可理解为左闭右开区间。

Trades 事件记号：

```text
P_i: 第 i 笔成交价格
V_i: 第 i 笔成交 amount
S_i: 第 i 笔成交方向，buy 或 sell
TV_i = P_i * V_i
dP_i = P_i - P_{i-1}
```

Quotes 事件记号：

```text
B_i: best bid price
A_i: best ask price
QB_i: best bid amount
QA_i: best ask amount
```

25 档 orderbook 快照记号：

```text
AP_l: ask l price, l = 1..25
AS_l: ask l size
BP_l: bid l price
BS_l: bid l size
```

时间窗口记号：

```text
W_t(w) = {t-w+1, ..., t}
mean_w(X)_t = mean(X over W_t(w))
std_w(X)_t = pandas rolling std(X over W_t(w)), ddof=1
max_w(X)_t = max(X over W_t(w))
min_w(X)_t = min(X over W_t(w))
q80_w(X)_t = 80% rolling quantile
q20_w(X)_t = 20% rolling quantile
```

代码中的小常数：

```text
eps_cross = 1e-15
eps_time = 1e-12
```

如果某个桶没有数据，很多中间列会产生 NaN，后续代码通常用 `ffill()` 或 `fillna(0)` 处理。公式只写非空桶里的原始计算。

另一个重要点：`merge_concat/concat.py` 会对 `FUTURE_FEATURE` 整体做 `shift(+1)`。也就是说，基础状态特征、K 线截面特征、quote 归一化特征在进入最终 `CONCAT_FEATURE/ALL_FEATURE` 后，时间 `t` 上看到的是上一桶 `t-1` 的值。下面公式是生成阶段的原始值。

### Reward/environment 公式

代码位置：

- `operator_futures/orderbook_25/down_scale_single_shot.py`
- `operator_futures/derivative_ticker/down_scale_single_shot.py`

25 档 orderbook 降采样：

```text
ask{l}_price_t = first(AP_l in I_t)
ask{l}_size_t  = first(AS_l in I_t)
bid{l}_price_t = first(BP_l in I_t)
bid{l}_size_t  = first(BS_l in I_t)
```

其中 `l = 1..25`。列名来自：

```text
asks[l-1].price  -> ask{l}_price
asks[l-1].amount -> ask{l}_size
bids[l-1].price  -> bid{l}_price
bids[l-1].amount -> bid{l}_size
```

Derivative ticker 降采样：

```text
funding_timestamp_t = first(funding_timestamp in I_t)
funding_rate_t      = first(funding_rate in I_t)
index_price_t       = first(index_price in I_t)
mark_price_t        = first(mark_price in I_t)
```

### Trades 基础特征公式

代码位置：

- `operator_futures/features_related/feature_util.py`

对桶内所有 trades：

```text
open_t  = first(P_i, i in I_t)
high_t  = max(P_i, i in I_t)
low_t   = min(P_i, i in I_t)
close_t = last(P_i, i in I_t)

volume_t   = sum(V_i, i in I_t)
tradeval_t = sum(P_i * V_i, i in I_t)
ntrade_t   = count(i in I_t)

vwap_t = tradeval_t / volume_t
awap_t = mean(P_i, i in I_t)
```

TWAP 使用事件间隔权重。令 `dt_i = timestamp_i - timestamp_{i-1}`，单位秒：

```text
twap_t = sum(P_i * dt_i, i in I_t) / sum(dt_i, i in I_t)
```

成交方向计数基于全局相邻成交价差 `dP_i = P_i - P_{i-1}`：

```text
ntrade_up_t   = sum(1[dP_i > 0], i in I_t)
ntrade_down_t = sum(1[dP_i < 0], i in I_t)
ntrade_flat_t = sum(1[dP_i = 0], i in I_t)
```

对每个成交方向 `side in {buy, sell}`，只保留 `S_i = side` 的 trades：

```text
open_{side,t}  = first(P_i, i in I_t, S_i = side)
high_{side,t}  = max(P_i, i in I_t, S_i = side)
low_{side,t}   = min(P_i, i in I_t, S_i = side)
close_{side,t} = last(P_i, i in I_t, S_i = side)

volume_{side,t}   = sum(V_i, i in I_t, S_i = side)
tradeval_{side,t} = sum(P_i * V_i, i in I_t, S_i = side)
ntrade_{side,t}   = count(i in I_t, S_i = side)

awap_{side,t} = mean(P_i, i in I_t, S_i = side)
vwap_{side,t} = tradeval_{side,t} / volume_{side,t}
```

Side 分组里的 `ntrade_up/down/flat_{side}` 仍然使用全局 `dP_i`：

```text
ntrade_up_{side,t}   = sum(1[dP_i > 0], i in I_t, S_i = side)
ntrade_down_{side,t} = sum(1[dP_i < 0], i in I_t, S_i = side)
ntrade_flat_{side,t} = sum(1[dP_i = 0], i in I_t, S_i = side)
```

Side 分组里的 TWAP 使用同一个 side 且同一个桶内的相邻事件时间差：

```text
twap_{side,t} = sum(P_i * dt_i^{side}, i in I_t, S_i = side)
                / sum(dt_i^{side}, i in I_t, S_i = side)
```

### Quotes 计数特征公式

代码位置：

- `operator_futures/features_related/feature_util.py`

设桶内 quotes 按时间排序。桶内 `i-1` 表示同一桶内前一个 quote；代码中第一条 quote 与 NaN 比较，因此 `!=` 类计数第一条会记为 1，`>`、`<`、`==` 类第一条通常为 0。

基础 quote 数量：

```text
nquote_t = count(i in I_t)
```

Best bid/ask price 变化：

```text
nquote_bid_t      = sum(1[B_i != B_{i-1}], i in I_t)
nquote_ask_t      = sum(1[A_i != A_{i-1}], i in I_t)
nquote_bid_up_t   = sum(1[B_i >  B_{i-1}], i in I_t)
nquote_bid_down_t = sum(1[B_i <  B_{i-1}], i in I_t)
nquote_ask_up_t   = sum(1[A_i >  A_{i-1}], i in I_t)
nquote_ask_down_t = sum(1[A_i <  A_{i-1}], i in I_t)
```

Best bid/ask size 变化：

```text
nquote_bidsize_t      = sum(1[QB_i != QB_{i-1}], i in I_t)
nquote_asksize_t      = sum(1[QA_i != QA_{i-1}], i in I_t)
nquote_bidsize_up_t   = sum(1[QB_i >  QB_{i-1}], i in I_t)
nquote_bidsize_down_t = sum(1[QB_i <  QB_{i-1}], i in I_t)
nquote_asksize_up_t   = sum(1[QA_i >  QA_{i-1}], i in I_t)
nquote_asksize_down_t = sum(1[QA_i <  QA_{i-1}], i in I_t)
```

一边价格变化且另一边价格不变：

```text
nquote_bid_askflat_t =
    sum(1[B_i != B_{i-1}] * 1[A_i = A_{i-1}], i in I_t)

nquote_bidup_askflat_t =
    sum(1[B_i > B_{i-1}] * 1[A_i = A_{i-1}], i in I_t)

nquote_biddown_askflat_t =
    sum(1[B_i < B_{i-1}] * 1[A_i = A_{i-1}], i in I_t)

nquote_ask_bidflat_t =
    sum(1[A_i != A_{i-1}] * 1[B_i = B_{i-1}], i in I_t)

nquote_askup_bidflat_t =
    sum(1[A_i > A_{i-1}] * 1[B_i = B_{i-1}], i in I_t)

nquote_askdown_bidflat_t =
    sum(1[A_i < A_{i-1}] * 1[B_i = B_{i-1}], i in I_t)
```

### Quotes 一档盘口 OHLC/TWAP/AWAP 公式

代码位置：

- `operator_futures/features_related/feature_util.py`

先为每条 quote 定义 9 个基础序列：

```text
spread_i = A_i - B_i
mid_i = (A_i + B_i) / 2
imblance_volume_i = (QB_i - QA_i) / (QB_i + QA_i)

makav_rev_i = (QA_i * A_i + QB_i * B_i) / (QA_i + QB_i)
makav_ori_i = (QA_i * B_i + QB_i * A_i) / (QA_i + QB_i)

bid_i = B_i
bidsize_i = QB_i
ask_i = A_i
asksize_i = QA_i
```

对任意序列 `X_i`，其中 `X` 属于：

```text
spread, mid, imblance_volume, makav_rev, makav_ori, bid, bidsize, ask, asksize
```

生成：

```text
open_X_t  = first(X_i, i in I_t)
high_X_t  = max(X_i, i in I_t)
low_X_t   = min(X_i, i in I_t)
close_X_t = last(X_i, i in I_t)
awap_X_t  = mean(X_i, i in I_t)
twap_X_t  = sum(X_i * dt_i, i in I_t) / sum(dt_i, i in I_t)
```

其中 `dt_i = timestamp_i - timestamp_{i-1}`，单位秒。

### K 线截面特征公式

代码位置：

- `operator_futures/cross_section/base_feature_util.py`

对任意 OHLC/TWAP/AWAP 组合，定义：

```text
O = open
H = high
L = low
C = close
TW = twap
AW = awap
VW = vwap, 如果该组存在 vwap
eps = eps_cross
```

基础 K 线形态：

```text
klen  = (H - L) / (O + eps)
kmid  = (C - O) / (O + eps)
kmid2 = (C - O) / (H - L + eps)

kup  = (H - max(O, C)) / (O + eps)
kup2 = (H - max(O, C)) / (H - L + eps)

klow  = (min(O, C) - L) / (O + eps)
klow2 = (min(O, C) - L) / (H - L + eps)

ksft  = (2*C - H - L) / (O + eps)
ksft2 = (2*C - H - L) / (H - L + eps)
```

TWAP/AWAP 相对位置：

```text
kotwap  = (O - TW) / (O + eps)
kotwap2 = (O - TW) / (H - L + eps)
kctwap  = (C - TW) / (C + eps)
kctwap2 = (C - TW) / (H - L + eps)

koawap  = (O - AW) / (O + eps)
koawap2 = (O - AW) / (H - L + eps)
kcawap  = (C - AW) / (C + eps)
kcawap2 = (C - AW) / (H - L + eps)
```

如果该组存在 VWAP，再生成：

```text
kovwap  = (O - VW) / (O + eps)
kovwap2 = (O - VW) / (H - L + eps)
kcvwap  = (C - VW) / (C + eps)
kcvwap2 = (C - VW) / (H - L + eps)
```

列名规则：

```text
origin 组: klen, kmid, ...
buy 组:    klen_buy, kmid_buy, ...
sell 组:   klen_sell, kmid_sell, ...
quote 组:  klen_spread, kmid_spread, ... 或 klen_asksize, ...
```

### Buy/sell、up/down、bid/ask 归一化公式

代码位置：

- `operator_futures/cross_section/base_feature_util.py`

Buy/sell 归一化。对任意三元组：

```text
X_all, X_buy, X_sell
```

生成：

```text
X_buy_bsnorm = X_buy / (X_all + eps_cross)
X_sell_bsnorm = X_sell / (X_all + eps_cross)
X_buysell_imbalance_bsnorm = (X_buy - X_sell) / (X_all + eps_cross)
```

例如：

```text
volume_buy_bsnorm = volume_buy / (volume + eps_cross)
volume_buysell_imbalance_bsnorm = (volume_buy - volume_sell) / (volume + eps_cross)
```

Up/down 归一化。对三元组：

```text
X_all, X_up, X_down
```

生成：

```text
X_up_udnorm = X_up / (X_all + eps_cross)
X_down_udnorm = X_down / (X_all + eps_cross)
X_updown_imbalance_udnorm = (X_up - X_down) / (X_all + eps_cross)
```

对四元组：

```text
X_all, X_up, X_down, X_flat
```

额外生成：

```text
X_flat_udnorm = X_flat / (X_all + eps_cross)
X_updownflat_vol_udnorm = (X_up + X_down - X_flat) / (X_all + eps_cross)
```

Bid/ask 归一化。对二元组：

```text
X_bid, X_ask
```

生成：

```text
X_bid_abnorm = X_bid / (X_bid + X_ask + eps_cross)
X_ask_abnorm = X_ask / (X_bid + X_ask + eps_cross)
X_bid_imbalance_abnorm = (X_bid - X_ask) / (X_bid + X_ask + eps_cross)
```

对三元组：

```text
X_all, X_bid, X_ask
```

生成：

```text
X_bid_abnorm = X_bid / (X_all + eps_cross)
X_ask_abnorm = X_ask / (X_all + eps_cross)
X_askbid_imbalance_abnorm = (X_bid - X_ask) / (X_all + eps_cross)
```

实际列名由代码里的原始特征名拼出来，例如：

```text
nquote_bid_abnorm = nquote_bid / (nquote + eps_cross)
nquote_askbid_imbalance_abnorm = (nquote_bid - nquote_ask) / (nquote + eps_cross)
```

### 25 档 orderbook 截面公式

代码位置：

- `operator_futures/cross_section/base_feature_util.py`

对某个时刻 `t`，省略下标 `t`。定义：

```text
sum_ask_size = sum(AS_l, l=1..25)
sum_bid_size = sum(BS_l, l=1..25)

nAS_l = AS_l / sum_ask_size
nBS_l = BS_l / sum_bid_size
```

价格类特征：

```text
midprice = (AP_1 + BP_1) / 2

wap_1 = (AS_1 * BP_1 + BS_1 * AP_1) / (AS_1 + BS_1)
wap_2 = (AS_2 * BP_2 + BS_2 * AP_2) / (AS_2 + BS_2)
wap_balance = wap_1 - wap_2

sell_wap = sum(nAS_l * AP_l, l=1..25)
buy_wap  = sum(nBS_l * BP_l, l=1..25)
buy_sell_wap_spread = buy_wap - sell_wap

buy_spread_oe_max  = abs(BP_1 - BP_25)
sell_spread_oe_max = abs(AP_1 - AP_25)
```

Top-k 大挂单特征。令 `a_r` 是 `np.argsort(AS)[-topk:]` 得到的第 `r` 个 ask 档位索引，`b_r` 是 `np.argsort(BS)[-topk:]` 得到的第 `r` 个 bid 档位索引。也就是说，代码取 size 最大的 top-k 档，但 top-k 内部顺序是从较小到较大，不再反转。

默认 `topk = 5`，对 `r = 1..5`：

```text
ask_price_topk_size_{r}_increments = AP_{a_r} - AP_1
bid_price_topk_size_{r}_increments = BP_{b_r} - BP_1

ask_size_topk_size_{r}_increments = AS_{a_r} - AS_1
bid_size_topk_size_{r}_increments = BS_{b_r} - BS_1
```

挂单量类特征：

```text
buy_volume_oe = sum(BS_l, l=1..25)
sell_volume_oe = sum(AS_l, l=1..25)

imblance_volume_oe =
    (buy_volume_oe - sell_volume_oe)
    / (buy_volume_oe + sell_volume_oe + eps_cross)

ask{l}_size_n = AS_l / sell_volume_oe
bid{l}_size_n = BS_l / buy_volume_oe
```

其中 `l = 1..25`。

### 单列时间滚动特征公式

代码位置：

- `operator_futures/time_operator/create_feature_multi_processing.py`
- `operator_futures/time_operator/multi_processing_util.py`

对任意单列序列 `X_t`，其中 `X` 来自 110 个输入列：

```text
bid1_price ... bid25_price
ask1_price ... ask25_price
buy_spread_oe_max, sell_spread_oe_max,
wap_1, wap_2, buy_wap, sell_wap,
mark_price,
buy_volume_oe, sell_volume_oe, imblance_volume_oe,
ask1_size_n ... ask25_size_n,
bid1_size_n ... bid25_size_n
```

对每个窗口 `w in {2, 6, 12, 16, 24, 48}`：

```text
X_log_return_{w,t} = log(X_t / (X_{t-1} + eps_time)) * 1000

X_trend_{w,t} =
    (X_t - mean_w(X)_t) / (std_w(X)_t + eps_time)
```

注意：代码里的 `log_return_{w}` 虽然带窗口后缀，但实际用的是 `X_t / X_{t-1}`，不是 `X_t / X_{t-w}`。

### OHLCV 时间滚动特征公式

代码位置：

- `operator_futures/time_operator/multi_processing_util.py`

OHLCV 输入组包括：

```text
origin: open, high, low, close, volume
buy:    open_buy, high_buy, low_buy, close_buy, volume_buy
sell:   open_sell, high_sell, low_sell, close_sell, volume_sell
```

对任意一组，记：

```text
O_t, H_t, L_t, C_t, V_t
R_t = C_t / C_{t-1} - 1
Rpos_t = max(R_t, 0)
Rabs_t = abs(R_t)

dV_t = V_t - V_{t-1}
dVpos_t = max(dV_t, 0)
dVabs_t = abs(dV_t)

LV_t = log(V_t + 1)
```

窗口无关列：

```text
log_volume_t = log(V_t + 1)
```

对每个窗口 `w`：

```text
roc_{w,t} = C_{t-w} / C_t
roc_{w}_std_norm_t = C_{t-w} / (std_w(C)_t + eps_time)

ma_{w,t} = mean_w(C)_t / C_t
ma_{w}_std_norm_t = mean_w(C)_t / (std_w(C)_t + eps_time)

std_{w,t} = std_w(C)_t / C_t

beta_{w,t} = (C_{t-w} - C_t) / (w * C_t)
beta_{w}_std_norm_t = (C_{t-w} - C_t) / (w * (std_w(C)_t + eps_time))

max_{w,t} = max_w(C)_t / C_t
max_{w}_std_norm_t = max_w(C)_t / (std_w(C)_t + eps_time)

min_{w,t} = min_w(C)_t / C_t
min_{w}_std_norm_t = min_w(C)_t / (std_w(C)_t + eps_time)

qtlu_{w,t} = q80_w(C)_t / C_t
qtlu_{w}_std_norm_t = q80_w(C)_t / (std_w(C)_t + eps_time)

qtld_{w,t} = q20_w(C)_t / C_t
qtld_{w}_std_norm_t = q20_w(C)_t / (std_w(C)_t + eps_time)
```

Rank 和高低点位置：

```text
rank_{w,t} = pct_rank(C_t within W_t(w)) / w

imax_{w,t} = argmax(H over W_t(w)) / w
imin_{w,t} = argmin(L over W_t(w)) / w
imxd_{w,t} = (argmax(H over W_t(w)) - argmin(L over W_t(w))) / w
```

这里 `argmax/argmin` 是窗口内从 0 开始的位置。`pct_rank` 已经是 pandas 的百分比排名，代码又除以 `w`，所以公式按代码写为 `pct_rank / w`。

RSV：

```text
low_ref_t = min(L_t, C_{t-w})
high_ref_t = max(H_t, C_{t-w})

rsv_{w,t} = (C_t - low_ref_t) / (high_ref_t - low_ref_t + eps_time)
rsv_{w}_std_norm_t = (C_t - low_ref_t) / (std_w(C)_t + eps_time)
```

收益方向计数：

```text
cntp_{w,t} = sum(1[R_j > 0], j in W_t(w)) / w
cntn_{w,t} = sum(1[R_j < 0], j in W_t(w)) / w
cntd_{w,t} = cntp_{w,t} - cntn_{w,t}
```

价量相关：

```text
corr_{w,t} = corr(C_j, LV_j, j in W_t(w))

PR_t = C_t / C_{t-1}
PV_t = log(V_t / V_{t-1} + 1)
cord_{w,t} = corr(PR_j, PV_j, j in W_t(w))
```

收益正负强度：

```text
sump_{w,t} = sum(Rpos_j, j in W_t(w))
             / (sum(Rabs_j, j in W_t(w)) + eps_time)

sumn_{w,t} = 1 - sump_{w,t}
sumd_{w,t} = 2 * sump_{w,t} - 1
```

成交量滚动：

```text
vma_{w,t} = mean_w(V)_t / (V_t + eps_time)
vma_{w}_std_norm_t = mean_w(V)_t / (std_w(V)_t + eps_time)

vstd_{w,t} = std_w(V)_t / (V_t + eps_time)
```

带成交量的波动：

```text
Shift_t = abs(C_t / C_{t-1} - 1) * V_t
wvma_{w,t} = std_w(Shift)_t / (mean_w(Shift)_t + eps_time)
```

成交量变化方向：

```text
vsump_{w,t} = sum(dVpos_j, j in W_t(w))
              / (sum(dVabs_j, j in W_t(w)) + eps_time)

vsumn_{w,t} = 1 - vsump_{w,t}
vsumd_{w,t} = 2 * vsump_{w,t} - 1
```

列名规则：

```text
origin: log_volume_origin, roc_2_origin, ...
buy:    log_volume_buy, roc_2_buy, ...
sell:   log_volume_sell, roc_2_sell, ...
```

### OHLC 时间滚动特征公式

代码位置：

- `operator_futures/time_operator/multi_processing_util.py`

OHLC 输入组包括：

```text
spread, mid, imblance_volume, makav_rev, makav_ori,
bid, bidsize, ask, asksize
```

对任意一组，记：

```text
O_t, H_t, L_t, C_t
R_t = C_t / C_{t-1} - 1
Rpos_t = max(R_t, 0)
Rabs_t = abs(R_t)
```

对每个窗口 `w`：

```text
roc_{w,t} = C_{t-w} / C_t
roc_{w}_std_norm_t = C_{t-w} / (std_w(C)_t + eps_time)

ma_{w,t} = mean_w(C)_t / C_t
ma_{w}_std_norm_t = mean_w(C)_t / (std_w(C)_t + eps_time)

std_{w,t} = std_w(C)_t / C_t

beta_{w,t} = (C_{t-w} - C_t) / (w * C_t)
beta_{w}_std_norm_t = (C_{t-w} - C_t) / (w * (std_w(C)_t + eps_time))

max_{w,t} = max_w(C)_t / C_t
max_{w}_std_norm_t = max_w(C)_t / (std_w(C)_t + eps_time)

min_{w,t} = min_w(C)_t / C_t
min_{w}_std_norm_t = min_w(C)_t / (std_w(C)_t + eps_time)

qtlu_{w,t} = q80_w(C)_t / C_t
qtlu_{w}_std_norm_t = q80_w(C)_t / (std_w(C)_t + eps_time)

qtld_{w,t} = q20_w(C)_t / C_t
qtld_{w}_std_norm_t = q20_w(C)_t / (std_w(C)_t + eps_time)
```

Rank、高低点位置、RSV：

```text
rank_{w,t} = pct_rank(C_t within W_t(w)) / w

imax_{w,t} = argmax(H over W_t(w)) / w
imin_{w,t} = argmin(L over W_t(w)) / w
imxd_{w,t} = (argmax(H over W_t(w)) - argmin(L over W_t(w))) / w

low_ref_t = min(L_t, C_{t-w})
high_ref_t = max(H_t, C_{t-w})
rsv_{w,t} = (C_t - low_ref_t) / (high_ref_t - low_ref_t + eps_time)
```

收益方向和强度：

```text
cntp_{w,t} = sum(1[R_j > 0], j in W_t(w)) / w
cntn_{w,t} = sum(1[R_j < 0], j in W_t(w)) / w
cntd_{w,t} = cntp_{w,t} - cntn_{w,t}

sump_{w,t} = sum(Rpos_j, j in W_t(w))
             / (sum(Rabs_j, j in W_t(w)) + eps_time)

sumn_{w,t} = 1 - sump_{w,t}
sumd_{w,t} = 2 * sump_{w,t} - 1
```

列名规则：

```text
roc_2_spread, roc_2_std_norm_spread, ...
roc_48_asksize, sumd_48_asksize, ...
```

### 特征选择 target 公式

代码位置：

- `operator_futures/feature_selection/ic_correlation.py`
- `operator_futures/feature_selection/rank_ic_correlation.py`
- `operator_futures/feature_selection/catbooost.py`
- `operator_futures/feature_selection/lasso_linear.py`

IC、RankIC、CatBoost 的预测目标：

```text
y_t^{h} = mark_price_{t+h} - mark_price_t
```

默认：

```text
h in {1, 6, 12}
```

Lasso 版本只用：

```text
y_t = mark_price_{t+1} - mark_price_t
```

### 缩放公式

代码位置：

- `operator_futures/scale_describe_save/scale_save.py`

对每个已选状态特征列 `X`，先按标准差缩放。默认 `base = 10`：

```text
scale_std = base ^ floor(log10(std(X)) * log10(base) / log10(10))
X' = X / scale_std
```

默认 `base = 10` 时等价于：

```text
scale_std = 10 ^ floor(log10(std(X)))
X' = X / scale_std
```

然后按均值平移。默认 `clip_threshold = 10`：

```text
mu = mean(X')

if abs(mu) > clip_threshold:
    adjust_power = base ^ round(log10(abs(mu)) * log10(base) / log10(10))
else:
    adjust_power = 0

if mu > 0:
    X'' = X' - adjust_power
else:
    X'' = X' + adjust_power
```

最终 `df.feather` 里保存的是：

```text
前 106 列 reward/environment 原样保留
已选 state_features.npy 对应列使用 X''
```

## 特征选择规则

代码位置：

- `operator_futures/feature_selection/ic_correlation.py`
- `operator_futures/feature_selection/rank_ic_correlation.py`
- `operator_futures/feature_selection/catbooost.py`
- `operator_futures/feature_selection/lasso_linear.py`
- `operator_futures/feature_selection/cor_util.py`

### 候选特征范围

所有选择脚本都先做：

```python
reward_features = df.columns[:106]
state_feature = [col for col in df.columns if col not in reward_features]
```

所以：

- 前 106 列不参与选择
- 第 107 列开始都是候选状态特征
- 候选包括：snapshot 截面特征、base feature、quotes/kline 截面特征、time feature

### Target

IC、RankIC、CatBoost 使用多窗口未来 `mark_price` 差分：

```python
target = df["mark_price"].shift(-window_length) - df["mark_price"]
```

默认窗口：

```text
1, 6, 12
```

### IC

`ic_correlation.py`：

1. 对每个 target window，计算每个状态特征与未来 `mark_price` 差分的 Pearson correlation。
2. 按 `abs(correlation)` 从大到小排序。
3. 保留 `abs(correlation) > ic_theshold` 的列，默认阈值 `0.01`。
4. 多个 window 的结果按顺序合并，并保持第一次出现的顺序。
5. 对候选列计算相关性矩阵。
6. 调用 `select_feature(..., cor_theshold)` 做相关性去重，默认阈值 `0.7`。
7. 输出：
   - `df.feather`
   - `state_features.npy`
   - `ic_window_{1,6,12}.json`
   - `correlation.csv`

### Rank IC

`rank_ic_correlation.py`：

逻辑与 IC 一样，但先对 feature 和 target 做 rank，再算 Pearson correlation。输出：

- `df_rank.feather`
- `state_features_rank.npy`
- `rank_ic_window_{1,6,12}.json`
- `rank_correlation.csv`

### CatBoost

`catbooost.py`：

1. 对每个 target window，用全部状态候选特征训练 `CatBoostRegressor`。
2. 读取 CatBoost feature importance。
3. 保留 `Importance > ic_theshold` 的列，默认阈值 `0.01`。
4. 多个 window 的结果按顺序合并，并保持第一次出现的顺序。
5. 对候选列计算相关性矩阵。
6. 调用 `select_feature(..., cor_theshold)` 做相关性去重。
7. 输出：
   - `df_catboost.feather`
   - `state_features_catboost.npy`
   - `cat_boost_feature_importance_{1,6,12}.csv`
   - `correlation_catboost.csv`

CatBoost 是 README 中说明的 FineFT 默认市场状态构建方式；但脚本文件强制 GPU，如果机器没有可用 GPU 会失败。

### Lasso

`lasso_linear.py`：

1. target 只用一步未来 `mark_price` 差分。
2. 对所有状态候选做 `StandardScaler`。
3. 用 `LassoCV` 拟合。
4. 保留非零系数特征。
5. 输出：
   - `df_lasso.feather`
   - `state_features_lasso.npy`

它没有调用 `cor_util.select_feature` 做二次相关性去重。

### 相关性去重

`cor_util.select_feature()` 的规则是贪心的：

1. 输入特征已经按重要性排序。
2. 从前往后遍历。
3. 当前特征如果还没被删除，就加入最终特征。
4. 删除所有与当前特征绝对相关性 `> cor_theshold` 的后续特征。

所以最终特征数量取决于：

- `ic_theshold`
- `cor_theshold`
- feature importance 排序
- 数据本身
- target windows

## 最终产物在哪里

特征选择中间结果：

```text
PREPROCESS_DATASET/binance-futures/IC_RESULT/{symbol}/{target_freq}/{start_date}-{end_date}/
```

可能包含：

```text
df.feather
state_features.npy
df_rank.feather
state_features_rank.npy
df_catboost.feather
state_features_catboost.npy
df_lasso.feather
state_features_lasso.npy
```

最终给 FineFT 使用的缩放后结果：

```text
PREPROCESS_DATASET/binance-futures/SCALE_SAVE/{symbol}/{target_freq}/{start_date}-{end_date}/df.feather
PREPROCESS_DATASET/binance-futures/SCALE_SAVE/{symbol}/{target_freq}/{start_date}-{end_date}/state_features.npy
PREPROCESS_DATASET/binance-futures/SCALE_SAVE/{symbol}/{target_freq}/{start_date}-{end_date}/df_describe.csv
```

`scale_save.py` 根据 `--ic_choice` 决定读哪个中间结果：

| `--ic_choice` | 读取的 df | 读取的特征列表 | 最终输出 |
| --- | --- | --- | --- |
| `ic` | `df.feather` | `state_features.npy` | `SCALE_SAVE/.../df.feather` 和 `state_features.npy` |
| `rank_ic` | `df_rank.feather` | `state_features_rank.npy` | 同上 |
| `catboost` | `df_catboost.feather` | `state_features_catboost.npy` | 同上 |

无论选择方式是什么，最终保存给 FineFT 的文件名都统一叫：

```text
df.feather
state_features.npy
```

## 如果要查看某次实际最终选择了哪些列

等预处理跑完后，直接读取对应 npy：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ArchetypeTrade
python - <<'PY'
import numpy as np

path = "PREPROCESS_DATASET/binance-futures/SCALE_SAVE/BNBUSDT/5min/2021-04-01-2024-01-01/state_features.npy"
features = np.load(path, allow_pickle=True)
print(len(features))
for f in features:
    print(f)
PY
```

如果要看特征选择阶段、尚未 scale 的结果，可以换成：

```text
PREPROCESS_DATASET/binance-futures/IC_RESULT/{symbol}/{freq}/{start}-{end}/state_features.npy
PREPROCESS_DATASET/binance-futures/IC_RESULT/{symbol}/{freq}/{start}-{end}/state_features_rank.npy
PREPROCESS_DATASET/binance-futures/IC_RESULT/{symbol}/{freq}/{start}-{end}/state_features_catboost.npy
```
