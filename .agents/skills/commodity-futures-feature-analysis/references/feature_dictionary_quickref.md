# Commodity Futures Feature Dictionary & Code Mapping Quick Reference

This reference maps feature categories and naming conventions to their generating Python modules in `data_preprocess/operator_futures/commodity/` and their specification sections in `openspec/specs/commodity-futures-feature-engineering/spec.md`.

---

## 1. Feature Category Mapping

| Category | Typical Feature Naming Patterns | Code Location | Spec Section | Key Metrics / Purpose |
|---|---|---|---|---|
| **Base & Reference Price** | `mark_price`, `vwap`, `twap`, `awap`, `midprice`, `open_interest` | `downscale.py`, `downscale_single_day.py` | Sec 3.1 | Core benchmark prices, volume-weighted prices |
| **Quote & Multi-Depth Pressure** | `wap_1`, `wap_2`, `wap_balance`, `buy_wap`, `sell_wap`, `bid_ask_spread`, `ask1_price_log_return_w` | `downscale.py`, `cross_section/base_feature_util.py` | Sec 3.2 | Multi-depth order book imbalance, spread, price returns |
| **Order Flow Imbalance (OFI)** | `ofi_h1`..`ofi_h5`, `ofi_sum_5`, `ofi_weighted_5` | `downscale.py` | Sec 3.3 | Microstructure order flow pressure across 5 orderbook levels |
| **Queue & Microstructure** | `micro_depth_imbalance`, `bid_queue_pressure`, `ask_queue_pressure` | `downscale.py` | Sec 3.4 | Depth-5 queue length imbalances and queue pressure ratios |
| **Cross-Section KLine Geometry** | `klen`, `kmid`, `kup`, `ksft`, `open`, `high`, `low`, `close`, `volume` | `cross_section/create_feature.py` | Sec 3.5 | Candle geometry and normalized candle body/shadow ratios |
| **Base Time Features** | `session_progress`, `is_night_session`, `sin_delivery_month`, `cos_delivery_month` | `base_time_feature.py` | Sec 3.6 | Mandatory protected features (`BASE_TIME_FEATURE`) for session & lifecycle |
| **Mixed-Frequency Daily** | `prev_day_return`, `prev_day_range_pct`, `prev_2_day_trade_imbalance`, `prev_10_day_turnover_rate` | `daily_base_feature.py`, `daily_mixed_frequency_feature.py` | Sec 3.7 | Higher time-frame (daily) history metrics mapped back to intraday |
| **Mixed-Frequency Weekly** | `prev_week_return`, `prev_week_range_pct`, `prev_2_week_trade_up_ratio`, `prev_4_week_open_interest_change` | `weekly_base_feature.py`, `weekly_mixed_frequency_feature.py` | Sec 3.7 | Weekly history metrics mapped back to intraday samples |
| **Cross-Month Contract** | `main_sub_spread`, `relative_spread`, `volume_ratio_main_sub`, `oi_ratio_main_sub`, `main_switch_indicator` | `cross_month_feature.py` | Sec 3.8 | Main contract vs term structure/sub-main contract spreads and liquidity shift |
| **Rolling Risk & Liquidity** | `atr_pct_w`, `parkinson_volatility_w`, `garman_klass_volatility_w`, `relative_open_interest_w`, `oi_change_rate_norm_10m` | `downscale_continuous_by_trading_day.py` | Sec 3.9 | Volatility estimators (Parkinson, Garman-Klass) and liquidity dynamics |
| **Time-Operator Rolling OHLCV** | `ma_w_origin`, `bollinger_lower_w_origin`, `pivot_s1_w_origin`, `rsv_w_std_norm_origin`, `roc_w_origin` | `time_operator/time_operator_util.py` | Sec 3.10 | Multi-window (2, 6, 12, 16, 24, 48, 96, 192) rolling technical operators |
| **RL Trading Process** | `position_exposure`, `single_holding_return_rate`, `single_holding_max_drawdown`, `current_holding_duration_norm` | FineFT Environment | Sec 3.11 | Agent state tracking during execution |

---

## 2. Directory Structure & Key Output Files

```
PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/
└── {target_freq}/               # e.g., 30min, 10min
    └── {symbol}/                # e.g., fu (Fuel Oil)
        ├── train/
        │   ├── feature_selection_manifest.json  # Feature selection metadata, filters, drop breakdown
        │   ├── aggregate_metrics.csv            # IC, Rank IC, Permutation Importance, CatBoost, Sharpe
        │   ├── state_features.npy               # Final selected state feature list (numpy string array)
        │   ├── per_contract/                    # Contract-specific metrics CSVs
        │   │   ├── fu2305_metrics.csv
        │   │   └── fu2401_metrics.csv
        │   └── {contract}/                      # Individual contract feature feather files
        │       └── df.feather
        └── valid/
            ├── feature_selection_manifest.json
            ├── aggregate_metrics.csv
            └── per_contract/
```

---

## 3. Mandatory / Protected Features

As specified in `openspec/specs/commodity-futures-feature-engineering/spec.md` Section 2 & 3.6, mandatory features (such as `BASE_TIME_FEATURE`) MUST NOT be dropped during correlation or performance filtering because RL models require explicit temporal anchoring.

Mandatory features list:
- `session_progress`
- `is_night_session`
- `sin_delivery_month`
- `cos_delivery_month`
- `contract_lifecycle_ratio`
