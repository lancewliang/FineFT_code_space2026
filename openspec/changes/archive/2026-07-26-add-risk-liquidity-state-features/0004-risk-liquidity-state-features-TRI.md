---
status: draft
owner: FineFT
---

# Risk and Liquidity State Features — TRI

## Context

- User story: [0004-risk-liquidity-state-features.md](0004-risk-liquidity-state-features.md)
- Full glossary: [../../LANGUAGE.md](../../LANGUAGE.md)
- Full system map: [../../MAP.md](../../MAP.md)

### Key Domain Terms

| Term | Definition |
|------|------------|
| 风险状态特征 | 基于 OHLC 和收益率历史窗口计算的波动率类 State Feature。 |
| 流动性状态特征 | 基于成交量、成交额和持仓量历史窗口计算的市场活跃度 State Feature。 |
| 日内 Bar 数 | 根据商品期货 `Trading Session` 总交易时长和 `target_freq` 推导的每日 bar 数。 |
| 滚动窗口特征 | 基于历史窗口滚动计算的衍生特征，由 `time_operator` 生成。 |
| State Feature | 经过 Feature Selection 后用于 RL agent 观测的训练特征。 |

### Components Involved

| Component | Role in this feature |
|-----------|----------------------|
| Commodity Preprocessing (`data_preprocess/operator_futures/commodity/`) | 在 `BASE_FEATURE` 输出窗口末尾 `open_interest`，并在缺少原始 `OpenInterest` 时 fail-fast。 |
| Rolling Window Feature Generator (`data_preprocess/operator_futures/time_operator/`) | 从合并后的 5min 特征生成带窗口后缀的风险状态特征和流动性状态特征。 |
| Commodity Config (`data_preprocess/operator_futures/commodity/config.py`) | 提供 `Trading Session`，用于推导 `bars_per_day`。 |
| DataQualityValidator | 校验新输出不包含 `NaN`、`inf` 或 `-inf`。 |
| Feature Selection / Scale Save | 将新列作为普通 candidate state feature 消费，不作为 mandatory 或 passthrough。 |

### Relevant Decisions

| Decision | Summary |
|----------|---------|
| [0003-base-time-feature-mandatory-passthrough.md](../decisions/0003-base-time-feature-mandatory-passthrough.md) | Base_Time_feature 是 mandatory passthrough；本特征明确不沿用该规则，而是普通候选特征。 |

---

## What Is Being Built

为商品期货预处理增加 `open_interest` 基础列，并在 `time_operator` 中生成 10 类带窗口后缀的风险与流动性滚动 State Feature。

## Functional Requirements

1. `downscale_continuous_by_trading_day.py` 的秒级下采样必需列必须包含原始 `OpenInterest`。
2. `downscale_base_features()` 必须在 `BASE_FEATURE` 输出 `open_interest`，取当前右闭右标下采样窗口内最后一条快照的 `OpenInterest`。
3. 缺少 `OpenInterest`、`OpenInterest` 为 null、非数值或非有限值时必须 fail-fast。
4. `time_operator` 必须对每个 `--windows` 窗口输出以下风险状态特征：`atr_pct_{window}`、`historical_volatility_{window}`、`rolling_volatility_{window}`、`parkinson_volatility_{window}`、`garman_klass_volatility_{window}`、`realized_volatility_{window}`。
5. `time_operator` 必须对每个 `--windows` 窗口输出以下流动性状态特征：`relative_volume_{window}`、`relative_amount_{window}`、`relative_open_interest_{window}`、`open_interest_change_ratio_{window}`。
6. 所有新滚动特征必须只使用当前及过去 bar，不读取未来 bar，不跨合约混合窗口。
7. 所有新滚动特征必须沿用现有 `time_operator` warmup 裁剪行为，前 `max(window) + 1` 行不进入 `TIME_FEATURE`。
8. `historical_volatility_{window}` 必须使用 `std(log_return, window) * sqrt(bars_per_day)`；`bars_per_day` 由商品 `Trading Session` 总交易分钟数和 `target_freq` 推导。
9. `rolling_volatility_{window}` 必须使用 EWMA 收益率标准差，`window` 作为 EWM span。
10. 相对流动性特征的 rolling mean 分母必须包含当前 bar。
11. `open_interest_change_ratio_{window}` 遇到 `open_interest_{t-window} <= 0` 时输出 `0`。
12. 风险状态公式涉及的 `open/high/low/close` 非正、null、非数值或非有限值时必须 fail-fast。
13. `garman_klass_volatility_{window}` 在 `sqrt` 前必须对滚动均值做 lower-bound 0 裁剪。
14. `relative_amount_{window}` 使用 `tradeval`，不得新增重复的 `amount` 基础列。
15. 新特征必须作为普通 candidate state feature 进入后续流程，不加入 `mandatory_state_features`，不作为 Scale Save passthrough。

## Non-Functional Requirements

- **Performance**: 新特征应在现有 Polars `time_operator` 路径中向量化计算；不得引入逐行 Python 循环处理整列行情。
- **Reliability**: 输入异常必须在 downscale 或 time feature 阶段 fail-fast；输出必须通过现有非有限值校验。
- **Scalability**: 实现必须支持多合约、多窗口、多 `target_freq` 和不同商品 `Trading Session`。
- **Security**: 无外部网络、认证、权限或密钥变化。

---

## Design

### Architecture

```text
second-level commodity snapshots
  require: OpenInterest
  |
  v
commodity/downscale.py
  BASE_FEATURE:
    timestamp, open, high, low, close, volume, tradeval, open_interest, ...
  |
  v
MERGE_CONCAT / CONCAT_FEATURE
  |
  v
time_operator/create_feature_multi_processing.py
  uses --windows and --target_freq
  |
  v
TIME_FEATURE:
  atr_pct_{window}
  historical_volatility_{window}
  rolling_volatility_{window}
  parkinson_volatility_{window}
  garman_klass_volatility_{window}
  realized_volatility_{window}
  relative_volume_{window}
  relative_amount_{window}
  relative_open_interest_{window}
  open_interest_change_ratio_{window}
```

### Data Model

```yaml
BASE_FEATURE:
  open_interest:
    source: OpenInterest
    aggregation: last snapshot in right-closed right-labeled window
    dtype: numeric finite

TIME_FEATURE additions:
  risk_state_features:
    - atr_pct_{window}
    - historical_volatility_{window}
    - rolling_volatility_{window}
    - parkinson_volatility_{window}
    - garman_klass_volatility_{window}
    - realized_volatility_{window}
  liquidity_state_features:
    - relative_volume_{window}
    - relative_amount_{window}
    - relative_open_interest_{window}
    - open_interest_change_ratio_{window}
```

### API Design

No external API changes.

Internal changes:

- `SECOND_LEVEL_DOWNSCALE_REQUIRED_COLUMNS` includes `OpenInterest`.
- `downscale_base_features(second_df, target_freq, symbol)` returns `open_interest`.
- `create_feature_multi_processing.py` passes commodity symbol and `target_freq` context to the helper that computes the new columns.
- `multi_processing_util.py` adds a Polars helper for risk/liquidity state features and joins it by `timestamp`.

### User Interface

Type: CLI.

Existing commodity preprocessing and `time_operator` CLI invocations continue to use `--windows` and `--target_freq`. No new user-facing parameter is required.

---

## Implementation Plan

### Phase 1: BASE_FEATURE `open_interest`

- [ ] Add `OpenInterest` to commodity second-level required columns.
- [ ] Validate `OpenInterest` in second-level snapshots / downscale input using existing fail-fast validation style.
- [ ] Extend `downscale_base_features()` to aggregate `pl.col("OpenInterest").last().alias("open_interest")`.
- [ ] Include `open_interest` in the returned `BASE_FEATURE` select list.
- [ ] Update commodity downscale tests and fixtures to include `OpenInterest`.

### Phase 2: Risk/Liquidity Helper

- [ ] Add a Polars helper in `time_operator/multi_processing_util.py` for these 10 feature families.
- [ ] Implement price input validation for `open/high/low/close`.
- [ ] Implement `bars_per_day` calculation from `get_commodity_config(symbol).trading_sessions` and `target_freq`.
- [ ] Implement all formulas with output names `{feature_name}_{window}`.
- [ ] Join the helper output into `time_feature_list_all` in `create_feature_multi_processing.py`.

### Phase 3: Pipeline Integration

- [ ] Ensure time feature input must contain `open_interest` before generating OI-based features.
- [ ] Preserve existing warmup row slicing and timestamp alignment.
- [ ] Ensure output CSV and feather include the new columns.
- [ ] Confirm Feature Selection and Scale Save need no mandatory/passthrough changes.

---

## New Dependencies

| Dependency | Purpose | Validated |
|------------|---------|-----------|
| None | Existing Polars/Numpy stack is sufficient. | yes |

---

## Testing Strategy

### Unit Tests

- `downscale_base_features()` outputs `open_interest` as the last `OpenInterest` value in each target window.
- Commodity downscale fails when `OpenInterest` is missing.
- Risk feature helper computes expected values for a small deterministic OHLC series.
- `historical_volatility_{window}` uses session-derived `bars_per_day`, not a hardcoded 24h bar count.
- `rolling_volatility_{window}` differs from simple rolling std by using EWMA semantics.
- `garman_klass_volatility_{window}` clips negative sqrt input to 0.
- `open_interest_change_ratio_{window}` outputs 0 when the shifted denominator is non-positive.
- Price inputs with non-positive `open/high/low/close` fail-fast.

### Integration Tests

- `create_feature_multi_processing.py` on a commodity fixture with `open_interest` writes all 10 feature families for each configured window.
- `--windows 12,20` creates both `_12` and `_20` suffixes and no unsuffixed risk/liquidity columns.
- Output retains existing `time_operator` warmup slicing and validates no illegal values.

### End-to-End Tests

- Commodity preprocessing path from downscale through time feature generation produces `open_interest` in `BASE_FEATURE` and the new risk/liquidity columns in `TIME_FEATURE`.

---

## Security Considerations

- No new external calls, credentials, or file permissions.

## Performance Considerations

- Use Polars expressions and rolling operations.
- Avoid pandas-only additions in `create_feature_multi_processing.py` and `multi_processing_util.py`; existing tests assert these files do not import pandas.
- Avoid per-row Python loops over full market data.

---

## Known Risks

- `time_operator` currently has existing generic features such as `std_{window}` and `vma_{window}`; the new explicit names may be correlated but are intentionally clearer state features.
- EWMA rolling volatility in Polars may require careful implementation to match expected span semantics.
- `OpenInterest` availability depends on the raw commodity source; making it required can break older fixtures or datasets until they are updated.

## Open Questions

- [ ] Should legacy input datasets without `OpenInterest` be migrated, or is fail-fast sufficient for all supported commodity runs?

## References

- [0004-risk-liquidity-state-features.md](0004-risk-liquidity-state-features.md)
- [../../LANGUAGE.md](../../LANGUAGE.md)
- [../../MAP.md](../../MAP.md)
- [../decisions/0003-base-time-feature-mandatory-passthrough.md](../decisions/0003-base-time-feature-mandatory-passthrough.md)
