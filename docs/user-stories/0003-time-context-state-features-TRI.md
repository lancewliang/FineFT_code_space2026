---
status: draft
owner: FineFT
---

# BASE_TIME_FEATURE State Features — TRI

## Context

- User story: [0003-time-context-state-features.md](0003-time-context-state-features.md)
- Full glossary: [../../LANGUAGE.md](../../LANGUAGE.md)
- Full system map: [../../MAP.md](../../MAP.md)

### Key Domain Terms

| Term | Definition |
|------|------------|
| BASE_TIME_FEATURE | 与 Base Feature 平级的商品期货时间编码特征产物；必须进入 State Feature，不参与 Feature Selection 指标计算，也不参与 Scale Save 缩放。 |
| Trading Session 内进度 | 当前 timestamp 在所属 Trading Session 内按 session 持续分钟数归一化得到的进度，不跨午休或非交易空档连续计算。 |
| 合约交割月份 | 合约代码中表示交割月份的月份字段，如 `fu2605` 中的 `05`。 |
| 合约最后交易日 | 在选择合约数据文件时，从原始下载中该合约全部 `TradingDay` 取最大值得到的真实最后交易日。 |
| 合约完整交易日数量 | 在选择合约数据文件时，从原始下载中该合约全部不同 `TradingDay` 计数得到的交易日总数。 |
| State Feature | 经过特征选择后用于 RL agent 观测的训练特征，由 `state_features.npy` 记录。 |

### Components Involved

| Component | Role in this feature |
|-----------|----------------------|
| Commodity Preprocessing (`data_preprocess/operator_futures/commodity/`) | 在 `Main Contract Summary` 中记录 `last_trading_day` 和 `total_trading_day_count`，并提供 Trading Session 配置。 |
| BASE_TIME_FEATURE Generator | 新增与 `BASE_FEATURE` 平级的产物层，按 `BASE_FEATURE` timestamp 生成 9 个非绝对时间/合约生命周期特征。 |
| Daily Merge (`data_preprocess/operator_futures/merge_concat/merge.py`) | 强校验并将同日期 `BASE_TIME_FEATURE` join 到 `FUTURE_FEATURE`。 |
| Feature Selection (`data_preprocess/operator_futures/feature_selection/`) | 排除 BASE_TIME_FEATURE 的指标计算，但将其作为 mandatory state feature 追加到最终 `state_features.npy`。 |
| Scale Save (`data_preprocess/operator_futures/scale_describe_save/muti_contract_scale_save.py`) | 对普通 state feature 做 robust scaling，对 BASE_TIME_FEATURE passthrough，并在 manifest 记录。 |

### Relevant Decisions

| Decision | Summary |
|----------|---------|
| [0003-base-time-feature-mandatory-passthrough.md](../decisions/0003-base-time-feature-mandatory-passthrough.md) | BASE_TIME_FEATURE 是业务必需上下文，不参与 Feature Selection 指标，也不参与 Scale Save robust scaling。 |

---

## What Is Being Built

为商品期货预处理增加 `BASE_TIME_FEATURE` 产物层，并让它以 mandatory passthrough state feature 的身份贯穿 daily merge、Feature Selection 和 commodity multi-contract Scale Save。

## Functional Requirements

1. `MainContractSummaryContract` 必须新增并序列化 `last_trading_day` 和 `total_trading_day_count`。
2. `stitch_main_contract.py`/主力合约选择逻辑必须在读取合约源文件时，从该合约原始下载的全部不同 `TradingDay` 计算生命周期元数据。
3. 新增唯一常量 `BASE_TIME_FEATURE_COLUMNS`，按以下顺序定义 9 列：`trading_minute_progress`, `morning_session`, `afternoon_session`, `night_session`, `is_opening_30m`, `is_closing_30m`, `contract_month_sin`, `contract_month_cos`, `contract_life_remaining_ratio`。
4. `BASE_TIME_FEATURE` 生成必须以同日期同合约 `BASE_FEATURE` 的 `timestamp` 为主输入，输出只包含 `timestamp` 和 `BASE_TIME_FEATURE_COLUMNS`。
5. `trading_minute_progress` 必须按所属 Trading Session 内进度计算，不按全天分钟或自然时间间隔计算。
6. `morning_session`、`afternoon_session`、`night_session` 必须为互斥 one-hot；`is_opening_30m`、`is_closing_30m` 必须按每个 Trading Session 独立计算。
7. `contract_month_sin`/`contract_month_cos` 必须从合约交割月份计算，不使用当前交易日自然月。
8. `contract_life_remaining_ratio` 必须使用包含当前交易日的剩余交易日数量除以 `total_trading_day_count`；最后交易日当天为 `1 / total_trading_day_count`。
9. Daily merge 缺少 `BASE_TIME_FEATURE` 文件或 timestamp 集合不一致时必须 fail-fast。
10. Feature Selection 指标计算 universe 必须排除 `BASE_TIME_FEATURE_COLUMNS`。
11. 最终 `state_features.npy` 必须将普通筛选特征放在前面，并按常量顺序追加去重后的 `BASE_TIME_FEATURE_COLUMNS`。
12. Feature Selection Manifest 必须记录 `mandatory_state_features`。
13. Commodity multi-contract Scale Save 必须对 `BASE_TIME_FEATURE_COLUMNS` passthrough，不参与 robust scaler fit/transform/clip。
14. Scale Manifest 必须记录 `passthrough_state_features`。

## Non-Functional Requirements

- **Performance**: 生命周期元数据在主力合约选择阶段计算一次；`BASE_TIME_FEATURE` 生成不得按日期重复扫描原始下载全量数据。
- **Reliability**: 缺失文件、timestamp 错位、缺失 summary 生命周期字段、缺失 mandatory 列、blacklist 冲突都必须 fail-fast。
- **Scalability**: 实现应支持多合约、多 target_freq 和不同商品 Trading Session 配置。
- **Security**: 无外部网络、权限或认证变化。

---

## Design

### Architecture

```text
raw contract files
  |
  v
stitch_main_contract.py
  Main Contract Summary
    - last_trading_day
    - total_trading_day_count
  |
  v
BASE_FEATURE + summary + commodity trading sessions
  |
  v
BASE_TIME_FEATURE/{symbol}/{contract}/{target_freq}/{date}.feather
  |
  v
daily merge -> FUTURE_FEATURE -> concat -> split
  |
  v
Feature Selection
  metrics on non-BASE_TIME_FEATURE candidates
  state_features.npy = selected + BASE_TIME_FEATURE_COLUMNS
  |
  v
muti_contract_scale_save.py
  scale selected normal features
  passthrough BASE_TIME_FEATURE
```

### Data Model

```yaml
MainContractSummaryContract:
  last_trading_day: str  # raw TradingDay string, YYYYMMDD
  total_trading_day_count: int

BASE_TIME_FEATURE file:
  path: PREPROCESS_DATASET/commodity-futures/BASE_TIME_FEATURE/{symbol}/{contract}/{target_freq}/{date}.feather
  columns:
    - timestamp
    - trading_minute_progress
    - morning_session
    - afternoon_session
    - night_session
    - is_opening_30m
    - is_closing_30m
    - contract_month_sin
    - contract_month_cos
    - contract_life_remaining_ratio

FeatureSelectionManifest:
  mandatory_state_features: list[str]

ScaleManifest:
  passthrough_state_features: list[str]
```

### API Design

No external API changes.

Internal CLI/function changes:

- Add a commodity BASE_TIME_FEATURE generation entrypoint or runner in the commodity preprocessing flow.
- Daily merge reads the expected `BASE_TIME_FEATURE` path when `market_type=commodity_futures` or when running the commodity scripts.
- `run_feature_selection(...)` uses `BASE_TIME_FEATURE_COLUMNS` as mandatory excluded-from-metrics columns.
- `muti_contract_scale_save.py` uses `BASE_TIME_FEATURE_COLUMNS` as passthrough state features.

### User Interface

Type: CLI.

Commodity full process scripts should run the new BASE_TIME_FEATURE generation before daily merge. No new user-facing parameter is required unless an implementation needs an explicit `--summary` path for the generator.

---

## Implementation Plan

### Phase 1: Summary Lifecycle Metadata

- [ ] Extend `MainContractSummaryContract` dataclass with `last_trading_day` and `total_trading_day_count`.
- [ ] Update `from_dict()` and `to_dict()` with validation.
- [ ] Calculate lifecycle metadata during contract file loading/selection from all available raw `TradingDay` values for each contract.
- [ ] Update summary tests and CLI fixture expectations.

### Phase 2: BASE_TIME_FEATURE Generation

- [ ] Add `BASE_TIME_FEATURE_COLUMNS` in a shared commodity feature module.
- [ ] Implement generator using `BASE_FEATURE` timestamp, commodity Trading Session config, contract code, and Main Contract Summary lifecycle fields.
- [ ] Write `BASE_TIME_FEATURE/{symbol}/{contract}/{target_freq}/{date}.feather` with only `timestamp` plus the 9 feature columns.
- [ ] Add commodity full process runner step before daily merge.

### Phase 3: Daily Merge

- [ ] Read the same-date same-contract BASE_TIME_FEATURE file during commodity daily merge.
- [ ] Fail-fast when the file is missing.
- [ ] Fail-fast when timestamp sets differ from `BASE_FEATURE`.
- [ ] Join BASE_TIME_FEATURE columns into `FUTURE_FEATURE`.

### Phase 4: Feature Selection

- [ ] Exclude `BASE_TIME_FEATURE_COLUMNS` from metric calculation universe.
- [ ] Reject `feature_blacklist` entries that target BASE_TIME_FEATURE columns.
- [ ] Append de-duplicated mandatory BASE_TIME_FEATURE columns after selected normal features in `state_features.npy`.
- [ ] Add `mandatory_state_features` to `FeatureSelectionManifest`.
- [ ] Ensure filtered per-contract outputs include mandatory columns.

### Phase 5: Commodity Multi-Contract Scale Save

- [ ] Add `passthrough_state_features` to `ScaleManifest`.
- [ ] Fit robust scaler only on non-BASE_TIME_FEATURE state features.
- [ ] Transform/clip only scaled features; concatenate passthrough BASE_TIME_FEATURE columns unchanged.
- [ ] Keep final output column order aligned with `state_features.npy`.
- [ ] Scope required passthrough support to `muti_contract_scale_save.py`; legacy `scale_save.py` is not required for this feature.

---

## New Dependencies

| Dependency | Purpose | Validated |
|------------|---------|-----------|
| None | Use existing Polars, NumPy, commodity config, and pipeline code. | yes |

---

## Testing Strategy

### Unit Tests

- Main Contract Summary:
  - serializes/deserializes `last_trading_day` and `total_trading_day_count`;
  - computes lifecycle metadata from all raw TradingDay values, not selected main-contract window only.
- BASE_TIME_FEATURE generator:
  - outputs exactly `timestamp` plus `BASE_TIME_FEATURE_COLUMNS`;
  - computes session progress inside each Trading Session;
  - marks morning/afternoon/night one-hot correctly;
  - marks opening/closing 30 minutes per session;
  - computes delivery month sin/cos from contract code;
  - computes last-day `contract_life_remaining_ratio = 1 / total_trading_day_count`.
- Feature Selection:
  - excludes BASE_TIME_FEATURE columns from metrics;
  - appends mandatory columns in constant order;
  - rejects blacklist conflicts;
  - writes `mandatory_state_features` to manifest.
- Scale Save:
  - robust scaler manifest excludes passthrough columns from `features`;
  - output preserves BASE_TIME_FEATURE values exactly;
  - `passthrough_state_features` appears in manifest.

### Integration Tests

- Commodity daily merge joins BASE_TIME_FEATURE into `FUTURE_FEATURE`.
- Daily merge fails when BASE_TIME_FEATURE file is missing.
- Daily merge fails when timestamp sets differ.
- Commodity split -> Feature Selection -> Scale Save mini-flow keeps all mandatory columns through final output without scaling them.

### End-to-End Tests

- Update commodity full-process script/CLI tests to verify the BASE_TIME_FEATURE step is included before merge and before Feature Selection.

---

## Security Considerations

- No new external service, credential, user input surface, or network dependency.

## Performance Considerations

- Avoid repeated raw file scans during per-date BASE_TIME_FEATURE generation; lifecycle metadata belongs in Main Contract Summary.
- BASE_TIME_FEATURE generation is linear in `BASE_FEATURE` row count and should remain cheap relative to downscale/cross-section generation.

---

## Known Risks

- Trading Session classification must handle night sessions that cross calendar dates.
- Historical fixtures and tests that assert Main Contract Summary shape will need updates for new required fields.
- Keeping legacy `scale_save.py` out of scope means only commodity multi-contract Scale Save is guaranteed to passthrough BASE_TIME_FEATURE.

## Open Questions

- None.

## References

- [0003-time-context-state-features.md](0003-time-context-state-features.md)
- [../decisions/0003-base-time-feature-mandatory-passthrough.md](../decisions/0003-base-time-feature-mandatory-passthrough.md)
