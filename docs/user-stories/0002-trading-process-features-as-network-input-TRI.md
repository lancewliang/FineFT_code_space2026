---
status: draft
owner: FineFT
---

# Trading Process Features as Network Input — TRI

## Context

- User story: [0002-trading-process-features-as-network-input.md](0002-trading-process-features-as-network-input.md)
- Full glossary: [../../LANGUAGE.md](../../LANGUAGE.md)
- Full system map: [../../MAP.md](../../MAP.md)

### Key Domain Terms

| Term | Definition |
|------|------------|
| Trading Process Feature | Agent 侧动作执行后的实时交易状态输入，由归一化 signed position exposure 和当前持仓的收益率/最大回撤率组成。 |
| 当前持仓 | 从 position 由 0 变为非 0 或持仓方向改变时开始，并在平仓到 0 或持仓方向再次改变时结束；同方向变仓不结束。 |
| Previous Action | 现有低层 Q 网络输入名，实际表示当前 position/leverage 映射到 action space 的编码。 |
| State Feature | 经过特征选择后用于 RL agent 观测的训练特征，由 `state_features.npy` 记录。 |
| Reverse Position | 一步内先平掉当前仓位再反向开仓，采用 best-effort 语义。 |

### Components Involved

| Component | Role in this feature |
|-----------|----------------------|
| Futures Trading Environment (`FineFT/env/`) | 计算并在 `reset()` / `step()` 的 `info["trading_info"]` 暴露动作执行后的 Trading Process Feature。 |
| Replay Buffer (`FineFT/RL/util/replay_buffer_DQN.py`) | 在采样结果中保留 `trading_info`，供当前状态和 next state 的网络输入使用。 |
| Low-level Q Network (`FineFT/model/low_level.py`) | 新增 `fc_trading` 编码连续 Trading Process Feature，并与现有输入融合。 |
| Stage I Low-level Training (`FineFT/RL/DiHFT/low_level/`) | 从 replay buffer 的 `info` / `next_info` 提取 `trading_info` 并传入 eval/target network。 |
| Stage II/III, Ablation, Baselines | 一次性迁移所有 `Qnet` / `ensemble_Qnet` 构造和调用点，避免旧签名残留。 |

### Relevant Decisions

| Decision | Summary |
|----------|---------|
| [0001-reverse-position-semantics.md](../decisions/0001-reverse-position-semantics.md) | Reverse Position 是 best-effort 先平后开；方向改变会结束当前持仓。 |
| [0002-require-trading-info-qnet-input.md](../decisions/0002-require-trading-info-qnet-input.md) | `Qnet` / `ensemble_Qnet` 必须显式接收 `trading_info`，不提供默认零向量兼容路径。 |

---

## What Is Being Built

把环境已计算但未进入网络的当前持仓暴露、当前持仓收益率和当前持仓最大回撤打包为 `trading_info`，经 replay buffer 传递，并作为低层 Q 网络第五路输入参与决策。

## Functional Requirements

1. `Base_Env.reset()` 和所有 `Base_Env.step()` 返回路径必须包含 `info["trading_info"]`。
2. `trading_info` 字段顺序必须唯一固定为 `("position_exposure", "single_holding_return_rate", "single_holding_max_drawdown")`。
3. `position_exposure = position / max(abs(p) for p in self.position_list)`，不得新增 CLI 参数或硬编码最大仓位。
4. `trading_info` 表示动作执行后的可观测状态；空仓、正常平仓和爆仓后必须为 `[0, 0, 0]`。
5. 当前持仓的收益率和最大回撤在同方向加仓/减仓时延续，在平仓或持仓方向改变时重置。
6. `single_holding_return_rate` 分母使用当前持仓生命周期内的累计投入/占用口径，沿用 `calculate_required_money(...)`；持仓结束时重置相关 history。
7. Replay buffer 所有网络输入采样路径必须返回 `trading_info`，包括 `sample()`、`sample_evaluate()` 和 sunrise 分支。
8. `Qnet.forward()`、`ensemble_Qnet.forward()` 和 `ensemble_Qnet.get_best_q()` 必须显式要求 `trading_info` 参数。
9. Repo 内所有 `Qnet` / `ensemble_Qnet` 构造和调用点必须一次性迁移，不得留下旧签名调用。
10. 旧 checkpoint 与新增 `fc_trading` 权重不兼容；实现不得通过 `trading_info=None` 或默认零向量静默兼容旧路径。

## Non-Functional Requirements

- **Performance**: 每个 Qnet 只新增一个 `Linear(3, hidden_nodes)` 和一次 concat；不得引入 per-step 显著额外 I/O 或全局扫描。
- **Reliability**: 缺少 `trading_info` 的网络调用必须失败；字段顺序和 replay buffer 白名单必须用常量收敛，避免静默错位。
- **Scalability**: `position_exposure` 必须随 `position_list` 动态归一化，支持未来不同 `position_choices` / `max_holding_number` 配置。
- **Security**: 无外部输入、认证、权限或网络调用变化。

---

## Design

### Architecture

```text
Base_Env.reset()/step()
  info["trading_info"] = [position_exposure, return_rate, max_drawdown]
        |
        v
Multi_step_ReplayBuffer*_multi_info.sample()
  infos["trading_info"], next_infos["trading_info"]
        |
        v
training / testing / routing callers
  trading_info.float().reshape(batch_size, -1).to(device)
        |
        v
ensemble_Qnet.forward(..., trading_info)
        |
        v
Qnet.forward(..., trading_info)
  state -> fc1
  previous_action -> fc3
  time -> fc_time
  trading_info -> fc_trading
  concat -> fc2 -> out -> availability mask
```

### Data Model

```yaml
info.trading_info:
  type: numpy.ndarray
  shape: [3]
  dtype: numeric float-compatible
  order:
    - position_exposure
    - single_holding_return_rate
    - single_holding_max_drawdown

position_exposure:
  formula: position / max(abs(p) for p in position_list)
  range: [-1, 1]

NETWORK_INFO_KEYS:
  includes:
    - avaliable_action
    - previous_action
    - q_value
    - high_level_state
    - funding_count_down_hour
    - funding_count_down_minute
    - trading_info
```

### API Design

No external API changes.

Internal Python interface changes:

```python
Qnet(
    N_STATES,
    N_ACTIONS,
    hidden_nodes,
    TIME_INFO_DIM,
    TRADING_INFO_DIM=3,
)

Qnet.forward(
    state,
    time,
    previous_action,
    avaliable_action,
    trading_info,
)

ensemble_Qnet(
    N_STATES,
    N_ACTIONS,
    hidden_nodes,
    TIME_INFO_DIM,
    ensemble_number,
    TRADING_INFO_DIM=3,
)

ensemble_Qnet.forward(
    state,
    time,
    previous_action,
    avaliable_action,
    trading_info,
)

ensemble_Qnet.get_best_q(
    state,
    time,
    previous_action,
    avaliable_action,
    trading_info,
)
```

### User Interface

No user-facing UI changes. Training CLIs may expose `--trading_info_dim` where they already construct `ensemble_Qnet`; default must remain `3`.

---

## Implementation Plan

### Phase 1: Environment Contract

- [ ] Add `TRADING_INFO_KEYS` and a small helper in `FineFT/env/env_class/base_env.py`.
- [ ] Compute `position_exposure` from `self.position_list`.
- [ ] Add `info["trading_info"]` to `reset()` and every `step()` return path.
- [ ] Reset current-holding histories when position becomes zero or direction changes.
- [ ] Preserve current-holding histories when same-direction position changes.

### Phase 2: Replay Buffer Propagation

- [ ] Add shared `NETWORK_INFO_KEYS` in `FineFT/RL/util/replay_buffer_DQN.py`.
- [ ] Replace duplicated hard-coded info-key lists in network-input sample paths with the shared constant.
- [ ] Ensure `trading_info` is stacked as float-compatible tensors for both `infos` and `next_infos`.

### Phase 3: Q Network Input

- [ ] Add `TRADING_INFO_DIM=3` to `Qnet` and `ensemble_Qnet` constructors.
- [ ] Add `self.fc_trading = nn.Linear(TRADING_INFO_DIM, hidden_nodes)`.
- [ ] Increase `fc2` input size to `N_ACTIONS + 2 * hidden_nodes + time_bedding`.
- [ ] Require `trading_info` in `forward()` / `get_best_q()` and pass it to every child Qnet.
- [ ] Do not add a `None` fallback or zero-vector compatibility path.

### Phase 4: Call-Site Migration

- [ ] Migrate Stage I training and test files.
- [ ] Migrate Stage II/III high-level routing files.
- [ ] Migrate ablation files.
- [ ] Migrate baseline DQN/SUNRISE files.
- [ ] Search for remaining old-signature `Qnet` / `ensemble_Qnet` construction, `forward()`, and `get_best_q()` calls.

### Phase 5: Checkpoint Handling

- [ ] Document or update load paths so old checkpoints are not treated as compatible by accident.
- [ ] Use `strict=False` only in explicit checkpoint migration utilities, not in normal training/evaluation paths to hide missing `fc_trading` weights.

---

## New Dependencies

| Dependency | Purpose | Validated |
|------------|---------|-----------|
| None | This feature uses existing NumPy, PyTorch, and project code only. | yes |

---

## Testing Strategy

### Unit Tests

- Environment tests in `FineFT/tests/env/`:
  - `reset()` returns `[0, 0, 0]`.
  - Normal held position returns normalized `position_exposure`.
  - Same-direction add/reduce does not reset return/drawdown.
  - Closing to zero resets `trading_info` to `[0, 0, 0]`.
  - Direction change ends old current holding and starts a new one.
  - Liquidation path returns action-after state `[0, 0, 0]`.

- Replay buffer tests in `FineFT/tests/rl/`:
  - `sample()` returns `infos["trading_info"]` and `next_infos["trading_info"]` with shape `(batch_size, 3)`.
  - `sample_evaluate()` returns the same keys and shape.
  - Sunrise buffer sample path includes `trading_info`.
  - `NETWORK_INFO_KEYS` is the only whitelist used by these network-input sample paths.

- Model tests in `FineFT/tests/rl/`:
  - `Qnet.forward(..., trading_info)` returns `(batch, N_ACTIONS)`.
  - `ensemble_Qnet.forward(..., trading_info)` returns `(batch, ensemble_number, N_ACTIONS)`.
  - `ensemble_Qnet.get_best_q(..., trading_info)` returns `(batch, ensemble_number)`.
  - Missing `trading_info` raises a Python call error.
  - `fc_trading` weights are present in `state_dict`.

### Integration Tests

- Minimal training/update-path tests should verify `info["trading_info"]` and `next_info["trading_info"]` are extracted and passed to eval/target networks.
- Static or lightweight runtime check should confirm repo-local Qnet call sites no longer use old signatures.

### End-to-End Tests

- No new end-to-end user flow is required. Existing low-level train/test smoke tests should continue to run after full migration.

---

## Security Considerations

- No new external inputs, credentials, network calls, or file permissions are introduced.

## Performance Considerations

- The added model cost is linear in `hidden_nodes * 3 * ensemble_number`, small relative to existing state feature encoding.
- `trading_info` construction must remain O(1) per environment step except for computing `max_abs_position`; compute/cache `max_abs_position` once from `position_list` if needed.

---

## Resilience Checks

- **Failure points**: The main risks are missing `trading_info` in an environment return path, replay buffer key filtering, and old Qnet call signatures. Tests must cover each layer.
- **Duplication**: `previous_action` already carries current position/leverage as action-space encoding, but it does not carry normalized exposure, return rate, or drawdown; no existing equivalent replaces `trading_info`.
- **Dependency failure**: No new dependency is introduced.
- **API failure**: No external API calls are introduced.
- **Supply chain**: No open-source dependency is added.

## Known Risks

- Current-holding history reset logic may expose existing assumptions in `single_holding_return_rate` and `single_holding_max_drawdown` calculations.
- Full migration touches many scripts; the highest risk is an untested baseline or ablation path retaining an old call signature.
- Old checkpoints are intentionally incompatible; experiment reruns must be planned.

## Open Questions

- [ ] Exact test fixture for direction-change and liquidation `trading_info` may need to reuse existing reverse-position fixtures or add a small deterministic environment fixture.

## References

- [0002 user story](0002-trading-process-features-as-network-input.md)
- [LANGUAGE.md](../../LANGUAGE.md)
- [MAP.md](../../MAP.md)
- [0001 reverse position semantics](../decisions/0001-reverse-position-semantics.md)
- [0002 require trading info qnet input](../decisions/0002-require-trading-info-qnet-input.md)
