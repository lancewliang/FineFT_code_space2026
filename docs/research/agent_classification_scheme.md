# Agent Classification Scheme: `label_{} + agent{version} + bin_{n} => {tag, profitability}`

**Research date**: 2026-08-06
**Scope**: Map every `(label, agent_version, bin_index)` triple produced by the DiHFT low-level test pipeline to one of 6 proposed strategy-archetype tags plus a profitability record, grounding each rule in code/data that actually exists.
**Method**: All claims traced to primary sources (scripts, Python modules, JSON manifests, CSV outputs). Where a rule does not yet exist in code it is explicitly marked **(proposed, not in source)**.

---

## 1. Summary

Two scripts drive the pipeline. [`FineFT/script/data/commodity_data_handler_30min_fu.sh`](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/script/data/commodity_data_handler_30min_fu.sh) builds the train/valid/test dataset (by date split + chunking), then runs [`slice_model.py`](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/datahandler/slice_model.py) per valid contract feather, which is what actually creates the `label_0..label_6` directories — there are **7 labels**, not 5 (memory of "7" is correct; "5" confuses `dynamic_number=5` with the 2 extra limit-state labels). [`FineFT/script/test/DiHFT/low_level/test_util_fu_30.sh`](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/script/test/DiHFT/low_level/test_util_fu_30.sh) wraps [`test_agent_index.py`](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/test_agent_index.py), which iterates the ensemble Q-net over every `(label, initial_action, bin_index)` cube and writes per-step `trading_action_detail` CSV plus an aggregate `analysis_result` CSV. Agent candidate selection per label is done by [`FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py`](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py) via a Semantic Guard + positive-reward filter then `trans_reward_mean` ranking, producing [`selection_manifest.json`](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/analysis_result/DiHFT/low_level/fu/30min_multi/selection_manifest.json).

The 6-type taxonomy (3 trend-following + 3 mean-reversion second-order patterns) is **a proposed design, not a derivation** from existing diagnostics. The repo already records per-step position/action/PnL trajectories in the trading-detail CSV, which is sufficient raw material to compute the shape signals (step-function snaps, right-shifted lag, exponential position-vs-cumPnL, U-bottom, linear reverse slope, divergence filters). However, **no script currently computes those shape features**, and Sharpe/Calmar/MDD/win-rate are only computed at the high-level portfolio stage — not per `(label, agent, bin)`. **Scope decision (2026-08-06)**: this round implements only the 6-class second-order grouping; reconciliation with the prior 12-archetype taxonomy in [`docs/research/label_agent_selection_logic.md`](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/docs/research/label_agent_selection_logic.md) / ADR [`docs/adr/0005-label-agent-selection-meta-routing.md`](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/docs/adr/0005-label-agent-selection-meta-routing.md) is deferred. **Locked decisions**: `bin_{n}` = ensemble Q-net head index (existing, not a new price bucket); `--labeling_method` = `slope` (not `DTW`).

---

## 2. Script 1: `test_util_fu_30.sh`

File: [`FineFT/script/test/DiHFT/low_level/test_util_fu_30.sh`](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/script/test/DiHFT/low_level/test_util_fu_30.sh)

### 2.1 What it actually does

The script defines two bash functions and calls one. The active call is `run_ddqn_context` with the `fu`/`30min` defaults ([lines 97-104](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/script/test/DiHFT/low_level/test_util_fu_30.sh#L97-L104)).

`run_ddqn_context` ([lines 2-49](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/script/test/DiHFT/low_level/test_util_fu_30.sh#L2-L49)):
- Takes `dataset_name`, `max_holding_number`, `epoch_start`, `epoch_end`, `base_path`, `experiment_name`.
- `ensemble_number` defaults to **7** ([line 9](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/script/test/DiHFT/low_level/test_util_fu_30.sh#L9)).
- Loops `epoch` from `epoch_start` to `epoch_end` ([line 21](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/script/test/DiHFT/low_level/test_util_fu_30.sh#L21)) and launches `python FineFT/RL/DiHFT/low_level/test_agent_index.py` per epoch in background, max 4 parallel ([lines 16, 35-40](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/script/test/DiHFT/low_level/test_util_fu_30.sh#L16)).
- Key CLI args passed ([lines 24-30](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/script/test/DiHFT/low_level/test_util_fu_30.sh#L24-L30)): `--initial_wallet_balance 10000`, `--order_book_depth 5`, `--position_choices 3`, `--N "${ensemble_number}"` (=7), `--transcation_cost 0.0004`, `--allow_reverse_position`.
- Each epoch's stdout/stderr goes to `log/DiHFT/${dataset_name}/low_level/test/${experiment_name}/epoch_${epoch}.log` ([lines 13, 30](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/script/test/DiHFT/low_level/test_util_fu_30.sh#L13)).

### 2.2 Label iteration, agent selection, bin usage — all inside `test_agent_index.py`

The shell script itself does NOT iterate labels or bins; it delegates to [`test_agent_index.py`](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/test_agent_index.py). The iteration structure inside `weighted_trader.test()` is:

- **Label discovery**: `_iter_valid_feather_files(self.valid_data_path)` ([lines 346-376](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/test_agent_index.py#L346-L376)) walks `valid/<contract>/label_*/df_*.feather`, where `label_*` must match `LABEL_DIR_PATTERN = re.compile(r"^label_\d+$")` ([line 256](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/test_agent_index.py#L256)). So labels are discovered from the on-disk directories created by `slice_model.py` (see Section 3). `label_list = sorted({entry["label"] for entry in df_entries})` ([line 611](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/test_agent_index.py#L611)).
- **Three nested loops** per label ([lines 612-619](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/test_agent_index.py#L612-L619)):
  1. `for label in label_list`
  2. `for initial_action in self.initial_action_list` — `initial_action_list = range((position_choices - 1) * len(leverage_choices) + 1)` ([lines 563-565](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/test_agent_index.py#L563-L565)). With `position_choices=3` and `leverage_choices=[1]` this is `range(3) = {0,1,2}` = the starting action (short / flat / long) the episode begins from.
  3. `for bin_index in range(self.N)` — `N` is the ensemble context number (=7 from the shell). **`bin_index` is the ensemble-Q-net head index** (which of the 7 ensemble Q-nets is active), confirmed by `act_test(self, state, info, context_index)` which asserts `context_index in range(self.N)` ([line 568](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/test_agent_index.py#L568)) and selects `self.eval_net.qnet_list[context_index]` ([line 593](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/test_agent_index.py#L593)). The call site passes `bin_index` as `context_index` ([line 700](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/test_agent_index.py#L700)).

  **Reconciliation with the user's `bin_{n}`**: the user's `bin_n` corresponds to this `bin_index` ∈ {0..6}. It is NOT a price/volatility bucket; it is the ensemble-head selector. The user's `agent{version}` corresponds to the epoch checkpoint (`epoch_path` / `trained_model.pkl`), and `label_{}` is `label_0..label_6`.

### 2.3 Per-step trading detail (rich trajectory data)

When `--save_trading_detail_csv` is on (default True, [line 176-180](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/test_agent_index.py#L176-L180)), every step appends a row via `build_trading_detail_row` ([lines 400-460](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/test_agent_index.py#L400-L460)). Each row contains: `label, df_path, initial_action, bin_index, timestep, open/high/low/close/volume/mark_price, action, target_position, target_leverage, position_before, leverage_before, position_after, leverage_after, action_change_step, trade_count_step, cumulative_action_change_count, cumulative_trade_count, step_reward, realized_pnl_step, cumulative_realized_pnl, commission_fee_step, cumulative_commission_fee, slippage_step, cumulative_slippage, wallet_balance, unrealized_pnl, margin_balance, notional_asset_value, total_value`.

Output path: `trading_action_detail_epoch_{epoch_num}.csv` under the epoch dir ([lines 338-339, 946-950](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/test_agent_index.py#L338-L339)). **This CSV is the primary raw material for any action-curve / position-trajectory classifier** (see Section 6).

### 2.4 Aggregate metrics per (label, initial_action, bin_index)

After each episode, per-(label, initial_action, bin_index) aggregate metrics are computed ([lines 838-934](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/test_agent_index.py#L838-L934)) and collected into `_overall_result` ([lines 911-935](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/test_agent_index.py#L911-L935)). The `AGGREGATE_JSON_COLUMNS` list ([lines 234-255](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/test_agent_index.py#L234-L255)) defines the schema:

`contract, df_path, reward_sum, df_length, turnover, mean_position, mean_abs_position, long_step_ratio, short_step_ratio, flat_step_ratio, long_reward_sum, short_reward_sum, flat_reward_sum, net_position_exposure, limit_up_step_ratio, limit_down_step_ratio, limit_up_long_reward_sum, limit_down_short_reward_sum, limit_up_reverse_short_ratio, limit_down_reverse_long_ratio`.

- `net_position_exposure = mean_position / max_holding_number` ([line 862](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/test_agent_index.py#L862)).
- Limit-state detection per step via `_detect_step_limit_states` ([lines 197-232](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/test_agent_index.py#L197-L232)) using `limit_up_single_sided_ratio`, `is_limit_up`, `UpperLimitPrice`, etc.
- Outputs: `analysis_result.npy` + `analysis_result.csv` ([lines 941-945](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/test_agent_index.py#L941-L945)).

### 2.5 Profitability computation at this stage

Profitability at this stage is **reward-based only** (no Sharpe/Calmar/MDD/win-rate here):
- `reward_sum` = sum of step rewards per episode.
- The pick-agent stage ([Section 5](#5-agent-candidate-pool--ranking)) derives `normalized_reward = reward_sum / df_length` and `trans_reward_mean` / `trans_reward_std` across the contracts in that label ([lines 422-431 of pick_agent](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py#L422-L431)).
- Sharpe / Calmar / MDD / SoR / required-money are computed only at the **high-level portfolio stage** by [`FineFT/analysis/calculate_metric/calculate_metric.py`](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/analysis/calculate_metric/calculate_metric.py) and surfaced in [`analysis_result/DiHFT/high_level_heurstic/fu/30min_multi/best_result.csv`](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/analysis_result/DiHFT/high_level_heurstic/fu/30min_multi/best_result.csv) (columns `tr, portfolio_tr, daily_vol, mdd, downside_deviation_daily, annual_sr, daily_cr, daily_SoR, required_money`). See [`conclude_metric.py` lines 160-173](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/analysis/calculate_metric/conclude_metric.py#L160-L173).

### 2.6 Commented-out trajectory dump (important gap)

Lines [762-820](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/test_agent_index.py#L762-L820) contain a commented-out block that would have saved, per (label, bin_index, initial_action, df_path): `micro_action_history.npy`, `reward_history.npy`, `initial_margin_history.npy`, `wallet_balance_history.npy`, `unrealized_pnl_history.npy`, `maintain_marigine_history.npy`, `new_position_required_money_history.npy`. **This is currently disabled**, so the only per-step trajectory artifact is the `trading_action_detail_*.csv`. The high-level stage separately re-runs episodes and does dump these `.npy` files (see [`HANDOFF_dihft_analysis.md`](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/HANDOFF_dihft_analysis.md)).

---

## 3. Script 2: `commodity_data_handler_30min_fu.sh`

File: [`FineFT/script/data/commodity_data_handler_30min_fu.sh`](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/script/data/commodity_data_handler_30min_fu.sh)

### 3.1 Three stages

1. **Build train/valid/test datasets** ([lines 14-22](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/script/data/commodity_data_handler_30min_fu.sh#L14-L22)): runs [`FineFT/datahandler/commodity_contract_dataset.py`](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/datahandler/commodity_contract_dataset.py) with `--dataset_split_manifest_path`, `--input_root PREPROCESS_DATASET/commodity-futures/SCALE_SAVE`, `--state_features_path .../state_features.npy`, `--output_root dataset/30min`, `--symbol fu`, `--target_freq 30min`, `--chunk_length 8000`, `--early_stop 2`. Defaults at [lines 4-8](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/script/data/commodity_data_handler_30min_fu.sh#L4-L8).
2. **Slice valid feathers into labels** ([lines 24-27](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/script/data/commodity_data_handler_30min_fu.sh#L24-L27)): for each `dataset/30min/fu/valid/*.feather`, runs `FineFT/datahandler/slice_model.py --data_path <file> --timestamp timestamp`. **This is the step that creates `label_0..label_6` directories.**
3. **VAE data creation** ([lines 29-32](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/script/data/commodity_data_handler_30min_fu.sh#L29-L32)): runs `FineFT/datahandler/vae_data_creation.py` over the base path.

### 3.2 Where labels are actually split — `slice_model.py`

File: [`FineFT/datahandler/slice_model.py`](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/datahandler/slice_model.py)

- `dynamic_number` arg defaults to **5** ([lines 98-102](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/datahandler/slice_model.py#L98-L102)); `save_limit_labels` defaults to True ([lines 134-138](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/datahandler/slice_model.py#L134-L138)).
- Limit-state detection: `detect_limit_states` ([lines 188-257](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/datahandler/slice_model.py#L188-L257)) sets `is_limit_up`, `is_limit_down`, `is_near_limit_up`, `is_near_limit_down`.
- Label assignment when `save_limit_labels=True` ([lines 450-460](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/datahandler/slice_model.py#L450-L460)):
  - `limit_down_label = 0`
  - `limit_up_label = self.dynamic_number + 1` (= 6 when dynamic_number=5)
  - middle dynamics are shifted `+1`: `merged_data["label"] = merged_data["label"] + 1`
  - `total_label_count = self.dynamic_number + 2` = **7**
- The 5 middle labels come from `Dynamic_labeler` in [`label_util.py`](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/datahandler/label_util.py) using `slope` / `quantile` / `DTW` ([lines 41-90](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/datahandler/label_util.py#L41-L90)); segments are derived from turning points of a Butterworth-filtered price series and linear-regression slopes ([lines 510-566](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/datahandler/label_util.py#L510-L566)).
- Output directories: `label_{i}` for `i in range(total_label_count)` ([lines 465-466](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/datahandler/slice_model.py#L465-L466)); contiguous same-label runs are written as `df_{counter}.feather` segments by `write_segment` ([lines 472-504](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/datahandler/slice_model.py#L472-L504)).

### 3.3 Data flow into the agent pipeline

`commodity_contract_dataset.run_dataset_generation` ([lines 225-260](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/datahandler/commodity_contract_dataset.py#L225-L260)) reads the split manifest, builds per-contract feather outputs, then `rebuild_train_slice_plan` + `write_train_slices` chunk the train set (chunk_length=8000, early_stop=2). The split manifest itself ([`dataset_split_manifest.json`](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/30min/fu/dataset_split_manifest.json)) defines date boundaries: train `2023-01-03`→`2024-07-23`, valid `2024-07-23`→`2025-06-30`, test `2025-06-30`→`2026-02-06` (split ratio 5:3:2) — **split is by date across contracts, not by contract month**.

So the label files that `test_agent_index.py` walks (`valid/<contract>/label_*/df_*.feather`) are produced by stage 2 above; e.g. `fu2505/label_0/df_0.feather` (visible in [`result_all.csv`](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/analysis_result/DiHFT/low_level/fu/30min_multi/result_all.csv) row 0).

### 3.4 Binning logic already present

**No price/volatility binning exists in this script.** The only "bin" in the pipeline is the ensemble-head `bin_index` in `test_agent_index.py` (Section 2.2). The `min_length_limit`/`merging_threshold`/`merging_dynamic_constraint` knobs in `slice_model.py` ([lines 67-90](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/datahandler/slice_model.py#L67-L90)) control segment merging, not binning.

---

## 4. Label semantics

Source: [`analysis_result/DiHFT/low_level/fu/30min_multi/label_semantics.json`](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/analysis_result/DiHFT/low_level/fu/30min_multi/label_semantics.json). `label_number = 7`, `labeling_method = "ordered_default"`.

| Label | direction | sign | strength | description | limit_state | limit_sign | Source |
|:---|:---|:---:|:---:|:---|:---|:---:|:---|
| `label_0` | strong_down | -1 | 2 | 跌停 | limit_down | -1 | [L6-L14](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/analysis_result/DiHFT/low_level/fu/30min_multi/label_semantics.json#L6-L14) |
| `label_1` | down | -1 | 2.0 | 下跌 | none | 0 | [L15-L23](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/analysis_result/DiHFT/low_level/fu/30min_multi/label_semantics.json#L15-L23) |
| `label_2` | down | -1 | 1.0 | 下跌 | none | 0 | [L24-L32](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/analysis_result/DiHFT/low_level/fu/30min_multi/label_semantics.json#L24-L32) |
| `label_3` | sideways | 0 | 0.0 | 震荡 | none | 0 | [L33-L41](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/analysis_result/DiHFT/low_level/fu/30min_multi/label_semantics.json#L33-L41) |
| `label_4` | up | 1 | 1.0 | 上涨 | none | 0 | [L42-L50](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/analysis_result/DiHFT/low_level/fu/30min_multi/label_semantics.json#L42-L50) |
| `label_5` | up | 1 | 2.0 | 上涨 | none | 0 | [L51-L59](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/analysis_result/DiHFT/low_level/fu/30min_multi/label_semantics.json#L51-L59) |
| `label_6` | strong_up | 1 | 2 | 涨停 | limit_up | 1 | [L60-L68](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/analysis_result/DiHFT/low_level/fu/30min_multi/label_semantics.json#L60-L68) |

**Memory contradiction resolved**: memory recalled "7, not 5" — **7 is correct**. The "5" comes from `slice_model.py`'s `dynamic_number=5` default ([line 100](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/datahandler/slice_model.py#L100)); the 2 extra labels (`label_0` limit_down, `label_6` limit_up) are appended by `save_limit_labels=True` ([lines 450-458](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/datahandler/slice_model.py#L450-L458)). This 7-label semantics is also generated/validated by [`FineFT_single_agent_with_different_position.py` `generate_default_label_semantics`](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py#L170-L240) and [`load_label_semantics`](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py#L242-L319) which enforces label_0=limit_down and last label=limit_up.

**Decision (2026-08-06): use `slope`, not `DTW`.** The data slicer [`commodity_data_handler_30min_fu.sh`](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/script/data/commodity_data_handler_30min_fu.sh) does not pass `--labeling_method` to [`slice_model.py`](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/datahandler/slice_model.py), so slicing already runs the **`slope` default** ([slice_model.py L62-L66](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/datahandler/slice_model.py#L62-L66)). The pick-agent launchers previously set `LABELING_METHOD=DTW`, which forced the explicit-manifest load path ([load_label_semantics L259-L263](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py#L259-L263)) and left the `labeling_method` field in `label_semantics.json` stale (`ordered_default`). Both launchers have been updated to `slope` ([low_level_fu_30.sh L17](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/script/analysis/pick_agent/low_level_fu_30.sh#L17), [low_level_fu_30_half.sh L17](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/script/analysis/pick_agent/low_level_fu_30_half.sh#L17)), aligning pick-agent with the actual slicing method. The stale `ordered_default` string in `label_semantics.json` will self-correct on the next re-slice.

---

## 5. Agent candidate pool & ranking

### 5.1 Candidate pool = every epoch × bin_index × initial_action row

The picker [`FineFT_single_agent_with_different_position.py`](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py) does NOT receive an external "agent version" manifest. It scans every epoch checkpoint under `result/DiHFT/low_level/<dataset>/<experiment>/weights_advantage_pretrain/epoch_*` produced by `test_agent_index.py`. The "agent version" is therefore the tuple **(epoch_path, bin_index)** — epoch = training checkpoint, bin_index = ensemble head.

- `conclude_single_parameter` iterates epochs `range(45, epoch_num)` ([lines 451-459](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py#L451-L459)); the launcher sets `--epoch_num 100` ([low_level_fu_30_half.sh line 33](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/script/analysis/pick_agent/low_level_fu_30_half.sh#L33)), so epochs 46..100 are scanned.
- Each epoch's `analysis_result.npy` is loaded, validated, and transformed ([lines 417-445](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py#L417-L445)); per-row derived fields: `normalized_reward = reward_sum/df_length`, `trans_reward_mean`, `trans_reward_std`, `mean_turnover`, `candidate_mean_exposure`, `candidate_long_ratio`, `candidate_short_ratio`, `candidate_long_reward_mean`, `candidate_short_reward_mean`, `candidate_limit_up_long_reward_mean`, `candidate_limit_down_short_reward_mean`, `candidate_limit_up_reverse_short_ratio`, `candidate_limit_down_reverse_long_ratio`.
- `get_all_parameter_result` ([lines 517-541](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py#L517-L541)) writes `result.csv` (best per (label,initial_action,bin_index,epoch)) and `result_all.csv` (every row).

### 5.2 Ranking = Semantic Guard + positive-reward filter + mean-minus-std score

`pick_best_agent_regarding_dynamics_bin_index_path` ([lines 543-656](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py#L543-L656)) per label:
1. **Positive-reward pool**: `trans_reward_mean > 0` rows first; if none, fall back to all rows ([lines 559-565](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py#L559-L565)).
2. **Semantic Guard** (`check_semantic_alignment`, [lines 321-377](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py#L321-L377)) with thresholds `min_directional_exposure=0.10`, `min_directional_step_ratio=0.35`, `max_neutral_abs_exposure=0.20`, `max_limit_reverse_ratio=0.20` ([lines 150-155](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py#L150-L155)). For a bullish label (`direction_sign=1`) it requires `candidate_mean_exposure >= 0.10`, `candidate_long_ratio >= 0.35`, `candidate_long_reward_mean > 0`; bearish is the mirror; sideways requires `|exposure| <= 0.20`. Limit-up/down states add their own checks.
3. **Rank by** `trans_reward_mean` grouped by `(label, bin_index, epoch_path)`, taking the group with max mean ([lines 589-607](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py#L589-L607)). The earlier `pick_best_index_from_single_epoch` uses `trans_reward_mean - std_preference*trans_reward_std` with `std_preference=0.1` ([lines 482-486](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py#L482-L486)); the final selection collapses `initial_action` via the groupby mean.

### 5.3 Outputs

- `best_index_info_by_dynamics_with_different_position.csv` — one row per label: `(label, epoch_path, bin_index, reward_max, source_rows, behavior_summary)`. See the actual file at [`analysis_result/DiHFT/low_level/fu/30min_multi/best_index_info_by_dynamics_with_different_position.csv`](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/analysis_result/DiHFT/low_level/fu/30min_multi/best_index_info_by_dynamics_with_different_position.csv).
- `selection_manifest.json` — one entry per label with `epoch_path`, `model_path`, `bin_index`, `score`, `behavior_summary`, `selection_reason`. See [`selection_manifest.json`](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/analysis_result/DiHFT/low_level/fu/30min_multi/selection_manifest.json). Example: `label_0` → epoch_46/bin_0/score -3.56 (fallback pool, bearish), `label_4` → epoch_54/bin_6/score 0.57 (passed strict gate, bullish), `label_6` → epoch_58/bin_5/score 0.53 (limit_up).
- A merged ensemble model `model.pth` is written via `create_new_ensemble_qnet_from_different_save_path` ([lines 709-737](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py#L709-L737)).

So the **single selected agent per label** is `(epoch_path, bin_index)` from `selection_manifest.json`. There is no persisted multi-agent pool per label in code — the 12-archetype "profile pool" in [`docs/research/label_agent_selection_logic.md`](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/docs/research/label_agent_selection_logic.md) is a design proposal, not an implemented data structure.

---

## 6. Existing diagnostics usable for the 6-type taxonomy

### 6.1 What already exists and is directly usable

| Signal needed by the 6-type taxonomy | Existing field/artifact | Source |
|:---|:---|:---|
| Position trajectory `position_after[t]` vs price `mark_price[t]` | `trading_action_detail_epoch_{N}.csv` columns `position_after, mark_price, timestep` | [test_agent_index.py L400-460](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/test_agent_index.py#L400-L460) |
| Action sequence (step function shape) | same CSV column `action` + `action_change_step` | [L442](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/test_agent_index.py#L442) |
| Cumulative realized PnL | `cumulative_realized_pnl` | [L448](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/test_agent_index.py#L448) |
| Floating PnL | `unrealized_pnl` | [L454](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/test_agent_index.py#L454) |
| Mean position / net exposure | aggregate `mean_position`, `mean_abs_position`, `net_position_exposure` | [L845-L862](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/test_agent_index.py#L845-L862) |
| Long/short/flat step ratios | `long_step_ratio`, `short_step_ratio`, `flat_step_ratio` | [L851-L853](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/test_agent_index.py#L851-L853) |
| Directional reward split | `long_reward_sum`, `short_reward_sum`, `flat_reward_sum` | [L855-L857](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/test_agent_index.py#L855-L857) |
| Turnover (trade frequency) | `turnover` | [L828-L830](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/test_agent_index.py#L828-L830) |
| Limit-state behavior | `limit_up_step_ratio`, `limit_down_step_ratio`, `limit_up_reverse_short_ratio`, `limit_down_reverse_long_ratio` | [L864-L878](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/test_agent_index.py#L864-L878) |
| Profitability (reward-space) | `reward_sum`, `normalized_reward`, `trans_reward_mean`, `trans_reward_std` | [test_agent_index L822-L824](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/test_agent_index.py#L822-L824); [pick_agent L422-L431](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py#L422-L431) |
| Profitability (risk-adjusted) | `tr, mdd, annual_sr, daily_cr, daily_SoR, daily_vol, required_money` — **only at portfolio/contract level** | [`best_result.csv`](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/analysis_result/DiHFT/high_level_heurstic/fu/30min_multi/best_result.csv); [conclude_metric.py L160-L173](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/analysis/calculate_metric/conclude_metric.py#L160-L173) |

### 6.2 What does NOT exist (be honest)

- **No action-curve shape analysis** (step-function detection, right-shift lag, exponential fit of position-vs-cumPnL, U-bottom steepness, linear reverse slope, divergence filters) is implemented anywhere in the repo. The 6-type taxonomy's defining signals are **not computed**.
- **No per-(label, agent, bin) Sharpe / Calmar / MDD / win-rate**. `calculate_metric` is only invoked at the high-level stage over the merged ensemble's `.npy` histories. `win_rate` is not produced at any stage — only `long_step_ratio`/`short_step_ratio` and reward sums exist as proxies.
- **No "RSI/volume divergence" feature is confirmed in `state_features.npy` by this audit** — the 12-archetype doc references `rsv_192_std_norm_origin`, `bollinger_lower_96_origin`, `price_oi_vol_interaction_10m` ([label_agent_selection_logic.md L25, L29](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/docs/research/label_agent_selection_logic.md#L25)), but the actual `state_features.npy` contents were not enumerated here. The divergence-enhanced rule therefore leans on features whose exact names need verification against `dataset/30min/fu/state_features.npy`.
- The per-step `.npy` trajectory dump in `test_agent_index.py` is **commented out** ([L762-L820](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/test_agent_index.py#L762-L820)); only the CSV is available at the low-level stage.

**Conclusion**: the taxonomy is a **proposed design**, not a derivation from existing diagnostics. The raw material (per-step CSV) exists and is sufficient to build a classifier, but the shape-feature extraction code must be written.

---

## 7. Proposed classification scheme

Map key: `label_{}` from Section 4; `agent{version}` = `epoch_path` (= epoch checkpoint, e.g. `epoch_54`); `bin_{n}` = `bin_index` ∈ {0..6} (ensemble Q-net head, Section 2.2). The classifier consumes the `trading_action_detail_epoch_{N}.csv` for the triple plus the aggregate `analysis_result.csv` row.

### 7.1 Profitability (reuse existing metrics)

`profitability = { reward_sum, normalized_reward, trans_reward_mean, trans_reward_std, mean_turnover, net_position_exposure, long_step_ratio, short_step_ratio, long_reward_sum, short_reward_sum }` (low-level, per triple) **plus** `{ tr, mdd, annual_sr, daily_cr, daily_SoR, required_money }` (high-level, when the triple is promoted into the routed ensemble) — see Section 6.1. **(proposed)**: also compute `win_rate = (# positive step_reward steps) / total_steps` and per-triple Sharpe/Calmar from `cumulative_realized_pnl` and `unrealized_pnl`; these are trivially derivable from the existing CSV but are not yet produced.

### 7.2 Diagnostic rules per tag

All rules below are **(proposed, not in source)** unless marked DERIVED. Each rule operates on the per-step trajectory of one `(label, epoch, bin_index)` triple joined with the per-step `mark_price`.

#### Trend-following (momentum)

**T1. 突破即时型 (Breakout-immediate)** — *position snaps 0→1 on price breaking MA/N-day high; action curve is a step function; big wins/losses, slippage-sensitive.*
- DERIVED signals available: `action_change_step` spikes from 0 to 1 at a single timestep; `position_after` jumps from 0 to ±max in one step; low `mean_abs_position` duration but high `long_reward_sum`/`short_reward_sum` magnitude; `turnover` concentrated in few large changes.
- PROPOSED rule: `count(action_change_step==1 & |Δposition_after|>=max_hold) / total_steps` is small but each such step has `|realized_pnl_step|` in the top decile; `corr(position_after[t], mark_price[t])` strongly positive (sign matches label direction). Slippage sensitivity = `cumulative_slippage / |cumulative_realized_pnl|` is high.
- Trend alignment: only assignable to `label_1..label_5` (directional, non-limit) — `label_0/6` are limit states.

**T2. 回调加仓型 / Smart Entry (Pullback add-position)** — *waits for small pullback after breakout before adding; action curve right-shifted lagged; sensitive to price 2nd derivative (acceleration).*
- PROPOSED rule: cross-correlation `corr(position_after[t], mark_price[t-k])` peaks at k>0 (right shift); `position_after` increases only after a local pullback (price derivative sign flip while the label trend remains). The 2nd-derivative signal requires computing `Δ²mark_price[t] = mark_price[t] - 2*mark_price[t-1] + mark_price[t-2]` from the existing `mark_price` column; `corr(Δposition_after[t], Δ²mark_price[t])` should be significant.
- DERIVED support: `mean_abs_position` builds gradually (not a single step); `long_step_ratio` (for up labels) grows over the episode.

**T3. 金字塔递增型 / Pyramid-increasing** — *scales position with accumulating floating profit; action correlates exponentially with cumulative return; extreme P/L ratio (>5:1), fast profit giveback on drawdown.*
- PROPOSED rule: fit `position_after[t] = a * exp(b * cumulative_realized_pnl[t]) + c`; require `b > 0` and `R² > 0.6`. P/L ratio = `max(cumulative_realized_pnl) / |min(cumulative_realized_pnl)|` > 5. Giveback = `(peak(cumulative_realized_pnl) - final) / peak` is large.
- DERIVED support: `unrealized_pnl` and `position_after` both in the CSV; `mean_position` grows with cumulative PnL.

#### Mean-reversion (reversal)

**M1. 硬边界抄底型 (Hard-boundary bottom-fishing)** — *only acts when price deviates 2σ, heavy reverse position; U-shaped bottom steepening; high win rate (>60%) but cliff loss if boundary breaks.*
- PROPOSED rule: `position_after` is non-zero only when `|mark_price - rolling_mean(mark_price, W)| / rolling_std(mark_price, W) >= 2` (Z-score from `mark_price` column). Heavy reverse position = `|position_after|` near `max_hold` at those times. U-bottom steepening = `position_after` peaks at the Z-score extremum and decays symmetrically.
- DERIVED support: `limit_up_reverse_short_ratio` / `limit_down_reverse_long_ratio` already capture reverse-at-limit behavior ([L873-L878](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/test_agent_index.py#L873-L878)) — generalize this to 2σ deviation. Assignable primarily to `label_0` (limit_down) and `label_6` (limit_up) and to deep `label_1`/`label_5` segments.
- **(proposed)** win_rate threshold > 0.60 requires computing win_rate from `step_reward` signs (not currently produced).

**M2. 网格微调型 (Grid fine-tuning, high-freq)** — *adjusts 1% position per 0.5σ deviation, linear reverse slope; highest win rate (>70%) but thin per-trade profit, fee-sensitive.*
- PROPOSED rule: linear regression `position_after[t] = -α * Z_score[t] + β` with `R² > 0.7` and small α (≈ 0.01*max_hold per 0.5σ). High `turnover` (many small `action_change_step`), small per-step `|realized_pnl_step|`, `commission_fee_step / |realized_pnl_step|` ratio is high (fee-sensitive).
- DERIVED support: `turnover`, `trade_count_step`, `commission_fee_step`, `cumulative_commission_fee` all exist in the CSV. `flat_step_ratio` low, `mean_abs_position` small.
- Assignable primarily to `label_3` (sideways).

**M3. 背离增强型 (Divergence-enhanced)** — *uses volume/RSI divergence to filter false breakouts; reduces choppy-market wear, improves Calmar.*
- PROPOSED rule: requires volume/RSI features in `state_features.npy` (e.g. `rsv_*`, `bollinger_*` — see caveat in Section 6.2). Behavioral signature: fewer `action_change_step` during periods where `mark_price` makes a new high but `volume` does not (negative price-volume divergence). Improved Calmar = high `annual_sr/daily_cr` ratio at the high-level stage.
- DERIVED support: `volume` is in the CSV ([DETAIL_MARKET_COLUMNS L379](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/test_agent_index.py#L379)); divergence = `sign(Δmark_price) != sign(Δvolume)` at new-extreme bars. **No RSI column confirmed in the CSV** — must come from `state_features.npy` features fed to the net, not from the trajectory CSV.
- Assignable as a *modifier* on top of T1/T2 (filters false breakouts).

### 7.3 Profitability measurement summary

- **DERIVED**: use `trans_reward_mean` (primary), `trans_reward_std` (stability), `mean_turnover` (cost exposure), `net_position_exposure` (directional conviction), directional reward sums, and limit-reverse ratios — all from [`result_all.csv`](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/analysis_result/DiHFT/low_level/fu/30min_multi/result_all.csv) and [`best_index_info_by_dynamics_with_different_position.csv`](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/analysis_result/DiHFT/low_level/fu/30min_multi/best_index_info_by_dynamics_with_different_position.csv).
- **DERIVED (high-level only)**: `tr, mdd, annual_sr, daily_cr, daily_SoR, required_money` from [`best_result.csv`](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/analysis_result/DiHFT/high_level_heurstic/fu/30min_multi/best_result.csv) — but only for the *selected* ensemble, not per candidate triple.
- **PROPOSED**: per-triple `win_rate`, `calmar = annual_sr / mdd`, per-triple Sharpe from the CSV's PnL columns.

---

## 8. Gaps & open questions

1. **Shape-feature extractor is missing.** No script computes step-function lags, exponential position-vs-cumPnL fits, Z-score-gated action windows, linear reverse slopes, or price-volume divergence. A new analysis module (e.g. `FineFT/analysis/classify_agent/action_curve_features.py`) reading `trading_action_detail_epoch_*.csv` is required to make the 6-type classifier operational.
2. **`bin_index` semantics — confirmed.** The user's `bin_{n}` = the existing `bin_index` (ensemble Q-net head ∈ {0..6}, [test_agent_index.py L617-L618](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/low_level/test_agent_index.py#L617-L618)), **not** a price/volatility bucket. Decision (2026-08-06): no new binning dimension is needed; the classifier keys on the existing ensemble-head index.
3. **Per-triple risk metrics absent.** Sharpe/Calmar/MDD/win-rate exist only at the portfolio/contract level ([conclude_metric.py L160-L173](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/analysis/calculate_metric/conclude_metric.py#L160-L173)). The classifier's profitability record for a *candidate* triple can only be reward-based unless the high-level harness is run per triple.
4. **5 vs 7 labels resolved.** 7 is correct; "5" is `dynamic_number` ([slice_model.py L100](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/datahandler/slice_model.py#L100)). `label_0` and `label_6` are limit-state labels appended by `save_limit_labels=True`.
5. **Prior 12-archetype taxonomy — deferred.** [`docs/research/label_agent_selection_logic.md`](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/docs/research/label_agent_selection_logic.md) and ADR [`docs/adr/0005-label-agent-selection-meta-routing.md`](file:///home/lanceliang/opt/aiwork/FineFT_code_space2026/docs/adr/0005-label-agent-selection-meta-routing.md) define 12 finer strategy archetypes (Trend_Following, Momentum_Acceleration, Mean_Reverting, Fade_Breakout, Order_Flow_Imbalance, Scalping_Grid, Open_Interest_Drive, Volume_Price_Divergence, Calendar_Spread_Arbitrage, Volatility_Breakout, Session_Time_Pattern, Risk_Averse_Neutral). Per the 2026-08-06 scope decision, this round implements only the coarser 6-class second-order grouping; an explicit 6↔12 mapping (e.g. T1≈Trend_Following+Momentum_Acceleration, T3≈Trend_Following, M2≈Scalping_Grid, M3≈Volume_Price_Divergence+Fade_Breakout) is parked for a later pass.
6. **`labeling_method` — resolved to `slope`.** Both pick-agent launchers now use `slope` (see §4). The stale `ordered_default` string in `label_semantics.json` will refresh on the next re-slice; no further code change required.
7. **No persisted multi-agent pool per label.** `selection_manifest.json` stores exactly one `(epoch_path, bin_index)` per label. The 12-archetype "profile pool" of multiple agents per label is a design proposal not yet backed by a data file; the 6-type classifier would initially operate on the candidate space in `result_all.csv` (every epoch×bin×initial_action), which is the only place a "pool" exists today.
8. **Feature-name verification needed.** The divergence-enhanced rule (M3) and several 12-archetype descriptions reference specific features (`rsv_192_std_norm_origin`, `bollinger_lower_96_origin`, `price_oi_vol_interaction_10m`, `limit_depth_imbalance_ratio_5`) whose presence must be confirmed against `dataset/30min/fu/state_features.npy` before the rule can be grounded.
