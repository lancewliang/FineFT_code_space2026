# fineft-low-level-agent-selection Delta

## ADDED Requirements

### Requirement: Low-level picker SHALL bind agents to explicit label semantics

系统 SHALL 在低层 agent 最终选择时读取机器可审计的 label 语义，并保证 `label_i` 绑定的 agent 行为与该 label 的市场方向语义和涨跌停语义一致。

#### Scenario: Picker loads complete label semantic manifest

- **GIVEN** picker 运行在商品期货数据集 `30min/fu/fu2305` 的 validation 选择阶段
- **AND** `dynamic_number = 5`
- **AND** `--num_label 7`
- **AND** `--label_semantics_path` 指向一个 `label_semantics.json`
- **WHEN** picker 开始选择低层 agent
- **THEN** manifest SHALL include `dataset_name`, `labeling_method`, `label_number`, and `labels`
- **AND** `labels` SHALL exactly cover `label_0` through `label_6`
- **AND** each label entry SHALL include `label`, `direction`, `direction_sign`, `strength`, `description`, `limit_state`, and `limit_state_sign`
- **AND** `direction_sign` SHALL be one of `-1`, `0`, or `1`
- **AND** `limit_state` SHALL be one of `none`, `near_limit_up`, `limit_up`, `near_limit_down`, or `limit_down`
- **AND** `limit_state_sign` SHALL be one of `-1`, `0`, or `1`

#### Scenario: Picker enforces current edge-label price-limit convention

- **GIVEN** picker runs with `dynamic_number = 5`
- **AND** picker runs with `--num_label 7`
- **AND** the current commodity label convention reserves edge labels for price-limit states
- **WHEN** picker validates `label_semantics.json`
- **THEN** `label_0` SHALL have `limit_state = "limit_down"` or `limit_state = "near_limit_down"`
- **AND** `label_0` SHALL have `limit_state_sign = -1`
- **AND** `label_6` SHALL have `limit_state = "limit_up"` or `limit_state = "near_limit_up"`
- **AND** `label_6` SHALL have `limit_state_sign = 1`
- **AND** middle labels `label_1` through `label_5` SHALL have `limit_state = "none"` unless a later spec explicitly introduces additional price-limit sublabels

#### Scenario: Picker rejects swapped or missing edge-label price-limit semantics

- **GIVEN** picker runs with `dynamic_number = 5`
- **AND** picker runs with `--num_label 7`
- **AND** `label_0` is not marked as a down-limit label or `label_6` is not marked as an up-limit label
- **WHEN** picker validates label semantics
- **THEN** picker SHALL fail fast
- **AND** the error message SHALL state that current commodity labels require `label_0` for down-limit and the last label for up-limit
- **AND** no `model.pth` SHALL be written

#### Scenario: Picker rejects DTW labels without explicit semantics

- **GIVEN** validation labels were produced by `labeling_method = "DTW"`
- **WHEN** picker starts without an explicit `label_semantics.json`
- **THEN** picker SHALL fail fast
- **AND** the error message SHALL state that DTW cluster ids have no stable bullish/bearish ordering
- **AND** no `model.pth` SHALL be written

#### Scenario: Picker can audit generated slope or quantile semantics

- **GIVEN** validation labels were produced by `labeling_method = "slope"` or `labeling_method = "quantile"`
- **AND** the user enables default semantic generation
- **WHEN** picker generates label semantics from ordered label ids
- **THEN** `label_0` SHALL map to down-limit semantics
- **AND** the last label SHALL map to up-limit semantics
- **AND** only middle label ids SHALL map to ordinary bearish, neutral, or bullish directions
- **AND** the generated semantics SHALL be written to a `label_semantics.json` artifact before selection

### Requirement: Low-level picker SHALL filter candidates by semantic alignment before financial ranking

系统 SHALL 在按收益指标排序前，先用 label 语义过滤候选 agent；收益指标只在语义匹配候选集合内比较。

#### Scenario: Bullish label selects long-profitable candidate

- **GIVEN** `label_5` has `direction_sign = 1` and `description = "大涨"`
- **AND** candidate A has higher `trans_reward_mean` but `candidate_mean_exposure < 0`
- **AND** candidate B has lower `trans_reward_mean`, `candidate_mean_exposure >= min_directional_exposure`, `candidate_long_ratio >= min_directional_step_ratio`, and `candidate_long_reward_mean > 0`
- **WHEN** picker selects the final agent for `label_5`
- **THEN** candidate A SHALL be rejected for bullish semantic mismatch
- **AND** candidate B SHALL remain eligible
- **AND** picker SHALL choose among eligible bullish candidates using the existing financial ranking logic

#### Scenario: Bearish label selects short-profitable candidate

- **GIVEN** `label_0` has `direction_sign = -1` and `description = "大跌"`
- **AND** candidate A has higher `trans_reward_mean` but `candidate_mean_exposure > 0`
- **AND** candidate B has lower `trans_reward_mean`, `candidate_mean_exposure <= -min_directional_exposure`, `candidate_short_ratio >= min_directional_step_ratio`, and `candidate_short_reward_mean > 0`
- **WHEN** picker selects the final agent for `label_0`
- **THEN** candidate A SHALL be rejected for bearish semantic mismatch
- **AND** candidate B SHALL remain eligible
- **AND** picker SHALL choose among eligible bearish candidates using the existing financial ranking logic

#### Scenario: Sideways label rejects strong directional exposure

- **GIVEN** `label_2` has `direction_sign = 0` and `description = "震荡"`
- **AND** candidate A has `abs(candidate_mean_exposure) > max_neutral_abs_exposure`
- **AND** candidate B has `abs(candidate_mean_exposure) <= max_neutral_abs_exposure`
- **WHEN** picker selects the final agent for `label_2`
- **THEN** candidate A SHALL be rejected for neutral semantic mismatch
- **AND** candidate B SHALL remain eligible
- **AND** picker SHALL choose among eligible neutral candidates using the existing financial ranking logic

#### Scenario: Limit-up label requires limit-state long profitability

- **GIVEN** `label_6` has `limit_state = "limit_up"` and `description` includes `涨停`
- **AND** candidate A has higher `trans_reward_mean` but `candidate_limit_up_reverse_short_ratio > max_limit_reverse_ratio`
- **AND** candidate B has lower `trans_reward_mean`, `candidate_limit_up_long_reward_mean > 0`, and `candidate_limit_up_reverse_short_ratio <= max_limit_reverse_ratio`
- **WHEN** picker selects the final agent for `label_6`
- **THEN** candidate A SHALL be rejected for limit-up semantic mismatch
- **AND** candidate B SHALL remain eligible
- **AND** picker SHALL choose among eligible limit-up candidates using the existing financial ranking logic

#### Scenario: Limit-down label requires limit-state short profitability

- **GIVEN** `label_0` has `limit_state = "limit_down"` and `description` includes `跌停`
- **AND** candidate A has higher `trans_reward_mean` but `candidate_limit_down_reverse_long_ratio > max_limit_reverse_ratio`
- **AND** candidate B has lower `trans_reward_mean`, `candidate_limit_down_short_reward_mean > 0`, and `candidate_limit_down_reverse_long_ratio <= max_limit_reverse_ratio`
- **WHEN** picker selects the final agent for `label_0`
- **THEN** candidate A SHALL be rejected for limit-down semantic mismatch
- **AND** candidate B SHALL remain eligible
- **AND** picker SHALL choose among eligible limit-down candidates using the existing financial ranking logic

#### Scenario: Picker fails when no candidate matches label semantics

- **GIVEN** a label has a valid semantic entry
- **AND** every candidate for that label fails the semantic filter
- **WHEN** picker attempts final selection
- **THEN** picker SHALL fail fast
- **AND** the error SHALL include the label, semantic direction, limit state, configured thresholds, and a summary of rejected candidate behavior metrics
- **AND** no `model.pth` SHALL be written

### Requirement: Selection manifest SHALL record semantic selection evidence

系统 SHALL 在 `selection_manifest.json` 中记录每个 label 的语义、过滤阈值、候选行为摘要和最终选择原因。

#### Scenario: Manifest includes selected label semantic context

- **WHEN** picker writes `selection_manifest.json`
- **THEN** each label entry SHALL include `label`, `description`, `direction`, `direction_sign`, `limit_state`, `limit_state_sign`, `epoch_path`, `model_path`, `bin_index`, `score`, and `source_rows`
- **AND** each label entry SHALL include `semantic_filter`
- **AND** each label entry SHALL include `behavior_summary`
- **AND** each label entry SHALL include `selection_reason`
- **AND** the manifest SHALL still be written under `analysis_result/DiHFT/low_level/<dataset>/<experiment>/`

#### Scenario: Potential model keeps semantic label order

- **WHEN** picker assembles `result/DiHFT/potential_model/<dataset>/<experiment>/model.pth`
- **THEN** the qnet at index `i` SHALL still correspond to `label_i`
- **AND** the same `label_i` entry in `selection_manifest.json` SHALL describe that qnet's semantic direction, limit state, and behavior evidence
- **AND** label order SHALL NOT be changed by semantic filtering
