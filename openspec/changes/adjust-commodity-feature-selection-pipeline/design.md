# Design: adjust-commodity-feature-selection-pipeline

## Context

The commodity futures full process currently performs per-contract `scale_save` before the global `dataset_split` step. That keeps `scale_save` tied to the older single-contract `IC_RESULT` path and leaves no independent stage for evaluating feature quality on the split train and valid datasets.

The target flow moves feature evaluation after dataset splitting:

1. Build contract-level `ALL_FEATURE` files as today.
2. Run `dataset_split` once across all summary contracts.
3. Evaluate and select features on split `train` files, producing candidate features.
4. Re-evaluate and select features on split `valid` files using only the train candidates, producing final features.
5. Rebuild filtered contract-level `df.feather` files and run `scale_save` against those filtered files.

## Decisions

### New module boundary

Create a new multi-contract feature selection package under `data_preprocess/operator_futures/feature_selection/muti_contract/`. The package owns split-dataset input discovery, per-contract metrics, cross-contract aggregation, filtering, manifest writing, and filtered contract output generation.

The shell remains an orchestrator. It should call the module twice, once for `train` and once for `valid`, and should not contain feature metric or filter logic.

### Output roots

- `PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/${target_freq}` remains the split dataset root.
- `PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/${target_freq}` becomes the feature evaluation and selection root.
- `PREPROCESS_DATASET/commodity-futures/SCALE_SAVE` remains the final scaled output root.

### Train and valid semantics

The `train` run reads all state features from split train contract files and writes `state_features_candidate.npy`. The `valid` run reads split valid contract files, restricts evaluation to the train candidate list, and writes final `state_features.npy`.

Train and valid each write their own per-contract metric detail, aggregate statistics, selected feature list, filtered contract-level `df.feather` files, and manifest. The two stages do not share final selected features; valid only consumes train candidates as its allowed feature universe.

### Metric and filter semantics

For every contract and stage, compute these metrics per state feature: Permutation Importance, CatBoost Importance, IC, RankIC, and Sharpe. Aggregate metrics across contracts with Mean, Std, and Median.

Metrics use the same target construction as the original feature-selection scripts: `mark_price.shift(-window) - mark_price`, with the last `window` rows removed. The default window list is `[1, 6, 12]`; per-contract metric detail records the window, and aggregate metrics roll up all contract/window observations by feature.

IC follows `ic_correlation.py`: drop NaN pairs, return `np.nan` for insufficient samples or zero standard deviation, otherwise return Pearson correlation. RankIC follows `rank_ic_correlation.py`: reject empty or constant original arrays with `0.0`, then correlate `np.argsort(np.argsort(...))` ranks and convert NaN/inf to `0.0`. CatBoost Importance follows `catbooost.py`: `CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, loss_function="MAE", task_type="GPU", random_seed=42)`, fit with `eval_set` and `verbose=100`, and no IC fallback when CatBoost is unavailable. Sharpe uses the single-feature pseudo-return convention described below. Permutation Importance uses the absolute IC loss after a deterministic one-step roll of the feature values, floored at `0.0`.

The selection pipeline applies Hard Filter, Stability Filter, Composite Score, and Correlation Filter in that order. Hard Filter keeps features where `abs(IC_Mean) >= min_abs_ic`; Stability Filter keeps features where `IC_Std <= max_metric_std`. Composite Score sorts by priority rather than plain summation: first `abs(RankIC_Mean)`, then `abs(Sharpe_Mean) + Permutation Importance_Mean` plus optional `SHAP Importance_Mean` if present, then `CatBoost Importance_Mean`. After this sort, the pipeline drops the bottom `composite_drop_ratio` fraction, defaulting to `0.1`, while preserving at least one feature. The Correlation Filter then removes highly correlated features using `max_correlation`.

The manifest records `windows_list`, `composite_drop_ratio`, each filter stage output, and `Composite Score Dropped`. If any required input is missing, the train candidate list is empty, or the final valid selected list is empty, the process fails before writing downstream-ready outputs.

### Scale-save compatibility

Keep the scale/save algorithm unchanged. Extend its input routing so commodity full process can point it at filtered `FEATURE_SELECTION` contract outputs while still writing final outputs under `SCALE_SAVE`.

## Risks

- CatBoost importance can be expensive or unavailable on GPU in local CI. The implementation should keep tests focused on deterministic helpers and use small fixtures, while preserving the production metric contract.
- Existing tests assert the old `merge_clean -> scale_save -> dataset_split` order. The spec expects those tests to be updated before implementation.
