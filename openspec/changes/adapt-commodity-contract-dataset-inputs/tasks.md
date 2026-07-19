## 1. Implementation

- [ ] 1.1 Update `FineFT/tests/datahandler/test_commodity_contract_dataset.py` for the new manifest-driven input contract, stage file naming, state feature path, train slices, and failure cases.
- [ ] 1.2 Refactor `FineFT/datahandler/commodity_contract_dataset.py` to read `dataset_split_manifest.json`, build a FineFT manifest from stage/contract metadata, copy staged SCALE_SAVE files, copy `--state_features_path`, and remove internal split-boundary filtering from the main path.
- [ ] 1.3 Keep train slice generation working from `train/{contract}.feather`, with continuous slice indices and manifest row counts.
- [ ] 1.4 Update `FineFT/script/data/commodity_data_handler_fu.sh` and `FineFT/script/data/commodity_data_handler_al.sh` to pass `--dataset_split_manifest_path` and `--state_features_path`, and to scan `valid/*.feather` for `slice_model.py`.
- [ ] 1.5 Update any FineFT commodity dataset tests that assert old `df_<contract>.feather`, `--summary_path`, `--feature_union_path`, or valid `df_*.feather` contracts.

## 2. Validation

- [ ] 2.1 Run `conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py -q`.
- [ ] 2.2 Run `openspec validate adapt-commodity-contract-dataset-inputs --strict`.
