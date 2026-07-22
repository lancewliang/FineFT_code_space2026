## 1. Implementation

- [x] 1.1 Add focused tests for commodity main contract object return types, object attribute access, and JSON payload compatibility. <!-- 已实现: tests now assert object returns, object attributes, and JSON/to_dict compatibility -->
- [x] 1.2 Add internal dataclass models in main_contract.py for source discovery and build state so summary assembly no longer depends on nested dict handoff. <!-- 已实现: added source-file and build-state dataclasses in main_contract.py -->
- [x] 1.3 Refactor build_main_contract_summary_for_date_range() and write_main_contract_summary_for_date_range() to return MainContractSummary and serialize only at the JSON boundary. <!-- 已实现: builder returns MainContractSummary and writer dumps summary.to_dict() -->
- [x] 1.4 Update the existing commodity main-contract tests to assert object access on source discovery, summary assembly, and written JSON equality. <!-- 已实现: commodity main-contract tests use object attribute assertions -->

## 2. Verification

- [x] 2.1 Run `conda activate finetf && pytest data_preprocess/tests/test_commodity_main_contract.py`. <!-- 已实现: `conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py -q` passed with 20 tests -->
- [x] 2.2 Run `conda activate finetf && python -m py_compile data_preprocess/operator_futures/commodity/main_contract.py`. <!-- 已实现: `conda run -n finetf python -m py_compile data_preprocess/operator_futures/commodity/main_contract.py` passed -->
- [x] 2.3 Run `openspec validate refactor-commodity-main-contract-objects --strict`. <!-- 已实现: OpenSpec strict validation passed -->
