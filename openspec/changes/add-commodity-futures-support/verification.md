# 验证记录：add-commodity-futures-support

## 已运行命令

- `PYTHONPATH=data_preprocess python -m pytest data_preprocess/tests/test_commodity_config_schema.py data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_downscale.py data_preprocess/tests/test_commodity_feature_pipeline.py data_preprocess/tests/test_commodity_scripts_docs.py data_preprocess/tests/test_commodity_main_contract_cli.py -q`
  - 结果：23 passed
- `cd FineFT && PYTHONPATH=. python -m pytest tests/env/test_commodity_env.py -q`
  - 结果：3 passed
- `conda run -n finetf python -m py_compile data_preprocess/operator_futures/feature_selection/ic_correlation.py`
  - 结果：退出码 0
- `source data_preprocess/script_preprocess/future_upgraded/commodity/commodity_process.sh && run_commodity_smoke_fu "$(pwd)" "5min" && test -f PREPROCESS_DATASET/commodity-futures/fu/5min/sample/orderbook_5.feather`
  - 结果：退出码 0
- `PYTHONPATH=data_preprocess python - <<'PY' ... process_snapshot_features(..., depth=25) ... PY`
  - 结果：输出 `depth-25 regression smoke passed`
- `openspec validate add-commodity-futures-support --strict`
  - 结果：`Change 'add-commodity-futures-support' is valid`

## 已知跳过项

- `cd FineFT && PYTHONPATH=. python -m pytest env/test_env.py -q`
  - 原因：该既有测试在 import 阶段读取 `/data2/mlqin/HFT4Ind2/dataset/BNBUSDT/train/df_0.feather`，当前机器没有该本地数据文件，无法作为自动化回归项运行。
- `conda run -n finetf bash -lc 'PYTHONPATH=data_preprocess python -m pytest ...'`
  - 原因：`finetf` conda 环境当前未安装 `pytest`；已使用该环境执行变更 Python 文件编译检查，并用仓库默认 Python 运行 pytest。
