# Close 验证记录：add-commodity-futures-date-range-support

## 结论

- CRITICAL：无
- WARNING：无
- SUGGESTION：无

## 验证证据

- `rg -n "^- \[ \]" openspec/changes/add-commodity-futures-date-range-support/tasks.md openspec/changes/add-commodity-futures-date-range-support/plan-ready.md docs/superpowers/plans/2026-06-21-add-commodity-futures-date-range-support.md`：无未完成 checkbox。
- `PYTHONPATH=data_preprocess conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py -q`：12 passed。
- `PYTHONPATH=data_preprocess:FineFT:. conda run -n finetf pytest data_preprocess/tests/test_commodity_config_schema.py data_preprocess/tests/test_commodity_downscale.py data_preprocess/tests/test_commodity_feature_pipeline.py data_preprocess/tests/test_commodity_scripts_docs.py FineFT/tests/env/test_commodity_env.py -q`：17 passed。
- `openspec validate add-commodity-futures-date-range-support --strict`：valid。
- `git diff --check`：通过。

## 规格对照

- Completeness：`tasks.md`、`plan-ready.md`、superpowers plan 均已勾选完成。
- Correctness：实现覆盖日期范围年份推导、跨年 `previous_frames` 连续、`--start_date` / `--end_date` CLI、主流程日期范围命名和左闭右开语义。
- Coherence：实现遵循 `design.md` 决策，未修改商品期货 funding、手续费、盘口深度或特征契约。

## 备注

- 当前仓库另有活跃变更 `migrate-operator-futures-to-polars`，本次 close 只处理 `add-commodity-futures-date-range-support`。
- 当前工作区存在用户运行生成目录 `PREPROCESS_DATASET/`、`log_futures/`，以及用户本地 `main.sh` 默认日期调整；这些不属于 OpenSpec 归档阻塞项。
