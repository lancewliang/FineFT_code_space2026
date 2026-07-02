## 1. Full-df warmup implementation

- [x] 1.0 Complete full-df warmup implementation. <!-- 已实现: 默认启用 full-df warmup、cache 扩展、空仓 action 解析、训练前 warmup loop 和日志摘要 -->
- [x] 1.1 Add CLI defaults and switches for full-df warmup, and change `pretrain_epoch` default to `0`. <!-- 已实现: --full_df_warmup/--no_full_df_warmup 与 pretrain_epoch=0 -->
- [x] 1.2 Add qtable/df cache extension support so full-df warmup can cover every `df_index` while reusing diagnostics caches. <!-- 已实现: extend_q_table_cache -->
- [x] 1.3 Add empty-position action resolution without hard-coding the action number. <!-- 已实现: _resolve_empty_initial_action -->
- [x] 1.4 Add the full-df warmup training loop before the sample loop, reusing existing pretrain rollout and `update_pretrain()` behavior. <!-- 已实现: _run_full_df_warmup 接入 train loop 前 -->
- [x] 1.5 Add per-df warmup logs, warning behavior for unprofitable paths, and a full-df warmup summary. <!-- 已实现: per-df info/warning 与 complete summary -->

## 2. Verification

- [x] 2.0 Complete verification. <!-- 已实现: focused tests、py_compile、OpenSpec strict validate 与实现证据检查均通过 -->
- [x] 2.1 Add focused tests for defaults, disable switch, empty-position action resolution, cache reuse, and one-warmup-per-df behavior. <!-- 已实现: 新增 RL focused tests -->
- [x] 2.2 Run `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py`. <!-- 已执行: conda finetf 环境 py_compile 通过 -->
- [x] 2.3 Run `openspec validate add-full-df-warmup-pretrain --strict`. <!-- 已执行: strict valid -->
