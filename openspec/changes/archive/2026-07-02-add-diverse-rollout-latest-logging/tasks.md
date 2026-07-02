## 1. Latest metrics helpers

- [x] 1.0 Complete latest metrics helper implementation. <!-- 已实现: 新增 latest metrics helper 与 focused tests -->
- [x] 1.1 Add focused tests for recording the latest metric, overwriting repeated `df_index + rollout_index`, sorting log output, profit/loss labels, and empty-cache behavior. <!-- 已实现: 覆盖覆盖语义、排序、盈亏标签和空缓存 -->
- [x] 1.2 Add minimal internal helpers in `weight_advantage_pretrain.py` to record latest diverse rollout metrics and log them at epoch boundaries. <!-- 已实现: record_diverse_rollout_latest_metric 和 log_diverse_rollout_latest_metrics -->

## 2. Training loop integration

- [x] 2.0 Complete training loop integration. <!-- 已实现: train 级缓存、diverse rollout 更新和 epoch 边界明细日志接入 -->
- [x] 2.1 Initialize the train-level latest metrics cache before the sample loop in `Weighted_Contexts_DQN.train()`. <!-- 已实现: sample loop 前初始化 diverse_rollout_latest_metrics_by_df -->
- [x] 2.2 Update the latest metrics cache after each diverse rollout completes, without changing pretrain or full-df warmup behavior. <!-- 已实现: 仅多样化分支调用 record helper -->
- [x] 2.3 Print latest diverse rollout metrics when `len(epoch_reward_sum_train_list) == epoch_number`, before the existing epoch summary log. <!-- 已实现: epoch summary 前调用 log helper -->

## 3. Verification

- [x] 3.0 Complete verification. <!-- 已实现: focused pytest、py_compile、OpenSpec strict 和 diff 检查完成 -->
- [x] 3.1 Run focused pytest for `FineFT/tests/rl/test_weight_advantage_pretrain_logging.py`. <!-- 已执行: 13 passed -->
- [x] 3.2 Run `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`. <!-- 已执行: py_compile 退出 0 -->
- [x] 3.3 Run `openspec validate add-diverse-rollout-latest-logging --strict`. <!-- 已执行: strict valid -->
