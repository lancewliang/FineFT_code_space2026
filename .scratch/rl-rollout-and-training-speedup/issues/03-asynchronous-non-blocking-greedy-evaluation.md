# 03: Asynchronous Non-blocking Greedy Evaluation

**What to build:** Decouple periodic greedy evaluation probes from the synchronous epoch training critical path in `FineFT/RL/DiHFT/low_level/parallel_diverse_train.py`. At epoch gradient update completion, capture a model parameter state snapshot (<2ms) and dispatch the 15 evaluation episodes to a dedicated background evaluator thread/worker. Eliminate 165 seconds (~2.75 minutes) of serial blocking delay per probe epoch while preserving complete metric logging to console and TensorBoard.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] `AsyncGreedyEvaluator` manages background evaluation lifecycle, accepting an epoch index, model weights snapshot, and market data handle via a non-blocking queue.
- [ ] Main training loop in `run_parallel_diverse_training` dispatches evaluation requests and immediately proceeds to the next rollout epoch without sleeping or waiting.
- [ ] Evaluation worker computes greedy probes across evaluation datasets and context indices, writing metrics (`mean_return_rate`, `profit_ratio`, `mean_trades`) to loggers and `SummaryWriter`.
- [ ] Thread-safe early stopping flag mechanism allows background evaluator to signal main training loop if early stopping criteria are met.
- [ ] Unit tests verify asynchronous evaluation completes cleanly and records expected metrics while the main execution thread advances.
