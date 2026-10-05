# 04: Native Continuous Tensor Stratified Replay Buffer

**What to build:** Replace the Python-list-of-tuples representation in `FineFT/RL/util/regime_stratified_replay_buffer.py` with pre-allocated contiguous PyTorch tensors for each regime grid. Eliminate the costly dynamic re-stacking loop (`extract_stacked_tensor_dicts` and `np.stack`), shrinking replay buffer memory from 2.5GB+ fragmented heap objects to ~1.01 GB contiguous numerical tensors while supporting O(1) circular FIFO eviction and balanced regime sampling.

**Blocked by:** None (can start immediately)

**Status:** completed

- [x] `RegimeStratifiedReplayBuffer` pre-allocates contiguous PyTorch tensors for `states`, `actions`, `rewards`, `next_states`, `dones`, `trading_info`, `time_info`, and `q_values` across all regime grids.
- [x] Transition insertion and regime routing write directly into tensor slots with in-place circular write pointers, achieving O(1) insertion without dynamic Python list appending.
- [x] Dynamic epoch re-stacking (`extract_stacked_tensor_dict`) is eliminated, saving 15-20 seconds of CPU chunking and `np.stack` allocations every epoch.
- [x] Active regime sampler slices directly from pre-allocated tensors with balanced grid quotas and replacement handling.
- [x] Unit tests verify circular FIFO eviction, regime quota balancing, and sampled batch tensor dimensions matching model update signatures.
