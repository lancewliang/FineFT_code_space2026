# 02: Worker Pool Right-Sizing and Shared Q-Table

**What to build:** Govern exploration worker concurrency to eliminate OS disk swap thrashing, and share precomputed dynamic programming Q-tables across worker processes. Update default `--diverse_num_workers` in `FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py` and `FineFT/script/train/train_commodity_fu_10.sh` to 40. Augment `SharedMarketDataPack` in `FineFT/RL/DiHFT/low_level/shared_data_manager.py` to hold precomputed Q-table tensors in POSIX shared memory, injecting them directly into `Demo_Env` on creation to eliminate repeated 51,813-iteration dynamic programming loops.

**Blocked by:** None (can start immediately)

**Status:** completed

- [x] Default `--diverse_num_workers` in `parallel_weight_advantage_pretrain.py` and `train_commodity_fu_10.sh` is adjusted to 40, capping total worker RSS under 25GB and ensuring 0 MB active swap on a 62GB host.
- [x] `EnvTensorPack` and `SharedMarketDataPack` store precomputed `q_table_tensor` in POSIX shared memory (`share_memory_()`).
- [x] `create_demo_env_from_pack` passes the precomputed Q-table array into `Demo_Env.__init__`, bypassing the Python `create_optimal_q_table` loop when an existing table is supplied.
- [x] Unit tests verify that `Demo_Env` initialized with a shared Q-table references the correct day-action slice identically to one computed in-place, with zero recomputation overhead.
