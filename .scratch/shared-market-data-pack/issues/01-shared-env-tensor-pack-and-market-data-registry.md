# 01: Shared Env Tensor Pack and Market Data Registry

**What to build:** Implement `EnvTensorPack` and `SharedMarketDataPack` in `FineFT/RL/DiHFT/low_level/shared_data_manager.py`. Pre-extract all raw Pandas DataFrame columns from `train_df_cache` into contiguous PyTorch CPU tensors in POSIX shared memory (`tensor.share_memory_()`). Provide zero-copy NumPy views (`tensor.numpy()`) for trading environments, eliminating 90-worker duplicate DataFrame memory overhead and Pandas string column index overhead.

**Blocked by:** None (can start immediately)

**Status:** completed

- [x] Define `EnvTensorPack` dataclass holding contiguous PyTorch CPU tensors: `state_tensor`, `ask_prices_tensor`, `bid_prices_tensor`, `ask_qtys_tensor`, `bid_qtys_tensor`, `markprice_tensor`, `timestamp_tensor`, `funding_rate_tensor`, `funding_timestamp_tensor`, plus optional limit columns (`is_limit_up_tensor`, `is_limit_down_tensor`, `limit_up_ask_depth_ratio_5_tensor`, `limit_down_bid_depth_ratio_5_tensor`, `upper_limit_prices_tensor`, `lower_limit_prices_tensor`, `regime_grid_ids_tensor`).
- [x] Implement `to_env_kwargs()` on `EnvTensorPack` returning zero-copy `np.ndarray` views (`tensor.numpy()`) compatible with `Demo_Env` initialization.
- [x] Define `SharedMarketDataPack` class containing `packs: dict[int, EnvTensorPack]`.
- [x] Implement `SharedMarketDataPack.from_dataframes(train_df_cache: dict[int, pd.DataFrame], env_kwargs: dict[str, Any]) -> SharedMarketDataPack` factory method that pre-extracts numeric arrays once, converts to contiguous PyTorch CPU tensors, and calls `tensor.share_memory_()`.
- [x] Implement fail-fast validation in `from_dataframes`: if `enable_limit_reward=True` and any required limit columns are missing from any DataFrame, raise `ValueError` immediately.
- [x] Implement dictionary-like indexing `__getitem__(self, df_index: int) -> EnvTensorPack` and `__len__`, `keys()`, `values()`.
