# 02: In-place Trading Environment State Reset Protocol

**What to build:** Extend `Base_Env.reset` and `Demo_Env.reset` to accept an optional `initial_state: tuple[float, float, float, float, float] | None = None` parameter. Enable in-place reset of environment episode cursors, wallet balances, positions, margins, and metric histories without reallocating or mutating the underlying shared market data arrays. Provide a direct environment factory `create_demo_env_from_pack` to construct `Demo_Env` directly from `EnvTensorPack` without touching Pandas DataFrames.

**Blocked by:** 01 (needs `EnvTensorPack`)

**Status:** completed

- [x] Extend `Base_Env.reset(self, initial_state: tuple[float, float, float, float, float] | None = None)` in `FineFT/env/env_class/base_env.py` to update `self.initial_state = initial_state` if provided, and reset day cursor `self.day = 0`, terminal state, wallet balances, margins, and all tracking histories in-place.
- [x] Ensure underlying arrays (`state_array`, `ask_prices_array`, `bid_prices_array`, etc.) remain completely untouched during in-place resets.
- [x] Extend `Demo_Env.reset(self, initial_state: tuple[float, float, float, float, float] | None = None)` in `FineFT/env/env_class/demo_env.py` to forward `initial_state` to `super().reset(...)` and re-evaluate `info["q_value"]` for the reset step.
- [x] Implement `create_demo_env_from_pack(pack: EnvTensorPack, env_kwargs: dict[str, Any], initial_state: tuple[float, float, float, float, float] | None = None) -> Demo_Env` in `FineFT/env/env_initiate/demo_initiate.py` (or `shared_data_manager.py`) avoiding any Pandas DataFrame conversions.
- [x] Verify numerical exactness: calling `env.reset(initial_state=s)` on a reused environment returns identical `(state, info)` to a newly created environment with `initial_state=s`.
