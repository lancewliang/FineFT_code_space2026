import numpy as np

from common.trading_observation import TRADING_INFO_DIM
from env.env_class.base_env import Base_Env


def _make_env():
    n = 25
    mark = np.arange(100.0, 125.0)
    book = np.repeat(mark[:, None], 25, axis=1)
    qty = np.full((n, 25), 100.0)
    timestamps = np.datetime64("2026-01-01T00:00") + np.arange(n).astype("timedelta64[m]")
    scale = np.zeros(n, dtype=np.float64)
    scale[18:] = 1.0
    valid = np.zeros(n, dtype=bool)
    valid[18:] = True
    return Base_Env(
        state_array=np.zeros((n, 2), dtype=np.float32),
        ask_prices_array=book + 1,
        bid_prices_array=book - 1,
        ask_qtys_array=qty,
        bid_qtys_array=qty,
        markprice_array=mark,
        timestamp_array=timestamps,
        funding_rate_array=np.zeros(n),
        funding_timestamp_array=timestamps + np.timedelta64(1, "h"),
        max_holding_number=4,
        position_choices=3,
        leverage_choice=[1],
        initial_state=(100_000.0, 0.0, 0.0, 0.0, 1.0),
        exit_price_scale_floor=5.0,
        causal_price_scale_array=scale,
        causal_price_scale_valid_array=valid,
    )


def test_environment_starts_at_first_valid_row_and_keeps_real_exit_memory():
    env = _make_env()
    _, info = env.reset()
    assert env.day == 18
    assert info["trading_info"].shape == (TRADING_INFO_DIM,)
    long_action = env.env_map_position_leverage_to_action(4.0, 1.0)
    flat_action = env.env_map_position_leverage_to_action(0.0, 1.0)
    env.step(long_action)
    _, _, _, info = env.step(flat_action)
    observed = info["trading_info"]
    assert observed[0] == 0.0
    assert observed[4] == 0.0
    assert observed[5] == 1.0
    assert observed[6] == 0.0
    assert observed[7] > 0.0
    assert env.last_long_exit_record.observation_step == env.day
    assert env.last_long_exit_record.price_scale == 5.0


def test_reset_clears_directional_memory_and_invalid_scale_rows_are_rejected():
    env = _make_env()
    env.reset()
    assert np.all(env._calculate_trading_info() == 0.0)
    with np.testing.assert_raises(ValueError):
        Base_Env(
            np.zeros((3, 1)),
            np.ones((3, 1)), np.ones((3, 1)), np.ones((3, 1)), np.ones((3, 1)),
            np.ones(3), np.arange(3), np.zeros(3), np.arange(3),
            causal_price_scale_array=np.ones(3),
            causal_price_scale_valid_array=np.array([True, True, True]),
        )
