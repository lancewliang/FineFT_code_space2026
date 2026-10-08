import numpy as np
import pandas as pd
import pytest

from env.env_initiate.base_initiate import initiate_base_env
from env.env_class.futures_util import compute_directional_turnover_penalty_rate


def _make_sample_df(prices):
    n = len(prices)
    timestamps = pd.date_range("2026-01-01", periods=n, freq="10min")
    data = {
        "timestamp": timestamps,
        "funding_timestamp": timestamps + pd.Timedelta(hours=8),
        "funding_rate": np.zeros(n),
        "mark_price": np.array(prices, dtype=np.float64),
        "feat1": np.zeros(n),
    }
    for level in range(1, 26):
        data[f"ask{level}_price"] = data["mark_price"]
        data[f"ask{level}_size"] = np.full(n, 10.0)
        data[f"bid{level}_price"] = data["mark_price"]
        data[f"bid{level}_size"] = np.full(n, 10.0)
    return pd.DataFrame(data)


def test_drawdown_reward_shaping_holding_bleed():
    # Price path:
    # 0: 100.0 (reset)
    # 1: 100.0 (open long pos=1 at 100)
    # 2: 110.0 (hold long: profit +10%, peak=0.10 >= 0.08, retrace=0)
    # 3: 107.0 (hold long: profit +7%, retrace=(0.10-0.07)/0.10 = 0.30 > 0.15)
    prices = [100.0, 100.0, 110.0, 107.0]
    df = _make_sample_df(prices)

    profit_min = 0.08
    allow_ratio = 0.15
    penalty_weight = 0.01

    env_plain = initiate_base_env(
        df,
        feature_list=["feat1"],
        max_holding_number=1,
        position_choices=3,
        leverage_choice=[1],
        initial_state=(1e5, 0.0, 0.0, 0.0, 1),
        turnover_penalty_rate=0.0, commission_rate=0.0,
        enable_drawdown_reward_shaping=False,
    )
    env_shaped = initiate_base_env(
        df,
        feature_list=["feat1"],
        max_holding_number=1,
        position_choices=3,
        leverage_choice=[1],
        initial_state=(1e5, 0.0, 0.0, 0.0, 1),
        turnover_penalty_rate=0.0, commission_rate=0.0,
        enable_drawdown_reward_shaping=True,
        drawdown_profit_min=profit_min,
        drawdown_allow_ratio=allow_ratio,
        drawdown_penalty_weight=penalty_weight,
    )

    env_plain.reset()
    env_shaped.reset()

    # Step 1: Open Long (pos 0 -> 1)
    # actions: 0: short (-1), 1: flat (0), 2: long (+1)
    env_plain.step(2)
    env_shaped.step(2)

    # Step 2: Hold Long (1 -> 1) at price 110.0. Gain = 10%. Retrace = 0.
    _, r_plain_2, _, _ = env_plain.step(2)
    _, r_shaped_2, _, _ = env_shaped.step(2)
    assert r_shaped_2 == pytest.approx(r_plain_2)

    # Step 3: Hold Long (1 -> 1) at price 107.0. Gain = 7%.
    # Retrace = (0.10 - 0.07) / 0.10 = 0.30.
    # Excess = 0.30 - 0.15 = 0.15.
    # Notional = 1.0 * 107.0 = 107.0.
    # Penalty = 0.01 * (0.15 ** 2) * 107.0 = 0.01 * 0.0225 * 107.0 = 0.024075.
    expected_penalty = penalty_weight * ((0.30 - allow_ratio) ** 2) * (1.0 * 107.0)
    _, r_plain_3, _, _ = env_plain.step(2)
    _, r_shaped_3, _, _ = env_shaped.step(2)

    assert r_shaped_3 == pytest.approx(r_plain_3 - expected_penalty)
    # Wallet balance must be completely untouched (pure reward shaping)
    assert env_shaped.wallet_balance == pytest.approx(env_plain.wallet_balance)


def test_drawdown_reward_shaping_flat_exit_ceases_penalty():
    # Same price path as above, but at Step 3 the agent exits to Flat (pos 1 -> 0)
    prices = [100.0, 100.0, 110.0, 107.0]
    df = _make_sample_df(prices)

    env_plain = initiate_base_env(
        df,
        feature_list=["feat1"],
        max_holding_number=1,
        position_choices=3,
        leverage_choice=[1],
        initial_state=(1e5, 0.0, 0.0, 0.0, 1),
        turnover_penalty_rate=0.0, commission_rate=0.0,
        enable_drawdown_reward_shaping=False,
    )
    env_shaped = initiate_base_env(
        df,
        feature_list=["feat1"],
        max_holding_number=1,
        position_choices=3,
        leverage_choice=[1],
        initial_state=(1e5, 0.0, 0.0, 0.0, 1),
        turnover_penalty_rate=0.0, commission_rate=0.0,
        enable_drawdown_reward_shaping=True,
        drawdown_profit_min=0.08,
        drawdown_allow_ratio=0.15,
        drawdown_penalty_weight=0.01,
    )

    env_plain.reset()
    env_shaped.reset()

    # Step 1: Open Long
    env_plain.step(2)
    env_shaped.step(2)

    # Step 2: Hold Long at 110.0
    env_plain.step(2)
    env_shaped.step(2)

    # Step 3: Exit to Flat (action 1: 1 -> 0)
    # Penalty must immediately cease (0.0) upon closing!
    _, r_plain_3, _, _ = env_plain.step(1)
    _, r_shaped_3, _, _ = env_shaped.step(1)

    assert r_shaped_3 == pytest.approx(r_plain_3)


def test_drawdown_reward_shaping_inactive_below_profit_threshold():
    # Price path: gain peaks at only 4% (104.0), below threshold 8%
    prices = [100.0, 100.0, 104.0, 101.0]
    df = _make_sample_df(prices)

    env_shaped = initiate_base_env(
        df,
        feature_list=["feat1"],
        max_holding_number=1,
        position_choices=3,
        leverage_choice=[1],
        initial_state=(1e5, 0.0, 0.0, 0.0, 1),
        turnover_penalty_rate=0.0, commission_rate=0.0,
        enable_drawdown_reward_shaping=True,
        drawdown_profit_min=0.08,
        drawdown_allow_ratio=0.15,
        drawdown_penalty_weight=0.01,
    )
    env_plain = initiate_base_env(
        df,
        feature_list=["feat1"],
        max_holding_number=1,
        position_choices=3,
        leverage_choice=[1],
        initial_state=(1e5, 0.0, 0.0, 0.0, 1),
        turnover_penalty_rate=0.0, commission_rate=0.0,
        enable_drawdown_reward_shaping=False,
    )

    env_plain.reset()
    env_shaped.reset()

    env_plain.step(2)
    env_shaped.step(2)

    env_plain.step(2)
    env_shaped.step(2)

    _, r_plain_3, _, _ = env_plain.step(2)
    _, r_shaped_3, _, _ = env_shaped.step(2)

    # Below threshold -> no penalty
    assert r_shaped_3 == pytest.approx(r_plain_3)


def test_trading_info_retracement_observation():
    prices = [100.0, 100.0, 104.0, 110.0, 107.0, 107.0]
    df = _make_sample_df(prices)

    env = initiate_base_env(
        df,
        feature_list=["feat1"],
        max_holding_number=1,
        position_choices=3,
        leverage_choice=[1],
        initial_state=(1e5, 0.0, 0.0, 0.0, 1),
        enable_drawdown_reward_shaping=True,
        drawdown_profit_min=0.08,
        commission_rate=0.0,
    )

    # Reset: pos=0 -> trading_info[2] must be 0.0
    _, info = env.reset()
    assert info["trading_info"][2] == 0.0

    # Step 1: Open long at 100.0
    _, _, _, info = env.step(2)
    assert info["trading_info"][2] == 0.0

    # Step 2: Hold long at 104.0 (gain 4% < 8% -> inactive -> 0.0)
    _, _, _, info = env.step(2)
    assert info["trading_info"][2] == 0.0

    # Step 3: Hold long at 110.0 (gain 10% >= 8%, peak=10%, retrace=0.0)
    _, _, _, info = env.step(2)
    assert info["trading_info"][2] == 0.0

    # Step 4: Hold long at 107.0 (gain 7%, retrace = (0.10 - 0.07)/0.10 = 0.30)
    _, _, _, info = env.step(2)
    assert info["trading_info"][2] == pytest.approx(0.30, abs=1e-4)

    # Step 5: Exit to flat
    _, _, _, info = env.step(1)
    assert info["trading_info"][2] == 0.0


def test_take_profit_turnover_penalty_exemption_unit():
    # Unit test for compute_directional_turnover_penalty_rate with is_take_profit_exit
    base_rate = 0.0002
    adverse_ratio = 6.0

    # In Uptrend (slope_bin == 2):
    # Long -> Flat (1.0 -> 0.0)
    # Without exemption: adverse rate (6x)
    rate_normal = compute_directional_turnover_penalty_rate(
        old_position=1.0,
        new_position=0.0,
        regime_grid_id=8,  # 8 % 3 == 2 (uptrend)
        turnover_base_rate=base_rate,
        turnover_adverse_ratio=adverse_ratio,
        is_take_profit_exit=False,
    )
    assert rate_normal == pytest.approx(base_rate * adverse_ratio)

    # With take-profit exemption: de-escalated to base rate (1x)
    rate_exempt = compute_directional_turnover_penalty_rate(
        old_position=1.0,
        new_position=0.0,
        regime_grid_id=8,
        turnover_base_rate=base_rate,
        turnover_adverse_ratio=adverse_ratio,
        is_take_profit_exit=True,
    )
    assert rate_exempt == pytest.approx(base_rate)


def test_base_env_take_profit_turnover_exemption_integration():
    # Prices:
    # 0: 100 (reset)
    # 1: 100 (open long pos=1)
    # 2: 110 (hold long: reaches 10% gain >= 8%)
    # 3: 107 (close to flat: pos 1 -> 0)
    prices = [100.0, 100.0, 110.0, 107.0]
    df = _make_sample_df(prices)
    # Regime grid 8 (uptrend)
    df["regime_grid_id"] = np.full(len(prices), 8, dtype=np.int64)

    base_rate = 0.0001
    adverse_ratio = 6.0

    # env_no_exemption: take profit exemption disabled -> pays 6x on exit
    env_no_exemption = initiate_base_env(
        df,
        feature_list=["feat1"],
        max_holding_number=1,
        position_choices=3,
        leverage_choice=[1],
        initial_state=(1e5, 0.0, 0.0, 0.0, 1),
        turnover_base_rate=base_rate,
        turnover_adverse_ratio=adverse_ratio,
        enable_drawdown_reward_shaping=True,
        enable_take_profit_turnover_exemption=False,
    )

    # env_exemption: take profit exemption enabled -> pays 1x on exit
    env_exemption = initiate_base_env(
        df,
        feature_list=["feat1"],
        max_holding_number=1,
        position_choices=3,
        leverage_choice=[1],
        initial_state=(1e5, 0.0, 0.0, 0.0, 1),
        turnover_base_rate=base_rate,
        turnover_adverse_ratio=adverse_ratio,
        enable_drawdown_reward_shaping=True,
        enable_take_profit_turnover_exemption=True,
    )

    env_no_exemption.reset()
    env_exemption.reset()

    # Step 1: Open Long
    env_no_exemption.step(2)
    env_exemption.step(2)

    # Step 2: Hold Long at 110 (profit reached 10%)
    env_no_exemption.step(2)
    env_exemption.step(2)

    # Step 3: Close to flat at 107
    # Nominal trade turnover = 1.0 * 110.0 = 110.0
    # No exemption penalty = 0.0006 * 1.0 * 110 = 0.066
    # Exemption penalty = 0.0001 * 1.0 * 110 = 0.011
    # Diff = 0.055
    _, r_no_ex, _, _ = env_no_exemption.step(1)
    _, r_ex, _, _ = env_exemption.step(1)

    assert r_ex == pytest.approx(r_no_ex + (base_rate * (adverse_ratio - 1.0) * 1.0 * 110.0))


def test_drawdown_reward_shaping_short_position():
    # Price path for Short:
    # 0: 100.0 (reset)
    # 1: 100.0 (open short pos=-1 at 100)
    # 2: 90.0 (hold short: price dropped 10 points -> profit = (100 - 90)/100 = +10% >= 8%, peak=0.10)
    # 3: 93.0 (hold short: price bounced to 93 -> profit = (100 - 93)/100 = +7%, retrace = (0.10 - 0.07)/0.10 = 0.30 > 0.15)
    prices = [100.0, 100.0, 90.0, 93.0]
    df = _make_sample_df(prices)

    profit_min = 0.08
    allow_ratio = 0.15
    penalty_weight = 0.01

    env_plain = initiate_base_env(
        df,
        feature_list=["feat1"],
        max_holding_number=1,
        position_choices=3,
        leverage_choice=[1],
        initial_state=(1e5, 0.0, 0.0, 0.0, 1),
        turnover_penalty_rate=0.0,
        commission_rate=0.0,
        enable_drawdown_reward_shaping=False,
    )
    env_shaped = initiate_base_env(
        df,
        feature_list=["feat1"],
        max_holding_number=1,
        position_choices=3,
        leverage_choice=[1],
        initial_state=(1e5, 0.0, 0.0, 0.0, 1),
        turnover_penalty_rate=0.0,
        commission_rate=0.0,
        enable_drawdown_reward_shaping=True,
        drawdown_profit_min=profit_min,
        drawdown_allow_ratio=allow_ratio,
        drawdown_penalty_weight=penalty_weight,
    )

    env_plain.reset()
    env_shaped.reset()

    # Step 1: Open Short (action 0: -1.0)
    env_plain.step(0)
    env_shaped.step(0)

    # Step 2: Hold Short at 90.0. Gain = +10%. Retrace = 0.
    _, r_plain_2, _, _ = env_plain.step(0)
    _, r_shaped_2, _, _ = env_shaped.step(0)
    assert r_shaped_2 == pytest.approx(r_plain_2)

    # Step 3: Hold Short at 93.0. Gain = +7%. Retrace = 0.30 > 0.15.
    # Notional = |-1.0| * 93.0 = 93.0.
    # Penalty = 0.01 * (0.15 ** 2) * 93.0 = 0.01 * 0.0225 * 93.0 = 0.020925.
    expected_penalty = penalty_weight * ((0.30 - allow_ratio) ** 2) * (1.0 * 93.0)
    _, r_plain_3, _, _ = env_plain.step(0)
    _, r_shaped_3, _, _ = env_shaped.step(0)

    assert r_shaped_3 == pytest.approx(r_plain_3 - expected_penalty)


def test_get_info_field_diagnostics():
    prices = [100.0, 100.0, 110.0, 107.0]
    df = _make_sample_df(prices)

    env = initiate_base_env(
        df,
        feature_list=["feat1"],
        max_holding_number=1,
        position_choices=3,
        leverage_choice=[1],
        initial_state=(1e5, 0.0, 0.0, 0.0, 1),
        commission_rate=0.0,
        enable_drawdown_reward_shaping=True,
        drawdown_profit_min=0.08,
        drawdown_allow_ratio=0.15,
        drawdown_penalty_weight=0.01,
    )

    env.reset()
    assert env.get_info_field("drawdown_penalty") == 0.0
    assert env.get_info_field("instant_profit_retracement") == 0.0

    env.step(2)  # Open long at 100
    env.step(2)  # Hold at 110
    env.step(2)  # Hold at 107 -> triggers penalty

    assert env.get_info_field("instant_profit_retracement") == pytest.approx(0.30)
    assert env.get_info_field("drawdown_penalty") > 0.0


def test_base_env_take_profit_turnover_no_exemption_when_peak_below_threshold():
    # Prices:
    # 0: 100 (reset)
    # 1: 100 (open long pos=1)
    # 2: 104 (hold long: reaches 4% gain < 8% profit threshold)
    # 3: 102 (close to flat: pos 1 -> 0 in uptrend regime)
    prices = [100.0, 100.0, 104.0, 102.0]
    df = _make_sample_df(prices)
    df["regime_grid_id"] = np.full(len(prices), 8, dtype=np.int64)  # 8 % 3 == 2 (uptrend)

    base_rate = 0.0001
    adverse_ratio = 6.0

    env = initiate_base_env(
        df,
        feature_list=["feat1"],
        max_holding_number=1,
        position_choices=3,
        leverage_choice=[1],
        initial_state=(1e5, 0.0, 0.0, 0.0, 1),
        commission_rate=0.0,
        turnover_base_rate=base_rate,
        turnover_adverse_ratio=adverse_ratio,
        enable_drawdown_reward_shaping=True,
        drawdown_profit_min=0.08,
        enable_take_profit_turnover_exemption=True,
    )

    env.reset()
    env.step(2)  # Open Long at 100
    env.step(2)  # Hold Long at 104 (profit is 4% < 8%)

    # Step 3: Close to flat at 102
    # Because peak profit was 4% (< 8%), take-profit exemption does NOT trigger.
    # Adverse turnover penalty (6x) must apply: 6 * 0.0001 * 1.0 * 104.0 = 0.0624
    _, reward, _, _ = env.step(1)
    expected_turnover = adverse_ratio * base_rate * 1.0 * 104.0
    assert reward == pytest.approx(-expected_turnover)
