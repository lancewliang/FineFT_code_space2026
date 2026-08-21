import sys
from pathlib import Path

import pytest


FINEFT_ROOT = Path(__file__).resolve().parents[2]
if str(FINEFT_ROOT) not in sys.path:
    sys.path.insert(0, str(FINEFT_ROOT))


def test_policy_direction_metrics_describe_forward_returns_and_position_lifecycle():
    from RL.DiHFT.low_level.policy_diagnostics import (
        calculate_policy_direction_metrics,
    )

    metrics = calculate_policy_direction_metrics(
        positions=[1.0, 1.0, 0.0, -1.0, -1.0, 1.0],
        mark_prices=[100.0, 101.0, 102.0, 101.0, 100.0, 99.0, 100.0],
    )

    assert metrics["position_forward_return_corr"] == pytest.approx(
        0.9284766909
    )
    assert metrics["position_flip_rate"] == pytest.approx(0.2)
    assert metrics["mean_holding_duration"] == pytest.approx(5.0 / 3.0)
    assert metrics["long_forward_return_mean"] == pytest.approx(1.0)
    assert metrics["short_forward_return_mean"] == pytest.approx(-1.0)


def test_policy_direction_metrics_return_zero_for_insufficient_data():
    from RL.DiHFT.low_level.policy_diagnostics import (
        calculate_policy_direction_metrics,
    )

    metrics = calculate_policy_direction_metrics(
        positions=[0.0],
        mark_prices=[100.0, 101.0],
    )

    assert metrics == {
        "position_forward_return_corr": 0.0,
        "position_flip_rate": 0.0,
        "mean_holding_duration": 0.0,
        "long_forward_return_mean": 0.0,
        "short_forward_return_mean": 0.0,
    }
