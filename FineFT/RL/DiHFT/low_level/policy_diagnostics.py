from __future__ import annotations

from collections.abc import Sequence

import numpy as np


ZERO_POLICY_DIRECTION_METRICS = {
    "position_forward_return_corr": 0.0,
    "position_flip_rate": 0.0,
    "mean_holding_duration": 0.0,
    "long_forward_return_mean": 0.0,
    "short_forward_return_mean": 0.0,
}


def calculate_policy_direction_metrics(
    positions: Sequence[float], mark_prices: Sequence[float]
) -> dict[str, float]:
    """Measure whether a policy follows or opposes the next market move."""

    position_array = np.asarray(positions, dtype=float).reshape(-1)
    price_array = np.asarray(mark_prices, dtype=float).reshape(-1)
    usable = min(position_array.size, max(price_array.size - 1, 0))
    if usable == 0:
        return dict(ZERO_POLICY_DIRECTION_METRICS)

    position_array = position_array[:usable]
    forward_returns = np.diff(price_array)[:usable]
    finite = np.isfinite(position_array) & np.isfinite(forward_returns)
    position_array = position_array[finite]
    forward_returns = forward_returns[finite]
    if position_array.size == 0:
        return dict(ZERO_POLICY_DIRECTION_METRICS)

    if (
        position_array.size >= 2
        and np.std(position_array) > 0.0
        and np.std(forward_returns) > 0.0
    ):
        correlation = float(np.corrcoef(position_array, forward_returns)[0, 1])
    else:
        correlation = 0.0

    if position_array.size >= 2:
        flip_rate = float(
            np.mean(position_array[:-1] * position_array[1:] < 0.0)
        )
    else:
        flip_rate = 0.0

    run_lengths: list[int] = []
    current_sign = 0
    current_length = 0
    for position in position_array:
        sign = int(np.sign(position))
        if sign == 0:
            if current_length:
                run_lengths.append(current_length)
                current_sign = 0
                current_length = 0
            continue
        if sign == current_sign:
            current_length += 1
        else:
            if current_length:
                run_lengths.append(current_length)
            current_sign = sign
            current_length = 1
    if current_length:
        run_lengths.append(current_length)

    long_returns = forward_returns[position_array > 0.0]
    short_returns = forward_returns[position_array < 0.0]
    return {
        "position_forward_return_corr": correlation,
        "position_flip_rate": flip_rate,
        "mean_holding_duration": (
            float(np.mean(run_lengths)) if run_lengths else 0.0
        ),
        "long_forward_return_mean": (
            float(np.mean(long_returns)) if long_returns.size else 0.0
        ),
        "short_forward_return_mean": (
            float(np.mean(short_returns)) if short_returns.size else 0.0
        ),
    }
