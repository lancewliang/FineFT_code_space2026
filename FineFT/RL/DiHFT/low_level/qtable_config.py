from __future__ import annotations


def build_optimal_qtable_kwargs(
    *,
    max_holding_number: float,
    order_book_depth: int,
    position_choices: int,
    leverage_choice: list[int],
    long_estimated_rate: float,
    short_estimated_rate: float,
    commission_rate: float,
    gamma: float,
    allow_reverse_position: bool,
    max_punishment: float = 1e10,
) -> dict[str, object]:
    """Build the DP teacher configuration from the active training config."""

    return {
        "max_holding_number": max_holding_number,
        "order_book_depth": order_book_depth,
        "position_choices": position_choices,
        "leverage_choice": leverage_choice,
        "long_estimated_rate": long_estimated_rate,
        "short_estimated_rate": short_estimated_rate,
        "commission_rate": commission_rate,
        "max_punishment": max_punishment,
        "gamma": gamma,
        "allow_reverse_position": allow_reverse_position,
    }
