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
    max_punishment: float = 1e10,
    enable_limit_reward: bool = True,
    limit_hold_bonus: float = 1.0,
    limit_stay_bonus: float = 0.5,
    limit_reverse_penalty: float = 1.5,
    near_limit_threshold: float = 0.003,
    turnover_penalty_rate: float = 0.0,
    turnover_base_rate: float = 0.0,
    turnover_adverse_ratio: float = 1.0,
) -> dict[str, object]:
    """Build the DP teacher configuration from the active training config."""
    effective_base = turnover_base_rate if turnover_base_rate > 0.0 else turnover_penalty_rate

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
        "enable_limit_reward": enable_limit_reward,
        "limit_hold_bonus": limit_hold_bonus,
        "limit_stay_bonus": limit_stay_bonus,
        "limit_reverse_penalty": limit_reverse_penalty,
        "near_limit_threshold": near_limit_threshold,
        "turnover_penalty_rate": turnover_penalty_rate,
        "turnover_base_rate": effective_base,
        "turnover_adverse_ratio": turnover_adverse_ratio,
    }
