from __future__ import annotations


class RoutingParamColumns:
    SLOPE_WINDOW_LENGTH: str = "slope_window_length"
    VOLATILITY_WINDOW_LENGTH: str = "volatility_window_length"
    SLOPE_GAMMA: str = "slope_gamma"
    VOLATILITY_GAMMA: str = "volatility_gamma"
    SLOPE_RULE_BASE_THRESHOLD: str = "slope_rule_base_threshold"
    VOLATILITY_RULE_BASE_THRESHOLD: str = "volatility_rule_base_threshold"

    PARAMS_SLOPE_WINDOW_LENGTH: str = "params_slope_window_length"
    PARAMS_VOLATILITY_WINDOW_LENGTH: str = "params_volatility_window_length"
    PARAMS_SLOPE_GAMMA: str = "params_slope_gamma"
    PARAMS_VOLATILITY_GAMMA: str = "params_volatility_gamma"
    PARAMS_SLOPE_RULE_BASE_THRESHOLD: str = "params_slope_rule_base_threshold"
    PARAMS_VOLATILITY_RULE_BASE_THRESHOLD: str = (
        "params_volatility_rule_base_threshold"
    )

    PARAMS_WINDOW_LENGTH: str = "params_window_length"
    PARAMS_GAMMA: str = "params_gamma"
    PARAMS_RULE_BASE_THRESHOLD: str = "params_rule_base_threshold"
    NUMBER: str = "number"

    GATING_STRATEGY: str = "gating_strategy"
    PARAMS_GATING_STRATEGY: str = "params_gating_strategy"

    OOD_THRESHOLD: str = "ood_threshold"
    PARAMS_OOD_THRESHOLD: str = "params_ood_threshold"

    SLOPE_MARGIN_THRESHOLD: str = "slope_margin_threshold"
    PARAMS_SLOPE_MARGIN_THRESHOLD: str = "params_slope_margin_threshold"

    VOLATILITY_MARGIN_THRESHOLD: str = "volatility_margin_threshold"
    PARAMS_VOLATILITY_MARGIN_THRESHOLD: str = (
        "params_volatility_margin_threshold"
    )
