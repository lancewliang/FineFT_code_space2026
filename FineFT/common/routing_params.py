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
