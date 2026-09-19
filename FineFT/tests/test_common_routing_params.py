from common.routing_params import RoutingParamColumns


def test_routing_param_columns_literals() -> None:
    assert RoutingParamColumns.SLOPE_WINDOW_LENGTH == "slope_window_length"
    assert (
        RoutingParamColumns.VOLATILITY_WINDOW_LENGTH
        == "volatility_window_length"
    )
    assert RoutingParamColumns.SLOPE_GAMMA == "slope_gamma"
    assert RoutingParamColumns.VOLATILITY_GAMMA == "volatility_gamma"
    assert (
        RoutingParamColumns.SLOPE_RULE_BASE_THRESHOLD
        == "slope_rule_base_threshold"
    )
    assert (
        RoutingParamColumns.VOLATILITY_RULE_BASE_THRESHOLD
        == "volatility_rule_base_threshold"
    )

    assert (
        RoutingParamColumns.PARAMS_SLOPE_WINDOW_LENGTH
        == "params_slope_window_length"
    )
    assert (
        RoutingParamColumns.PARAMS_VOLATILITY_WINDOW_LENGTH
        == "params_volatility_window_length"
    )
    assert RoutingParamColumns.PARAMS_SLOPE_GAMMA == "params_slope_gamma"
    assert (
        RoutingParamColumns.PARAMS_VOLATILITY_GAMMA
        == "params_volatility_gamma"
    )
    assert (
        RoutingParamColumns.PARAMS_SLOPE_RULE_BASE_THRESHOLD
        == "params_slope_rule_base_threshold"
    )
    assert (
        RoutingParamColumns.PARAMS_VOLATILITY_RULE_BASE_THRESHOLD
        == "params_volatility_rule_base_threshold"
    )

    assert RoutingParamColumns.PARAMS_WINDOW_LENGTH == "params_window_length"
    assert RoutingParamColumns.PARAMS_GAMMA == "params_gamma"
    assert (
        RoutingParamColumns.PARAMS_RULE_BASE_THRESHOLD
        == "params_rule_base_threshold"
    )
    assert RoutingParamColumns.NUMBER == "number"
