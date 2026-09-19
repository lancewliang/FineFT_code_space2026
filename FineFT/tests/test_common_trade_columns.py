from common.trade_columns import (
    AGGREGATE_JSON_COLUMNS,
    CSV_HEADER_LABELS,
    TradeColumns,
)


def test_trade_columns_literals() -> None:
    assert TradeColumns.POSITION == "position"
    assert TradeColumns.INITIAL_ACTION == "initial_action"
    assert TradeColumns.PREVIOUS_ACTION == "previous_action"
    assert TradeColumns.AVALIABLE_ACTION == "avaliable_action"
    assert TradeColumns.BIN_INDEX == "bin_index"
    assert TradeColumns.LABEL == "label"
    assert TradeColumns.EPOCH_NUMBER == "epoch_number"
    assert TradeColumns.REALIZED_PNL_STEP == "realized_pnl_step"
    assert TradeColumns.UNREALIZED_PNL == "unrealized_pnl"
    assert TradeColumns.CUMULATIVE_REALIZED_PNL == "cumulative_realized_pnl"
    assert TradeColumns.COMMISSION_FEE_STEP == "commission_fee_step"
    assert TradeColumns.CUMULATIVE_COMMISSION_FEE == "cumulative_commission_fee"
    assert TradeColumns.SLIPPAGE_STEP == "slippage_step"
    assert TradeColumns.CUMULATIVE_SLIPPAGE == "cumulative_slippage"
    assert TradeColumns.WALLET_BALANCE == "wallet_balance"
    assert TradeColumns.CASH_BALANCE == "cash_balance"
    assert TradeColumns.TOTAL_VALUE == "total_value"
    assert TradeColumns.LEVERAGE == "leverage"
    assert TradeColumns.MARK_PRICE == "mark_price"
    assert TradeColumns.CLOSE == "close"
    assert TradeColumns.BID1_PRICE == "bid1_price"
    assert TradeColumns.LOWER_LIMIT_PRICE == "LowerLimitPrice"
    assert TradeColumns.UPPER_LIMIT_PRICE == "UpperLimitPrice"
    assert TradeColumns.IS_LIMIT_UP == "is_limit_up"
    assert TradeColumns.IS_LIMIT_DOWN == "is_limit_down"
    assert (
        TradeColumns.LIMIT_UP_SINGLE_SIDED_RATIO
        == "limit_up_single_sided_ratio"
    )
    assert (
        TradeColumns.LIMIT_DOWN_SINGLE_SIDED_RATIO
        == "limit_down_single_sided_ratio"
    )
    assert TradeColumns.TRADING_INFO == "trading_info"
    assert TradeColumns.MEAN_HOLDING_DURATION == "mean_holding_duration"
    assert TradeColumns.POSITION_FLIP_RATE == "position_flip_rate"
    assert (
        TradeColumns.POSITION_FORWARD_RETURN_CORR
        == "position_forward_return_corr"
    )
    assert (
        TradeColumns.LONG_FORWARD_RETURN_MEAN
        == "long_forward_return_mean"
    )
    assert (
        TradeColumns.SHORT_FORWARD_RETURN_MEAN
        == "short_forward_return_mean"
    )


def test_csv_header_labels_and_json_columns() -> None:
    assert isinstance(CSV_HEADER_LABELS, dict)
    assert CSV_HEADER_LABELS["mean_position"] == "平均仓位"
    assert CSV_HEADER_LABELS["wallet_balance"] == "结算总价值"
    assert CSV_HEADER_LABELS["unrealized_pnl"] == "浮动盈亏"
    assert CSV_HEADER_LABELS["position_after"] == "执行后仓位"
    assert "contract" in AGGREGATE_JSON_COLUMNS
    assert "reward_sum" in AGGREGATE_JSON_COLUMNS
    assert "limit_down_reverse_long_ratio" in AGGREGATE_JSON_COLUMNS
