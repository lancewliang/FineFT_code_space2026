from __future__ import annotations


class TradeColumns:
    POSITION: str = "position"
    INITIAL_ACTION: str = "initial_action"
    PREVIOUS_ACTION: str = "previous_action"
    AVALIABLE_ACTION: str = "avaliable_action"
    BIN_INDEX: str = "bin_index"
    LABEL: str = "label"
    EPOCH_NUMBER: str = "epoch_number"
    REALIZED_PNL_STEP: str = "realized_pnl_step"
    UNREALIZED_PNL: str = "unrealized_pnl"
    CUMULATIVE_REALIZED_PNL: str = "cumulative_realized_pnl"
    COMMISSION_FEE_STEP: str = "commission_fee_step"
    CUMULATIVE_COMMISSION_FEE: str = "cumulative_commission_fee"
    SLIPPAGE_STEP: str = "slippage_step"
    CUMULATIVE_SLIPPAGE: str = "cumulative_slippage"
    WALLET_BALANCE: str = "wallet_balance"
    CASH_BALANCE: str = "cash_balance"
    TOTAL_VALUE: str = "total_value"
    LEVERAGE: str = "leverage"
    MARK_PRICE: str = "mark_price"
    CLOSE: str = "close"
    BID1_PRICE: str = "bid1_price"
    LOWER_LIMIT_PRICE: str = "LowerLimitPrice"
    UPPER_LIMIT_PRICE: str = "UpperLimitPrice"
    IS_LIMIT_UP: str = "is_limit_up"
    IS_LIMIT_DOWN: str = "is_limit_down"
    LIMIT_UP_SINGLE_SIDED_RATIO: str = "limit_up_single_sided_ratio"
    LIMIT_DOWN_SINGLE_SIDED_RATIO: str = "limit_down_single_sided_ratio"
    TRADING_INFO: str = "trading_info"
    MEAN_HOLDING_DURATION: str = "mean_holding_duration"
    POSITION_FLIP_RATE: str = "position_flip_rate"
    POSITION_FORWARD_RETURN_CORR: str = "position_forward_return_corr"
    LONG_FORWARD_RETURN_MEAN: str = "long_forward_return_mean"
    SHORT_FORWARD_RETURN_MEAN: str = "short_forward_return_mean"


CSV_HEADER_LABELS: dict[str, str] = {
    "mean_position": "平均仓位",
    "mean_abs_position": "平均绝对仓位",
    "long_step_ratio": "多头步数占比",
    "short_step_ratio": "空头步数占比",
    "flat_step_ratio": "空仓步数占比",
    "long_reward_sum": "多头奖励总和",
    "short_reward_sum": "空头奖励总和",
    "flat_reward_sum": "空仓奖励总和",
    "net_position_exposure": "净仓位敞口",
    "position_forward_return_corr": "仓位与下一期收益相关",
    "position_flip_rate": "仓位换向率",
    "mean_holding_duration": "平均持仓时长",
    "long_forward_return_mean": "多头下一期平均收益",
    "short_forward_return_mean": "空头下一期平均收益",
    "limit_up_step_ratio": "涨停步数占比",
    "limit_down_step_ratio": "跌停步数占比",
    "limit_up_long_reward_sum": "涨停多头奖励总和",
    "limit_down_short_reward_sum": "跌停空头奖励总和",
    "limit_up_reverse_short_ratio": "涨停反向空头占比",
    "limit_down_reverse_long_ratio": "跌停反向多头占比",
    "label": "标签",
    "initial_action": "初始动作",
    "bin_index": "分箱索引",
    "contract": "合约",
    "df_path": "数据文件",
    "reward_sum": "奖励总和",
    "df_length": "数据长度",
    "turnover": "换手率",
    "timestep": "时间步",
    "timestamp": "时间戳",
    "open": "开盘价",
    "high": "最高价",
    "low": "最低价",
    "close": "收盘价",
    "volume": "成交量",
    "mark_price": "标记价格",
    "action": "动作",
    "target_position": "目标仓位",
    "target_leverage": "目标杠杆",
    "position_before": "执行前仓位",
    "leverage_before": "执行前杠杆",
    "position_after": "执行后仓位",
    "leverage_after": "执行后杠杆",
    "action_change_step": "动作变化",
    "trade_count_step": "交易计数",
    "cumulative_action_change_count": "累计动作变化次数",
    "cumulative_trade_count": "累计交易次数",
    "step_reward": "单步奖励",
    "realized_pnl_step": "单步实现盈亏",
    "cumulative_realized_pnl": "累计已实现盈亏",
    "commission_fee_step": "单步手续费",
    "cumulative_commission_fee": "累计手续费",
    "slippage_step": "单步滑点",
    "cumulative_slippage": "累计滑点",
    "wallet_balance": "结算总价值",
    "unrealized_pnl": "浮动盈亏",
    "margin_balance": "保证金余额",
    "notional_asset_value": "持仓资产",
    "cash_balance": "结算总价值",
    "total_value": "浮动总价值",
}

AGGREGATE_JSON_COLUMNS: list[str] = [
    "contract",
    "df_path",
    "reward_sum",
    "df_length",
    "turnover",
    "mean_position",
    "mean_abs_position",
    "long_step_ratio",
    "short_step_ratio",
    "flat_step_ratio",
    "long_reward_sum",
    "short_reward_sum",
    "flat_reward_sum",
    "net_position_exposure",
    "position_forward_return_corr",
    "position_flip_rate",
    "mean_holding_duration",
    "long_forward_return_mean",
    "short_forward_return_mean",
    "limit_up_step_ratio",
    "limit_down_step_ratio",
    "limit_up_long_reward_sum",
    "limit_down_short_reward_sum",
    "limit_up_reverse_short_ratio",
    "limit_down_reverse_long_ratio",
]
