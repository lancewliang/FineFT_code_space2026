import numpy as np
import pandas as pd

"""
if markprice can be guranateed to be between ask1_price and bid1_price, the open losses and close losses could be integrated into the market order
calculation function and make the overall process faster.
"""
min_orderbook = 1e-5


def change_of_wallet(
    markprice,
    ask_prices,
    ask_qtys,
    bid_prices,
    bid_qtys,
    long_estimated_rate,
    short_estimated_rate,
    commission_rate,
    # before action
    previous_leverage,
    previous_position,
    previous_initial_margine,
    previous_unrealized_pnL,
    previous_wallet_balance,
    # target after the action
    current_leverage,
    current_position,
    silent=True,
):
    # if current_leverage == previous_leverage:
    if current_position == previous_position:
        (
            previous_leverage,
            previous_position,
            previous_initial_margine,
            previous_unrealized_pnL,
            previous_wallet_balance,
            slippage,
        ) = change_of_leverage(
            markprice,
            previous_leverage,
            previous_position,
            previous_initial_margine,
            previous_unrealized_pnL,
            previous_wallet_balance,
            current_leverage,
            silent=silent,
        )
        return (
            previous_leverage,
            previous_position,
            previous_initial_margine,
            previous_unrealized_pnL,
            previous_wallet_balance,
            slippage,
        )
    else:
        if current_position * previous_position < 0:
            if not silent:
                print(
                    "You can not turn over the position in just one step, the position will stick to the previous situation (the position and the leverage)"
                )
            return (
                previous_leverage,
                previous_position,
                previous_initial_margine,
                previous_unrealized_pnL,
                previous_wallet_balance,
                0,
            )
        elif max(current_position, previous_position) > 0:
            # 多头情况，分close long 和 open long
            if current_position - previous_position > 0:
                # open long position
                # first change the leverage, then open the position
                (
                    previous_leverage,
                    previous_position,
                    previous_initial_margine,
                    previous_unrealized_pnL,
                    previous_wallet_balance,
                    slippage,
                ) = change_of_leverage(
                    markprice,
                    previous_leverage,
                    previous_position,
                    previous_initial_margine,
                    previous_unrealized_pnL,
                    previous_wallet_balance,
                    current_leverage,
                    silent=silent,
                )
                return open_long_position(
                    markprice,
                    ask_prices,
                    ask_qtys,
                    long_estimated_rate,
                    commission_rate,
                    previous_leverage,
                    previous_position,
                    previous_initial_margine,
                    previous_unrealized_pnL,
                    previous_wallet_balance,
                    current_leverage,
                    current_position,
                    silent=silent,
                )
            else:
                # close long position and then change the leverage
                (
                    previous_leverage,
                    previous_position,
                    previous_initial_margine,
                    previous_unrealized_pnL,
                    previous_wallet_balance,
                    slippage,
                ) = close_long_position(
                    markprice,
                    bid_prices,
                    bid_qtys,
                    commission_rate,
                    previous_leverage,
                    previous_position,
                    previous_initial_margine,
                    previous_unrealized_pnL,
                    previous_wallet_balance,
                    previous_leverage,
                    current_position,
                    silent=silent,
                )
                return change_of_leverage(
                    markprice,
                    previous_leverage,
                    previous_position,
                    previous_initial_margine,
                    previous_unrealized_pnL,
                    previous_wallet_balance,
                    current_leverage,
                    silent=silent,
                )
        elif min(current_position, previous_position) < 0:
            # 空头情况, 分close short 和 open short
            if current_position - previous_position < 0:
                # open short position
                # first change the leverage, then open the position
                (
                    previous_leverage,
                    previous_position,
                    previous_initial_margine,
                    previous_unrealized_pnL,
                    previous_wallet_balance,
                    slippage,
                ) = change_of_leverage(
                    markprice,
                    previous_leverage,
                    previous_position,
                    previous_initial_margine,
                    previous_unrealized_pnL,
                    previous_wallet_balance,
                    current_leverage,
                    silent=silent,
                )
                return open_short_position(
                    markprice,
                    bid_prices,
                    bid_qtys,
                    short_estimated_rate,
                    commission_rate,
                    previous_leverage,
                    previous_position,
                    previous_initial_margine,
                    previous_unrealized_pnL,
                    previous_wallet_balance,
                    current_leverage,
                    current_position,
                    silent=silent,
                )
            else:
                # close short position
                # first close the position, then change the leverage
                (
                    previous_leverage,
                    previous_position,
                    previous_initial_margine,
                    previous_unrealized_pnL,
                    previous_wallet_balance,
                    slippage,
                ) = close_short_position(
                    markprice,
                    ask_prices,
                    ask_qtys,
                    commission_rate,
                    previous_leverage,
                    previous_position,
                    previous_initial_margine,
                    previous_unrealized_pnL,
                    previous_wallet_balance,
                    previous_leverage,
                    current_position,
                    silent=silent,
                )
                return change_of_leverage(
                    markprice,
                    previous_leverage,
                    previous_position,
                    previous_initial_margine,
                    previous_unrealized_pnL,
                    previous_wallet_balance,
                    current_leverage,
                    silent=silent,
                )


def change_of_leverage(
    markprice,
    previous_leverage,
    previous_position,
    previous_initial_margine,
    previous_unrealized_pnL,
    previous_wallet_balance,
    current_leverage,
    silent=True,
):
    assert previous_initial_margine == np.abs(
        markprice * previous_position / previous_leverage
    )
    # markprice does not change, so the estimated holding does not change
    # change leverage, the pnl and position will not change
    # the only thing change is the margine, and consequently the avalibale balance
    current_margine_requirement = np.abs(
        markprice * previous_position / current_leverage
    )
    current_margine_balance = previous_wallet_balance + previous_unrealized_pnL
    if (
        current_margine_balance < current_margine_requirement
        and previous_leverage != current_leverage
    ):
        if not silent:
            print(
                "Not enough balance to change leverage, we will hold on to the previous leverage"
            )
        current_leverage = previous_leverage
    else:
        current_leverage = current_leverage

    current_initial_margine = np.abs(markprice * previous_position / current_leverage)
    current_position = previous_position
    current_unrealized_pnL = previous_unrealized_pnL
    current_wallet_balance = previous_wallet_balance
    slippage = 0
    return (
        current_leverage,
        current_position,
        current_initial_margine,
        current_unrealized_pnL,
        current_wallet_balance,
        slippage,
    )


def open_short_position(
    markprice,
    bid_prices,
    bid_qtys,
    short_estimated_rate,
    commission_rate,
    # before action
    previous_leverage,
    previous_position,
    previous_initial_margine,
    previous_unrealized_pnL,
    previous_wallet_balance,
    # target after the action
    current_leverage,
    current_position,
    silent=True,
):
    assert previous_position <= 0
    assert previous_leverage == current_leverage
    assert (previous_position - current_position) > 0

    assert previous_initial_margine == np.abs(
        markprice * previous_position / previous_leverage
    )

    open_short_position = previous_position - current_position
    open_losses, actual_changed_position = calculate_open_short_loss(
        bid_prices, bid_qtys, open_short_position, markprice, short_estimated_rate
    )
    open_value, actual_position_change = calculate_open_short_close_long_position(
        bid_prices, bid_qtys, open_short_position
    )
    assert actual_changed_position == actual_position_change
    if actual_changed_position != open_short_position:
        if not silent:
            print(
                "the short position could not be opened all clear since further level of ask price is needed. We will stick to the most order we can take"
            )
        open_short_position = actual_changed_position
        current_position = previous_position - open_short_position
    openinig_margine = markprice * open_short_position / current_leverage
    open_margine = openinig_margine + open_losses
    commission_fee = commission_rate * open_value
    # 如果开成了，开仓current_wallet_balance的变化,注意到在判断是否可以开仓的时候，并不需要考虑手续费 所以这个时候current_wallet_balance是不包含手续费的
    current_wallet_balance = previous_wallet_balance
    available_balance = (
        current_wallet_balance + previous_unrealized_pnL - previous_initial_margine
    )
    if open_margine > available_balance:
        if not silent:
            print(
                "Not enough balance to open short position, will stick to the previous position"
            )
        # 之前的手续费要退回来 重新算balance
        current_wallet_balance = previous_wallet_balance
        current_position = previous_position
        current_unrealized_pnL = previous_unrealized_pnL
        current_initial_margine = previous_initial_margine
        slippage = 0
    else:
        # 开成了要判断一下是不是瞬间就被强平了 还是可以继续下去
        current_wallet_balance = previous_wallet_balance - commission_fee
        current_position = current_position
        opening_pnL = -(openinig_margine * current_leverage - open_value)
        current_unrealized_pnL = previous_unrealized_pnL + opening_pnL
        current_initial_margine = np.abs(
            current_position * markprice / current_leverage
        )
        # notice that is loss is not natrual and not directly deducted from the wallet balance, yet it is viewed as a part of of unrealized pnl
        slippage = openinig_margine * current_leverage - open_value
    return (
        current_leverage,
        current_position,
        current_initial_margine,
        current_unrealized_pnL,
        current_wallet_balance,
        slippage,
    )


def close_short_position(
    markprice,
    ask_prices,
    ask_qtys,
    commission_rate,
    # before action
    previous_leverage,
    previous_position,
    previous_initial_margine,
    previous_unrealized_pnL,
    previous_wallet_balance,
    # target after the action
    current_leverage,
    current_position,
    silent=True,
):
    assert current_position <= 0
    assert previous_leverage == current_leverage
    assert (current_position - previous_position) > 0
    assert previous_initial_margine == np.abs(
        markprice * previous_position / previous_leverage
    )
    close_short_position = current_position - previous_position
    value, actual_changed_position = calculate_open_long_close_short_position(
        ask_prices, ask_qtys, close_short_position
    )
    commission_fee = commission_rate * value
    if actual_changed_position != close_short_position:
        if not silent:
            print(
                "the short position could not be closed all clear since further level of ask price is needed. We will stick to the most order we can take"
            )

        close_short_position = actual_changed_position
        current_position = previous_position + close_short_position
    realized_pnL = (
        previous_unrealized_pnL
        * np.abs(close_short_position / previous_position)  # 对应的unrealized pnl的变化
        - value  # 还币真正花的钱
        + markprice * close_short_position  # 之前欠的预估的钱
    )
    current_unrealized_pnL = previous_unrealized_pnL * (
        current_position / previous_position
    )
    current_initial_margine = np.abs(markprice * current_position / current_leverage)
    current_wallet_balance = previous_wallet_balance + realized_pnL - commission_fee
    slippage = value - markprice * close_short_position
    return (
        current_leverage,
        current_position,
        current_initial_margine,
        current_unrealized_pnL,
        current_wallet_balance,
        slippage,
    )


def open_long_position(
    markprice,
    ask_prices,
    ask_qtys,
    long_estimated_rate,
    commission_rate,
    # before action
    previous_leverage,
    previous_position,
    previous_initial_margine,
    previous_unrealized_pnL,
    previous_wallet_balance,
    # target after the action
    current_leverage,
    current_position,
    silent=True,
):
    # the leveage should be consisten and the position should be originally semi-postive and increased.
    assert previous_initial_margine == markprice * previous_position / previous_leverage
    assert previous_position >= 0
    assert (current_position - previous_position) > 0
    assert previous_leverage == current_leverage
    open_long_position = current_position - previous_position
    open_losses, actual_changed_position = calculate_open_long_loss(
        ask_prices, ask_qtys, open_long_position, markprice, long_estimated_rate
    )
    open_value, actual_position_change = calculate_open_long_close_short_position(
        ask_prices, ask_qtys, open_long_position
    )
    assert actual_changed_position == actual_position_change
    # 要不要直接统一成直接开不成了 直接维持现状
    if actual_changed_position != open_long_position:
        if not silent:
            print(
                "the long position could not be opened all clear since further level of ask price is needed. We will stick to the most order we can take"
            )
        open_long_position = actual_changed_position
        current_position = previous_position + open_long_position
    openinig_margine = markprice * open_long_position / current_leverage
    open_margine = openinig_margine + open_losses
    commission_fee = commission_rate * open_value

    # 如果开成了，开仓current_wallet_balance的变化
    current_wallet_balance = previous_wallet_balance
    available_balance = (
        current_wallet_balance + previous_unrealized_pnL - previous_initial_margine
    )
    if open_margine > available_balance:
        if not silent:
            print(
                "Not enough balance to open long position, will stick to the previous position"
            )
        # 之前的手续费要退回来 重新算balance
        current_wallet_balance = previous_wallet_balance
        current_position = previous_position
        current_unrealized_pnL = previous_unrealized_pnL
        current_initial_margine = previous_initial_margine
        slippage = 0
    else:
        # 开成了要判断一下是不是瞬间就被强平了 还是可以继续下去
        current_wallet_balance = previous_wallet_balance - commission_fee
        current_position = current_position
        opening_pnL = openinig_margine * current_leverage - open_value
        current_unrealized_pnL = previous_unrealized_pnL + opening_pnL
        current_initial_margine = current_position * markprice / current_leverage
        slippage = open_value - openinig_margine
    return (
        current_leverage,
        current_position,
        current_initial_margine,
        current_unrealized_pnL,
        current_wallet_balance,
        slippage,
    )


def close_long_position(
    markprice,
    bid_prices,
    bid_qtys,
    commission_rate,
    # before action
    previous_leverage,
    previous_position,
    previous_initial_margine,
    previous_unrealized_pnL,
    previous_wallet_balance,
    # target after the action
    current_leverage,
    current_position,
    silent=True,
):
    # * 没有判断是否被强暴仓了
    assert current_position >= 0
    assert previous_leverage == current_leverage
    assert (previous_position - current_position) > 0
    assert previous_initial_margine == markprice * previous_position / previous_leverage
    close_long_position = previous_position - current_position
    value, actual_changed_position = calculate_open_short_close_long_position(
        bid_prices, bid_qtys, close_long_position
    )
    commission_fee = commission_rate * value
    if actual_changed_position != close_long_position:
        if not silent:
            print(
                "the long position could not be closed all clear since further level of bid price is needed. We will stick to the most order we can take"
            )
        close_long_position = actual_changed_position
        current_position = previous_position - close_long_position
    realized_pnL = (
        previous_unrealized_pnL
        * close_long_position
        / previous_position  # 对应的unrealized pnl的变化
        + value  # 真正手里拿到的钱
        - markprice * close_long_position  # 之前预估的钱
    )
    current_unrealized_pnL = previous_unrealized_pnL * (
        current_position / previous_position
    )
    current_initial_margine = markprice * current_position / current_leverage
    current_wallet_balance = previous_wallet_balance + realized_pnL - commission_fee
    slippage = markprice * close_long_position - value
    return (
        current_leverage,
        current_position,
        current_initial_margine,
        current_unrealized_pnL,
        current_wallet_balance,
        slippage,
    )


def calculate_open_short_loss(
    bid_prices, bid_qtys, position, mark_price, short_estimated_rate, slient=True
):
    """
    bid_prices and bid_qtys should be form level 1 to level 25
    calculate the open losses for a short position, the position should be positive, indicating the absolute value of the open position
    it output the open losses, which indicates the price difference between the mark price and the market price, indicating that besides
    initial margine, how much more we should own in the avaible balance.
    """

    assert position > 0
    assert len(bid_prices) == len(bid_qtys)
    open_losses = 0
    orgional_position = position

    for level in range(len(bid_prices) + 1):
        if level == len(bid_prices) or position <= bid_qtys[level]:
            break
        else:
            position -= bid_qtys[level]
            open_losses += bid_qtys[level] * (
                np.abs(
                    min(
                        0,
                        -1
                        * (mark_price - bid_prices[level] * (1 + short_estimated_rate)),
                    )
                )
            )
    if position > min_orderbook and level == len(bid_prices):
        if not slient:
            print(
                "the long position could not be opened all clear since further level of bid price is needed"
            )
        # 执行的单量
        actual_changed_position = orgional_position - position
    else:
        if position <= min_orderbook:
            actual_changed_position = orgional_position
            return open_losses, actual_changed_position
        open_losses += (
            np.abs(
                min(
                    0,
                    -1 * (mark_price - bid_prices[level] * (1 + short_estimated_rate)),
                )
            )
            * position
        )
        actual_changed_position = orgional_position
    return open_losses, actual_changed_position


def calculate_open_long_loss(
    ask_prices, ask_qtys, position, mark_price, long_estimated_rate, slient=True
):
    """
    ask_prices and ask_qtys should be form level 1 to level 25
    calculate the open losses for a long position, the position should be positive, indicating the absolute value of the open position
    it output the open losses, which indicates the price difference between the mark price and the market price, indicating that besides
    initial margine, how much more we should own in the avaible balance.
    """

    assert position > 0
    assert len(ask_prices) == len(ask_qtys)
    open_losses = 0
    orgional_position = position
    for level in range(len(ask_prices) + 1):
        if level == len(ask_prices) or position <= ask_qtys[level]:
            break
        else:
            position -= ask_qtys[level]
            open_losses += ask_qtys[level] * (
                np.abs(
                    min(
                        0,
                        1
                        * (mark_price - ask_prices[level] * (1 + long_estimated_rate)),
                    )
                )
            )
    if position > min_orderbook and level == len(ask_prices):
        if not slient:
            print(
                "the long position could not be opened all clear since further level of ask price is needed"
            )
        # 执行的单量
        actual_changed_position = orgional_position - position
    else:
        if position < min_orderbook:
            actual_changed_position = orgional_position
            return open_losses, actual_changed_position
        open_losses += (
            np.abs(
                min(
                    0,
                    1 * (mark_price - ask_prices[level] * (1 + long_estimated_rate)),
                )
            )
            * position
        )

        actual_changed_position = orgional_position
    return open_losses, actual_changed_position


def calculate_open_short_close_long_position(
    bid_prices, bid_qtys, position, slient=True
):
    """
    bid_prices and bid_qtys should be form level 1 to level 25
    calculate open real value for opening a short position, the position should be positive,
    indicating the absolute value of the open position
    it output the the actual position change, detemrined both by the position and the orderbooksnapshot.
    as well as the total executed price, excluding commission fee.
    """
    assert position > 0
    assert len(bid_prices) == len(bid_qtys)
    orgional_position = position
    value = 0
    for level in range(len(bid_prices) + 1):
        if level == len(bid_prices) or position <= bid_qtys[level]:
            break
        else:
            position -= bid_qtys[level]
            value += bid_qtys[level] * bid_prices[level]
    if position > min_orderbook and level == len(bid_prices):
        if not slient:
            print(
                "the long position could not be opened all clear since further level of ask price is needed"
            )
        actual_changed_position = orgional_position - position
    else:
        if position <= min_orderbook:
            actual_changed_position = orgional_position
            return value, actual_changed_position
        value += bid_prices[level] * position
        actual_changed_position = orgional_position
    return value, actual_changed_position


def calculate_open_long_close_short_position(
    ask_prices, ask_qtys, position, slient=True
):
    """
    ask_prices and ask_qtys should be form level 1 to level 25
    calculate open real value for opening a long position, the position should be positive,
    indicating the absolute value of the open position
    it output the the actual position change, detemrined both by the position and the orderbooksnapshot.
    as well as the total executed price, excluding commission fee.
    """
    assert position > 0
    assert len(ask_prices) == len(ask_qtys)
    orgional_position = position
    # use ask price and size to evaluate
    value = 0
    for level in range(len(ask_prices) + 1):
        if level == len(ask_prices) or position <= ask_qtys[level]:
            break
        else:
            position -= ask_qtys[level]
            value += ask_qtys[level] * ask_prices[level]
    if position > min_orderbook and level == len(ask_prices):
        if not slient:
            print(
                "the long position could not be opened all clear since further level of ask price is needed"
            )
        # 执行的单量
        actual_changed_position = orgional_position - position
    else:
        if position <= min_orderbook:
            actual_changed_position = orgional_position
            return value, actual_changed_position
        value += ask_prices[level] * position
        actual_changed_position = orgional_position
    return value, actual_changed_position


def get_maintenance_margin(
    value,
    maintenance_margin_ratio_dict={
        "50000": [0.004, 0],
        "500000": [0.005, 50],
        "10000000": [0.01, 2550],
    },
):
    # 转换给定的字典键为整型以便比较
    keys_as_int = sorted([int(key) for key in maintenance_margin_ratio_dict.keys()])
    level_maintenance = keys_as_int
    # 根据值返回对应的维持保证金率
    for key in keys_as_int:
        if value < key:
            return maintenance_margin_ratio_dict[str(key)]
    return maintenance_margin_ratio_dict[str(keys_as_int[-1])]


def calculate_maintenance_margin(
    value,
    maintenance_margin_ratio_dict={
        "50000": [0.004, 0],
        "500000": [0.005, 50],
        "10000000": [0.01, 2550],
    },
):
    ratio, sub = get_maintenance_margin(value, maintenance_margin_ratio_dict)
    holding_margine = value * ratio - sub
    return holding_margine


def judge_liquidation(
    markprice,
    position,
    unrealized_pnL,
    wallet_balance,
    maintenance_margin_ratio_dict={
        "50000": [0.004, 0],
        "500000": [0.005, 50],
        "10000000": [0.01, 2550],
    },
):
    margine_balance = wallet_balance + unrealized_pnL
    esitmated_holding = np.abs(markprice * position)
    ratio, sub = get_maintenance_margin(
        esitmated_holding, maintenance_margin_ratio_dict
    )
    holding_margine = esitmated_holding * ratio - sub
    if margine_balance <= holding_margine:
        return True
    else:
        return False


def calculate_avaiable_action(
    markprice,
    ask_prices,
    ask_qtys,
    bid_prices,
    bid_qtys,
    long_estimated_rate,
    short_estimated_rate,
    commission_rate,
    # before action
    leverage,
    position,
    initial_margine,
    unrealized_pnL,
    wallet_balance,
    # action space setting
    leverage_choices=[1, 2, 5],
    position_choices=[-8, -6, -4, -2, 0, 2, 4, 6, 8],
):
    # TODO modify that if the margine balance equals to dilvering the liqudation, then the action is not actually avaiable
    assert leverage in leverage_choices
    if position not in position_choices:
        print("position not in position_choices", position)
    assert position in position_choices

    assert initial_margine == np.abs(markprice * position / leverage)
    # repetition is allowed in the following two list
    # they should have the same length and each correponding element in the pair should be corresponding to avaliable action
    avaiable_position_choices = []
    avaiable_leverage_choices = []

    buy_size_max = np.sum(ask_qtys)
    sell_size_max = np.sum(bid_qtys)
    # 由orderbook限定的范围
    position_upper = min(position + buy_size_max, max(position_choices))
    position_lower = max(position - sell_size_max, min(position_choices))
    # 由position限定的范围
    if position == 0:
        position_upper = position_upper
        position_lower = position_lower
    elif position > 0:
        position_lower = max(0, position_lower)
    else:
        position_upper = min(0, position_upper)

    available_positions = [
        position
        for position in position_choices
        if position_upper >= position >= position_lower
    ]
    # 由当前leverage限定的范围
    margine_balalnce = wallet_balance + unrealized_pnL
    for available_position in available_positions:
        if available_position == position:
            # 只换杠杆
            for available_leverage in leverage_choices:
                if available_leverage == leverage:
                    avaiable_position_choices.append(available_position)
                    avaiable_leverage_choices.append(available_leverage)
                elif margine_balalnce >= np.abs(
                    markprice * available_position / available_leverage
                ):
                    avaiable_position_choices.append(available_position)
                    avaiable_leverage_choices.append(available_leverage)
        elif np.abs(available_position) > np.abs(position):
            # 开仓，一般先换杠杆后开仓
            for available_leverage in leverage_choices:
                # 同杠杆或者换杠杆后margine balance足够继续
                if available_leverage == leverage or margine_balalnce >= np.abs(
                    markprice * position / available_leverage
                ):
                    new_available_balance = margine_balalnce - np.abs(
                        markprice * position / available_leverage
                    )
                    open_position = np.abs(available_position - position)
                    if available_position > 0:
                        # 开多
                        assert position >= 0
                        open_losses, _ = calculate_open_long_loss(
                            ask_prices,
                            ask_qtys,
                            open_position,
                            markprice,
                            long_estimated_rate,
                        )
                    else:
                        # 开空
                        assert position <= 0
                        open_losses, _ = calculate_open_short_loss(
                            bid_prices,
                            bid_qtys,
                            open_position,
                            markprice,
                            short_estimated_rate,
                        )
                    assert open_position > 0
                    new_margine = open_position * markprice / available_leverage
                    if new_available_balance >= open_losses + new_margine:
                        avaiable_position_choices.append(available_position)
                        avaiable_leverage_choices.append(available_leverage)
        else:
            # 平仓，一般先平仓后换杠杆
            if position > 0:
                # 平多
                assert available_position >= 0
                close_position = np.abs(available_position - position)
                value, _ = calculate_open_short_close_long_position(
                    bid_prices, bid_qtys, close_position
                )
                unrealized_pnL_new = unrealized_pnL * available_position / position
                realized_pnL = (
                    unrealized_pnL * np.abs(close_position / position)
                    + value
                    - markprice * close_position
                    - value * commission_rate
                )

            else:
                # 平空
                assert available_position <= 0
                close_position = np.abs(available_position - position)
                value, _ = calculate_open_long_close_short_position(
                    ask_prices, ask_qtys, close_position
                )
                unrealized_pnL_new = unrealized_pnL * available_position / position
                realized_pnL = (
                    unrealized_pnL * np.abs(close_position / position)
                    - value
                    + markprice * close_position
                    - value * commission_rate
                )
            wallet_balance_new = wallet_balance + realized_pnL
            margine_balance_new = wallet_balance_new + unrealized_pnL_new
            for available_leverage in leverage_choices:
                # 同杠杆或者换杠杆后margine balance足够继续
                if available_leverage == leverage or margine_balance_new >= np.abs(
                    markprice * available_position / available_leverage
                ):
                    avaiable_position_choices.append(available_position)
                    avaiable_leverage_choices.append(available_leverage)
    return avaiable_position_choices, avaiable_leverage_choices


def map_action_to_position_leverage(action, leverage_choices, position_list):
    leverage_length = len(leverage_choices)
    position_length = len(position_list)
    zero_position_action = leverage_length * (position_length // 2)
    if action == zero_position_action:
        return 0, leverage_choices[0]
    elif action > zero_position_action:
        action = action + leverage_length - 1
    else:
        action = action
    # 返回对应的仓位和杠杆倍率
    position_index = action // len(leverage_choices)
    leverage_index = action % len(leverage_choices)
    position = position_list[position_index]
    leverage = leverage_choices[leverage_index]
    return position, leverage


def map_position_leverage_to_action(
    position, leverage, leverage_choices, position_list
):
    # 找到 position 在 position_list 中的索引
    position_index = position_list.index(position)

    # 找到 leverage 在 leverage_choices 中的索引
    leverage_index = leverage_choices.index(leverage)

    # 计算并返回 action
    if position < 0:
        action = position_index * len(leverage_choices) + leverage_index
    elif position == 0:
        action = (position_index) * len(leverage_choices)
    else:
        action = (position_index - 1) * len(leverage_choices) + leverage_index + 1
    return action


# 不考虑现金流的损失，只做reward最优切换，也不考虑杠杆，直接做大，只调仓
def create_optimal_q_table(
    ask_prices_array,
    bid_prices_array,
    ask_qtys_array,
    bid_qtys_array,
    markprice_array,
    timestamp_array,
    funding_rate_array,
    funding_timestamp_array,
    max_holding_number=8,
    position_choices=9,  # (must be an odd number, the minum of trading equals to (max_holder_number)/((action_dim-1)/2)s))
    leverage_choice=[
        5
    ],  # recommend only use one leverage choice, because the leverage does not influence the return directly, the position
    # itself is enough to show the risk preference
    long_estimated_rate=0.0005,
    short_estimated_rate=0,
    commission_rate=0.0002,
    # the default is for btcusdt perpetual contract
    max_punishment=1e10,
    gamma=1,
):
    assert (
        len(ask_prices_array)
        == len(bid_prices_array)
        == len(ask_qtys_array)
        == len(bid_qtys_array)
        == len(markprice_array)
        == len(timestamp_array)
        == len(funding_rate_array)
        == len(funding_timestamp_array)
    )
    total_length = len(ask_prices_array)

    num_action = (position_choices - 1) * len(leverage_choice) + 1

    single_side_action_num = int((position_choices - 1) / 2)
    position_list = (
        [
            max_holding_number / single_side_action_num * i
            for i in range(1, single_side_action_num + 1)
        ]
        + [0]
        + [
            max_holding_number / single_side_action_num * -i
            for i in range(1, single_side_action_num + 1)
        ]
    )
    position_list.sort()
    q_table = np.zeros((total_length, num_action, num_action))
    # max punishment 为了限制单边量变化超过snapshot提供的量以及仓量反转的情况
    for t in range(2, total_length + 1):
        # variable initialization
        current_timestamp_index = -t
        future_timestamp_index = -t + 1

        current_timestamp = timestamp_array[current_timestamp_index]
        future_timestamp = timestamp_array[future_timestamp_index]

        current_markprice = markprice_array[current_timestamp_index]
        future_markprice = markprice_array[future_timestamp_index]

        current_funding_rate = funding_rate_array[current_timestamp_index]
        future_funding_rate = funding_rate_array[future_timestamp_index]

        current_funding_timestamp = funding_timestamp_array[current_timestamp_index]
        future_funding_timestamp = funding_timestamp_array[future_timestamp_index]

        current_ask_prices = ask_prices_array[current_timestamp_index]
        future_ask_prices = ask_prices_array[future_timestamp_index]

        current_bid_prices = bid_prices_array[current_timestamp_index]
        future_bid_prices = bid_prices_array[future_timestamp_index]

        current_ask_qtys = ask_qtys_array[current_timestamp_index]
        future_ask_qtys = ask_qtys_array[future_timestamp_index]

        current_bid_qtys = bid_qtys_array[current_timestamp_index]
        future_bid_qtys = bid_qtys_array[future_timestamp_index]
        for current_action in range(num_action):
            current_position, current_leverage = map_action_to_position_leverage(
                current_action, leverage_choice, position_list
            )
            previous_initial_margine = np.abs(
                current_markprice * current_position / current_leverage
            )
            previous_unrealized_pnL = 0
            previous_wallet_balance = 1e5
            for future_action in range(num_action):

                future_position, future_leverage = map_action_to_position_leverage(
                    future_action, leverage_choice, position_list
                )
                if future_position * current_position < 0:
                    #   仓位反转，直接惩罚
                    q_table[current_timestamp_index, current_action, future_action] = (
                        -max_punishment
                    )
                else:
                    previous_margine_balance = (
                        previous_wallet_balance + previous_unrealized_pnL
                    )
                    # 仓位未反转，计算reward
                    (
                        changed_leverage,
                        changed_position,
                        current_initial_margine,
                        current_unrealized_pnL,
                        current_wallet_balance,
                        slippage,
                    ) = change_of_wallet(
                        markprice=current_markprice,
                        ask_prices=current_ask_prices,
                        ask_qtys=current_ask_qtys,
                        bid_prices=current_bid_prices,
                        bid_qtys=current_bid_qtys,
                        long_estimated_rate=long_estimated_rate,
                        short_estimated_rate=short_estimated_rate,
                        commission_rate=commission_rate,
                        # before action
                        previous_leverage=current_leverage,
                        previous_position=current_position,
                        previous_initial_margine=previous_initial_margine,
                        previous_unrealized_pnL=previous_unrealized_pnL,
                        previous_wallet_balance=previous_wallet_balance,
                        # target after the action
                        current_leverage=future_leverage,
                        current_position=future_position,
                        silent=True,
                    )
                    if (
                        changed_leverage != future_leverage
                        or changed_position != future_position
                    ):
                        q_table[
                            current_timestamp_index, current_action, future_action
                        ] = -max_punishment
                    else:
                        if future_timestamp == current_funding_timestamp:
                            current_wallet_balance = (
                                current_wallet_balance
                                - future_position
                                * future_markprice
                                * future_funding_rate
                            )
                        current_unrealized_pnL = (
                            changed_position * (future_markprice - current_markprice)
                            + current_unrealized_pnL
                        )
                        current_margine_balance = (
                            current_wallet_balance + current_unrealized_pnL
                        )
                        reward = current_margine_balance - previous_margine_balance
                        q_table[
                            current_timestamp_index, current_action, future_action
                        ] = reward + gamma * np.max(
                            q_table[future_timestamp_index][future_action][:]
                        )
    return q_table


def create_optimal_q_table_from_df(
    df: pd.DataFrame,
    max_holding_number=8,
    position_choices=9,  # (must be an odd number, the minum of trading equals to (max_holder_number)/((action_dim-1)/2)s))
    leverage_choice=[
        5
    ],  # recommend only use one leverage choice, because the leverage does not influence the return directly, the position
    # itself is enough to show the risk preference
    long_estimated_rate=0.0005,
    short_estimated_rate=0,
    commission_rate=0.0002,
    # the default is for btcusdt perpetual contract
    max_punishment=1e10,
    gamma=1,
):
    bid_prices_names = ["bid{}_price".format(i) for i in range(1, 26)]
    ask_prices_names = ["ask{}_price".format(i) for i in range(1, 26)]
    bid_sizes_names = ["bid{}_size".format(i) for i in range(1, 26)]
    ask_sizes_names = ["ask{}_size".format(i) for i in range(1, 26)]

    markprice_array = df["mark_price"].values
    timestamp_array = df["timestamp"].values
    funding_rate_array = df["funding_rate"].values
    funding_timestamp_array = df["funding_timestamp"].values
    ask_prices_array = df[ask_prices_names].values
    bid_prices_array = df[bid_prices_names].values
    ask_qtys_array = df[ask_sizes_names].values
    bid_qtys_array = df[bid_sizes_names].values
    return create_optimal_q_table(
        ask_prices_array,
        bid_prices_array,
        ask_qtys_array,
        bid_qtys_array,
        markprice_array,
        timestamp_array,
        funding_rate_array,
        funding_timestamp_array,
        max_holding_number,
        position_choices,  # (must be an odd number, the minum of trading equals to (max_holder_number)/((action_dim-1)/2)s))
        leverage_choice,  # recommend only use one leverage choice, because the leverage does not influence the return directly, the position
        # itself is enough to show the risk preference
        long_estimated_rate,
        short_estimated_rate,
        commission_rate,
        # the default is for btcusdt perpetual contract
        max_punishment,
        gamma,
    )


def get_dp_action_from_qtable(q_table, initial_action):
    action_list = []
    initial_q_table = q_table[0][initial_action][:]
    first_action = np.argmax(initial_q_table)
    action_list.append(first_action)
    for i in range(1, len(q_table)):
        first_action = np.argmax(q_table[i, first_action, :])
        action_list.append(first_action)
    return action_list


# for rejection state policy


def find_nearest_number(target, position_list):
    if target > 0:
        best_match = None  # 寻找最大的比target小的数
        comparison_function = lambda x, best: x < target and (best is None or x > best)
    elif target < 0:
        best_match = None  # 寻找最小的比target大的数
        comparison_function = lambda x, best: x > target and (best is None or x < best)
    else:
        return 0
    for num in position_list:
        # 使用lambda函数来决定是否更新best_match
        if comparison_function(num, best_match):
            best_match = num

    # 返回找到的数，如果没有找到则返回None
    return best_match


def find_closest_action_target(
    available_actions, target_position, leverage, leverage_choices, position_list
):
    # 检查zero_position_action是否在列表中
    target_position = find_nearest_number(target_position, position_list)
    wanted_action = map_position_leverage_to_action(
        target_position, leverage, leverage_choices, position_list
    )
    if wanted_action in available_actions:
        return wanted_action

    # 计算每个元素与zero_position_action的差的绝对值
    differences = {action: abs(action - wanted_action) for action in available_actions}

    # 找到最小差值对应的动作
    closest_action = min(differences, key=differences.get)

    return closest_action


def rule_based_close(info, zero_position_action, leverage_choices, position_list):
    # one step close position (step by step)
    ask_qyts = info["ask_qyts"]
    bid_qyts = info["bid_qyts"]
    bid1_size = bid_qyts[0]
    ask1_size = ask_qyts[0]
    current_position, current_leverage = (
        info["personal_state"][-2],
        info["personal_state"][-1],
    )
    if current_position > 0:
        if current_position <= bid1_size:
            action = zero_position_action
        else:
            target_position = current_position - bid1_size
            action = find_closest_action_target(
                info["avaiable_action_list"],
                target_position,
                current_leverage,
                leverage_choices,
                position_list,
            )
    else:
        if current_position >= -ask1_size:
            action = zero_position_action
        else:
            target_position = current_position + ask1_size
            action = find_closest_action_target(
                info["avaiable_action_list"],
                target_position,
                current_leverage,
                leverage_choices,
                position_list,
            )
    return action
