import numpy as np
import os


def calculate_differences(lst):
    return [lst[i + 1] - lst[i] for i in range(len(lst) - 1)]


def calculate_single_holsing_max_draw_down(
    holding_money_history,
    initial_margin_history,
    maintrain_margine_history,
    new_position_required_moeny_history,
    unrealized_pnl_history,
    wallet_balance_history,
):
    # the input is the return money for each timestamp
    cash_flow_history = np.cumsum(holding_money_history)
    look_back_window = len(holding_money_history)
    # for i in range(1, len(holding_money_history)):
    #     cash_flow_history.append(cash_flow_history[-1] + holding_money_history[i])
    trade_initial_margin_history = np.array(initial_margin_history[-look_back_window:])
    trade_maintrain_margine_history = np.array(
        maintrain_margine_history[-look_back_window:]
    )
    trade_new_position_required_moeny_history = np.array(
        new_position_required_moeny_history[-look_back_window:]
    )
    trade_unrealized_pnl_history = np.array(unrealized_pnl_history[-look_back_window:])
    trade_wallet_balance_history = np.array(wallet_balance_history[-look_back_window:])
    single_trade_require_money = calculate_required_money(
        trade_initial_margin_history,
        trade_maintrain_margine_history,
        trade_new_position_required_moeny_history,
        trade_unrealized_pnl_history,
        trade_wallet_balance_history,
    )
    single_trade_require_money = single_trade_require_money + 1e-12
    total_asset_value_list = (
        single_trade_require_money + np.array(cash_flow_history)
    ).tolist()
    mdd = 0
    peak = total_asset_value_list[0]
    for value in total_asset_value_list:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        if dd > mdd:
            mdd = dd
    return mdd


def calculate_required_money(
    initial_margin_history,
    maintrain_margine_history,
    new_position_required_moeny_history,
    unrealized_pnl_history,
    wallet_balance_history,
):
    assert (
        len(initial_margin_history)
        == len(maintrain_margine_history)
        == len(new_position_required_moeny_history)
        == len(unrealized_pnl_history)
        == len(wallet_balance_history)
    )
    margine_balance_history = wallet_balance_history + unrealized_pnl_history
    avaible_balance_history = margine_balance_history - initial_margin_history
    aboudant_in_open_new_position = min(
        avaible_balance_history[:-1] - new_position_required_moeny_history[1:]
    )
    aboudant_in_matain = min(margine_balance_history - maintrain_margine_history)
    required_money = wallet_balance_history[0] - min(
        aboudant_in_open_new_position, aboudant_in_matain
    )
    return required_money


def calculate_metric(required_money, reward_list, freq=6 * 60 * 24):
    # freq is calculated as the number of groups for the given period in one day, default is 10 seconds
    reward_sum_list = [reward_list[0]]
    for i in range(1, len(reward_list)):
        reward_sum_list.append(reward_sum_list[-1] + reward_list[i])
    total_asset_value_list = (required_money + np.array(reward_sum_list)).tolist()
    return_rate_list = [
        total_asset_value_list[i + 1] / (total_asset_value_list[i] + 1e-12) - 1
        for i in range(len(new_func(total_asset_value_list)) - 1)
    ]
    daily_return_rate_list = [
        total_asset_value_list[i + freq] / (total_asset_value_list[i] + 1e-12) - 1
        for i in range(0, len(total_asset_value_list) - freq, freq)
    ]
    tr = total_asset_value_list[-1] / (total_asset_value_list[0] + 1e-12) - 1
    vol = np.std(return_rate_list)
    daily_vol = np.std(daily_return_rate_list)
    mdd = 0
    peak = total_asset_value_list[0]
    for value in total_asset_value_list:
        if value > peak:
            peak = value
        dd = (peak - value) / (peak + 1e-12)
        if dd > mdd:
            mdd = dd
    negative_second_return_rate_list = [x for x in return_rate_list if x < 0]
    downside_deviation = np.std(negative_second_return_rate_list)
    daily_negative_second_return_rate = [x for x in daily_return_rate_list if x < 0]
    downside_deviation_daily = np.std(daily_negative_second_return_rate)
    sr = np.mean(return_rate_list) / np.std(return_rate_list)
    annual_sr = (
        np.mean(daily_return_rate_list) / np.std(daily_return_rate_list) * np.sqrt(365)
    )
    cr = np.mean(return_rate_list) / mdd
    daily_cr = np.mean(daily_return_rate_list) * 365 / mdd
    SoR = np.mean(return_rate_list) / downside_deviation
    daily_SoR = tr / downside_deviation_daily
    return tr, daily_vol, mdd, downside_deviation_daily, annual_sr, daily_cr, daily_SoR


def calculate_behavior_metric(
    required_money,
    reward_list,
    micro_action_list,
    max_action=8,
    min_action=0,
):
    # return wining rate and turn over
    zero_position_action = (min_action + max_action) / 2
    assert len(reward_list) == len(micro_action_list)
    turn_over = 0
    previous_action = micro_action_list[0]
    for action in micro_action_list:
        if action != previous_action:
            turn_over += abs(action - previous_action) / ((max_action - min_action) / 2)
            previous_action = action
    wining_rate = 0


def new_func(total_asset_value_list):
    return total_asset_value_list


# behavior metric (Wining Rate, Risk-Reward Ratio, Turn Over)
def calculate_wining_rate(micro_action_history, reward_history, zero_position=4):
    single_trade_return_list = []
    single_trade_return = 0
    turn_over = 0
    change_position_times = 0
    previous_action = zero_position
    for micro_action, reward in zip(micro_action_history, reward_history):
        if micro_action != previous_action:
            turn_over += np.abs(previous_action - micro_action) / 4
            change_position_times += 1
        if micro_action == zero_position and single_trade_return != 0:
            single_trade_return += reward
            single_trade_return_list.append(single_trade_return)
            single_trade_return = 0
        else:
            single_trade_return += reward
        previous_action = micro_action
    single_trade_return_list.append(single_trade_return)
    TTN = len(single_trade_return_list)
    TO = turn_over
    TT = change_position_times
    WR = len([i for i in single_trade_return_list if i > 0]) / (
        len(single_trade_return_list) + 1e-12
    )
    negative_returns = [
        negative for negative in single_trade_return_list if negative < 0
    ]
    positive_returns = [
        positive for positive in single_trade_return_list if positive > 0
    ]

    sum_negative_return = np.sum(negative_returns) if negative_returns else 0
    sum_positive_return = np.sum(positive_returns) if positive_returns else 0
    print(sum_negative_return, sum_positive_return)

    mean_negative_return = np.mean(negative_returns) if negative_returns else 0
    mean_positive_return = np.mean(positive_returns) if positive_returns else 0
    print(mean_negative_return, mean_positive_return)

    RRR = sum_positive_return / (np.abs(sum_negative_return) + 1e-12)
    print(RRR)
    ARR = (mean_positive_return) / (np.abs(mean_negative_return) + 1e-12)
    print(ARR)
    return TO, TTN, TT, WR, RRR, ARR


if __name__ == "__main__":
    pass
