import pandas as pd
import numpy as np
import argparse
import os
import torch
import matplotlib.pyplot as plt

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"
import sys

sys.path.append(".")
from analysis.calculate_metric.calculate_metric import (
    calculate_metric,
    calculate_required_money,
)

parser = argparse.ArgumentParser()
# replay buffer coffient
parser.add_argument(
    "--dataset_name",
    type=str,
    default="BTCUSDT",
    help="the number of transcation we store in one memory",
)

parser.add_argument(
    "--context_number",
    type=int,
    default=5,
    help="the number of initial_position",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="analysis_result/DiHFT/single_agent",
    help="the number of initial_position",
)


class Picker:
    def __init__(self, args) -> None:
        self.dataset_name = args.dataset_name
        self.context_number = args.context_number
        self.save_path = os.path.join(args.save_path, args.dataset_name)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.plot_data = {}

    def analysis_single_context(self, context_path):
        initial_margin_history = np.load(
            os.path.join(context_path, "initial_margin_history.npy")
        )
        maintain_marigine_history = np.load(
            os.path.join(context_path, "maintain_marigine_history.npy")
        )
        new_position_required_money_history = np.load(
            os.path.join(context_path, "new_position_required_money_history.npy")
        )
        micro_action_history = np.load(
            os.path.join(context_path, "micro_action_history.npy")
        )
        reward_history = np.load(os.path.join(context_path, "reward_history.npy"))
        total_asset_history = np.load(
            os.path.join(context_path, "total_asset_history.npy")
        )
        unrealized_pnl_history = np.load(
            os.path.join(context_path, "unrealized_pnl_history.npy")
        )
        wallet_balance_history = np.load(
            os.path.join(context_path, "wallet_balance_history.npy")
        )
        requred_money = calculate_required_money(
            initial_margin_history,
            maintain_marigine_history,
            new_position_required_money_history,
            unrealized_pnl_history,
            wallet_balance_history,
        )
        tr, daily_vol, mdd, downside_deviation_daily, annual_sr, daily_cr, daily_SoR = (
            calculate_metric(requred_money, reward_history, freq=12 * 24)
        )
        result_dict = {}
        result_dict["path"] = context_path
        result_dict["context_index"] = context_path.split("/")[-1]
        result_dict["tr"] = tr
        result_dict["daily_vol"] = daily_vol
        result_dict["mdd"] = mdd
        result_dict["downside_deviation_daily"] = downside_deviation_daily
        result_dict["annual_sr"] = annual_sr
        result_dict["daily_cr"] = daily_cr
        result_dict["daily_SoR"] = daily_SoR

        accummulative_reward_sum = [reward_history[0]]
        for i in range(len(reward_history) - 1):
            accummulative_reward_sum.append(
                accummulative_reward_sum[-1] + reward_history[i + 1]
            )

        self.get_plot_data(
            context_index=result_dict["context_index"],
            return_array=accummulative_reward_sum,
            required_money=requred_money,
        )

        return result_dict

    def analysis_all_contexts(self):
        result_list = []
        model_root_path = "result/DiHFT/high_level/{}/single_agent".format(
            self.dataset_name
        )
        for i in range(self.context_number):
            context_path = os.path.join(model_root_path, f"context_{i}")
            result = self.analysis_single_context(context_path)
            result_list.append(result)
        result_df = pd.DataFrame(result_list)
        self.result_df = result_df
        result_df.to_csv(os.path.join(self.save_path, "result.csv"))
        self.result_df = result_df

    def get_plot_data(self, context_index, return_array, required_money):
        self.plot_data[context_index] = return_array / required_money

    def plot(self):
        np.save(os.path.join(self.save_path, "plot_data.npy"), self.plot_data)
        plot_data = self.plot_data
        test_df = pd.read_feather("dataset/{}/test.feather".format(self.dataset_name))
        timestamp = test_df["timestamp"].values
        color_list = [
            "#DCE125",
            "#5086C4",
            "#FFF0BC",
            "#4C6C43",
            "#F0A19A",
            "#7C7CBA",
            "#00A664",
            "#F9ED1D",
        ]
        fig, ax = plt.subplots(figsize=(10, 5))
        for color, context in zip(color_list, plot_data):
            return_rate = plot_data[context] * 100
            plt.plot(
                timestamp[: len(return_rate)], return_rate, label=context, color=color
            )
        plt.xlabel("Trading Timestamp", size=18)
        plt.ylabel("Total Return(%)", size=18)
        plt.grid(ls="--")
        plt.legend(loc="upper center", fontsize=18)
        ax = plt.gca()
        # y 轴用科学记数法
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.16),
            ncol=4,
            fontsize=18,
            frameon=True,
        )
        plt.savefig(os.path.join(self.save_path, "single_agent_return_rate.pdf"))


if __name__ == "__main__":
    args = parser.parse_args()
    picker = Picker(args)
    picker.analysis_all_contexts()
    picker.plot()
