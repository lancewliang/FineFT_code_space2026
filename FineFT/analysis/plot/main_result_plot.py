import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"
import sys

sys.path.append(".")
from analysis.calculate_metric.calculate_metric import (
    calculate_metric,
    calculate_required_money,
    calculate_differences,
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
    "--dataset_name_list",
    type=list,
    default=["BNBUSDT", "BTCUSDT", "DOTUSDT", "ETHUSDT"],
    help="the number of transcation we store in one memory",
)
parser.add_argument(
    "--earnHFT_path",
    type=str,
    default="result/EarnHFT/final_result",
    help="the number of transcation we store in one memory",
)
parser.add_argument(
    "--FineFT_path",
    type=str,
    default="result/DiHFT/final_result",
    help="the number of transcation we store in one memory",
)
parser.add_argument(
    "--base_path",
    type=str,
    default="result/base/final_result",
    help="the number of transcation we store in one memory",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="analysis_result/table_result",
    help="the number of transcation we store in one memory",
)


class ploter:
    def __init__(self, args) -> None:
        self.base_algorithm_list = [
            "cdqn_rp",
            "dqn",
            "dra",
            "ppo",
            "macd",
            "imbalace_volume",
        ]
        self.algorithm_list_dict = {
            "policy_based": ["dra", "ppo"],
            "value_based": ["cdqn_rp", "dqn"],
            "rule_based": ["macd", "imbalace_volume"],
            # "hierarchical_RL": ["FineFT", "EarnHFT", "MacroHFT"],
            "hierarchical_RL": [ "EarnHFT","FineFT",],
        }
        self.color_list_dict = {
            "policy_based": sns.color_palette("Blues", 2),
            "value_based": sns.color_palette("Greens", 2),
            "rule_based": sns.color_palette("Purples", 2),
            "hierarchical_RL": sns.color_palette("Reds", 3),
        }
        self.dataset_name = args.dataset_name
        self.dataset_name_list = args.dataset_name_list
        self.base_path = args.base_path
        self.FineFT_path = args.FineFT_path
        self.earnHFT_path = args.earnHFT_path
        self.plot_path = os.path.join(args.save_path, "plot_result")
        os.makedirs(self.plot_path, exist_ok=True)

        # plot data for selected dataset
        self.accumulative_reward_list_dict = {}
        self.require_money_dict = {}

        # plot data for all datasets
        self.accumulative_reward_list_dict_all = {}
        self.require_money_dict_all = {}

    def collect_single_algorithm_result(self, algorithm_result_path, algorithm_name):
        initial_margin_history = np.load(
            os.path.join(algorithm_result_path, "initial_margin_history.npy")
        )
        maintain_marigine_history = np.load(
            os.path.join(algorithm_result_path, "maintain_marigine_history.npy")
        )
        new_position_required_money_history = np.load(
            os.path.join(
                algorithm_result_path, "new_position_required_money_history.npy"
            )
        )

        reward_history = np.load(
            os.path.join(algorithm_result_path, "reward_history.npy")
        )

        unrealized_pnl_history = np.load(
            os.path.join(algorithm_result_path, "unrealized_pnl_history.npy")
        )
        wallet_balance_history = np.load(
            os.path.join(algorithm_result_path, "wallet_balance_history.npy")
        )
        requred_money = calculate_required_money(
            initial_margin_history,
            maintain_marigine_history,
            new_position_required_money_history,
            unrealized_pnl_history,
            wallet_balance_history,
        )

        # accumaltive reward history result
        accummulative_reward_sum = [reward_history[0]]
        for i in range(len(reward_history) - 1):
            accummulative_reward_sum.append(
                accummulative_reward_sum[-1] + reward_history[i + 1]
            )
        self.accumulative_reward_list_dict[algorithm_name] = accummulative_reward_sum
        self.require_money_dict[algorithm_name] = requred_money

    def conclude_all_algorithms_result_single_dataset(self):
        Finft_path = os.path.join(self.FineFT_path, self.dataset_name)
        EarnHFT_path = os.path.join(self.earnHFT_path, self.dataset_name)
        base_path_list = [
            os.path.join(
                self.base_path,
                algorithm_name,
                self.dataset_name,
            )
            for algorithm_name in self.base_algorithm_list
        ]
        algorithm_name_list = ["FineFT", "EarnHFT"] + self.base_algorithm_list
        algorithm_path_list = [Finft_path, EarnHFT_path] + base_path_list
        for path, name in zip(algorithm_path_list, algorithm_name_list):
            self.collect_single_algorithm_result(path, name)

    def single_plot(self):
        df = pd.read_feather(os.path.join("dataset", self.dataset_name, "test.feather"))
        max_length = max(
            len(arr) for arr in self.accumulative_reward_list_dict.values()
        )
        df = df.iloc[-max_length:]
        timestamp = df.timestamp[:]
        for key, value in self.accumulative_reward_list_dict.items():
            if len(value) < max_length:
                value = [0] * (max_length - len(value)) + value
                self.accumulative_reward_list_dict[key] = value

        fig, ax = plt.subplots(figsize=(16, 5))
        for algo_type in self.color_list_dict:
            color_list = self.color_list_dict[algo_type]
            algorithm_name_list = self.algorithm_list_dict[algo_type]
            for algorithm_name, color in zip(algorithm_name_list, color_list):
                accumlative_reward = self.accumulative_reward_list_dict[algorithm_name]
                require_money = self.require_money_dict[algorithm_name]
                return_data = np.array(accumlative_reward) / require_money
                plt.plot(
                    timestamp,
                    return_data * 100,
                    color=color,
                    label=algorithm_name,
                    linewidth=2,
                )
        plt.xlabel("Trading Timestamp", size=18)
        plt.ylabel("Total Return(%)", size=18)
        plt.grid(ls="--")
        plt.legend(loc="upper center", fontsize=18)
        ax = plt.gca()
        # y 轴用科学记数法
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.4, 1.15, 0.1, 0.1),
            ncol=4,
            fontsize=18,
            frameon=True,
        )
        plt.savefig(
            os.path.join(self.plot_path, "{}.pdf".format(self.dataset_name)),
            bbox_inches="tight",
        )


if __name__ == "__main__":
    plter = ploter(parser.parse_args())
    plter.conclude_all_algorithms_result_single_dataset()
    plter.single_plot()
    print("done")
