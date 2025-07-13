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
from analysis.calculate_metric.calculate_metric import calculate_wining_rate

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
    "--MacroHFT_path",
    type=str,
    default="result/MacroHFT/final_result",
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


class Tabler:
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
            "hierarchical_RL": ["FineFT", "EarnHFT", "MacroHFT"],
            # "hierarchical_RL": ["FineFT", "EarnHFT"],
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
        self.macroHFT_path = args.MacroHFT_path
        self.save_path = os.path.join(args.save_path, "main_behavior_result")
        self.table_result_path = os.path.join(args.save_path, "present_behavior_result")
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.table_result_path, exist_ok=True)
        self.result_df_list = []

        # plot data
        self.accumulative_reward_list_dict = {}
        self.require_money_dict = {}

    def conclude_single_algorithm_result(self, algorithm_result_path, algorithm_name):
        print("algorithm name", algorithm_name)
        if os.path.exists(
            os.path.join(algorithm_result_path, "micro_action_history.npy")
        ):
            micro_action_history = np.load(
                os.path.join(algorithm_result_path, "micro_action_history.npy")
            )
        else:
            micro_action_history = np.load(
                os.path.join(algorithm_result_path, "action.npy")
            )
        reward_history = np.load(
            os.path.join(algorithm_result_path, "reward_history.npy")
        )

        TO, TTN, TT, WR, RRR, ARR = calculate_wining_rate(
            micro_action_history, reward_history, zero_position=4
        )

        result_dict = {}
        result_dict["Algorithm"] = algorithm_name
        result_dict["TO"] = TO
        result_dict["TTN"] = TTN
        result_dict["TT"] = TT
        result_dict["WR(%)"] = WR * 100
        result_dict["RRR(%)"] = RRR * 100
        result_dict["ARR(%)"] = ARR * 100

        return result_dict

    def conclude_all_algorithms_result_single_dataset(self):
        Finft_path = os.path.join(self.FineFT_path, self.dataset_name)
        EarnHFT_path = os.path.join(self.earnHFT_path, self.dataset_name)
        MacroHFT_path = os.path.join(self.macroHFT_path, self.dataset_name)
        base_path_list = [
            os.path.join(
                self.base_path,
                algorithm_name,
                self.dataset_name,
            )
            for algorithm_name in self.base_algorithm_list
        ]
        algorithm_name_list = [
            "FineFT",
            "EarnHFT",
            "MacroHFT",
        ] + self.base_algorithm_list
        algorithm_path_list = [Finft_path, EarnHFT_path, MacroHFT_path] + base_path_list
        for path, name in zip(algorithm_path_list, algorithm_name_list):
            result_dict = self.conclude_single_algorithm_result(path, name)
            self.result_df_list.append(result_dict)
        result_df = pd.DataFrame(self.result_df_list)
        result_df.to_csv(
            os.path.join(self.save_path, "{}.csv".format(self.dataset_name))
        )
        # change to the format more sutiatble for paper presentation

        result_df = result_df.round(2)
        result_df.to_csv(
            os.path.join(self.table_result_path, "{}.csv".format(self.dataset_name))
        )


if __name__ == "__main__":
    args = parser.parse_args()
    tabler = Tabler(args)
    tabler.conclude_all_algorithms_result_single_dataset()
