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
        self.MacroHFT_path = args.MacroHFT_path
        self.earnHFT_path = args.earnHFT_path
        self.save_path = os.path.join(args.save_path, "main_result")
        self.table_result_path = os.path.join(args.save_path, "present_result")
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.table_result_path, exist_ok=True)
        self.result_df_list = []

        # plot data
        self.accumulative_reward_list_dict = {}
        self.require_money_dict = {}

    def conclude_single_algorithm_result(self, algorithm_result_path, algorithm_name):
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
        wallet_balance_history = np.load(
            os.path.join(algorithm_result_path, "wallet_balance_history.npy")
        )
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
        if os.path.exists(os.path.join(algorithm_result_path, "reward_history.npy")):
            reward_history = np.load(
                os.path.join(algorithm_result_path, "reward_history.npy")
            )

        if not os.path.exists(
            os.path.join(algorithm_result_path, "total_asset_history.npy")
        ):
            total_asset_history = np.array(
                [wallet_balance_history[0]]
                + np.array(
                    np.cumsum(reward_history) + np.array([wallet_balance_history[0]])
                ).tolist()
            )
        else:
            total_asset_history = np.load(
                os.path.join(algorithm_result_path, "total_asset_history.npy")
            )
        reward_history = calculate_differences(total_asset_history)
        unrealized_pnl_history = np.load(
            os.path.join(algorithm_result_path, "unrealized_pnl_history.npy")
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
        print("the current algorithmi is:", algorithm_name)
        print("the length of the reward history is:", len(reward_history))
        result_dict = {}
        result_dict["name"] = algorithm_name
        result_dict["tr"] = tr
        result_dict["annual_sr"] = annual_sr
        result_dict["daily_cr"] = daily_cr
        result_dict["daily_SoR"] = daily_SoR
        result_dict["daily_vol"] = daily_vol
        result_dict["mdd"] = mdd
        result_dict["downside_deviation_daily"] = downside_deviation_daily

        # accumaltive reward history result
        accummulative_reward_sum = [reward_history[0]]
        for i in range(len(reward_history) - 1):
            accummulative_reward_sum.append(
                accummulative_reward_sum[-1] + reward_history[i + 1]
            )
        self.accumulative_reward_list_dict[algorithm_name] = accummulative_reward_sum
        self.require_money_dict[algorithm_name] = requred_money
        return result_dict

    def conclude_all_algorithms_result_single_dataset(self):
        Finft_path = os.path.join(self.FineFT_path, self.dataset_name)
        EarnHFT_path = os.path.join(self.earnHFT_path, self.dataset_name)
        MacroHFT_path = os.path.join(self.MacroHFT_path, self.dataset_name)
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
        result_df.rename(
            columns={
                "name": "Algorithm",
                "tr": "TR(%)",
                "annual_sr": "Annual SR",
                "daily_cr": "Daily CR",
                "daily_SoR": "Daily SoR",
                "daily_vol": "Daily Vol(%)",
                "mdd": "MDD(%)",
                "downside_deviation_daily": "Downside Deviation(%)",
            },
            inplace=True,
        )
        result_df["TR(%)"] = result_df["TR(%)"] * 100
        result_df["Daily Vol(%)"] = result_df["Daily Vol(%)"] * 100
        result_df["MDD(%)"] = result_df["MDD(%)"] * 100
        result_df = result_df.round(2)
        result_df.to_csv(
            os.path.join(self.table_result_path, "{}.csv".format(self.dataset_name))
        )


if __name__ == "__main__":
    args = parser.parse_args()
    tabler = Tabler(args)
    tabler.conclude_all_algorithms_result_single_dataset()
