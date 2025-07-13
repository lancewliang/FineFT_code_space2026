import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json

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
    "--base_path",
    type=str,
    default="result/SL",
    help="the number of transcation we store in one memory",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="analysis_result/SL",
    help="the number of transcation we store in one memory",
)


class picker:
    def __init__(self, args) -> None:

        self.dataset_name = args.dataset_name
        self.base_path = os.path.join(args.base_path, self.dataset_name)
        self.save_path = os.path.join(args.save_path, self.dataset_name)
        os.makedirs(self.save_path, exist_ok=True)

    def conclude_single_algorithm_result(self, algorithm_name):
        algorithm_result_path = os.path.join(self.base_path, algorithm_name)
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

        result_dict = {}
        result_dict["name"] = algorithm_name
        result_dict["tr"] = tr
        result_dict["annual_sr"] = annual_sr
        result_dict["daily_cr"] = daily_cr
        result_dict["daily_SoR"] = daily_SoR
        result_dict["daily_vol"] = daily_vol
        result_dict["mdd"] = mdd
        result_dict["downside_deviation_daily"] = downside_deviation_daily
        self.result_dict = result_dict
        with open(os.path.join(self.save_path, f"{algorithm_name}.json"), "w") as file:
            json.dump(result_dict, file)
        # accumaltive reward history result
        accummulative_reward_sum = [reward_history[0]]
        for i in range(len(reward_history) - 1):
            accummulative_reward_sum.append(
                accummulative_reward_sum[-1] + reward_history[i + 1]
            )
        self.accumulative_reward_list = accummulative_reward_sum
        self.require_money = requred_money
        return result_dict

    def plot(self, name):
        test_df = pd.read_feather(
            "dataset/{}/{}.feather".format(self.dataset_name, name)
        )
        length = len(self.accumulative_reward_list)
        timestamp = test_df.timestamp[:length]

        fig, ax = plt.subplots(figsize=(16, 5))

        plt.plot(
            timestamp,
            (np.array(self.accumulative_reward_list) / self.require_money) * 100,
            color="#8ECFC9",
            label="adaboost",
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
            bbox_to_anchor=(0.5, 1.16),
            ncol=3,
            fontsize=18,
            frameon=True,
        )
        plt.savefig(
            os.path.join(self.save_path, "{}.pdf".format(name)), bbox_inches="tight"
        )

    def total_process(self):
        for name in ["valid", "test"]:
            self.conclude_single_algorithm_result(name)
            self.plot(name)


if __name__ == "__main__":
    args = parser.parse_args()
    p = picker(args)
    p.total_process()
