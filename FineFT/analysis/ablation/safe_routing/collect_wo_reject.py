import pandas as pd
import argparse
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle

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
parser.add_argument(
    "--dataset_name",
    type=str,
    default="BTCUSDT",
    help="the number of transcation we store in one memory",
)
parser.add_argument(
    "--save_result_path",
    type=str,
    default="analysis_result/DiHFT/high_level_heurstic_wo_routing",
    help="the number of initial_position",
)
parser.add_argument(
    "--save_agent_path",
    type=str,
    default="result/DiHFT/high_level_wo_risk",
    help="the number of initial_position",
)
parser.add_argument(
    "--early_stop",
    type=int,
    default=0,
    help="the number of initial_position",
)


class Picker:
    def __init__(self, args) -> None:
        self.dataset_name = args.dataset_name
        self.save_result_path = os.path.join(args.save_result_path, args.dataset_name)
        if not os.path.exists(self.save_result_path):
            os.makedirs(self.save_result_path)
        self.save_agent_path = os.path.join(args.save_agent_path, args.dataset_name)
        if not os.path.exists(self.save_result_path):
            os.makedirs(self.save_result_path)

        self.early_stop = args.early_stop

    def analysis_single_epoch(self, epoch_path):
        initial_margin_history = np.load(
            os.path.join(epoch_path, "initial_margin_history.npy")
        )
        maintain_marigine_history = np.load(
            os.path.join(epoch_path, "maintain_marigine_history.npy")
        )
        new_position_required_money_history = np.load(
            os.path.join(epoch_path, "new_position_required_money_history.npy")
        )
        micro_action_history = np.load(
            os.path.join(epoch_path, "micro_action_history.npy")
        )
        reward_history = np.load(os.path.join(epoch_path, "reward_history.npy"))
        total_asset_history = np.load(
            os.path.join(epoch_path, "total_asset_history.npy")
        )
        unrealized_pnl_history = np.load(
            os.path.join(epoch_path, "unrealized_pnl_history.npy")
        )
        wallet_balance_history = np.load(
            os.path.join(epoch_path, "wallet_balance_history.npy")
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
        result_dict["path"] = epoch_path
        result_dict["tr"] = tr
        result_dict["daily_vol"] = daily_vol
        result_dict["mdd"] = mdd
        result_dict["downside_deviation_daily"] = downside_deviation_daily
        result_dict["annual_sr"] = annual_sr
        result_dict["daily_cr"] = daily_cr
        result_dict["daily_SoR"] = daily_SoR
        result_dict["required_money"] = requred_money
        return result_dict

    def analysis_all_epoch(self):
        result_list = []
        model_root_path = "result/DiHFT/high_level/{}/vae_routing_risk_ablation".format(
            self.dataset_name
        )
        parameter_list = os.listdir(model_root_path)
        for parameter in parameter_list:
            epoch_path = os.path.join(model_root_path, parameter)
            if not os.listdir(epoch_path):
                continue
            result = self.analysis_single_epoch(epoch_path)
            result_list.append(result)
        result_df = pd.DataFrame(result_list)
        self.result_df = result_df
        result_df.to_csv(os.path.join(self.save_result_path, "result.csv"))

    def analysis_best_epoch(self):
        best_results = []
        for indicator in ["tr", "annual_sr", "daily_cr", "daily_SoR"]:
            max_tr_index = self.result_df[indicator].idxmax()
            max_tr_row = self.result_df.loc[[max_tr_index]].copy()
            max_tr_row["indicator"] = indicator
            best_results.append(max_tr_row)
        for indicator in ["daily_vol", "mdd", "downside_deviation_daily"]:
            min_index = self.result_df[indicator].idxmin()
            min_row = self.result_df.loc[[min_index]].copy()
            min_row["indicator"] = indicator
            best_results.append(min_row)
        best_results_df = pd.concat(best_results)
        self.best_result_df = best_results_df
        best_results_df.to_csv(os.path.join(self.save_result_path, "best_result.csv"))

    def create_best_agent(self):
        path = self.best_result_df.iloc[0]["path"]
        para = os.path.basename(path)
        initial_margin_history = np.load(
            os.path.join(path, "initial_margin_history.npy")
        )
        maintain_marigine_history = np.load(
            os.path.join(path, "maintain_marigine_history.npy")
        )
        micro_action_history = np.load(os.path.join(path, "micro_action_history.npy"))
        new_position_required_money_history = np.load(
            os.path.join(path, "new_position_required_money_history.npy")
        )
        reward_history = np.load(os.path.join(path, "reward_history.npy"))
        total_asset_history = np.load(os.path.join(path, "total_asset_history.npy"))
        unrealized_pnl_history = np.load(
            os.path.join(path, "unrealized_pnl_history.npy")
        )
        wallet_balance_history = np.load(
            os.path.join(path, "wallet_balance_history.npy")
        )
        high_level_path = self.save_agent_path
        if not os.path.exists(high_level_path):
            os.makedirs(high_level_path)
        with open(
            os.path.join(high_level_path, "high_level_agent_para.txt"), "w"
        ) as file:
            file.write("%s\n" % para)
        np.save(
            os.path.join(high_level_path, "initial_margin_history.npy"),
            initial_margin_history,
        )
        np.save(
            os.path.join(high_level_path, "maintain_marigine_history.npy"),
            maintain_marigine_history,
        )
        np.save(
            os.path.join(high_level_path, "micro_action_history.npy"),
            micro_action_history,
        )
        np.save(
            os.path.join(high_level_path, "new_position_required_money_history.npy"),
            new_position_required_money_history,
        )
        np.save(os.path.join(high_level_path, "reward_history.npy"), reward_history)
        np.save(
            os.path.join(high_level_path, "total_asset_history.npy"),
            total_asset_history,
        )
        np.save(
            os.path.join(high_level_path, "unrealized_pnl_history.npy"),
            unrealized_pnl_history,
        )
        np.save(
            os.path.join(high_level_path, "wallet_balance_history.npy"),
            wallet_balance_history,
        )

    def get_plot_data(self, path):
        # return the pnL for a given path, the path should contain the trading history
        initial_margin_history = np.load(
            os.path.join(path, "initial_margin_history.npy")
        )
        maintain_marigine_history = np.load(
            os.path.join(path, "maintain_marigine_history.npy")
        )
        micro_action_history = np.load(os.path.join(path, "micro_action_history.npy"))
        new_position_required_money_history = np.load(
            os.path.join(path, "new_position_required_money_history.npy")
        )
        reward_history = np.load(os.path.join(path, "reward_history.npy"))
        total_asset_history = np.load(os.path.join(path, "total_asset_history.npy"))
        unrealized_pnl_history = np.load(
            os.path.join(path, "unrealized_pnl_history.npy")
        )
        wallet_balance_history = np.load(
            os.path.join(path, "wallet_balance_history.npy")
        )
        requred_money = calculate_required_money(
            initial_margin_history,
            maintain_marigine_history,
            new_position_required_money_history,
            unrealized_pnl_history,
            wallet_balance_history,
        )
        accummulative_reward_sum = [reward_history[0]]
        for i in range(len(reward_history) - 1):
            accummulative_reward_sum.append(
                accummulative_reward_sum[-1] + reward_history[i + 1]
            )
        return accummulative_reward_sum / requred_money

    def plot(self, single_agent_path_list=[]):
        # involves 3 results final result, result wo risk and single agent result
        result_dict = {}
        test_df = pd.read_feather("dataset/{}/test.feather".format(self.dataset_name))
        timestamp = test_df["timestamp"].values
        # final result path
        final_result_path = "result/DiHFT/final_result/{}".format(self.dataset_name)
        result_dict["FineFT"] = self.get_plot_data(final_result_path)
        wo_reject_path = "result/DiHFT/high_level_wo_risk/{}".format(self.dataset_name)
        result_dict["FineFT_wo_risk"] = self.get_plot_data(wo_reject_path)
        single_agent_path_list = [
            os.path.join(
                "result/DiHFT/high_level",
                self.dataset_name,
                "single_agent",
                "context_{}".format(i),
            )
            for i in range(5)
        ]
        fig, ax = plt.subplots(figsize=(12, 6))
        for name, index, path in zip(
            ["Bear", "Pullback", "Sideways", "Rally", "Bull"],
            range(5),
            single_agent_path_list,
        ):
            result_dict[name] = self.get_plot_data(path)
        with open(
            f"/data2/mlqin/FT_0618/analysis_result/DiHFT/high_level_heurstic_wo_routing/{self.dataset_name}/result_dict.pkl",
            "wb",
        ) as file:
            pickle.dump(result_dict, file)
        color_list = [
            "#DE7833",
            "#912C2C",
            "#F2BB6B",
            "#C2ABC8",
            "#329845",
            "#AED185",
            "#276C9E",
        ]
        for color, context in zip(color_list, result_dict):
            return_rate = result_dict[context] * 100
            plt.plot(
                timestamp[: len(return_rate)], return_rate, label=context, color=color
            )
        plt.xlabel("Trading Timestamp", size=25)
        plt.ylabel("Total Return(%)", size=25)
        ax.tick_params(axis="both", which="major", labelsize=18, width=2, color="black")
        ax.tick_params(
            axis="both", which="minor", labelsize=14, width=1.5, color="grey"
        )

        plt.grid(ls="--")
        plt.legend(loc="upper center", fontsize=21)
        ax = plt.gca()
        # y 轴用科学记数法
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.48, 1.30),
            ncol=4,
            fontsize=20,
            frameon=True,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.save_result_path, "ablation_routing_comparision.pdf"),
            bbox_inches="tight",
        )


if __name__ == "__main__":
    args = parser.parse_args()
    picker = Picker(args)
    picker.analysis_all_epoch()
    picker.analysis_best_epoch()
    picker.create_best_agent()
    picker.plot()
