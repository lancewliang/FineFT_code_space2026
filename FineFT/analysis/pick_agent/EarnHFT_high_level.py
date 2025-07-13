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
    default="DOTUSDT",
    help="the number of transcation we store in one memory",
)

parser.add_argument(
    "--epoch_num",
    type=int,
    default=50,
    help="the number of initial_position",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="analysis_result/EarnHFT/high_level",
    help="the number of initial_position",
)
parser.add_argument(
    "--topk",
    type=int,
    default=30,
    help="the number of initial_position",
)
parser.add_argument(
    "--selected_rank",
    type=int,
    default=1,
    help="the number of initial_position",
)


class Picker:
    def __init__(self, args) -> None:
        self.dataset_name = args.dataset_name
        self.epoch_num = args.epoch_num
        self.save_path = os.path.join(args.save_path, args.dataset_name)
        self.topk = args.topk
        self.selected_rank = args.selected_rank
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

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
        return result_dict

    def analysis_all_epoch(self):
        result_list = []
        model_root_path = "result/EarnHFT/high_level/{}/seed_12345".format(
            self.dataset_name
        )
        for i in range(self.epoch_num):
            epoch_path = os.path.join(model_root_path, f"epoch_{i+1}")
            result = self.analysis_single_epoch(epoch_path)
            result_list.append(result)
        result_df = pd.DataFrame(result_list)
        self.result_df = result_df
        result_df.to_csv(os.path.join(self.save_path, "result.csv"))

    def get_top_n(self, df, indicator, n=10, ascending=False):
        sorted_df = df.sort_values(by=indicator, ascending=ascending).head(n).copy()
        sorted_df["rank"] = range(1, n + 1)
        sorted_df["indicator"] = indicator
        return sorted_df

    def analysis_best_epoch(self):
        best_results = []
        max_indicators = ["tr", "annual_sr", "daily_cr", "daily_SoR"]
        min_indicators = ["daily_vol", "mdd", "downside_deviation_daily"]
        for indicator in max_indicators:
            top_n_rows = self.get_top_n(
                self.result_df, indicator, n=self.topk, ascending=False
            )
            best_results.append(top_n_rows)
        for indicator in min_indicators:
            top_n_rows = self.get_top_n(self.result_df, indicator, n=10, ascending=True)
            best_results.append(top_n_rows)
        best_results_df = pd.concat(best_results)
        self.best_result_df = best_results_df
        best_results_df.to_csv(os.path.join(self.save_path, "best_result.csv"))

    def create_best_agent(self):
        selected_position=self.selected_rank-1
        path = self.best_result_df.iloc[selected_position]["path"]
        para = torch.load(os.path.join(path, "trained_model.pkl"))
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
        high_level_path = "result/EarnHFT/final_result/{}".format(self.dataset_name)
        if not os.path.exists(high_level_path):
            os.makedirs(high_level_path)
        torch.save(para, os.path.join(high_level_path, "high_level_agent.pkl"))
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

    def plot(self):
        result_dict = {}
        high_level_path = "result/EarnHFT/final_result/{}".format(self.dataset_name)
        initial_margin_history = np.load(
            os.path.join(high_level_path, "initial_margin_history.npy")
        )
        maintain_marigine_history = np.load(
            os.path.join(high_level_path, "maintain_marigine_history.npy")
        )
        micro_action_history = np.load(
            os.path.join(high_level_path, "micro_action_history.npy")
        )
        new_position_required_money_history = np.load(
            os.path.join(high_level_path, "new_position_required_money_history.npy")
        )
        reward_history = np.load(os.path.join(high_level_path, "reward_history.npy"))
        total_asset_history = np.load(
            os.path.join(high_level_path, "total_asset_history.npy")
        )
        unrealized_pnl_history = np.load(
            os.path.join(high_level_path, "unrealized_pnl_history.npy")
        )
        wallet_balance_history = np.load(
            os.path.join(high_level_path, "wallet_balance_history.npy")
        )
        requred_money = calculate_required_money(
            initial_margin_history,
            maintain_marigine_history,
            new_position_required_money_history,
            unrealized_pnl_history,
            wallet_balance_history,
        )

        test_df = pd.read_feather("dataset/{}/test.feather".format(self.dataset_name))
        result_dict["Buy & Hold"] = (
            np.array(test_df.mark_price / (test_df.mark_price.iloc[0])) - 1
        )
        accummulative_reward_sum = [reward_history[0]]
        for i in range(len(reward_history) - 1):
            accummulative_reward_sum.append(
                accummulative_reward_sum[-1] + reward_history[i + 1]
            )
        result_dict["EarnHFT"] = accummulative_reward_sum / requred_money
        color_list = ["#002c53", "#ffa510"]
        # graph
        # ax.set_aspect(1)
        fig, ax = plt.subplots(figsize=(16, 5))
        for i, key in enumerate(result_dict.keys()):
            if i == 0:
                plt.plot(
                    test_df.timestamp[:],
                    result_dict[key] * 100,
                    color=color_list[i],
                    label=key,
                    linewidth=2,
                )
            else:
                plt.plot(
                    test_df.timestamp[:-1],
                    result_dict[key] * 100,
                    color=color_list[i],
                    label=key,
                    linewidth=2,
                )
        plt.xlabel("Trading Timestamp(s)", size=18)
        plt.ylabel("Total Return(%)", size=18)
        plt.grid(ls="--")
        plt.legend(loc="upper center", fontsize=18)
        ax = plt.gca()
        # y 轴用科学记数法
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.16),
            ncol=2,
            fontsize=18,
            frameon=True,
        )
        plt.savefig(
            os.path.join(self.save_path, "best_result.pdf"), bbox_inches="tight"
        )


if __name__ == "__main__":
    args = parser.parse_args()
    picker = Picker(args)
    picker.analysis_all_epoch()
    picker.analysis_best_epoch()
    picker.create_best_agent()
    picker.plot()
