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
    "--save_path",
    type=str,
    default="analysis_result/base/",
    help="the number of initial_position",
)
parser.add_argument(
    "--topk",
    type=int,
    default=10,
    help="the number of initial_position",
)
parser.add_argument(
    "--selected_rank",
    type=int,
    default=1,
    help="the number of initial_position",
)
parser.add_argument(
    "--epoch_num",
    type=int,
    default=50,
    help="the number of initial_position",
)


class Picker:
    def __init__(self, args) -> None:
        self.algorithm_list = ["cdqn_rp", "dqn", "dra", "ppo"]
        self.dataset_name = args.dataset_name
        self.epoch_num = args.epoch_num
        self.save_path = os.path.join(args.save_path, args.dataset_name)
        self.topk = args.topk
        self.selected_rank = args.selected_rank
        for algorithm_name in self.algorithm_list:
            if not os.path.exists(os.path.join(self.save_path, algorithm_name)):
                os.makedirs(os.path.join(self.save_path, algorithm_name))
        self.result_df_list = []

    def analysis_single_epoch_valid_test(self, epoch_path):
        valid_path = os.path.join(epoch_path, "valid")
        test_path = os.path.join(epoch_path, "test")
        result_dict = {}
        for vt, path in zip(["valid", "test"], [valid_path, test_path]):
            initial_margin_history = np.load(
                os.path.join(path, "initial_margin_history.npy")
            )
            maintain_marigine_history = np.load(
                os.path.join(path, "maintain_marigine_history.npy")
            )
            new_position_required_money_history = np.load(
                os.path.join(path, "new_position_required_money_history.npy")
            )
            micro_action_history = np.load(os.path.join(path, "action.npy"))

            total_asset_history = np.load(os.path.join(path, "total_asset_history.npy"))
            reward_history = calculate_differences(total_asset_history)
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
            (
                tr,
                daily_vol,
                mdd,
                downside_deviation_daily,
                annual_sr,
                daily_cr,
                daily_SoR,
            ) = calculate_metric(requred_money, reward_history, freq=12 * 24)

            result_dict["path"] = epoch_path
            result_dict["tr_{}".format(vt)] = tr
            result_dict["daily_vol_{}".format(vt)] = daily_vol
            result_dict["mdd_{}".format(vt)] = mdd
            result_dict["downside_deviation_daily_{}".format(vt)] = (
                downside_deviation_daily
            )
            result_dict["annual_sr_{}".format(vt)] = annual_sr
            result_dict["daily_cr_{}".format(vt)] = daily_cr
            result_dict["daily_SoR_{}".format(vt)] = daily_SoR
        return result_dict

    def analysis_all_epoch_single_algorithm(self, algorithm_name):
        result_list = []
        model_root_path = "result/base/{}/{}/seed_12345".format(
            algorithm_name,
            self.dataset_name,
        )
        for i in range(self.epoch_num):
            epoch_path = os.path.join(model_root_path, f"epoch_{i+1}")
            result = self.analysis_single_epoch_valid_test(epoch_path)
            result_list.append(result)
        result_df = pd.DataFrame(result_list)
        result_df.to_csv(os.path.join(self.save_path, algorithm_name, "result.csv"))
        self.result_df_list.append(result_df)

    def analysis_all_epoch(self):
        for algorithm_name in self.algorithm_list:
            self.analysis_all_epoch_single_algorithm(algorithm_name)

    def get_top_n(self, df, indicator, n=10, ascending=False):
        sorted_df = df.sort_values(by=indicator, ascending=ascending).head(n).copy()
        sorted_df["rank"] = range(1, n + 1)
        sorted_df["indicator"] = indicator
        return sorted_df

    def analysis_best_epcoh(self):
        for algorithm_name, result_df in zip(self.algorithm_list, self.result_df_list):
            best_results_df = self.analysis_best_epoch_single_algorithm(
                result_df, algorithm_name
            )
            self.create_best_agent_single_algorithm(best_results_df, algorithm_name)
            self.plot_single_algorithm(algorithm_name)

    def analysis_best_epoch_single_algorithm(self, result_df, algorithm_name):
        best_results = []
        max_indicators = [
            "tr_valid",
            "annual_sr_valid",
            "daily_cr_valid",
            "daily_SoR_valid",
        ]
        min_indicators = [
            "daily_vol_valid",
            "mdd_valid",
            "downside_deviation_daily_valid",
        ]
        for indicator in max_indicators:
            top_n_rows = self.get_top_n(
                result_df, indicator, n=self.topk, ascending=False
            )
            best_results.append(top_n_rows)
        for indicator in min_indicators:
            top_n_rows = self.get_top_n(result_df, indicator, n=10, ascending=True)
            best_results.append(top_n_rows)
        best_results_df = pd.concat(best_results)
        best_results_df.to_csv(
            os.path.join(self.save_path, algorithm_name, "best_result.csv")
        )
        return best_results_df

    def create_best_agent_single_algorithm(self, best_results_df, algorithm_name):
        selected_position = self.selected_rank - 1
        path = best_results_df.iloc[selected_position]["path"]
        para = torch.load(os.path.join(path, "trained_model.pkl"))
        initial_margin_history = np.load(
            os.path.join(path, "test", "initial_margin_history.npy")
        )
        maintain_marigine_history = np.load(
            os.path.join(path, "test", "maintain_marigine_history.npy")
        )
        micro_action_history = np.load(os.path.join(path, "test", "action.npy"))
        new_position_required_money_history = np.load(
            os.path.join(path, "test", "new_position_required_money_history.npy")
        )

        total_asset_history = np.load(
            os.path.join(path, "test", "total_asset_history.npy")
        )
        reward_history = calculate_differences(total_asset_history)
        unrealized_pnl_history = np.load(
            os.path.join(path, "test", "unrealized_pnl_history.npy")
        )
        wallet_balance_history = np.load(
            os.path.join(path, "test", "wallet_balance_history.npy")
        )
        high_level_path = "result/base/final_result/{}/{}".format(
            algorithm_name, self.dataset_name
        )
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
            os.path.join(high_level_path, "action.npy"),
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

    def plot_single_algorithm(self, algorithm_name):
        result_dict = {}
        high_level_path = "result/base/final_result/{}/{}".format(
            algorithm_name, self.dataset_name
        )
        initial_margin_history = np.load(
            os.path.join(high_level_path, "initial_margin_history.npy")
        )
        maintain_marigine_history = np.load(
            os.path.join(high_level_path, "maintain_marigine_history.npy")
        )
        micro_action_history = np.load(os.path.join(high_level_path, "action.npy"))
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
        reward_length = len(test_df) - 1
        accummulative_reward_sum = [reward_history[0]]
        for i in range(len(reward_history) - 1):
            accummulative_reward_sum.append(
                accummulative_reward_sum[-1] + reward_history[i + 1]
            )
        accummulative_reward_sum = [0] * (
            reward_length - len(accummulative_reward_sum)
        ) + accummulative_reward_sum
        result_dict[algorithm_name] = accummulative_reward_sum / requred_money
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
        plt.xlabel("Trading Timestamp", size=18)
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
            os.path.join(self.save_path, algorithm_name, "best_result.pdf"),
            bbox_inches="tight",
        )


if __name__ == "__main__":
    args = parser.parse_args()
    picker = Picker(args)
    picker.analysis_all_epoch()
    picker.analysis_best_epcoh()
    print("done")
