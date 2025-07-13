import pandas as pd
import numpy as np
import argparse
import os
import torch

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"
import sys

sys.path.append(".")
from model.low_level import create_new_ensemble_qnet_from_different_save_path

# * analysis the result of low level agent, consider the std along with the mean of the reward
# * the analysis is cater to the choosing a single agent that can handle well different dynamics using different preferenced number
# TODO for each dynamics, pick the agent with the highest reward sum and least std across different position
parser = argparse.ArgumentParser()
# replay buffer coffient
parser.add_argument(
    "--dataset_name",
    type=str,
    default="BTCUSDT",
    help="the number of transcation we store in one memory",
)

parser.add_argument(
    "--num_label",
    type=int,
    default=5,
    help="the number of label",
)
parser.add_argument(
    "--epoch_num",
    type=int,
    default=50,
    help="the number of initial_position",
)
parser.add_argument(
    "--initial_position",
    type=int,
    default=9,
    help="the number of initial_position",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="analysis_result/DiHFT/low_level",
    help="the number of initial_position",
)
parser.add_argument(
    "--model_save_path",
    type=str,
    default="result/DiHFT/potential_model",
    help="the number of initial_position",
)
parser.add_argument(
    "--std_preference",
    type=float,
    default=0.1,
    help="the number of initial_position",
)


class picker:
    def __init__(self, args) -> None:
        self.dataset_name = args.dataset_name
        self.num_label = args.num_label
        self.num_initial_position = args.initial_position
        self.label_list = ["label_{}".format(i) for i in range(self.num_label)]
        self.initial_position_list = range(self.num_initial_position)

        self.epoch_num = args.epoch_num
        self.save_path = args.save_path
        self.model_save_path = os.path.join(args.model_save_path, args.dataset_name)
        self.std_preference = args.std_preference

    def transform_single_epoch_result(self, result, epoch_path):
        # calculate the mean and std of the normalized return for each record and throw away the original return and length record
        new_result = []
        for single_result in result:
            single_result["normalized_reward"] = np.array(
                single_result["reward_sum"]
            ) / np.array(single_result["df_length"])
            single_result["trans_reward_mean"] = np.mean(
                single_result["normalized_reward"]
            )
            single_result["trans_reward_std"] = np.std(
                single_result["normalized_reward"]
            )
            single_result["mean_turnover"] = np.mean(single_result["turnover"])
            single_result.pop("normalized_reward")
            single_result.pop("reward_sum")
            single_result.pop("df_length")
            single_result.pop("turnover")
            single_result["epoch_path"] = epoch_path
            new_result.append(single_result)
        return new_result

    def conclude_single_parameter(self, parameter_path):
        single_parameter_result = []
        single_parameter_best_result = []
        for i in range(self.epoch_num):
            epoch_path = os.path.join(parameter_path, "epoch_{}".format(i + 1))
            best_result, result = self.analysis_single_epoch(epoch_path)
            single_parameter_result.extend(result)
            single_parameter_best_result.extend(best_result)
        return single_parameter_result, single_parameter_best_result

    def pick_best_index_from_single_epoch(
        self,
        result,
        epoch_path,
    ):
        # 找到这个epoch各种dynamics和initial position下最好的agent
        max_result = []
        for label in self.label_list:
            for initial_action in self.initial_position_list:
                single_condition_result = []
                for single_result in result:
                    if (
                        single_result["initial_action"] == initial_action
                        and single_result["label"] == label
                    ):
                        single_condition_result.append(single_result)
                max_item = max(
                    single_condition_result,
                    key=lambda x: x["trans_reward_mean"]
                    - self.std_preference * x["trans_reward_std"],
                )
                max_item["epoch_path"] = epoch_path
                max_result.append(max_item)
        return max_result

    def analysis_single_epoch(
        self,
        epoch_path,
    ):
        result = np.load(
            os.path.join(epoch_path, "analysis_result.npy"), allow_pickle=True
        )
        result = self.transform_single_epoch_result(result, epoch_path)
        result_best_single_agent = self.pick_best_index_from_single_epoch(
            result, epoch_path
        )
        return result_best_single_agent, result

    def conclude_all_parameter(self, root_path):
        parameter_list = os.listdir(root_path)
        all_parameter_result_all = []
        all_parameter_result_best = []
        for parameter in parameter_list:
            parameter_path = os.path.join(root_path, parameter)
            single_parameter_result, single_parameter_best_result = (
                self.conclude_single_parameter(parameter_path)
            )
            all_parameter_result_all.extend(single_parameter_result)
            all_parameter_result_best.extend(single_parameter_best_result)
        return all_parameter_result_all, all_parameter_result_best

    def get_all_parameter_result(self):
        root_path = os.path.join("result", "DiHFT", "low_level", self.dataset_name)
        all_parameter_result, all_parameter_result_best = self.conclude_all_parameter(
            root_path
        )
        df_best = pd.DataFrame(all_parameter_result_best)
        df_all = pd.DataFrame(all_parameter_result)
        df_all["epoch_number"] = (
            df_all["epoch_path"].str.extract(r"epoch_(\d+)").astype(int)
        )

        df_all = df_all.sort_values(
            by=["epoch_number", "label", "initial_action", "bin_index"],
            ascending=[True, True, True, True],
        )
        df_all = df_all.drop(columns="epoch_number")

        if not os.path.exists(os.path.join(self.save_path, self.dataset_name)):
            os.makedirs(os.path.join(self.save_path, self.dataset_name))
        df_best.to_csv(os.path.join(self.save_path, self.dataset_name, "result.csv"))
        df_all.to_csv(os.path.join(self.save_path, self.dataset_name, "result_all.csv"))
        self.result_df_best = df_best
        self.result_df_all = df_all
        return df_best, df_all

    def pick_best_agent_regarding_dynamics_bin_index_path(self, result_all):
        label_list = []
        epoch_path_list = []
        bin_index_list = []
        reward_max_list = []
        for label in result_all["label"].unique():
            print(label)
            selected_df = result_all[result_all["label"] == label]
            reward_mean_info = selected_information_based_reward_sum = (
                selected_df.groupby(["label", "bin_index", "epoch_path"])[
                    "trans_reward_mean"
                ].mean()
            )
            selected_information_based_reward_sum = reward_mean_info.idxmax()
            label = selected_information_based_reward_sum[0]
            bin_index = selected_information_based_reward_sum[1]
            epoch_path = selected_information_based_reward_sum[2]
            reward_max = reward_mean_info.max()
            label_list.append(label)
            epoch_path_list.append(epoch_path)
            bin_index_list.append(bin_index)
            reward_max_list.append(reward_max)
        best_agent_info = pd.DataFrame(
            {
                "label": label_list,
                "epoch_path": epoch_path_list,
                "bin_index": bin_index_list,
                "reward_max": reward_max_list,
            }
        )
        best_agent_info.to_csv(
            os.path.join(
                self.save_path,
                self.dataset_name,
                "best_index_info_by_dynamics_with_different_position.csv",
            )
        )
        return best_agent_info

    def create_potential_result(self, best_agent_df):
        n_state = len(
            np.load(os.path.join("dataset", self.dataset_name, "state_features.npy"))
        )
        n_action = 9
        n_hidden = 128
        label_list = best_agent_df["label"].unique().tolist()
        epoch_path_list = best_agent_df["epoch_path"].tolist()
        epoch_path_list = [
            os.path.join(epoch_path, "trained_model.pkl")
            for epoch_path in epoch_path_list
        ]
        bin_index_list = best_agent_df["bin_index"].tolist()
        assert len(label_list) == len(epoch_path_list) == len(bin_index_list)
        new_ensemble = create_new_ensemble_qnet_from_different_save_path(
            n_state,
            n_action,
            n_hidden,
            2,
            epoch_path_list,
            bin_index_list,
        )
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        torch.save(
            new_ensemble.state_dict(),
            os.path.join(self.model_save_path, "model.pth"),
        )


if __name__ == "__main__":
    args = parser.parse_args()
    dataset_name = args.dataset_name
    p = picker(args)
    single_parameter_result_best, single_parameter_result_all = (
        p.get_all_parameter_result()
    )

    df = pd.read_csv(
        "analysis_result/DiHFT/low_level/{}/result.csv".format(dataset_name),
        index_col=0,
    )
    df_all = pd.read_csv(
        "analysis_result/DiHFT/low_level/{}/result_all.csv".format(dataset_name),
        index_col=0,
    )
    best_agent_info = p.pick_best_agent_regarding_dynamics_bin_index_path(df_all)
    p.create_potential_result(best_agent_info)
