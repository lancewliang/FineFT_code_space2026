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
    "--algorith_name_list",
    type=list,
    default=[
        "EarnHFT_PES",
        "EarnHFT_Random",
        "FineFT",
        "FineFT_wo_pretrain",
    ],
    help="the number of initial_position",
)
parser.add_argument(
    "--save_agent_path",
    type=str,
    default="result/DiHFT/ablation",
    help="the number of initial_position",
)
parser.add_argument(
    "--conclude_result_path",
    type=str,
    default="analysis_result/DiHFT/ablation/converge_steps",
    help="the number of initial_position",
)

parser.add_argument(
    "--num_label",
    type=int,
    default=5,
    help="the number of label",
)
parser.add_argument(
    "--initial_position",
    type=int,
    default=9,
    help="the number of initial_position",
)
parser.add_argument(
    "--std_preference",
    type=float,
    default=0.1,
    help="the number of initial_position",
)
parser.add_argument(
    "--epoch_num",
    type=int,
    default=100,
    help="the number of initial_position",
)


def get_final_epoch_path(parameter_path, epoch_index):
    """
    根据给定的参数路径和epoch索引，获取最终的epoch路径。
    如果指定的epoch目录下存在子目录，则返回第一个子目录的路径，否则返回原路径。

    参数:
    parameter_path (str): 参数路径的基础路径。
    epoch_index (int): epoch的索引。

    返回:
    str: 最终的epoch路径。
    """
    epoch_path = os.path.join(parameter_path, f"epoch_{epoch_index + 1}")

    # 检查epoch_path是否是一个目录
    if os.path.isdir(epoch_path):
        # 获取epoch_path目录下的所有子目录
        subdirectories = [
            d
            for d in os.listdir(epoch_path)
            if os.path.isdir(os.path.join(epoch_path, d))
        ]

        # 如果存在子目录
        if subdirectories:
            # 选择第一个子目录（根据需要选择逻辑）
            first_subdirectory = subdirectories[0]
            # 更新epoch_path为子目录路径
            epoch_path = os.path.join(epoch_path, first_subdirectory)

    return epoch_path


# first pick agent, then based on the picked agent, calculate the convergence steps
class step_couter:
    def __init__(self, args):
        self.algorith_name_list = args.algorith_name_list
        self.dataset_name = args.dataset_name
        self.save_agent_path = args.save_agent_path
        self.conclude_result_path = args.conclude_result_path
        self.algorithm_path_list = [
            os.path.join(self.save_agent_path, algorithm_name, self.dataset_name)
            for algorithm_name in self.algorith_name_list
        ]
        self.save_algorithm_result_list = [
            os.path.join(self.conclude_result_path, algorithm_name, self.dataset_name)
            for algorithm_name in self.algorith_name_list
        ]
        for path in self.save_algorithm_result_list:
            os.makedirs(path, exist_ok=True)
        self.std_preference = args.std_preference
        self.num_initial_position = args.initial_position
        self.num_label = args.num_label
        self.label_list = ["label_{}".format(i) for i in range(self.num_label)]
        self.initial_position_list = range(self.num_initial_position)
        self.epoch_num = args.epoch_num

    def conclude_single_parameter(self, parameter_path):
        single_parameter_result = []
        single_parameter_best_result = []
        for i in range(self.epoch_num):
            epoch_path = os.path.join(parameter_path, f"epoch_{i+1}")
            if not os.path.exists(epoch_path):
                epoch_path = os.path.join(parameter_path, "seed_12345", f"epoch_{i+1}")
            best_result, result = self.analysis_single_epoch(epoch_path)
            single_parameter_result.extend(result)
            single_parameter_best_result.extend(best_result)
        return single_parameter_result, single_parameter_best_result

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
            single_result.pop("normalized_reward")
            single_result.pop("reward_sum")
            single_result.pop("df_length")
            single_result["epoch_path"] = epoch_path
            new_result.append(single_result)
        return new_result

    # single algorithm path
    def conclude_single_algorithm_path(self, root_path):
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

    def get_all_algorithm_result(self):
        for algorithm_path, save_path in zip(
            self.algorithm_path_list, self.save_algorithm_result_list
        ):
            all_parameter_result, all_parameter_result_best = (
                self.conclude_single_algorithm_path(algorithm_path)
            )
            df_best = pd.DataFrame(all_parameter_result_best)
            df_all = pd.DataFrame(all_parameter_result)
            df_all["epoch_number"] = (
                df_all["epoch_path"].str.extract(r"epoch_(\d+)").astype(int)
            )
            if "bin_index" not in df_all.columns:
                df_all = df_all.sort_values(
                    by=["epoch_number", "label", "initial_action"],
                    ascending=[True, True, True],
                )
            else:
                df_all = df_all.sort_values(
                    by=["epoch_number", "label", "initial_action", "bin_index"],
                    ascending=[True, True, True, True],
                )
            df_all = df_all.drop(columns="epoch_number")

            df_best.to_csv(os.path.join(save_path, "result.csv"))
            df_all.to_csv(os.path.join(save_path, "result_all.csv"))
            self.pick_best_agent(df_best, save_path)
            self.result_df_best = df_best
            self.result_df_all = df_all

    def pick_best_agent(self, best_result, save_path):
        # this best result df is only the best regarding different index, for EarnHFT where the agents are trained sperately,
        # the convergence steps are sumed up if picked from different parameter.
        best_results = []
        for label in self.label_list:
            for action in self.initial_position_list:
                result = best_result[
                    (best_result["label"] == label)
                    & (best_result["initial_action"] == action)
                ]
                max_tr_index = result["trans_reward_mean"].idxmax()
                max_tr_row = result.loc[[max_tr_index]].copy()
                max_tr_row = max_tr_row[
                    [
                        "label",
                        "initial_action",
                        "trans_reward_mean",
                        "trans_reward_std",
                        "epoch_path",
                    ]
                ]
                best_results.append(max_tr_row)
        best_results_df = pd.concat(best_results)
        best_results_df.reset_index(drop=True, inplace=True)
        best_results_df.to_csv(
            os.path.join(save_path, "best_result.csv")
        )  # save the best result


if __name__ == "__main__":
    args = parser.parse_args()
    step_counter = step_couter(args)
    step_counter.get_all_algorithm_result()
