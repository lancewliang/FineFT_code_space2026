import pandas as pd
import numpy as np
import argparse
import os
import torch

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"


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
    "--initial_position",
    type=int,
    default=9,
    help="the number of initial_position",
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
    default="analysis_result/EarnHFT/low_level",
    help="the number of initial_position",
)
parser.add_argument(
    "--model_save_path",
    type=str,
    default="result/EarnHFT/potential_model",
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

    def analysis_single_epoch(self, epoch_path):
        result = np.load(
            os.path.join(epoch_path, "analysis_result.npy"), allow_pickle=True
        )
        single_result_dict_list = []
        for r in result:
            found = False
            single_result_dict = {}
            single_result_dict["path"] = epoch_path
            for label in self.label_list:
                if found:  # 如果已找到匹配项，则跳出外层循环
                    break
                for initial_position in self.initial_position_list:
                    if r["label"] == label and r["initial_action"] == initial_position:
                        averaged_reward_sum = np.array(r["reward_sum"]) / np.array(
                            r["df_length"]
                        )
                        averaged_reward_sum_mean = np.mean(averaged_reward_sum)
                        single_result_dict["label"] = label
                        single_result_dict["initial_action"] = initial_position
                        single_result_dict["averaged_reward_sum_mean"] = (
                            averaged_reward_sum_mean
                        )
                        single_result_dict['std']=np.std(averaged_reward_sum)
                        single_result_dict_list.append(single_result_dict)
                        found = True  # 设置标志为True，表示已找到匹配项
                        break  # 找到匹配项，跳出内层循环
        return single_result_dict_list

    def conclude_single_parameter(self, parameter_path):
        single_parameter_result = []
        for i in range(self.epoch_num):
            epoch_path = os.path.join(
                parameter_path, "seed_12345", "epoch_{}".format(i + 1)
            )
            single_parameter_result.extend(self.analysis_single_epoch(epoch_path))
        return single_parameter_result

    def conclude_all_parameter(self, root_path):
        parameter_list = os.listdir(root_path)
        all_parameter_result = []
        for parameter in parameter_list:
            parameter_path = os.path.join(root_path, parameter)
            all_parameter_result.extend(self.conclude_single_parameter(parameter_path))
        return all_parameter_result

    def get_all_parameter_result(self):
        root_path = os.path.join("result", "EarnHFT", "low_level", self.dataset_name)
        all_parameter_result = self.conclude_all_parameter(root_path)
        df = pd.DataFrame(all_parameter_result)
        if not os.path.exists(os.path.join(self.save_path, self.dataset_name)):
            os.makedirs(os.path.join(self.save_path, self.dataset_name))
        df.to_csv(os.path.join(self.save_path, self.dataset_name, "result.csv"))
        self.result_df = df
        return df

    def pick_best_agent_single_condition(self, df, initial_action, label):
        df = df[(df["initial_action"] == initial_action) & (df["label"] == label)]
        df = df.sort_values(by="averaged_reward_sum_mean", ascending=False)
        para_path = df.iloc[0]["path"]
        reward_sum_mean = df.iloc[0]["averaged_reward_sum_mean"]
        std=df.iloc[0]['std']
        return para_path, reward_sum_mean,std

    def pick_best_result_all_condition(self):
        self.best_model_dict_list = []
        for label in self.label_list:
            for initial_action in self.initial_position_list:
                single_best_model_dict = {}
                para_path, reward_sum_mean,std = self.pick_best_agent_single_condition(
                    self.result_df, initial_action, label
                )
                single_best_model_dict["label"] = label
                single_best_model_dict["initial_action"] = initial_action
                single_best_model_dict["path"] = para_path
                single_best_model_dict["reward_sum_mean"] = reward_sum_mean
                single_best_model_dict["std"] = std
                self.best_model_dict_list.append(single_best_model_dict)
        df = pd.DataFrame(self.best_model_dict_list)
        df.to_csv(os.path.join(self.save_path, self.dataset_name, "best_result.csv"))
        self.best_result_df = df
        return df

    def create_potential_model(self):
        for initial_action in self.initial_position_list:
            for i, initial_dynamic in zip(range(len(self.label_list)), self.label_list):
                para_path = self.best_result_df[
                    (self.best_result_df["initial_action"] == initial_action)
                    & (self.best_result_df["label"] == initial_dynamic)
                ]["path"].values.item()
                print(para_path)
                para = torch.load(os.path.join(para_path, "trained_model.pkl"))
                if not os.path.exists(
                    os.path.join(
                        self.model_save_path, "initial_action_{}".format(initial_action)
                    )
                ):
                    os.makedirs(
                        os.path.join(
                            self.model_save_path,
                            "initial_action_{}".format(initial_action),
                        )
                    )
                torch.save(
                    para,
                    os.path.join(
                        self.model_save_path,
                        "initial_action_{}".format(initial_action),
                        "model_{}.pth".format(i),
                    ),
                )


if __name__ == "__main__":
    args = parser.parse_args()
    picker = picker(args)
    df = picker.get_all_parameter_result()
    best_df = picker.pick_best_result_all_condition()
    picker.create_potential_model()
