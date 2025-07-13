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
from model.low_level import create_new_ensemble_qnet

#! abandon this file, use FineFT_single_agent_with_different_position
# * analysis the result of low level agent, consider the std along with the mean of the reward
# * the analysis is cater to the choosing a single agent that can handle well different dynamics using different preferenced number
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

    def pick_best_agent_single_condition(self, df, initial_action, label):
        df = df[(df["initial_action"] == initial_action) & (df["label"] == label)]
        df = df.sort_values(by="averaged_reward_sum_mean", ascending=False)
        para_path = df.iloc[0]["path"]
        reward_sum_mean = df.iloc[0]["averaged_reward_sum_mean"]
        bin_index = df.iloc[0]["bin_index"]
        return para_path, reward_sum_mean, bin_index

    def pick_best_result_all_condition(self):
        self.best_model_dict_list = []
        for label in self.label_list:
            for initial_action in self.initial_position_list:
                single_best_model_dict = {}
                para_path, reward_sum_mean, bin_index = (
                    self.pick_best_agent_single_condition(
                        self.result_df, initial_action, label
                    )
                )
                single_best_model_dict["label"] = label
                single_best_model_dict["initial_action"] = initial_action
                single_best_model_dict["path"] = para_path
                single_best_model_dict["reward_sum_mean"] = reward_sum_mean
                single_best_model_dict["bin_index"] = bin_index
                self.best_model_dict_list.append(single_best_model_dict)
        df = pd.DataFrame(self.best_model_dict_list)
        df.to_csv(
            os.path.join(self.save_path, self.dataset_name, "best_seperate_result.csv")
        )
        self.best_result_df = df
        return df

    def pick_best_single_agent_regarding_initial_action(self, df):
        df = self.conclude_single_agent(df)
        max_row_list = []
        for initial_action in self.initial_position_list:
            initial_action_df = df[df["initial_action"] == initial_action]
            max_row = initial_action_df.loc[
                initial_action_df["average_std_preferenced_reward_sum"]
                .idxmax() : initial_action_df["average_std_preferenced_reward_sum"]
                .idxmax()
                + 1
            ]
            max_row_list.append(max_row)
        max_row_df = pd.concat(max_row_list, axis=0)
        max_row_df.reset_index(drop=True, inplace=True)
        max_row_df.to_csv(
            os.path.join(
                self.save_path, self.dataset_name, "best_initial_action_result.csv"
            )
        )
        return max_row_df

    def pick_best_single_agent_regarding_all_situation(self, df):
        print("pick best single agent")
        df = self.conclude_single_agent(df)
        epoch_list = df["epoch_path"].unique().tolist()
        agg_df_list = []
        for epoch in epoch_list:
            epoch_df = df[df["epoch_path"] == epoch]
            average_new_value = epoch_df["average_std_preferenced_reward_sum"].mean()
            new_df = pd.DataFrame(
                {
                    "epoch_path": [epoch],
                    "average_std_preferenced_reward_sum_mean": [average_new_value],
                }
            )
            agg_df_list.append(new_df)
        agg_df = pd.concat(agg_df_list, axis=0)
        agg_df.reset_index(drop=True, inplace=True)
        agg_df.to_csv(
            os.path.join(
                self.save_path,
                self.dataset_name,
                "best_all_condition_single_epoch_result.csv",
            )
        )
        return agg_df

    def pick_best_index_within_single_agent_regarding_all_situation(
        self, agg_df, df_all
    ):
        best_agent_index_list = []
        best_single_agent_epoch_path = agg_df.loc[
            agg_df["average_std_preferenced_reward_sum_mean"].idxmax()
        ]["epoch_path"]
        select_result = df_all[df_all["epoch_path"] == best_single_agent_epoch_path]
        label_list = select_result["label"].unique().tolist()
        bin_index_list = select_result["bin_index"].unique().tolist()
        for label in label_list:
            single_label_result_list = []
            for bin_index in bin_index_list:
                single_label_bin_index_result = select_result[
                    (select_result["bin_index"] == bin_index)
                    & (select_result["label"] == label)
                ]
                average_result = np.mean(
                    single_label_bin_index_result["trans_reward_mean"]
                )
                single_label_result_list.append(average_result)
            max_index = np.argmax(single_label_result_list)
            best_single_agent_epoch_result = single_label_bin_index_result = (
                select_result[
                    (select_result["bin_index"] == bin_index_list[max_index])
                    & (select_result["label"] == label)
                ]
            )
            best_agent_index_list.append(best_single_agent_epoch_result)
        best_agent_index_df = pd.concat(best_agent_index_list, axis=0)
        best_agent_index_df.reset_index(drop=True, inplace=True)
        best_agent_index_df.to_csv(
            os.path.join(
                self.save_path, self.dataset_name, "best_single_agent_index_result.csv"
            )
        )
        return best_agent_index_df

    def create_single_agent_all_condition(self, agg_df):
        best_single_agent_epoch_path = agg_df.loc[
            agg_df["average_std_preferenced_reward_sum_mean"].idxmax()
        ]["epoch_path"]
        print("the path is {}".format(best_single_agent_epoch_path))
        print(
            "the reward sum is {}".format(
                agg_df.loc[agg_df["average_std_preferenced_reward_sum_mean"].idxmax()][
                    "average_std_preferenced_reward_sum_mean"
                ]
            )
        )
        para = torch.load(
            os.path.join(best_single_agent_epoch_path, "trained_model.pkl")
        )
        if not os.path.exists(
            os.path.join(self.model_save_path, "single_agent_selection")
        ):
            os.makedirs(os.path.join(self.model_save_path, "single_agent_selection"))
        torch.save(
            para,
            os.path.join(self.model_save_path, "single_agent_selection", "model.pth"),
        )

    def conclude_single_agent(self, df):
        epoch_path_list = df["epoch_path"].unique()
        all_single_epoch_list = []
        for epoch in epoch_path_list:
            df_epoch = df[df["epoch_path"] == epoch].copy()
            for initial_action in self.initial_position_list:
                df_epoch_initial_action = df_epoch[
                    df_epoch["initial_action"] == initial_action
                ].copy()
                df_epoch_initial_action.loc[:, "new_value"] = (
                    df_epoch_initial_action["trans_reward_mean"]
                    - self.std_preference * df_epoch_initial_action["trans_reward_std"]
                )

                average_new_value = df_epoch_initial_action["new_value"].mean()
                new_df = pd.DataFrame(
                    {
                        "epoch_path": [epoch],
                        "initial_action": [initial_action],
                        "average_std_preferenced_reward_sum": [average_new_value],
                    }
                )
                all_single_epoch_list.append(new_df)
        single_epoch_df = pd.concat(all_single_epoch_list, axis=0)
        single_epoch_df.reset_index(drop=True, inplace=True)
        single_epoch_df.to_csv(
            os.path.join(
                self.save_path, self.dataset_name, "best_single_epoch_result.csv"
            )
        )
        return single_epoch_df

    def create_potential_agent(self, best_single_agent_index_result):
        path_list = best_single_agent_index_result["epoch_path"].unique().tolist()
        assert len(path_list) == 1
        print("path_list", path_list)
        label_list = best_single_agent_index_result["label"].unique().tolist()
        index_list = []
        for label in label_list:
            single_label_df = best_single_agent_index_result[
                best_single_agent_index_result["label"] == label
            ]
            bin_index_list = single_label_df["bin_index"].unique().tolist()
            assert len(bin_index_list) == 1
            print("bin_index_list", bin_index_list)
            bin_index = bin_index_list[0]
            index_list.append(bin_index)
        n_state = len(
            np.load(os.path.join("dataset", self.dataset_name, "state_features.npy"))
        )
        n_action = 9
        n_hidden = 128
        saved_model_path = os.path.join(path_list[0], "trained_model.pkl")
        new_ensemble = create_new_ensemble_qnet(
            n_state, n_action, n_hidden, 2, index_list, saved_model_path
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
    agg_df = p.pick_best_single_agent_regarding_all_situation(df)
    best_agent_index_df = p.pick_best_index_within_single_agent_regarding_all_situation(
        agg_df, df_all
    )
    p.create_potential_agent(best_agent_index_df)
