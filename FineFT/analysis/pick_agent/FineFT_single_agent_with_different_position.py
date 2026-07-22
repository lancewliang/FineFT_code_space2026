import pandas as pd
import numpy as np
import argparse
import json
import os
import re
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

parser.add_argument(
    "--experiment_name",
    type=str,
    default="default",
    help="experiment name used to namespace serial training outputs",
)
parser.add_argument(
    "--base_path",
    type=str,
    default="dataset",
    help="the number of action we have in the training and testing env",
)
parser.add_argument(
    "--position_choices",
    type=int,
    default=3,
    help="the transcation cost of not holding the same action as before",
)
parser.add_argument(
    "--hidden_nodes",
    type=int,
    default=128,
    help="the number of the hidden nodes",
)

LABEL_PATTERN = re.compile(r"^label_(\d+)$")
ARRAY_FIELDS = ["contract", "df_path", "reward_sum", "df_length", "turnover"]
REQUIRED_RESULT_FIELDS = ["label", "initial_action", "bin_index"] + ARRAY_FIELDS

class picker:
    def __init__(self, args) -> None:
        self.base_path = args.base_path
        self.dataset_name = args.dataset_name
        self.num_label = args.num_label
        print(self.num_label)
        self.num_initial_position = args.initial_position
        self.label_list = ["label_{}".format(i) for i in range(self.num_label)]
        print(self.label_list)
        self.initial_position_list = range(self.num_initial_position)
        self.position_choices = args.position_choices
        self.hidden_nodes = args.hidden_nodes
        
        self.epoch_num = args.epoch_num
        self.save_path = args.save_path
        self.model_save_path = os.path.join(args.model_save_path, args.dataset_name, args.experiment_name)
        self.std_preference = args.std_preference
        self.experiment_name = args.experiment_name

    def _result_output_dir(self):
        return os.path.join(self.save_path, self.dataset_name, self.experiment_name)

    def _label_sort_key(self, label):
        match = LABEL_PATTERN.fullmatch(str(label))
        if not match:
            raise ValueError(f"invalid label {label}; expected label_<integer>")
        return int(match.group(1))

    def _expected_label_set(self):
        return set(self.label_list)

    def _validate_result_record(self, single_result):
        missing_fields = [
            field for field in REQUIRED_RESULT_FIELDS if field not in single_result
        ]
        if missing_fields:
            raise ValueError(
                f"analysis_result record missing fields {missing_fields}; "
                "rerun test_agent_index.py to generate the new schema"
            )
        label = single_result["label"]
        if "/" in str(label) or "\\" in str(label):
            raise ValueError(
                f"legacy label schema {label}; rerun test_agent_index.py "
                "to generate the new schema"
            )
        self._label_sort_key(label)
        lengths = {field: len(single_result[field]) for field in ARRAY_FIELDS}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"aligned array fields have mismatched lengths: {lengths}")
        if lengths["reward_sum"] == 0:
            raise ValueError(f"record for {label} has no validation samples")
        reward_sum = np.asarray(single_result["reward_sum"], dtype=float)
        df_length = np.asarray(single_result["df_length"], dtype=float)
        if np.any(df_length <= 0):
            raise ValueError(f"record for {label} has non-positive df_length")
        if not np.all(np.isfinite(reward_sum)):
            raise ValueError(f"record for {label} has non-finite reward_sum")
        if len(set(single_result["df_path"])) != len(single_result["df_path"]):
            raise ValueError(f"record for {label} has duplicate df_path values")

    def _validate_label_coverage(self, labels):
        actual = set(labels)
        expected = self._expected_label_set()
        if actual != expected:
            missing = sorted(expected - actual, key=self._label_sort_key)
            extra = sorted(actual - expected, key=self._label_sort_key)
            raise ValueError(f"label coverage mismatch; missing={missing}, extra={extra}")

    def transform_single_epoch_result(self, result, epoch_path):
        # calculate the mean and std of the normalized return for each record and throw away the original return and length record
        new_result = []
        print("transform_single_epoch_result {} {}", epoch_path, len(result))
        for single_result in result:
            single_result = dict(single_result)
            self._validate_result_record(single_result)
            reward_sum = np.asarray(single_result["reward_sum"], dtype=float)
            df_length = np.asarray(single_result["df_length"], dtype=float)
            single_result["normalized_reward"] = reward_sum / df_length
            single_result["trans_reward_mean"] = np.mean(
                single_result["normalized_reward"]
            )
            single_result["trans_reward_std"] = np.std(
                single_result["normalized_reward"]
            )
            single_result["mean_turnover"] = np.mean(single_result["turnover"])
            # single_result.pop("normalized_reward")
            # single_result.pop("reward_sum")
            # single_result.pop("df_length")
            # single_result.pop("turnover")
            single_result["epoch_path"] = epoch_path
            new_result.append(single_result)
        return new_result

    def conclude_single_parameter(self, parameter_path):
        single_parameter_result = []
        single_parameter_best_result = []
        for i in range(11,self.epoch_num):
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
        print("pick_best_index_from_single_epoch {} {}", self.label_list, len(self.label_list))
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
                if not single_condition_result:
                    continue
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
        root_path = os.path.join("result", "DiHFT", "low_level", self.dataset_name, self.experiment_name)
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

        if not os.path.exists(os.path.join(self.save_path, self.dataset_name, self.experiment_name)):
            os.makedirs(os.path.join(self.save_path, self.dataset_name, self.experiment_name))
        df_best.to_csv(os.path.join(self.save_path, self.dataset_name, self.experiment_name, "result.csv"))
        df_all.to_csv(os.path.join(self.save_path, self.dataset_name, self.experiment_name, "result_all.csv"))
        self.result_df_best = df_best
        self.result_df_all = df_all
        return df_best, df_all

    def pick_best_agent_regarding_dynamics_bin_index_path(self, result_all):
        self._validate_label_coverage(result_all["label"].unique())
        label_list = []
        epoch_path_list = []
        bin_index_list = []
        reward_max_list = []
        source_rows_list = []
        for label in result_all["label"].unique():
            print(label)
            selected_df = result_all[result_all["label"] == label]
            reward_mean_info = (
                selected_df.groupby(["label", "bin_index", "epoch_path"])[
                    "trans_reward_mean"
                ]
                .agg(["mean", "count"])
                .dropna()
            )
            if reward_mean_info.empty:
                raise ValueError(f"label {label} has no finite selection candidates")
            selected_information_based_reward_sum = reward_mean_info["mean"].idxmax()
            label = selected_information_based_reward_sum[0]
            bin_index = selected_information_based_reward_sum[1]
            epoch_path = selected_information_based_reward_sum[2]
            reward_max = reward_mean_info.loc[
                selected_information_based_reward_sum, "mean"
            ]
            source_rows = int(
                reward_mean_info.loc[selected_information_based_reward_sum, "count"]
            )
            label_list.append(label)
            epoch_path_list.append(epoch_path)
            bin_index_list.append(bin_index)
            reward_max_list.append(reward_max)
            source_rows_list.append(source_rows)
        best_agent_info = pd.DataFrame(
            {
                "label": label_list,
                "epoch_path": epoch_path_list,
                "bin_index": bin_index_list,
                "reward_max": reward_max_list,
                "source_rows": source_rows_list,
            }
        )
        self._validate_label_coverage(best_agent_info["label"].tolist())
        best_agent_info.to_csv(
            os.path.join(
                self.save_path,
                self.dataset_name,
                self.experiment_name,
                "best_index_info_by_dynamics_with_different_position.csv",
            )
        )
        return best_agent_info

    def _ordered_best_agent_df(self, best_agent_df):
        labels = best_agent_df["label"].tolist()
        self._validate_label_coverage(labels)
        if best_agent_df["label"].duplicated().any():
            raise ValueError("each label must have exactly one selected agent")
        ordered_df = best_agent_df.copy()
        ordered_df["_label_index"] = ordered_df["label"].apply(self._label_sort_key)
        ordered_df = ordered_df.sort_values("_label_index").drop(columns="_label_index")
        return ordered_df

    def write_selection_manifest(self, best_agent_df):
        ordered_df = self._ordered_best_agent_df(best_agent_df)
        output_dir = self._result_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        labels = []
        for row in ordered_df.to_dict("records"):
            model_path = os.path.join(row["epoch_path"], "trained_model.pkl")
            labels.append(
                {
                    "label": row["label"],
                    "epoch_path": row["epoch_path"],
                    "model_path": model_path,
                    "bin_index": int(row["bin_index"]),
                    "score": float(row["reward_max"]),
                    "source_rows": int(row.get("source_rows", 0)),
                }
            )
        manifest = {
            "dataset_name": self.dataset_name,
            "experiment_name": self.experiment_name,
            "selection_method": "sample_equal_current_picker_logic",
            "labels": labels,
        }
        manifest_path = os.path.join(output_dir, "selection_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)
        return manifest_path

    def create_potential_result(self, best_agent_df):
        best_agent_df = self._ordered_best_agent_df(best_agent_df)
        n_state = len(
            np.load(os.path.join(self.base_path, self.dataset_name, "state_features.npy"))
        )
        n_action = self.position_choices
        n_hidden = self.hidden_nodes
        label_list = best_agent_df["label"].unique().tolist()
        epoch_path_list = best_agent_df["epoch_path"].tolist()
        epoch_path_list = [
            os.path.join(epoch_path, "trained_model.pkl")
            for epoch_path in epoch_path_list
        ]
        bin_index_list = best_agent_df["bin_index"].tolist()
        assert len(label_list) == len(epoch_path_list) == len(bin_index_list)
        # print(label_list)
        # print(epoch_path_list)        
        # print(bin_index_list)      
        # print(f"self.model_save_path: {self.model_save_path}")
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
        self.write_selection_manifest(best_agent_df)


if __name__ == "__main__":
    args = parser.parse_args()
    dataset_name = args.dataset_name
    p = picker(args)
    single_parameter_result_best, single_parameter_result_all = (
        p.get_all_parameter_result()
    )

    df = pd.read_csv(
        "analysis_result/DiHFT/low_level/{}/{}/result.csv".format(dataset_name, p.experiment_name),
        index_col=0,
    )
    df_all = pd.read_csv(
        "analysis_result/DiHFT/low_level/{}/{}/result_all.csv".format(dataset_name, p.experiment_name),
        index_col=0,
    )
    best_agent_info = p.pick_best_agent_regarding_dynamics_bin_index_path(df_all)
    p.create_potential_result(best_agent_info)
