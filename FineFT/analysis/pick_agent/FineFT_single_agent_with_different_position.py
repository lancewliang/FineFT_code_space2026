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
parser.add_argument(
    "--dataset_name",
    type=str,
    default="BTCUSDT",
    help="dataset name",
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
    help="the number of epoch",
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
    help="the number of hidden nodes",
)
parser.add_argument(
    "--label_semantics_path",
    type=str,
    default=None,
    help="path to label_semantics.json",
)
parser.add_argument(
    "--labeling_method",
    type=str,
    default="slope",
    help="labeling method used: slope, quantile, or DTW",
)

LABEL_PATTERN = re.compile(r"^label_(\d+)$")
ARRAY_FIELDS = [
    "contract",
    "df_path",
    "reward_sum",
    "df_length",
    "turnover",
    "mean_position",
    "mean_abs_position",
    "long_step_ratio",
    "short_step_ratio",
    "flat_step_ratio",
    "long_reward_sum",
    "short_reward_sum",
    "flat_reward_sum",
    "net_position_exposure",
    "limit_up_step_ratio",
    "limit_down_step_ratio",
    "limit_up_long_reward_sum",
    "limit_down_short_reward_sum",
    "limit_up_reverse_short_ratio",
    "limit_down_reverse_long_ratio",
]
REQUIRED_RESULT_FIELDS = ["label", "initial_action", "bin_index"] + ARRAY_FIELDS


class picker:
    def __init__(self, args) -> None:
        self.base_path = getattr(args, "base_path", "dataset")
        self.dataset_name = getattr(args, "dataset_name", "BTCUSDT")
        self.num_label = getattr(args, "num_label", 5)
        self.num_initial_position = getattr(args, "initial_position", 9)
        self.label_list = ["label_{}".format(i) for i in range(self.num_label)]
        self.initial_position_list = range(self.num_initial_position)
        self.position_choices = getattr(args, "position_choices", 3)
        self.hidden_nodes = getattr(args, "hidden_nodes", 128)

        self.epoch_num = getattr(args, "epoch_num", 50)
        self.save_path = getattr(args, "save_path", "analysis_result/DiHFT/low_level")
        self.model_save_path = os.path.join(
            getattr(args, "model_save_path", "result/DiHFT/potential_model"),
            self.dataset_name,
            getattr(args, "experiment_name", "default"),
        )
        self.std_preference = getattr(args, "std_preference", 0.1)
        self.experiment_name = getattr(args, "experiment_name", "default")
        self.label_semantics_path = getattr(args, "label_semantics_path", None)
        self.labeling_method = getattr(args, "labeling_method", "slope")
        self.semantic_filter_thresholds = {
            "min_directional_exposure": 0.10,
            "min_directional_step_ratio": 0.35,
            "max_neutral_abs_exposure": 0.20,
            "max_limit_reverse_ratio": 0.20,
        }
        self.label_semantics = None

    def _result_output_dir(self):
        return os.path.join(self.save_path, self.dataset_name, self.experiment_name)

    def _label_sort_key(self, label):
        match = LABEL_PATTERN.fullmatch(str(label))
        if not match:
            raise ValueError(f"invalid label {label}; expected label_<integer>")
        return int(match.group(1))

    def _expected_label_set(self):
        return set(self.label_list)

    def generate_default_label_semantics(self):
        num_label = self.num_label
        if num_label < 2:
            raise ValueError(f"num_label must be at least 2, got {num_label}")
        labels = []

        # label_0 is limit_down
        labels.append(
            {
                "label": "label_0",
                "direction": "strong_down",
                "direction_sign": -1,
                "strength": 2,
                "description": "跌停",
                "limit_state": "limit_down",
                "limit_state_sign": -1,
            }
        )

        num_middle = num_label - 2
        for i in range(1, num_label - 1):
            lbl_name = f"label_{i}"
            if num_middle == 1:
                direction, direction_sign, desc = "sideways", 0, "震荡"
            else:
                rel_pos = (i - 1) / (num_middle - 1)
                if rel_pos < 0.33:
                    direction, direction_sign, desc = "down", -1, "下跌"
                elif rel_pos > 0.67:
                    direction, direction_sign, desc = "up", 1, "上涨"
                else:
                    direction, direction_sign, desc = "sideways", 0, "震荡"
            labels.append(
                {
                    "label": lbl_name,
                    "direction": direction,
                    "direction_sign": direction_sign,
                    "strength": 1 if direction_sign != 0 else 0,
                    "description": desc,
                    "limit_state": "none",
                    "limit_state_sign": 0,
                }
            )

        # last label is limit_up
        last_lbl = f"label_{num_label - 1}"
        labels.append(
            {
                "label": last_lbl,
                "direction": "strong_up",
                "direction_sign": 1,
                "strength": 2,
                "description": "涨停",
                "limit_state": "limit_up",
                "limit_state_sign": 1,
            }
        )

        manifest_data = {
            "dataset_name": self.dataset_name,
            "labeling_method": self.labeling_method,
            "dynamic_number": num_label - 2,
            "label_number": num_label,
            "labels": labels,
        }
        out_dir = self._result_output_dir()
        os.makedirs(out_dir, exist_ok=True)
        gen_path = os.path.join(out_dir, "label_semantics.json")
        with open(gen_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, ensure_ascii=False, indent=2)
        return manifest_data

    def load_label_semantics(self):
        if self.label_semantics is not None:
            return self.label_semantics

        data = None
        sem_path = self.label_semantics_path
        if sem_path and os.path.exists(sem_path):
            with open(sem_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            default_dataset_sem = os.path.join(
                self.base_path, self.dataset_name, "label_semantics.json"
            )
            if os.path.exists(default_dataset_sem):
                with open(default_dataset_sem, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                if getattr(self, "labeling_method", "slope") == "DTW":
                    raise ValueError(
                        "DTW labeling method requires an explicit label_semantics.json manifest; "
                        "cluster ids have no stable bullish/bearish ordering"
                    )
                data = self.generate_default_label_semantics()

        labels = data.get("labels", [])
        if len(labels) != self.num_label:
            raise ValueError(
                f"label_semantics count ({len(labels)}) does not match num_label ({self.num_label})"
            )

        label_map = {item["label"]: item for item in labels}
        expected_set = set(self.label_list)
        if set(label_map.keys()) != expected_set:
            raise ValueError(
                f"label_semantics missing labels: {expected_set - set(label_map.keys())}"
            )

        l0 = label_map["label_0"]
        if (
            l0.get("limit_state") not in ["limit_down", "near_limit_down"]
            or l0.get("limit_state_sign") != -1
        ):
            raise ValueError(
                "current commodity labels require label_0 to be limit_down/near_limit_down with limit_state_sign=-1"
            )

        last_lbl_name = f"label_{self.num_label - 1}"
        ln = label_map[last_lbl_name]
        if (
            ln.get("limit_state") not in ["limit_up", "near_limit_up"]
            or ln.get("limit_state_sign") != 1
        ):
            raise ValueError(
                f"current commodity labels require the last label ({last_lbl_name}) to be limit_up/near_limit_up with limit_state_sign=1"
            )

        for item in labels:
            if item.get("direction_sign") not in [-1, 0, 1]:
                raise ValueError(
                    f"invalid direction_sign {item.get('direction_sign')} in label {item.get('label')}"
                )
            if item.get("limit_state") not in [
                "none",
                "near_limit_up",
                "limit_up",
                "near_limit_down",
                "limit_down",
            ]:
                raise ValueError(
                    f"invalid limit_state {item.get('limit_state')} in label {item.get('label')}"
                )
            if item.get("limit_state_sign") not in [-1, 0, 1]:
                raise ValueError(
                    f"invalid limit_state_sign {item.get('limit_state_sign')} in label {item.get('label')}"
                )

        self.label_semantics = label_map
        return self.label_semantics

    def check_semantic_alignment(self, candidate_record, semantic_entry):
        def _val(f):
            if isinstance(candidate_record, dict):
                return candidate_record[f]
            return candidate_record[f]

        cand_exp = float(_val("candidate_mean_exposure"))
        cand_l_ratio = float(_val("candidate_long_ratio"))
        cand_s_ratio = float(_val("candidate_short_ratio"))
        cand_l_rew = float(_val("candidate_long_reward_mean"))
        cand_s_rew = float(_val("candidate_short_reward_mean"))
        cand_lim_u_rew = float(_val("candidate_limit_up_long_reward_mean"))
        cand_lim_d_rew = float(_val("candidate_limit_down_short_reward_mean"))
        cand_lim_u_rev = float(_val("candidate_limit_up_reverse_short_ratio"))
        cand_lim_d_rev = float(_val("candidate_limit_down_reverse_long_ratio"))

        dir_sign = semantic_entry["direction_sign"]
        lim_state = semantic_entry["limit_state"]

        th = self.semantic_filter_thresholds
        min_dir_exp = th["min_directional_exposure"]
        min_dir_step = th["min_directional_step_ratio"]
        max_neut_exp = th["max_neutral_abs_exposure"]
        max_lim_rev = th["max_limit_reverse_ratio"]

        if dir_sign == 1:
            if cand_exp < min_dir_exp or cand_l_ratio < min_dir_step or cand_l_rew <= 0:
                return False, (
                    f"bullish semantic mismatch (exp={cand_exp:.3f} < {min_dir_exp}, "
                    f"long_ratio={cand_l_ratio:.3f} < {min_dir_step}, long_reward={cand_l_rew:.3f} <= 0)"
                )
        elif dir_sign == -1:
            if cand_exp > -min_dir_exp or cand_s_ratio < min_dir_step or cand_s_rew <= 0:
                return False, (
                    f"bearish semantic mismatch (exp={cand_exp:.3f} > {-min_dir_exp}, "
                    f"short_ratio={cand_s_ratio:.3f} < {min_dir_step}, short_reward={cand_s_rew:.3f} <= 0)"
                )
        elif dir_sign == 0:
            if abs(cand_exp) > max_neut_exp:
                return False, (
                    f"neutral semantic mismatch (abs_exp={abs(cand_exp):.3f} > {max_neut_exp})"
                )

        if lim_state in ["limit_up", "near_limit_up"]:
            if cand_lim_u_rew <= 0 or cand_lim_u_rev > max_lim_rev:
                return False, (
                    f"limit-up semantic mismatch (limit_up_long_reward={cand_lim_u_rew:.3f} <= 0 "
                    f"or reverse_short_ratio={cand_lim_u_rev:.3f} > {max_lim_rev})"
                )
        elif lim_state in ["limit_down", "near_limit_down"]:
            if cand_lim_d_rew <= 0 or cand_lim_d_rev > max_lim_rev:
                return False, (
                    f"limit-down semantic mismatch (limit_down_short_reward={cand_lim_d_rew:.3f} <= 0 "
                    f"or reverse_long_ratio={cand_lim_d_rev:.3f} > {max_lim_rev})"
                )

        return True, "passed"

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
        new_result = []
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

            single_result["candidate_mean_exposure"] = float(np.mean(single_result["net_position_exposure"]))
            single_result["candidate_long_ratio"] = float(np.mean(single_result["long_step_ratio"]))
            single_result["candidate_short_ratio"] = float(np.mean(single_result["short_step_ratio"]))
            single_result["candidate_long_reward_mean"] = float(np.mean(single_result["long_reward_sum"]))
            single_result["candidate_short_reward_mean"] = float(np.mean(single_result["short_reward_sum"]))
            single_result["candidate_limit_up_long_reward_mean"] = float(np.mean(single_result["limit_up_long_reward_sum"]))
            single_result["candidate_limit_down_short_reward_mean"] = float(np.mean(single_result["limit_down_short_reward_sum"]))
            single_result["candidate_limit_up_reverse_short_ratio"] = float(np.mean(single_result["limit_up_reverse_short_ratio"]))
            single_result["candidate_limit_down_reverse_long_ratio"] = float(np.mean(single_result["limit_down_reverse_long_ratio"]))

            single_result["epoch_path"] = epoch_path
            new_result.append(single_result)
        return new_result

    def transform_single_epoch_result_all(self, result, epoch_path):
        transformed = self.transform_single_epoch_result(result, epoch_path)
        return pd.DataFrame(transformed)

    def conclude_single_parameter(self, parameter_path):
        single_parameter_result = []
        single_parameter_best_result = []
        for i in range(45, self.epoch_num):
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
        semantics = self.load_label_semantics()
        max_result = []
        for label in self.label_list:
            sem_entry = semantics[label]
            for initial_action in self.initial_position_list:
                single_condition_result = []
                for single_result in result:
                    if (
                        single_result["initial_action"] == initial_action
                        and single_result["label"] == label
                    ):
                        passed, _ = self.check_semantic_alignment(single_result, sem_entry)
                        if passed:
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

        out_dir = self._result_output_dir()
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        df_best.to_csv(os.path.join(out_dir, "result.csv"))
        df_all.to_csv(os.path.join(out_dir, "result_all.csv"))
        self.result_df_best = df_best
        self.result_df_all = df_all
        return df_best, df_all

    def pick_best_agent_regarding_dynamics_bin_index_path(self, result_all):
        self._validate_label_coverage(result_all["label"].unique())
        semantics = self.load_label_semantics()

        label_list = []
        epoch_path_list = []
        bin_index_list = []
        reward_max_list = []
        source_rows_list = []
        behavior_summaries = []

        for label in result_all["label"].unique():
            selected_df = result_all[result_all["label"] == label].copy()
            sem_entry = semantics[label]

            # Step 1: Filter positive reward candidate models first
            pos_df = selected_df[selected_df["trans_reward_mean"] > 0]
            if not pos_df.empty:
                pool_df = pos_df
                pool_is_positive = True
            else:
                pool_df = selected_df
                pool_is_positive = False

            # Step 2: Directional Semantics check within candidate pool
            eligible_rows = []
            for idx, row in pool_df.iterrows():
                passed, reason = self.check_semantic_alignment(row, sem_entry)
                if passed:
                    eligible_rows.append(row)

            if eligible_rows:
                target_df = pd.DataFrame(eligible_rows)
                selection_note = (
                    "passed positive reward filter and strict directional semantics gate"
                    if pool_is_positive
                    else "passed strict directional semantics gate (fallback pool)"
                )
            else:
                target_df = pool_df
                selection_note = (
                    "positive reward model fallback (soft directional semantics)"
                    if pool_is_positive
                    else "all model fallback (soft directional semantics)"
                )

            reward_mean_info = (
                target_df.groupby(["label", "bin_index", "epoch_path"])[
                    "trans_reward_mean"
                ]
                .agg(["mean", "count"])
                .dropna()
            )
            if reward_mean_info.empty:
                raise ValueError(
                    f"label {label} has no finite selection candidates after candidate pool filtering"
                )

            selected_information_based_reward_sum = reward_mean_info["mean"].idxmax()
            sel_label = selected_information_based_reward_sum[0]
            sel_bin_index = selected_information_based_reward_sum[1]
            sel_epoch_path = selected_information_based_reward_sum[2]
            reward_max = reward_mean_info.loc[
                selected_information_based_reward_sum, "mean"
            ]
            source_rows = int(
                reward_mean_info.loc[selected_information_based_reward_sum, "count"]
            )

            matched_group = target_df[
                (target_df["bin_index"] == sel_bin_index)
                & (target_df["epoch_path"] == sel_epoch_path)
            ]
            summary = {
                "candidate_mean_exposure": float(matched_group["candidate_mean_exposure"].mean()),
                "candidate_long_ratio": float(matched_group["candidate_long_ratio"].mean()),
                "candidate_short_ratio": float(matched_group["candidate_short_ratio"].mean()),
                "candidate_long_reward_mean": float(matched_group["candidate_long_reward_mean"].mean()),
                "candidate_short_reward_mean": float(matched_group["candidate_short_reward_mean"].mean()),
                "candidate_limit_up_long_reward_mean": float(matched_group["candidate_limit_up_long_reward_mean"].mean()),
                "candidate_limit_down_short_reward_mean": float(matched_group["candidate_limit_down_short_reward_mean"].mean()),
                "candidate_limit_up_reverse_short_ratio": float(matched_group["candidate_limit_up_reverse_short_ratio"].mean()),
                "candidate_limit_down_reverse_long_ratio": float(matched_group["candidate_limit_down_reverse_long_ratio"].mean()),
                "selection_note": selection_note,
                "pool_is_positive": pool_is_positive,
            }

            label_list.append(sel_label)
            epoch_path_list.append(sel_epoch_path)
            bin_index_list.append(sel_bin_index)
            reward_max_list.append(reward_max)
            source_rows_list.append(source_rows)
            behavior_summaries.append(summary)

        best_agent_info = pd.DataFrame(
            {
                "label": label_list,
                "epoch_path": epoch_path_list,
                "bin_index": bin_index_list,
                "reward_max": reward_max_list,
                "source_rows": source_rows_list,
                "behavior_summary": behavior_summaries,
            }
        )
        self._validate_label_coverage(best_agent_info["label"].tolist())
        out_dir = self._result_output_dir()
        os.makedirs(out_dir, exist_ok=True)
        best_agent_info.to_csv(
            os.path.join(
                out_dir,
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
        semantics = self.load_label_semantics()

        labels = []
        for row in ordered_df.to_dict("records"):
            lbl_name = row["label"]
            sem = semantics[lbl_name]
            model_path = os.path.join(row["epoch_path"], "trained_model.pkl")
            b_summary = row.get("behavior_summary", {})
            labels.append(
                {
                    "label": lbl_name,
                    "description": sem.get("description", ""),
                    "direction": sem.get("direction", ""),
                    "direction_sign": sem.get("direction_sign", 0),
                    "limit_state": sem.get("limit_state", "none"),
                    "limit_state_sign": sem.get("limit_state_sign", 0),
                    "epoch_path": row["epoch_path"],
                    "model_path": model_path,
                    "bin_index": int(row["bin_index"]),
                    "score": float(row["reward_max"]),
                    "source_rows": int(row.get("source_rows", 0)),
                    "semantic_filter": self.semantic_filter_thresholds,
                    "behavior_summary": b_summary,
                    "selection_reason": b_summary.get("selection_note", "passed semantic gate, then ranked by trans_reward_mean"),
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
