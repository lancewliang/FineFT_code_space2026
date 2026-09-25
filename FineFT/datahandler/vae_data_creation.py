import pandas as pd
import numpy as np
import os
import shutil
import argparse
import sys
from pathlib import Path

FINEFT_ROOT = Path(__file__).resolve().parents[1]
if str(FINEFT_ROOT) not in sys.path:
    sys.path.insert(0, str(FINEFT_ROOT))

from common import ArtifactNames, get_vae_label_filename, get_vae_test_filename

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"

parser = argparse.ArgumentParser()
# replay buffer coffient

# here we only use valid dataset to do the speration
parser.add_argument(
    "--base_path",
    type=str,
    default="dataset",
    help="the number of transcation we store in one memory",
)
parser.add_argument(
    "--dataset_name",
    type=str,
    default="BTCUSDT",
    help="the number of transcation we store in one memory",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="dataset",
    help="the number of transcation we store in one memory",
)
parser.add_argument(
    "--source_split",
    choices=("train", "valid"),
    default="train",
    help="dataset split to read sliced labels from (default: train)",
)
parser.add_argument(
    "--labeling_method",
    choices=("slope", "volatility"),
    default="slope",
    help="valid dynamic-label directory to consume",
)


def _collect_label_files(source_path):
    label_files = {"legacy_labels": {}, "contracts": {}}
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"missing split path: {source_path}")

    for name in sorted(os.listdir(source_path)):
        path = os.path.join(source_path, name)
        if not os.path.isdir(path) or name in ("processed", "slice"):
            continue
        if name.startswith("label_"):
            label_files["legacy_labels"][name] = [
                os.path.join(path, file_name)
                for file_name in sorted(os.listdir(path))
                if file_name.endswith(".feather")
            ]
            continue
        contract_labels = label_files["contracts"].setdefault(name, {})
        for label in sorted(os.listdir(path)):
            label_path = os.path.join(path, label)
            if not os.path.isdir(label_path) or not label.startswith("label_"):
                continue
            df_paths = [
                os.path.join(label_path, file_name)
                for file_name in sorted(os.listdir(label_path))
                if file_name.endswith(".feather")
            ]
            if df_paths:
                contract_labels[label] = df_paths
        if not contract_labels:
            label_files["contracts"].pop(name, None)
    return label_files


_collect_valid_label_files = _collect_label_files


def _save_label_array(df_paths, state_features, output_path):
    single_label_data_list = []
    for df_path in df_paths:
        df = pd.read_feather(df_path)
        single_label_data = df[state_features].values
        single_label_data_list.append(single_label_data)
    if not single_label_data_list:
        return False
    single_label_data_all = np.concatenate(single_label_data_list, axis=0)
    np.save(output_path, single_label_data_all)
    return True


def make_data(args):
    source_split = args.source_split
    split_root = os.path.join(args.base_path, args.dataset_name, source_split)
    labeling_method = args.labeling_method
    method_path = os.path.join(split_root, labeling_method)
    if os.path.isdir(method_path):
        source_path = method_path
    elif labeling_method == "slope":
        source_path = split_root
    else:
        source_path = method_path
    state_name_path = os.path.join(
        args.base_path, args.dataset_name, ArtifactNames.STATE_FEATURES_NPY
    )
    state_features = np.load(state_name_path)
    vae_data_root = os.path.join(args.save_path, args.dataset_name, "VAE_data")
    method_save_path = os.path.join(vae_data_root, labeling_method)
    if os.path.isdir(method_save_path):
        for item in os.listdir(method_save_path):
            item_path = os.path.join(method_save_path, item)
            if os.path.isdir(item_path) and item not in ("train", "test", "processed"):
                shutil.rmtree(item_path)
    os.makedirs(method_save_path, exist_ok=True)
    label_files = _collect_label_files(source_path)
    for label, df_paths in label_files["legacy_labels"].items():
        if not _save_label_array(
            df_paths, state_features, os.path.join(method_save_path, get_vae_label_filename(label))
        ):
            print(f"skip empty label: {label}")
    for contract, labels in label_files["contracts"].items():
        contract_save_path = os.path.join(method_save_path, contract)
        os.makedirs(contract_save_path, exist_ok=True)
        for label, df_paths in labels.items():
            if not _save_label_array(
                df_paths,
                state_features,
                os.path.join(contract_save_path, get_vae_label_filename(label)),
            ):
                print(f"skip empty label: {contract}/{label}")
    test_path = os.path.join(args.base_path, args.dataset_name, ArtifactNames.TEST_FEATHER)
    if os.path.exists(test_path):
        test_frames = [pd.read_feather(test_path)]
        test_data = np.concatenate(
            [df[state_features].values for df in test_frames],
            axis=0,
        )
        np.save(os.path.join(vae_data_root, ArtifactNames.TEST_NPY), test_data)
    else:
        test_dir = os.path.join(args.base_path, args.dataset_name, "test")
        test_files = [
            file_name
            for file_name in sorted(os.listdir(test_dir))
            if file_name.endswith(".feather")
        ]
        if not test_files:
            raise FileNotFoundError(
                f"missing test.feather and no test/df_<contract>.feather files under {test_dir}"
            )
        test_save_path = os.path.join(vae_data_root, "test")
        os.makedirs(test_save_path, exist_ok=True)
        for file_name in test_files:
            df = pd.read_feather(os.path.join(test_dir, file_name))
            contract = os.path.splitext(file_name)[0]
            if contract.startswith("df_"):
                contract = contract[3:]
            np.save(
                os.path.join(test_save_path, get_vae_test_filename(contract)),
                df[state_features].values,
            )


if __name__ == "__main__":
    print('start to create VAE data')
    args = parser.parse_args()
    make_data(args)
