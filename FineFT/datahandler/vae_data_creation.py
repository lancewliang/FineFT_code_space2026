import pandas as pd
import numpy as np
import os
import argparse

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


def _collect_valid_label_files(valid_path):
    valid_files = {"legacy_labels": {}, "contracts": {}}
    if not os.path.exists(valid_path):
        raise FileNotFoundError(f"missing valid path: {valid_path}")

    for name in sorted(os.listdir(valid_path)):
        path = os.path.join(valid_path, name)
        if not os.path.isdir(path) or name == "processed":
            continue
        if name.startswith("label_"):
            valid_files["legacy_labels"][name] = [
                os.path.join(path, file_name)
                for file_name in sorted(os.listdir(path))
                if file_name.endswith(".feather")
            ]
            continue
        contract_labels = valid_files["contracts"].setdefault(name, {})
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
            valid_files["contracts"].pop(name, None)
    return valid_files


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
    valid_path = os.path.join(args.base_path, args.dataset_name, "valid")
    state_name_path = os.path.join(
        args.base_path, args.dataset_name, "state_features.npy"
    )
    state_features = np.load(state_name_path)
    save_path = os.path.join(args.save_path, args.dataset_name, "VAE_data")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    valid_files = _collect_valid_label_files(valid_path)
    for label, df_paths in valid_files["legacy_labels"].items():
        if not _save_label_array(
            df_paths, state_features, os.path.join(save_path, "{}.npy".format(label))
        ):
            print(f"skip empty label: {label}")
    for contract, labels in valid_files["contracts"].items():
        contract_save_path = os.path.join(save_path, contract)
        os.makedirs(contract_save_path, exist_ok=True)
        for label, df_paths in labels.items():
            if not _save_label_array(
                df_paths,
                state_features,
                os.path.join(contract_save_path, "{}.npy".format(label)),
            ):
                print(f"skip empty label: {contract}/{label}")
    test_path = os.path.join(args.base_path, args.dataset_name, "test.feather")
    if os.path.exists(test_path):
        test_frames = [pd.read_feather(test_path)]
        test_data = np.concatenate(
            [df[state_features].values for df in test_frames],
            axis=0,
        )
        np.save(os.path.join(save_path, "test.npy"), test_data)
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
        test_save_path = os.path.join(save_path, "test")
        os.makedirs(test_save_path, exist_ok=True)
        for file_name in test_files:
            df = pd.read_feather(os.path.join(test_dir, file_name))
            contract = os.path.splitext(file_name)[0]
            if contract.startswith("df_"):
                contract = contract[3:]
            np.save(
                os.path.join(test_save_path, f"test_{contract}.npy"),
                df[state_features].values,
            )


if __name__ == "__main__":
    print('start to create VAE data')
    args = parser.parse_args()
    make_data(args)
