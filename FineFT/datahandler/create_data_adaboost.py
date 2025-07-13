import pandas as pd
import numpy as np
import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader

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


class SL_data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.input_dim = self.X[0].shape[
            0
        ]  # assume the data shape is (NSamples, Sample_Length)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def calculate_target(df, reward_feature):
    # the length of the target = len(df)-1
    target = df[reward_feature].shift(-1) - df[reward_feature]
    target = target[:-1]
    return target


def make_data(args):
    train_data_path = os.path.join(args.base_path, args.dataset_name, "train.feather")
    valid_data_path = os.path.join(args.base_path, args.dataset_name, "valid.feather")
    test_data_path = os.path.join(args.base_path, args.dataset_name, "test.feather")
    state_name_path = os.path.join(
        args.base_path, args.dataset_name, "state_features.npy"
    )
    state_features = np.load(state_name_path)
    save_path = os.path.join(args.save_path, args.dataset_name, "SL_data")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    train_data = pd.read_feather(train_data_path)
    valid_data = pd.read_feather(valid_data_path)
    test_data = pd.read_feather(test_data_path)

    reward_feature = "mark_price"
    X_train, y_train = make_X_y(train_data, state_features, reward_feature)
    X_valid, y_valid = make_X_y(valid_data, state_features, reward_feature)
    X_test, y_test = make_X_y(test_data, state_features, reward_feature)
    np.save(os.path.join(save_path, "X_train.npy"), X_train)
    np.save(os.path.join(save_path, "y_train.npy"), y_train)
    np.save(os.path.join(save_path, "X_valid.npy"), X_valid)
    np.save(os.path.join(save_path, "y_valid.npy"), y_valid)
    np.save(os.path.join(save_path, "X_test.npy"), X_test)
    np.save(os.path.join(save_path, "y_test.npy"), y_test)


def make_X_y(data: pd.DataFrame, state_features: np.ndarray, reward_feature: str):
    X = data[state_features].values[:-1]
    y = calculate_target(data, reward_feature)
    return X, y


if __name__ == "__main__":
    args = parser.parse_args()
    make_data(args)
