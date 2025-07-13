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


def make_data(args):
    valid_path = os.path.join(args.base_path, args.dataset_name, "valid")
    state_name_path = os.path.join(
        args.base_path, args.dataset_name, "state_features.npy"
    )
    state_features = np.load(state_name_path)
    save_path = os.path.join(args.save_path, args.dataset_name, "VAE_data")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for label in os.listdir(valid_path):
        single_label_data_list = []
        label_path = os.path.join(valid_path, label)
        for df in os.listdir(label_path):
            if df.endswith(".feather"):
                df_path = os.path.join(label_path, df)
                df = pd.read_feather(df_path)
                single_label_data = df[state_features].values
                single_label_data_list.append(single_label_data)
        single_label_data_all = np.concatenate(single_label_data_list, axis=0)
        np.save(os.path.join(save_path, "{}.npy".format(label)), single_label_data_all)
    test_path=os.path.join(args.base_path, args.dataset_name, "test.feather")
    df=pd.read_feather(test_path)
    test_data=df[state_features].values
    np.save(os.path.join(save_path, "test.npy"), test_data)


if __name__ == "__main__":
    args = parser.parse_args()
    make_data(args)
