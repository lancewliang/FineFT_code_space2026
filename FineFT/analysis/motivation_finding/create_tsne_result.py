# the motivation is created by the t-SNE result of the data
# the motivation demonstration is made by BTCUSDT data
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from joblib import dump, load
import os
import argparse
import random
import torch

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
parser = argparse.ArgumentParser()
parser.add_argument(
    "--df_path",
    type=str,
    default="dataset/BTCUSDT/df.feather",
    help="df path",
)
parser.add_argument(
    "--state_feature_path",
    type=str,
    default="dataset/BTCUSDT/state_features.npy",
    help="state_feature path",
)
parser.add_argument(
    "--train_portion",
    type=float,
    default=0.6,
    help="train_portion",
)
parser.add_argument(
    "--valid_portion",
    type=float,
    default=0.2,
    help="valid_portion",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="analysis_result/motivation",
    help="save path",
)
# we add this because there are too many data points and it is hard to do the tsne with that much data point
parser.add_argument(
    "--total_sample_num",
    type=float,
    default=1e4,
    help="save path",
)
parser.add_argument(
    "--look_back",
    type=int,
    default=15,
    help="save path",
)
parser.add_argument(
    "--look_forward",
    type=int,
    default=3,
    help="save path",
)


def seed_torch(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(args):
    seed_torch(12345)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    look_back = args.look_back
    look_forward = args.look_forward
    num_sample = args.total_sample_num
    df = pd.read_feather(args.df_path)
    state_feature_name = np.load(args.state_feature_path)
    # pick proper indices
    single_df, single_state_df, single_df_target, small_dfs = create_proper_sample(
        df, look_back, look_forward, num_sample, state_feature_name
    )
    tsne_result_path = os.path.join(args.save_path, "tsne_result")
    sampled_path = os.path.join(args.save_path, "sample_result_tsne")
    if not os.path.exists(sampled_path):
        os.makedirs(sampled_path)
    single_df.to_feather(os.path.join(sampled_path, "single_df.feather"))
    single_state_df.to_feather(os.path.join(sampled_path, "single_state_df.feather"))
    np.save(os.path.join(sampled_path, "single_df_target.npy"), single_df_target)
    if not os.path.exists(os.path.join(sampled_path, "small_dfs")):
        os.makedirs(os.path.join(sampled_path, "small_dfs"))
    for idx, small_df in enumerate(small_dfs):
        small_df.to_feather(
            os.path.join(sampled_path, "small_dfs", f"small_df_{idx}.feather")
        )
    #
    x = single_state_df.values
    y = single_df_target
    timestamp_array = single_df["timestamp"].values
    state_array_1d, tsne = get_tsne_result(timestamp_array, x)
    if not os.path.exists(tsne_result_path):
        os.makedirs(tsne_result_path)
    np.save(os.path.join(tsne_result_path, "state_array_1d.npy"), state_array_1d)


def get_tsne_result(timestamp_array, x):
    timestamps = timestamp_array.ravel().astype("datetime64[s]").astype("int64")
    tsne = TSNE(n_components=1, random_state=0)
    state_array_1d = tsne.fit_transform(x).ravel()  # 使用.ravel()将结果展平为一维数组
    return state_array_1d, tsne


def create_proper_sample(df, look_back, look_forward, num_sample, state_feature_name):
    valid_indices = np.arange(look_back, len(df) - look_forward)
    indices = np.random.choice(valid_indices, size=int(num_sample), replace=False)
    indices.sort()
    indices = np.unique(indices)
    single_df = df.loc[indices, :]
    single_state_df = df.loc[indices, state_feature_name]
    single_df_target = (
        df.loc[indices + 1]["mark_price"].values - df.loc[indices]["mark_price"].values
    )
    all_indices = np.unique(
        np.concatenate(
            [np.arange(idx - look_back, idx + look_forward + 1) for idx in indices]
        )
    )
    extended_df = df.iloc[all_indices]

    small_dfs = []
    for idx in indices:
        start_idx = np.searchsorted(
            all_indices, idx - look_back
        )  # 找到扩展后DataFrame中的起始索引
        end_idx = (
            np.searchsorted(all_indices, idx + look_forward, side="right") - 1
        )  # 找到结束索引
        small_df = extended_df.iloc[start_idx : end_idx + 1]
        small_dfs.append(small_df)
    return single_df, single_state_df, single_df_target, small_dfs


def main_load(args):
    look_back = args.look_back
    look_forward = args.look_forward
    num_sample = args.total_sample_num
    df_all = pd.read_feather(args.df_path)
    df_state = pd.read_feather(
        os.path.join(args.save_path, "sample_result_tsne/single_state_df.feather")
    )
    df_target = np.load(
        os.path.join(args.save_path, "sample_result_tsne/single_df_target.npy")
    )
    df = pd.read_feather(
        os.path.join(args.save_path, "sample_result_tsne/single_df.feather")
    )
    small_df = df_all
    timestamp_array = df["timestamp"].values
    state_array_1d = np.load(
        os.path.join(args.save_path, "tsne_result/state_array_1d.npy")
    )
    state_feature_name = np.load(args.state_feature_path)
    total_length = len(df_all)
    split_idx_1 = int(total_length * 0.8)
    split_idx_2 = int(total_length * 0.9)
    df_all_timestamp = df_all["timestamp"].values
    split_date = df_all_timestamp[split_idx_1]
    split_date_2 = df_all_timestamp[split_idx_2]
    split_date = df[df["timestamp"] <= split_date]["timestamp"].values[-1]
    split_date_2 = df[df["timestamp"] <= split_date_2]["timestamp"].values[-1]
    split_idx_1 = int(len(df) * args.train_portion)
    split_idx_2 = int(len(df) * (args.train_portion + args.valid_portion))
    save_path = os.path.join(args.save_path, "tsne_result")
    graph(
        timestamp_array, state_array_1d, df_target, split_date, split_date_2, save_path
    )


def graph(timestamp_array, state_array_1d, y, split_date, split_date_2, save_path):
    lower_y = np.percentile(y, 20)
    upper_y = np.percentile(y, 80)
    y_clipped = y.clip(lower_y, upper_y)
    fig, ax = plt.subplots(
        figsize=(12, 4)
    )  # 这里可以调整图形的大小，例如10英寸宽和6英寸高

    # 将timestamp_array转换为matplotlib能理解的日期格式
    dates = mdates.date2num(timestamp_array)
    # 根据y的值选择颜色
    sc = ax.scatter(
        dates,
        state_array_1d,
        c=y_clipped,
        cmap="coolwarm",
        s=0.1,
        marker="h",
    )
    # 为测试集绘制散点图

    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.axvline(x=split_date, color="r", linestyle="--", label="Train/Valid Split")
    ax.axvline(x=split_date_2, color="r", linestyle="--", label="Valid/Test Split")
    plt.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2, fontsize=18
    )  # 添加颜色条
    cbar = plt.colorbar(sc, shrink=1.2, pad=0.01)
    cbar.set_label("Return", size=21)
    cbar.ax.tick_params(labelsize=15)
    plt.xlabel("Dates", fontsize=21)
    plt.ylabel("t-SNE result", fontsize=21)
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.tick_params(axis="both", which="minor", labelsize=10)
    plt.subplots_adjust(right=0.85)
    plt.tight_layout()
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(os.path.join(save_path, "tsne.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    main_load(args)
