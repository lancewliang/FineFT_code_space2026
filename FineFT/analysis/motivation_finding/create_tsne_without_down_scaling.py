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

# display method
# aleo uncertainty noraml + one x
# epi uncertainty normal

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
parser = argparse.ArgumentParser()
parser.add_argument(
    "--df_path",
    type=str,
    default="dataset/motivation",
    help="df path",
)
parser.add_argument(
    "--state_feature_path",
    type=str,
    default="dataset/motivation/state_features.npy",
    help="state_feature path",
)

parser.add_argument(
    "--save_path",
    type=str,
    default="analysis_result/motivation/tsne",
    help="save path",
)
parser.add_argument(
    "--uncertainty_type",
    type=str,
    default="aleatoric",
    choices=["aleatoric", "epistemic"],
    help="state_feature path",
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
    args.save_path = os.path.join(args.save_path, args.uncertainty_type)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    df_path = os.path.join(
        args.df_path, "{}_uncertainty.feather".format(args.uncertainty_type)
    )
    df = pd.read_feather(df_path)
    state_feature_name = np.load(args.state_feature_path)
    # pick proper indices
    states, target, timestamp = create_proper_sample(df, state_feature_name)

    state_array_1d, tsne = get_tsne_result(states)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    np.save(os.path.join(args.save_path, "state_array_1d.npy"), state_array_1d)
    np.save(os.path.join(args.save_path, "target.npy"), target)
    np.save(os.path.join(args.save_path, "timestamp.npy"), timestamp)


def get_tsne_result(x):
    tsne = TSNE(n_components=1, random_state=0)
    state_array_1d = tsne.fit_transform(x).ravel()  # 使用.ravel()将结果展平为一维数组
    return state_array_1d, tsne


def calculate_target(df, reward_feature, window_length=1):
    # the length of the target = len(df)-1
    target = df[reward_feature].shift(-window_length) - df[reward_feature]
    target = target[:-window_length]
    return target


def create_proper_sample(df, state_feature_name):
    target = calculate_target(df, "mark_price")
    states = df[state_feature_name].values[:-1]
    timestamp = df["timestamp"].values[:-1]
    print(
        "state shape",
        states.shape,
        "target shape",
        target.shape,
        "timestamp shape",
        timestamp.shape,
    )
    return states, target, timestamp


def main_load(args):
    if args.uncertainty_type in args.save_path:
        pass
    else:
        args.save_path = os.path.join(args.save_path, args.uncertainty_type)
    target = np.load(os.path.join(args.save_path, "target.npy"))
    state_1_d = np.load(os.path.join(args.save_path, "state_array_1d.npy"))
    timestamp = np.load(os.path.join(args.save_path, "timestamp.npy"))
    graph(timestamp, state_1_d, target, args.save_path)


def graph(timestamp_array, state_array_1d, y, save_path):
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
    # ax.axvline(x=split_date, color="r", linestyle="--", label="Train/Valid Split")
    # ax.axvline(x=split_date_2, color="r", linestyle="--", label="Valid/Test Split")
    # plt.legend(
    #     loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2, fontsize=18
    # )  # 添加颜色条
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
    # main(args)
    main_load(args)
