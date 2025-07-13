# design for plot for all 4 datasets and form in just 1 figure.
import pandas as pd
import argparse
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"
import sys

sys.path.append(".")
from analysis.calculate_metric.calculate_metric import (
    calculate_metric,
    calculate_required_money,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--store_result_path",
    type=str,
    default="analysis_result/DiHFT/high_level_heurstic_wo_routing",
    help="the number of initial_position",
)
parser.add_argument(
    "--data_path",
    type=str,
    default="dataset",
    help="the number of initial_position",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="analysis_result/DiHFT/ablation/routing_related",
    help="the number of initial_position",
)
parser.add_argument(
    "--dataset_name_list",
    type=list,
    default=[
        "BNBUSDT",
        "BTCUSDT",
        "DOTUSDT",
        "ETHUSDT",
    ],
    help="the number of initial_position",
)
parser.add_argument(
    "--early_stop",
    type=int,
    default=0,
    help="the number of initial_position",
)


class ploter:
    def __init__(self, args) -> None:
        self.dataset_name_list = args.dataset_name_list
        self.ealry_stop = args.early_stop
        self.dataset_path = args.data_path
        self.store_result_path = args.store_result_path
        self.save_path = args.save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.result_dict_all = {}
        for dataset_name in self.dataset_name_list:
            with open(
                os.path.join(self.store_result_path, dataset_name, "result_dict.pkl"),
                "rb",
            ) as file:
                data = pickle.load(file)
                self.result_dict_all[dataset_name] = data
        self.color_list = [
            "#DE7833",
            "#912C2C",
            "#F2BB6B",
            "#C2ABC8",
            "#329845",
            "#AED185",
            "#276C9E",
        ]
        self.test_df_list = [
            pd.read_feather(
                os.path.join(self.dataset_path, dataset_name, "test.feather")
            )
            for dataset_name in self.dataset_name_list
        ]

    def plot_performance_comparison(self):
        fig, axs = plt.subplots(2, 2, figsize=(15, 8))
        for i in range(2):
            for j in range(2):
                dataset_name = self.dataset_name_list[i * 2 + j]
                result_dict = self.result_dict_all[dataset_name]
                timestamp = self.test_df_list[i * 2 + j].timestamp.values[:-1]
                for k, key in enumerate(result_dict.keys()):
                    if i + j == 0:
                        axs[i, j].plot(
                            timestamp[-len(result_dict[key]) :],
                            result_dict[key] * 100,
                            color=self.color_list[k],
                            label=key,
                            linewidth=2,
                        )
                    else:
                        axs[i, j].plot(
                            timestamp[-len(result_dict[key]) :],
                            result_dict[key] * 100,
                            color=self.color_list[k],
                            linewidth=2,
                        )
                    axs[i, j].grid(ls="--")
        for ax in axs.flat:
            ax = plt.gca()
        plt.subplots_adjust(hspace=0)
        titles = ["(a) BNB", "(b) BTC", "(c) DOT", "(d) ETH"]

        for ax, title in zip(axs.flat, titles):
            ax.annotate(
                title,
                xy=(0.5, -0.18),
                xycoords="axes fraction",
                ha="center",
                va="center",
                fontsize=18,
            )
            ax.annotate(
                "Date",
                xy=(0.99, -0.09),
                xycoords="axes fraction",
                ha="center",
                va="center",
                fontsize=18,
            )
        fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=7, fontsize=15)
        # 添加标签
        for i in range(2):
            for j in range(2):
                # axs[i, j].set_xlabel("Trading Timestamp", fontsize=18)
                axs[i, j].set_ylabel("Total Return(%)", fontsize=18)

        plt.tight_layout(pad=1.5, rect=[0, 0, 1, 0.88])
        plt.savefig(
            os.path.join(self.save_path, "performance_comparison.pdf"),
            bbox_inches="tight",
        )
if __name__=="__main__":
    args = parser.parse_args()
    ploter = ploter(args)
    ploter.plot()
