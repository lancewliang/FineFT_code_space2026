# design for plot for all 4 datasets and form in just 1 figure.
import pandas as pd
import argparse
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
from collections import Counter, OrderedDict

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"
import sys

sys.path.append(".")

parser = argparse.ArgumentParser()
# data related
parser.add_argument(
    "--dataset_name",
    type=str,
    default="BNBUSDT",
    help="dataset name",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="analysis_result/DiHFT/ablation/routing_related",
    help="the routing result path",
)
parser.add_argument(
    "--collect_data_path",
    type=str,
    default="result/DiHFT/final_result",
    help="the routing result path",
)
parser.add_argument(
    "--dataset_path",
    type=str,
    default="dataset",
    help="the routing result path",
)
# plot related
parser.add_argument(
    "--chunk_number",
    type=int,
    default=20,
    help="the routing result path",
)
parser.add_argument(
    "--dynamic_number",
    type=int,
    default=5,
    help="the routing result path",
)
parser.add_argument(
    "--dates_tick_number",
    type=int,
    default=4,
    help="the routing result path",
)
parser.add_argument(
    "--minum_value",
    type=int,
    default=85,
    help="the routing result path",
)
parser.add_argument(
    "--width",
    type=float,
    default=0.9,
    help="the routing result path",
)


class analyzer_router:
    def __init__(self, args):
        self.dataset_name = args.dataset_name
        self.save_path = args.save_path
        self.collect_data_path = args.collect_data_path
        self.chunk_number = args.chunk_number
        # timestamp as hiroton data
        self.data_timestamp = pd.read_feather(
            os.path.join(args.dataset_path, self.dataset_name, "test.feather")
        )["timestamp"].values
        total_points = len(self.data_timestamp)
        chunk_size = total_points // 4
        mid_points = [
            self.data_timestamp[i * chunk_size + chunk_size // 2] for i in range(4)
        ]
        self.mid_points = [dt.astype("datetime64[D]") for dt in mid_points]
        self.routing_result = np.load(
            os.path.join(
                self.collect_data_path, self.dataset_name, "macro_action_history.npy"
            )
        )
        interval_number = int(len(self.routing_result) / args.chunk_number)
        self.chunk_list = [
            self.routing_result[i * interval_number : (i + 1) * interval_number]
            for i in range(self.chunk_number - 1)
        ] + [self.routing_result[(self.chunk_number - 1) * interval_number :]]
        self.save_list = []
        for chunk in self.chunk_list:
            self.save_list.append(self.analysis_single_chunk(chunk))
        self.save_list = self.sort_dict_list(self.save_list)
        np.save(
            os.path.join(self.save_path, "{}.npy".format(self.dataset_name)),
            self.save_list,
        )
        self.save_list = np.load(
            os.path.join(self.save_path, "{}.npy".format(self.dataset_name)),
            allow_pickle=True,
        )
        self.mean_ranks = np.array(
            list(list(self.save_list[i].values()) for i in range(len(self.save_list)))
        ).transpose()
        self.minum_value = args.minum_value
        self.width = args.width

    def sort_dict_list(self, dict_list, key_order=[5, 0, 1, 2, 3, 4]):
        ordered_dict_list = []
        for d in dict_list:
            ordered_dict = OrderedDict((k, d.get(k, 0)) for k in key_order)
            ordered_dict_list.append(ordered_dict)
        return ordered_dict_list

    def analysis_single_chunk(self, data):
        counter = Counter(data)
        # 计算总元素个数
        total_elements = len(data)
        percentages = {
            key: value / total_elements * 100 for key, value in counter.items()
        }
        for i in range(6):
            if i not in percentages.keys():
                percentages[i] = 0
        return percentages

    def plot(self):
        fig, ax = plt.subplots(figsize=(10, 4))
        bottom = np.zeros_like(self.mean_ranks[0])
        agents = ["Conserverative", "Bear", "Pullback", "Sideways", "Rally", "Bull"]
        colors = [
            "#000000",
            "#410E73",
            "#8C2A81",
            "#DF4A68",
            "#FC9A6B",
            "#FCF8BB",
        ]
        colors.reverse()
        color_idxs = [0, 1, 2, 3, 4, 5]
        DMC_COLOR_DICT = dict(zip(agents, [colors[idx] for idx in color_idxs]))
        labels = list(range(0, len(self.mean_ranks[0])))
        width = self.width
        min_value = self.minum_value
        fig, ax = plt.subplots(figsize=(10, 4))

        bottom = np.zeros_like(self.mean_ranks[0])

        for i, key in enumerate(agents):
            ax.bar(
                labels,
                self.mean_ranks[i],
                width,
                label=key,
                color=DMC_COLOR_DICT[key],
                bottom=bottom,
            )
            bottom += self.mean_ranks[i]

        # Set the aspect of the plot to 'box' to ensure bars start at the boundary
        ax.set_aspect("auto", adjustable="box")

        # Adjust the axis limits to ensure the bars start right at the edge
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.set_ylim(min_value, 100)

        # Set y-axis ticks closer to the plot
        ax.yaxis.set_tick_params(pad=2)

        # Set yticks and xticks
        yticks = np.linspace(min_value, 100, num=6)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{y:.1f}%" for y in yticks], fontsize="large")

        date_labels = [f"Date {i}" for i in range(0, len(labels), 5)]
        ax.set_xticks(range(3, len(labels), 5))
        ax.set_xticklabels(self.mid_points, fontsize=14)

        # Set labels and legend
        ax.set_ylabel("Fraction (in %)", fontsize=16)
        ax.set_xlabel("Date", fontsize=16)

        # Remove spines for a clean look
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        fig.tight_layout()
        fig.legend(ncol=6, fontsize=14, bbox_to_anchor=(0.999, 1.09))
        plt.savefig(
            os.path.join(self.save_path, "{}.pdf".format(self.dataset_name)),
            bbox_inches="tight",
        )


if __name__ == "__main__":
    args = parser.parse_args()
    analyzer = analyzer_router(args)
    analyzer.plot()
