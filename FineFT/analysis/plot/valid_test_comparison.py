# plot data contrast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import os
from matplotlib.ticker import ScalarFormatter
import argparse

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"

parser = argparse.ArgumentParser()
# replay buffer coffient
parser.add_argument(
    "--dataset_root_path",
    type=str,
    default="dataset",
    help="the root path of the dataset",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="analysis_result/motivation/dataset",
    help="the root path of the dataset",
)


class Dataset_plotter:
    def __init__(self, args):
        self.dataset_root_path = args.dataset_root_path
        self.valid_test_result_dict = {}
        self.train_result_dict = {}
        self.color_list = ["#f74d4d", "#2455a4"]
        self.dataset_name_list = os.listdir(self.dataset_root_path)
        self.save_path = args.save_path

    def load_single_dataset_valid_test(self, dataset_path):
        result_dict = {}
        valid_data_path = os.path.join(dataset_path, "valid.feather")
        valid_data = pd.read_feather(valid_data_path)
        valid_buy_hold_df_net_curve = (
            np.array(valid_data.mark_price / (valid_data.mark_price.iloc[0])) - 1
        )
        test_data_path = os.path.join(dataset_path, "test.feather")
        test_data = pd.read_feather(test_data_path)
        test_buy_hold_df_net_curve = (
            np.array(test_data.mark_price / (test_data.mark_price.iloc[0])) - 1
        )
        result_dict["Valid"] = valid_buy_hold_df_net_curve
        result_dict["Test"] = test_buy_hold_df_net_curve
        return result_dict

    def load_single_train_dataset(self, dataset_path):
        result_dict = {}
        train_data_path = os.path.join(dataset_path, "train.feather")
        train_data = pd.read_feather(train_data_path)
        train_buy_hold_df_net_curve = (
            np.array(train_data.mark_price / (train_data.mark_price.iloc[0])) - 1
        )

        result_dict["Train"] = train_buy_hold_df_net_curve
        return result_dict

    def load_all_datasets(self):
        for dataset in os.listdir(self.dataset_root_path):
            dataset_path = os.path.join(self.dataset_root_path, dataset)
            self.valid_test_result_dict[dataset] = self.load_single_dataset_valid_test(
                dataset_path
            )
            self.train_result_dict[dataset] = self.load_single_train_dataset(
                dataset_path
            )
        return self.valid_test_result_dict

    def plot(self,all_data_result_dict,name):
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        for i in range(2):
            for j in range(2):
                dataset_name = self.dataset_name_list[i * 2 + j]
                result_dict = all_data_result_dict[dataset_name]
                for k, key in enumerate(result_dict.keys()):
                    if i + j == 0:
                        axs[i, j].plot(
                            range(len(result_dict[key])),
                            result_dict[key] * 100,
                            color=self.color_list[k],
                            label=key,
                            linewidth=2,
                        )
                    else:
                        axs[i, j].plot(
                            range(len(result_dict[key])),
                            result_dict[key] * 100,
                            color=self.color_list[k],
                            linewidth=2,
                        )
                    axs[i, j].grid(ls="--")
        for ax in axs.flat:
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-1, 1))
            ax.xaxis.set_major_formatter(formatter)
        titles = [
            "(a) {}".format(self.dataset_name_list[0]),
            "(b) {}".format(self.dataset_name_list[1]),
            "(c) {}".format(self.dataset_name_list[2]),
            "(d) {}".format(self.dataset_name_list[3]),
        ]
        for ax, title in zip(axs.flat, titles):
            ax.annotate(
                title,
                xy=(0.5, -0.3),
                xycoords="axes fraction",
                ha="center",
                va="center",
                fontsize=18,
            )
        fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=6, fontsize=15)
        plt.subplots_adjust(hspace=0.3)
        # 添加标签
        for i in range(2):
            for j in range(2):
                axs[i, j].set_xlabel("Trading Timestamp", fontsize=21)
                axs[i, j].set_ylabel("Total Return(%)", fontsize=21)
        plt.tight_layout(pad=1.5, rect=[0, 0, 1, 0.94])
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        plt.savefig(os.path.join(self.save_path, "{}_comparison.pdf".format(name)))


if __name__ == "__main__":
    args = parser.parse_args()
    dataset_plotter = Dataset_plotter(args)
    dataset_plotter.load_all_datasets()
    dataset_plotter.plot(dataset_plotter.valid_test_result_dict,'valid_and_test')
    dataset_plotter.plot(dataset_plotter.train_result_dict,'train')
    print("plot done")
