import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"
import sys

sys.path.append(".")

parser = argparse.ArgumentParser()
# replay buffer coffient
parser.add_argument(
    "--dataset_name",
    type=str,
    default="BTCUSDT",
    help="the number of transcation we store in one memory",
)
parser.add_argument(
    "--data_save_path",
    type=str,
    default="dataset",
    help="the number of transcation we store in one memory",
)
parser.add_argument(
    "--action_save_path",
    type=str,
    default="result/DiHFT/final_result",
    help="the number of transcation we store in one memory",
)
parser.add_argument(
    "--plot_save_path",
    type=str,
    default="analysis_result/DiHFT/trade_case",
    help="the number of transcation we store in one memory",
)
parser.add_argument(
    "--selected_start",
    type=int,
    default=500,
    help="the chunk start",
)
parser.add_argument(
    "--selected_end",
    type=int,
    default=700,
    help="the chunk end",
)


class trade_plot:
    def __init__(self, args) -> None:
        self.dataset_name = args.dataset_name
        self.data_save_path = args.data_save_path
        self.action_save_path = args.action_save_path
        self.plot_save_path = args.plot_save_path
        self.test_df = pd.read_feather(
            os.path.join(args.data_save_path, self.dataset_name, "test.feather")
        )
        self.position = np.load(
            os.path.join(
                self.action_save_path, self.dataset_name, "micro_action_history.npy"
            )
        )
        self.price = self.test_df["mark_price"].values
        self.dates = self.test_df["timestamp"].values

        self.start = args.selected_start
        self.end = args.selected_end

        self.position_selected = self.position[self.start : self.end]
        self.price_selected = self.price[self.start : self.end]
        self.dates_selected = self.dates[self.start : self.end]
        self.buy_selected = np.where(np.diff(self.position_selected) > 0)[0]
        self.sell_selected = np.where(np.diff(self.position_selected) < 0)[0]

    def plot_buy_sell(
        self, dates_selected, price_selected, buy_selected, sell_selected
    ):
        plt.figure(figsize=(10, 5))
        plt.subplot(facecolor="#E7E6E6")
        plt.plot(dates_selected, price_selected, label="Price", color="#BF9000")

        plt.scatter(
            dates_selected[buy_selected],
            price_selected[buy_selected],
            label="Buy",
            color="#0ECB81",
            marker="^",
        )
        plt.scatter(
            dates_selected[sell_selected],
            price_selected[sell_selected],
            label="Sell",
            color="#F6465D",
            marker="v",
        )

        plt.gca().xaxis.set_major_locator(
            ticker.MaxNLocator(nbins=4)
        )  # Adjust number of bins as needed
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))

        plt.xlabel("Time (days)")
        plt.ylabel("Price (USD)")
        plt.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.1))
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.savefig(os.path.join(self.plot_save_path, f"{self.dataset_name}.pdf"))

    def plot(self):
        self.plot_buy_sell(
            self.dates_selected,
            self.price_selected,
            self.buy_selected,
            self.sell_selected,
        )


if __name__ == "__main__":
    args = parser.parse_args()
    plotter = trade_plot(args)
    plotter.plot()
