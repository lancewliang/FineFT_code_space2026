import os
import numpy as np
import pandas as pd
import os
import argparse
import shutil
import glob

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_name",
    type=str,
    default="ETHUSDT",
    help="the number of transcation we store in one memory",
)

parser.add_argument(
    "--base_path",
    type=str,
    default="result/base/cdqn_rp",
    help="the number of transcation we store in one memory",
)


def delete_outside_result(epoch_path):
    npy_files = glob.glob(os.path.join(epoch_path, "*.npy"))
    for npy_file in npy_files:
        try:
            os.remove(npy_file)
            print(f"Deleted {npy_file}")
        except OSError as e:
            print(f"Error deleting {npy_file}: {e}")


class deleter:
    def __init__(self, args):
        self.base_path = args.base_path
        self.dataset_name = args.dataset_name
        self.result_path = os.path.join(
            self.base_path,
            self.dataset_name,
            "seed_12345",
        )

    def delete(self, epoch_number):
        epoch_path = os.path.join(self.result_path, epoch_number)
        if os.path.exists(epoch_path):
            delete_outside_result(epoch_path)

    def main(self):
        epoch_list = os.listdir(self.result_path)
        for epoch in epoch_list:
            self.delete(epoch)


if __name__ == "__main__":
    args = parser.parse_args()
    deleter(args).main()
