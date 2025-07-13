import os
import numpy as np
import pandas as pd
import os
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_name",
    type=str,
    default="DOTUSDT",
    help="the number of transcation we store in one memory",
)

parser.add_argument(
    "--base_path",
    type=str,
    default="result/DiHFT/high_level",
    help="the number of transcation we store in one memory",
)


def delete(result):
    if os.path.isfile(result):
        os.remove(result)
        print(f"File {result} has been deleted.")
    elif os.path.isdir(result):
        if not os.listdir(result):  # 检查文件夹是否为空
            shutil.rmtree(result)
            print(f"Empty directory {result} has been deleted.")
        else:
            print(f"Directory {result} is not empty, not deleting.")
    else:
        print(f"{result} is neither a file nor a directory.")


class deleter:
    def __init__(self, args):
        self.base_path = args.base_path
        self.dataset_name = args.dataset_name
        self.result_path = os.path.join(
            self.base_path, self.dataset_name, "vae_risk_aware_routing"
        )

    def delete(self, params):
        para_path = os.path.join(self.result_path, params)
        if os.path.exists(para_path):
            delete(para_path)

    def main(self):
        para_list = os.listdir(self.result_path)
        for paras in para_list:
            self.delete(paras)


if __name__ == "__main__":
    args = parser.parse_args()
    deleter(args).main()
