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
    default="ETHUSDT",
    help="the number of transcation we store in one memory",
)
parser.add_argument(
    "--epoch_number",
    type=int,
    default=50,
    help="the number of transcation we store in one memory",
)
parser.add_argument(
    "--base_path",
    type=str,
    default="result/DiHFT/low_level",
    help="the number of transcation we store in one memory",
)
def delete(result):
    if os.path.isfile(result):
        os.remove(result)
    elif os.path.isdir(result):
        shutil.rmtree(result)
    else:
        print(f"{result} is neither a file nor a directory")

class deleter:
    def __init__(self, args):
        self.base_path = args.base_path
        self.dataset_name = args.dataset_name
        self.epoch_number = args.epoch_number
        self.result_path = os.path.join(
            self.base_path, self.dataset_name, "weights_advantage_pretrain"
        )

    def delete(self, epoch_number):
        epoch_path = os.path.join(self.result_path, "epoch_{}".format(epoch_number))
        # result = os.path.join(epoch_path, "test_dynamics")
        # if os.path.exists(result):
        #     delete(result)
        
        result_average = os.path.join(epoch_path, "analysis_result.npy")
        if os.path.exists(result_average):
            delete(result_average)

    def main(self):
        for i in range(self.epoch_number):
            self.delete(i+1)


if __name__ == "__main__":
    args = parser.parse_args()
    deleter(args).main()


