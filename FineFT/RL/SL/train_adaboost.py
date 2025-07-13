import sys
import os
import argparse
import joblib

sys.path.append(".")
from datahandler.create_data_adaboost import SL_data
import numpy as np

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"

parser = argparse.ArgumentParser()


parser.add_argument(
    "--data_path",
    type=str,
    default="dataset",
    help="the number of transcation we store in one memory",
)
parser.add_argument(
    "--dataset_name",
    type=str,
    default="BNBUSDT",
    help="the number of transcation we store in one memory",
)
parser.add_argument(
    "--result_save_path",
    type=str,
    default="result/SL",
    help="the number of transcation we store in one memory",
)


class adaboost_trainer:
    def __init__(self, args) -> None:
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.X_train = np.load(
            os.path.join(self.data_path, self.dataset_name, "SL_data", "X_train.npy")
        )
        self.y_train = np.load(
            os.path.join(self.data_path, self.dataset_name, "SL_data", "y_train.npy")
        )

        self.X_valid = np.load(
            os.path.join(self.data_path, self.dataset_name, "SL_data", "X_valid.npy")
        )
        self.y_valid = np.load(
            os.path.join(self.data_path, self.dataset_name, "SL_data", "y_valid.npy")
        )

        self.X_test = np.load(
            os.path.join(self.data_path, self.dataset_name, "SL_data", "X_test.npy")
        )
        self.y_test = np.load(
            os.path.join(self.data_path, self.dataset_name, "SL_data", "y_test.npy")
        )
        self.base_estimator = DecisionTreeRegressor(max_depth=4)
        self.adaboost_model = AdaBoostRegressor(
            estimator=self.base_estimator, n_estimators=100, learning_rate=0.1
        )
        self.result_save_path = os.path.join(args.result_save_path, self.dataset_name)
        os.makedirs(self.result_save_path, exist_ok=True)

    def train(self):
        # 训练模型
        print("training model...")
        self.adaboost_model.fit(self.X_train, self.y_train)
        joblib.dump(
            self.adaboost_model, os.path.join(self.result_save_path, "model.pkl")
        )


if __name__ == "__main__":
    args = parser.parse_args()
    trainer = adaboost_trainer(args)
    trainer.train()
