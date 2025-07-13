import argparse
import numpy as np
import sys
import os

sys.path.append(".")
from process import (
    prepare_model,
    train_test,
    analyze,
    cross_analyze,
    prepare_dataset_loader_list,
)

parser = argparse.ArgumentParser(
    description="PyTorch implementation of VAE for fitting 1d data"
)
# dataset
parser.add_argument(
    "--if_train",
    type=bool,
    default=False,
    help="where to load the id data",
)
parser.add_argument(
    "--if_cross_analyze",
    type=bool,
    default=True,
    help="where to load the id data",
)
parser.add_argument(
    "--dataset_name",
    type=str,
    default="BTCUSDT",
    help="where to load the id data",
)
parser.add_argument(
    "--data_base_path",
    type=str,
    default="dataset",
    help="where to load the id data",
)

parser.add_argument(
    "--label_index",
    type=int,
    default=0,
    help="where to load the id data",
)
parser.add_argument(
    "--total_label_number",
    type=int,
    default=5,
    help="where to load the id data",
)
# log
parser.add_argument(
    "--base_model_path",
    type=str,
    default="result/DiHFT",
    help="where to load the id data",
)
# vae setting
parser.add_argument(
    "--z_dim", type=int, default=512, help="dimension of hidden variable Z"
)
parser.add_argument(
    "--hidden_dims",
    type=list,
    default=[4096, 2048, 1024, 1024],
    help="dimension of each hidden layers",
)
parser.add_argument(
    "--sample_ratio",
    type=float,
    default=0.2,
    help="how big ratio of the dataset to sample.",
)
# general trainining setting
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    help="batch size for training (default: 128)",
)
parser.add_argument(
    "--loss",
    type=str,
    default="NLL",
    help="BCE | NLL : Loss function for computing the likelihood",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=2000,
    help="number of epochs to train (default: 20)",
)
parser.add_argument(
    "--log_interval",
    type=int,
    default=100,
    help="interval between logs about training status (default: 100)",
)
parser.add_argument(
    "--save_interval",
    type=str,
    default=50,
    # default=1,
    help="interval for saving the checkpoints",
)
parser.add_argument(
    "--learning_rate",
    type=int,
    default=1e-5,
    help="learning rate for Adam optimizer (default: 1e-3)",
)

# No need to look at the following values unless you use FMNIST or MNIST for debug
parser.add_argument(
    "--prr",
    type=bool,
    default=False,
    help="whether plot the interpolation results for 2D image data",
)
parser.add_argument(
    "--prr-z1-range",
    type=int,
    default=2,
    help="z1 range for plot-reproduce-result (default: 2)",
)
parser.add_argument(
    "--prr-z2-range",
    type=int,
    default=2,
    help="z2 range for plot-reproduce-result (default: 2)",
)
parser.add_argument(
    "--prr-z1-interval",
    type=int,
    default=0.2,
    help="interval of z1 for plot-reproduce-result (default: 0.2)",
)
parser.add_argument(
    "--prr-z2-interval",
    type=int,
    default=0.2,
    help="interval of z2 for plot-reproduce-result (default: 0.2)",
)


class Piplineruner:
    def __init__(self, args):
        self.args = args
        if not self.args.if_train:
            self.args.batch_size = 1
        label_name = "label_{}".format(self.args.label_index)
        self.single_label_save_path = os.path.join(
            args.base_model_path,
            "vae_results",
            self.args.dataset_name,
            label_name,
        )
        self.args.single_label_save_path = self.single_label_save_path
        train_data_path = os.path.join(
            self.args.data_base_path,
            self.args.dataset_name,
            "VAE_data",
            "{}.npy".format(label_name),
        )
        ood_test_dataset_path = os.path.join(
            self.args.data_base_path, self.args.dataset_name, "VAE_data", "test.npy"
        )
        hidden_dims = self.args.hidden_dims
        z_dim = self.args.z_dim
        loss = self.args.loss
        learning_rate = self.args.learning_rate
        batch_size = self.args.batch_size
        epochs = self.args.epochs
        log_interval = self.args.log_interval
        prr = self.args.prr
        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.test_loader,
            self.ood_test_loader,
            self.device,
        ) = prepare_model(
            train_data_path,
            ood_test_dataset_path,
            hidden_dims,
            z_dim,
            loss,
            learning_rate,
            batch_size,
            epochs,
            log_interval,
            prr,
        )

    def train(self):
        train_test(
            self.args,
            self.model,
            self.train_loader,
            self.test_loader,
            self.ood_test_loader,
            self.optimizer,
            self.device,
        )

    def analyze_test(self):
        analyze(
            pretrained_model_path=os.path.join(
                self.single_label_save_path,
                "model_latest.pth",
            ),
            model=self.model,
            train_loader=self.train_loader,
            test_loader=self.test_loader,
            ood_test_loader=self.ood_test_loader,
            target_trading_pair=self.args.dataset_name,
            device=self.device,
            save_path=self.args.single_label_save_path,
        )

    def analyze_cross_test(self):
        label_path_list = []
        for i in range(self.args.total_label_number):
            label_name = "label_{}".format(self.args.label_index)
            label_save_path = os.path.join(
                self.args.data_base_path,
                self.args.dataset_name,
                "VAE_data",
                "{}.npy".format(label_name),
            )
            label_path_list.append(label_save_path)
        dataloader_list = prepare_dataset_loader_list(label_path_list)
        cross_analyze(
            pretrained_model_path=os.path.join(
                self.single_label_save_path,
                "model_latest.pth",
            ),
            model=self.model,
            name_list=range(self.args.total_label_number),
            test_loader_list=dataloader_list,
            device=self.device,
            save_path=self.args.single_label_save_path,
        )


if __name__ == "__main__":
    args = parser.parse_args()
    print()
    if args.if_train:
        piplinerunner = Piplineruner(args)
        piplinerunner.train()
        args.if_train = False
        piplinerunner = Piplineruner(args)
        piplinerunner.analyze_test()
    elif args.if_cross_analyze:
        piplinerunner = Piplineruner(args)
        piplinerunner.analyze_cross_test()

    elif not args.if_cross_analyze:
        piplinerunner = Piplineruner(args)
        piplinerunner.analyze_test()
