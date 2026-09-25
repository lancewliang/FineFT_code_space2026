import argparse
import logging
import numpy as np
import sys
import os
import torch

sys.path.append(".")
from merge_vae_train import (
    label_name_from_index,
    materialize_label_training_data,
    vae_data_dir,
)
from process import (
    prepare_model,
    train_test,
    analyze_contract_tests,
    prepare_contract_dataset_loader_list,
)
from common import ArtifactNames, get_vae_label_filename
from RL.DiHFT.VAE.manifests import (
    LabelTrainingManifest,
    TestContractSource,
    TrainBaselineLogpx,
)
from RL.DiHFT.VAE.summary import maybe_write_routing_summary_after_analysis
import RL.DiHFT.VAE.vae as VAEs
from datahandler.vae_dataset import One_Dim_Dataset

logger = logging.getLogger(__name__)


def configure_logger(log_dir: str | None = None, label_index: int = 0) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    has_stream_handler = False
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            handler.setFormatter(formatter)
            has_stream_handler = True

    if not has_stream_handler:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.abspath(
            os.path.join(log_dir, f"train_label_{label_index}.log")
        )
        try:
            stdout_stat = os.fstat(sys.stdout.fileno())
            if os.path.exists(log_file_path):
                file_stat = os.stat(log_file_path)
                if (
                    stdout_stat.st_ino == file_stat.st_ino
                    and stdout_stat.st_dev == file_stat.st_dev
                ):
                    return
        except Exception:
            pass

        for handler in root_logger.handlers:
            if (
                isinstance(handler, logging.FileHandler)
                and handler.baseFilename == log_file_path
            ):
                return

        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


parser = argparse.ArgumentParser(
    description="PyTorch implementation of VAE for fitting 1d data"
)
parser.add_argument(
    "--train",
    action="store_true",
    help="materialize cross-contract training data, train the VAE, and analyze test contracts",
)
parser.add_argument(
    "--analyze-only",
    action="store_true",
    help="load model_latest.pth and analyze test contracts without retraining",
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
parser.add_argument(
    "--labeling_method",
    type=str,
    default="slope",
    help="dynamic labeling method to consume for training data (default: slope)",
)
parser.add_argument(
    "--log_dir",
    type=str,
    default=None,
    help="directory for training log files",
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
    default=256,
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
    default=1500,
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
    type=int,
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
parser.add_argument(
    "--experiment_name",
    type=str,
    default="default",
    help="experiment name used to namespace serial training outputs",
)
def discover_test_sources(data_base_path, dataset_name):
    root = vae_data_dir(data_base_path, dataset_name)
    test_dir = root / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"missing VAE test path: {test_dir}")
    sources = []
    for path in sorted(test_dir.glob("test_*.npy"), key=lambda item: item.name):
        contract = path.stem
        if contract.startswith("test_"):
            contract = contract[len("test_") :]
        sources.append(TestContractSource(contract=contract, source_file=str(path)))
    if not sources:
        raise FileNotFoundError(f"no test_*.npy files found under {test_dir}")
    return sources


class Piplineruner:
    def __init__(self, args):
        self.args = args
        if not self.args.train:
            self.args.batch_size = 1
        label_name = label_name_from_index(self.args.label_index)
        self.label_name = label_name
        self.single_label_save_path = os.path.join(
            args.base_model_path,
            "vae_results",
            self.args.dataset_name,
            self.args.experiment_name,
            label_name,
        )
        self.args.single_label_save_path = self.single_label_save_path
        if self.args.log_dir is not None:
            logger.info(f"Configured log directory: {self.args.log_dir}")
        labeling_method = self.args.labeling_method
        if self.args.train:
            train_manifest = materialize_label_training_data(
                self.args.data_base_path,
                self.args.dataset_name,
                self.args.label_index,
                labeling_method=labeling_method,
            )
            logger.info(
                f"Materialized and using training data file: {train_manifest.merged_path} "
                f"(total_samples={train_manifest.total_samples}, feature_dim={train_manifest.feature_dim})"
            )
            for contract_source in train_manifest.included_contracts:
                logger.info(
                    f"  Source contract array: contract={contract_source.contract}, "
                    f"file={contract_source.source_file}, samples={contract_source.sample_count}"
                )
            if train_manifest.missing_contracts:
                logger.warning(
                    f"  Missing contract arrays for {self.label_name}: {train_manifest.missing_contracts}"
                )
        else:
            train_path = (
                vae_data_dir(self.args.data_base_path, self.args.dataset_name)
                / "train"
                / labeling_method
                / get_vae_label_filename(label_name)
            )
            if not train_path.exists():
                raise FileNotFoundError(f"missing materialized training data: {train_path}")
            train_data = np.load(train_path)
            if train_data.ndim != 2 or train_data.shape[0] == 0:
                raise ValueError(f"invalid materialized training data: {train_path}")
            train_manifest = LabelTrainingManifest(
                dataset_name=self.args.dataset_name,
                label=label_name,
                merged_path=str(train_path),
                total_samples=int(train_data.shape[0]),
                feature_dim=int(train_data.shape[1]),
                included_contracts=[],
                missing_contracts=[],
            )
            logger.info(
                f"Using existing training baseline data file: {train_path} "
                f"(total_samples={train_manifest.total_samples}, feature_dim={train_manifest.feature_dim})"
            )
        self.train_manifest = train_manifest
        logger.info(f"VAE model checkpoint save path: {self.single_label_save_path}")
        train_data_path = train_manifest.merged_path
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
            None,
            hidden_dims,
            z_dim,
            loss,
            learning_rate,
            batch_size,
            epochs,
            log_interval,
            prr,
        )
        self.contract_loader_list = prepare_contract_dataset_loader_list(
            discover_test_sources(self.args.data_base_path, self.args.dataset_name),
            expected_feature_dim=train_manifest.feature_dim,
        )

    def train(self):
        logger.info(
            f"Starting VAE training for {self.label_name} using {self.train_manifest.merged_path} "
            f"on device {self.device} (epochs={self.args.epochs}, batch_size={self.args.batch_size})"
        )
        train_test(
            self.args,
            self.model,
            self.train_loader,
            self.test_loader,
            self.ood_test_loader,
            self.optimizer,
            self.device,
        )

    def analyze_contracts(self):
        model_path = os.path.join(
            self.single_label_save_path,
            ArtifactNames.MODEL_LATEST_PTH,
        )
        logger.info(
            f"Starting contract analysis for {self.label_name} using model {model_path} "
            f"against {len(self.contract_loader_list)} test contracts"
        )
        self.model.load_state_dict(torch.load(model_path))
        train_dataset = One_Dim_Dataset(self.train_manifest.merged_path)
        kwargs = (
            {"num_workers": 1, "pin_memory": True}
            if self.device.type == "cuda"
            else {}
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=False, **kwargs
        )
        _, train_logpx = VAEs.analyze(self.model, train_loader, self.device)
        return analyze_contract_tests(
            pretrained_model_path=model_path,
            model=self.model,
            contract_loader_list=self.contract_loader_list,
            device=self.device,
            save_path=self.args.single_label_save_path,
            dataset_name=self.args.dataset_name,
            label=self.label_name,
            train_baseline=TrainBaselineLogpx(
                source_file=self.train_manifest.merged_path,
                input_samples=self.train_manifest.total_samples,
                analyzed_samples=int(np.asarray(train_logpx).reshape(-1).size),
                logpx=np.asarray(train_logpx, dtype=float),
            ),
        )


if __name__ == "__main__":
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
    args = parser.parse_args()
    configure_logger(args.log_dir, args.label_index)
    if args.train and args.analyze_only:
        parser.error("--train and --analyze-only are mutually exclusive")
    if not args.train and not args.analyze_only:
        parser.error("choose --train or --analyze-only")
    piplinerunner = Piplineruner(args)
    if args.train:
        piplinerunner.train()
    piplinerunner.analyze_contracts()
    maybe_write_routing_summary_after_analysis(args)
