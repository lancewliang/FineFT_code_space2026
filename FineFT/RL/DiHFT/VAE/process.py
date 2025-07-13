import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torch import optim
from torchvision import datasets  # , transforms
import numpy as np
import torchvision
from scipy.stats import gaussian_kde
import tqdm
import sys

sys.path.append(".")
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
from RL.DiHFT.VAE.util import compute_roc_pr_metrics
import RL.DiHFT.VAE.vae as VAEs
from datahandler.vae_dataset import One_Dim_Dataset


def prepare_dataset_loader_list(ood_test_dataset_path_list):
    dataloader_list = []
    for path in ood_test_dataset_path_list:
        ood_test_data = One_Dim_Dataset(path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpus")
        kwargs = {"num_workers": 1, "pin_memory": True} if device == "cuda" else {}
        ood_test_loader = torch.utils.data.DataLoader(
            ood_test_data, batch_size=1, shuffle=False, **kwargs
        )
        dataloader_list.append(ood_test_loader)
    return dataloader_list


def prepare_model(
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
):
    # --- data loading --- #

    train_data = One_Dim_Dataset(train_data_path)
    test_data = One_Dim_Dataset(train_data_path)
    ood_test_data = One_Dim_Dataset(ood_test_dataset_path)
    INPUT_DIM = train_data.input_dim

    print(f"Input_dim: {INPUT_DIM} hidden_dim: {hidden_dims} z_dim: {z_dim}")

    # pin memory provides improved transfer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpus")
    kwargs = {"num_workers": 1, "pin_memory": True} if device == "cuda" else {}

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=True, **kwargs
    )
    ood_test_loader = torch.utils.data.DataLoader(
        ood_test_data, batch_size=batch_size, shuffle=True, **kwargs
    )

    model = VAEs.MLP_VAE(
        INPUT_DIM=INPUT_DIM,
        Z_DIM=z_dim,
        hidden_dims=hidden_dims,
        loss_func=loss,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    return model, optimizer, train_loader, test_loader, ood_test_loader, device


def train_test(
    args, model, train_loader, test_loader, ood_test_loader, optimizer, device
):
    for epoch in range(1, args.epochs + 1):
        VAEs.train(model, train_loader, optimizer, device, epoch, args)
        if epoch % args.log_interval == 0:
            VAEs.test(model, test_loader, ood_test_loader, device, epoch, args)


def analyze(
    pretrained_model_path,
    model,
    train_loader,
    test_loader,
    ood_test_loader,
    target_trading_pair,
    device,
    save_path,
):
    print("Start analyzing...")
    model.load_state_dict(torch.load(pretrained_model_path))
    train_mus, train_logpx = VAEs.analyze(model, train_loader, device)
    id_mus, id_logpx = VAEs.analyze(model, test_loader, device)
    ood_mus, ood_logpx = VAEs.analyze(model, ood_test_loader, device)

    reference_scores = id_logpx
    test_scores = ood_logpx
    np.save(
        os.path.join(save_path, "id_logpx.npy"),
        id_logpx,
    )
    np.save(
        os.path.join(save_path, "ood_logpx.npy"),
        ood_logpx,
    )

    bigger_is_id = 1
    # compute metrics
    # check bigger should be id or ood?
    min_len = min(len(reference_scores), len(test_scores))
    y_true = np.array([*[bigger_is_id] * min_len, *[1 - bigger_is_id] * min_len])
    y_score = np.concatenate([reference_scores[:min_len], test_scores[:min_len]])

    (
        (roc_auc, fpr, tpr, thresholds),
        (pr_auc, precision, recall, thresholds),
        fpr80,
    ) = compute_roc_pr_metrics(y_true=y_true, y_score=y_score, reference_class=0)

    results = dict(
        roc=dict(roc_auc=roc_auc, fpr=fpr, tpr=tpr, thresholds=thresholds),
        pr=dict(
            pr_auc=pr_auc, precision=precision, recall=recall, thresholds=thresholds
        ),
        fpr80=fpr80,
    )

    print(f" AUROC={roc_auc:6.4f}, AUPRC={pr_auc:6.4f}, FPR80={fpr80:6.4f}\n")
    return results


def cross_analyze(
    pretrained_model_path,
    model,
    name_list,
    test_loader_list,
    device,
    save_path,
):
    assert len(name_list) == len(test_loader_list)
    print("Start analyzing...")
    model.load_state_dict(torch.load(pretrained_model_path))
    for name, test_loader in zip(name_list, test_loader_list):
        ood_mus, ood_logpx = VAEs.analyze(model, test_loader, device)
        np.save(
            os.path.join(save_path, "ood_logpx_{}.npy".format(name)),
            ood_logpx,
        )
