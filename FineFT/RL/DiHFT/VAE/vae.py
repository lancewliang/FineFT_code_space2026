"""
Reference: https://github.com/pytorch/examples/blob/master/vae/main.py,
           https://github.com/hwalsuklee/tensorflow-mnist-VAE
"""

import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
import numpy as np
import sys
import pandas as pd

sys.path.append(".")
from RL.DiHFT.VAE.util import ensure_tensor_2d


# --- some utils ----------- #
def gaussian_nll(mu, log_sigma, x):
    return (
        0.5 * torch.pow((x - mu) / log_sigma.exp(), 2)
        + log_sigma
        + 0.5 * np.log(2 * np.pi)
    )


def softclip(tensor, min):
    """Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials"""
    result_tensor = min + F.softplus(tensor - min)
    return result_tensor


# --- defines the model and the optimizer --- #
class MLP_VAE(nn.Module):
    def __init__(
        self, INPUT_DIM=784, Z_DIM=10, hidden_dims=[500, 200], loss_func="NLL"
    ):
        super().__init__()
        self.INPUT_DIM = INPUT_DIM

        # encoder
        self.fc1 = nn.Linear(INPUT_DIM, hidden_dims[0])
        self.fc_enc = nn.ModuleList()
        last_h_dim = hidden_dims[0]
        for dim in hidden_dims[1:]:
            self.fc_enc.append(nn.Linear(last_h_dim, dim))
            last_h_dim = dim

        self.fc21 = nn.Linear(last_h_dim, Z_DIM)  # fc21 for mean of Z
        self.fc22 = nn.Linear(last_h_dim, Z_DIM)  # fc22 for log variance of Z

        # decoder
        self.fc3 = nn.Linear(Z_DIM, last_h_dim)

        self.fc_dec = nn.ModuleList()
        for dim in reversed(hidden_dims[:-1]):
            self.fc_dec.append(nn.Linear(last_h_dim, dim))
            last_h_dim = dim
        self.fc4 = nn.Linear(last_h_dim, INPUT_DIM)
        self.fc5 = nn.Linear(last_h_dim, INPUT_DIM)
        self.log_sigma = 0.0
        self.loss_func = loss_func
        # self.log_sigma = torch.nn.Parameter(torch.full((1,), 0, dtype=torch.float)[0])

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        for fc in self.fc_enc:
            h1 = F.relu(fc(h1))
        mu = self.fc21(h1)
        # I guess the reason for using logvar instead of std or var is that
        # the output of fc22 can be negative value (std and var should be positive)
        logvar = self.fc22(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        for dc in self.fc_dec:
            h3 = F.relu(dc(h3))
        if self.loss_func == "BCE":
            recon_mu = torch.sigmoid(self.fc4(h3))
        else:
            recon_mu = self.fc4(h3)
        recon_logsigma = self.fc5(h3)
        return recon_mu, recon_logsigma

    def forward(self, x):
        # x: [batch size, 1, 28,28] -> x: [batch size, 784]
        x = x.view(-1, self.INPUT_DIM)
        mu, logvar = self.encode(x)
        # recon_mu, recon_logsigma = self.decode(mu)
        z = self.reparameterize(mu, logvar)
        recon_mu, recon_logvar = self.decode(z)
        return recon_mu, recon_logvar, mu, logvar

    # --- defines the loss function --- #
    def loss_function(self, recon_mu, recon_logvar, x, mu, logvar):
        if self.loss_func == "BCE":
            rec = F.binary_cross_entropy(
                recon_mu, x.view(-1, self.INPUT_DIM), reduction="sum"
            )
        if self.loss_func == "NLL":
            # set recon_logvar learnable but clip it to avoid NaN
            recon_logvar = softclip(recon_logvar, -6.0)
            recon_logvar = -softclip(-recon_logvar, 0.0)

            # set recon_logvar non-learnable, i.e., recon_logvar = 0
            # recon_logvar = torch.zeros_like(recon_logvar, device=recon_logvar.device)

            rec = gaussian_nll(
                recon_mu, 0.5 * recon_logvar, x.view(-1, self.INPUT_DIM)
            ).sum()

        KLD = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)

        return rec + KLD

    def estimate_log_px(self, data):
        with torch.no_grad():
            recon_mu, recon_logsigma, mu, logvar = self.forward(data)
            cur_loss = self.loss_function(
                recon_mu, recon_logsigma, data, mu, logvar
            ).item()

        return -cur_loss


# --- train and test --- #
def train(model, train_loader, optimizer, device, epoch, args):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        if len(data) == 2:
            # data: (x, label)  we don't need the label even if it has
            data = data[0].to(device)
        else:
            # for 1D dataset
            data = data.to(device)

        recon_mu, recon_logsigma, mu, logvar = model(data)

        loss = model.loss_function(recon_mu, recon_logsigma, data, mu, logvar)
        loss.backward()
        cur_loss = loss.item()
        train_loss += cur_loss
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    cur_loss / len(data),
                )
            )

    if epoch % args.save_interval == 0:
        if not os.path.exists(args.single_label_save_path):
            os.makedirs(args.single_label_save_path)
        torch.save(
            model.state_dict(),
            os.path.join(
                args.single_label_save_path,
                str(epoch) + ".pth",
            ),
        )
        torch.save(
            model.state_dict(),
            os.path.join(
                args.single_label_save_path,
                "model_latest.pth",
            ),
        )
    print(
        "====> Epoch: {} Average loss: {:.4f}".format(
            epoch, train_loss / len(train_loader.dataset)
        )
    )


def test(model, test_loader, ood_test_loader, device, epoch, args):
    model.eval()
    test_loss = 0
    ood_test_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            if len(data) == 2:
                # data: (x, label)  we don't need the label even if it has
                data = data[0].to(device)
            else:
                # for 1D dataset
                data = data.to(device)
            recon_mu, recon_logsigma, mu, logvar = model(data)
            cur_loss = model.loss_function(
                recon_mu, recon_logsigma, data, mu, logvar
            ).item()
            test_loss += cur_loss

        for batch_idx, data in enumerate(ood_test_loader):
            if len(data) == 2:
                # data: (x, label)  we don't need the label even if it has
                data = data[0].to(device)
            else:
                # for 1D dataset
                data = data.to(device)
            recon_mu, recon_logsigma, mu, logvar = model(data)
            cur_loss = model.loss_function(
                recon_mu, recon_logsigma, data, mu, logvar
            ).item()
            ood_test_loss += cur_loss

    test_loss /= len(test_loader.dataset)
    ood_test_loss /= len(ood_test_loader.dataset)
    print(
        "====> Test set loss: ID {:.4f} / OOD {:.4f} ".format(test_loss, ood_test_loss)
    )


def analyze_single_sample(model: MLP_VAE, data, device):
    model.eval()
    data = ensure_tensor_2d(data)
    data = data.float()
    with torch.no_grad():
        data = data.to(device)
        recon_mu, recon_logsigma, mu, logvar = model(data)
        loss = model.loss_function(recon_mu, recon_logsigma, data, mu, logvar)
    loss = -loss.cpu().numpy()
    z_mu = mu.cpu().numpy()
    return z_mu, loss


def analyze_batch_sample(model: MLP_VAE, data, device, id_logp, lower_quantile=1):
    reject_theshold = np.percentile(id_logp, lower_quantile)
    loss_list = []
    for single_data in data:
        z_mu, loss = analyze_single_sample(model, single_data, device)
        loss_list.append(loss)
    reject_rate = np.sum(np.array(loss_list) < reject_theshold) / len(loss_list)
    return np.array(loss_list), reject_rate


def analyze(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        z_mus, x_logpx = [], []
        train_mus, train_logpx = [], []
        id_mus, id_logpx = [], []
        ood_mus, ood_logpx = [], []
        for batch_idx, data in enumerate(data_loader):
            if len(data) == 2:
                # data: (x, label)  we don't need the label even if it has
                data = data[0].to(device)
            else:
                # for 1D dataset
                data = data.to(device)
            recon_mu, recon_logsigma, mu, logvar = model(data)
            loss = model.loss_function(recon_mu, recon_logsigma, data, mu, logvar)
            z_mus.append(mu.cpu().numpy())
            x_logpx.append(-loss.cpu().numpy())
            if batch_idx == len(data_loader) - 2:
                break

    z_mus = np.stack(z_mus).reshape(-1, mu.shape[-1])
    x_logpx = np.stack(x_logpx).reshape(-1, 1)

    return z_mus, x_logpx
