import abc
from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


class EncoderDecoderModel(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.epoch_loss_dict = {
            "train_loss_epoch_ave": [],
            "valid_loss_epoch_ave": [],
        }

    def encode(self, tensor):
        raise NotImplementedError

    def decode(self, tensor):
        raise NotImplementedError

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def configure_optimizers(self):
        if self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        elif self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            raise NotImplementedError
        return optimizer

    def sample_neuron_firings(self, dataloader: torch.utils.data.DataLoader):
        """
        入力データに対する中間ニューロンの発火状況を numpy.array で取得. 中間ニューロンは, model.encode メソッドで
        得られることを仮定しています.

        Args:
            dataloader (torch.utils.data.DataLoader): Input dataloader.

        Returns:
            np.array: Neuron firing matrix, shape of (#data, vec_size).
        """
        firings = []
        for data, target in dataloader:
            vv = self.encode(data.view(data.size(0), -1)).detach()
            firings.append(vv)
        firings = torch.cat(firings).numpy()
        return firings

    def training_epoch_end(self, training_step_outputs):
        epoch_train_loss = torch.stack([x["loss"] for x in training_step_outputs]).mean()
        self.epoch_loss_dict["train_loss_epoch_ave"].append(epoch_train_loss)


class LitVanillaVAE(EncoderDecoderModel):
    """
    Original Paper: "Auto-Encoding Variational Bayes" Diederik et. al., 2014 https://arxiv.org/abs/1312.6114
    Implemetnation reference: https://github.com/pytorch/examples/tree/main/vae
    """

    def __init__(self, n_vis, n_hid=100, optimizer_name="sgd", lr=0.01):
        super().__init__()
        self.n_vis = n_vis
        self.n_hid = n_hid
        self.encoder_mean = nn.Sequential(nn.Linear(n_vis, n_hid), nn.ReLU())

        # https://stackoverflow.com/questions/49634488/keras-variational-autoencoder-nan-loss
        self.encoder_var = nn.Sequential(
            OrderedDict(
                [
                    ("enc_var_fc1", nn.Linear(n_vis, n_hid)),
                    ("enc_var_relu1", nn.ReLU()),
                ]
            )
        )
        torch.nn.init.zeros_(self.encoder_var.enc_var_fc1.weight)

        self.decoder = nn.Sequential(nn.Linear(n_hid, n_vis), nn.Sigmoid())
        self.optimizer_name = optimizer_name
        self.lr = lr

    def _sample_hidden(self, mean, log_var):
        epsilon = torch.randn(mean.shape).to(mean.device)
        # Next code sometimes makes NaN for large log_var,
        # so weights of `self.encoder_var` are initialized to zero.
        h = mean + epsilon * torch.exp(0.5 * log_var)
        return h

    def encode(self, v):
        mean, log_var = self.encoder_mean(v), self.encoder_var(v)
        h = self._sample_hidden(mean, log_var)
        return h

    def decode(self, h):
        return self.decoder(h)

    def train_loss(self, tensor):
        batch_size = tensor.size(0)
        x = tensor.view(batch_size, -1)
        mean, log_var = self.encoder_mean(x), self.encoder_var(x)
        delta = 1e-8
        KL = 0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))

        z = self._sample_hidden(mean, log_var)
        x_hat = self.decode(z)
        reconstruction = torch.sum(
            x * torch.log(x_hat + delta) + (1 - x) * torch.log(1 - x_hat + delta)
        )
        lower_bound = KL + reconstruction
        # VAE loss does not divide by batch: https://github.com/pytorch/examples/issues/652
        # When average: https://github.com/pytorch/examples/issues/234
        return -lower_bound

    def training_step(self, batch, batch_idx):
        tensor, label = batch
        loss = self.train_loss(tensor)
        recon_loss = self.recon_loss(tensor)
        self.log("train_loss", loss)
        self.log("recon_loss", recon_loss)

        return {"loss": loss, "recon_loss": loss}

    def recon_loss(self, tensor):
        batch_size = tensor.size(0)
        x = tensor.view(batch_size, -1)
        recon = self.forward(x)
        return F.mse_loss(x, recon) / batch_size


class LitScaleFreeVAE(LitVanillaVAE):
    def __init__(self, n_vis, n_hid=100, optimizer_name="sgd", lr=0.01, power_law_gamma=1):
        super().__init__(n_vis, n_hid, optimizer_name, lr)
        self.var_decay_1d = torch.tensor([n ** (-power_law_gamma) for n in range(1, n_hid + 1)])

    def _sample_hidden(self, mean, log_var):
        batch_size = mean.size(0)
        var_decay = self.var_decay_1d.repeat(batch_size, 1)
        epsilon = torch.mul(torch.randn(mean.shape), var_decay).to(mean.device)
        h = mean + epsilon * torch.exp(0.5 * log_var)
        return h


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, n_vis, n_hid=100, optimizer_name="sgd", lr=0.01):
        super().__init__()
        self.n_vis = n_vis
        self.n_hid = n_hid
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.encoder = nn.Sequential(nn.Linear(n_vis, n_hid), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(n_hid, n_vis), nn.Sigmoid())

    def encode(self, v):
        return self.encoder(v)

    def decode(self, h):
        return self.decoder(h)

    def train_loss(self, tensor):
        batch = tensor.size(0)
        x = tensor.view(batch, -1)
        recon = self.forward(x)
        return F.mse_loss(x, recon)

    def training_step(self, batch, batch_idx):
        tensor, label = batch
        loss = self.train_loss(tensor)
        self.log("train_loss", loss)
        return loss
